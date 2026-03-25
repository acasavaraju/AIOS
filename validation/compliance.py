#!/usr/bin/env python3
"""
AIOS Validation — Model Contract Compliance Checker

Checks a model against all five AIOS Model Contract requirements (R1-R5)
and reports the compliance tier.

Usage:
    python compliance.py --model path/to/model.gguf --output results/compliance.json

See spec/model_contract.md for the full specification.
Tracks: GitHub Issues #5, #6, #7
"""

import argparse
import json
import struct
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np


# ── Contract requirements ─────────────────────────────────────────────────────
COMPLIANT_ACTIVATIONS = {"relu", "relu2", "leaky_relu"}
NON_COMPLIANT_ACTIVATIONS = {"silu", "swish", "gelu"}
MAX_KV_MB_AT_4K = 64.0          # MB — R2 target
MIN_SPARSITY_CONSERVATIVE = 0.70 # R1 target
MAX_ACTIVATION_KB_PER_LAYER = 200 # R4 target (KB, at batch=1 single token)


@dataclass
class RequirementResult:
    requirement: str       # R1, R2, R3, R4, R5
    name: str
    passed: bool
    measured_value: Optional[float]
    target_value: Optional[float]
    unit: str
    evidence: str
    note: str


@dataclass
class ComplianceReport:
    model_path: str
    model_size_gb: float
    architecture: dict
    requirements: list
    tier: str              # Compatible | Optimized | Native | Native+
    tier_explanation: str
    requirements_met: dict
    notes: list


def parse_gguf_metadata(model_path: Path) -> dict:
    """Parse GGUF header for architecture metadata."""
    meta = {}
    try:
        with open(model_path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return {"error": "Not a valid GGUF file"}

            version = struct.unpack("<I", f.read(4))[0]
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]

            meta["gguf_version"] = version
            meta["tensor_count"] = tensor_count

            for _ in range(min(kv_count, 300)):
                try:
                    key_len = struct.unpack("<Q", f.read(8))[0]
                    if key_len > 512:
                        break
                    key = f.read(key_len).decode("utf-8", errors="ignore")
                    val_type = struct.unpack("<I", f.read(4))[0]

                    if val_type == 8:    # uint32
                        val = struct.unpack("<I", f.read(4))[0]
                        meta[key] = val
                    elif val_type == 10: # uint64
                        val = struct.unpack("<Q", f.read(8))[0]
                        meta[key] = val
                    elif val_type == 4:  # string
                        slen = struct.unpack("<Q", f.read(8))[0]
                        if slen > 2048:
                            break
                        val = f.read(slen).decode("utf-8", errors="ignore")
                        meta[key] = val
                    elif val_type == 6:  # float32
                        val = struct.unpack("<f", f.read(4))[0]
                        meta[key] = val
                    elif val_type == 9:  # int32
                        val = struct.unpack("<i", f.read(4))[0]
                        meta[key] = val
                    else:
                        break
                except Exception:
                    break
    except Exception as e:
        meta["parse_error"] = str(e)

    return meta


def extract_architecture(meta: dict) -> dict:
    """Extract normalized architecture parameters from raw metadata."""
    arch = {}

    for key, val in meta.items():
        lkey = key.lower()
        if "block_count" in lkey:
            arch["num_layers"] = val
        elif "embedding_length" in lkey:
            arch["hidden_size"] = val
        elif "feed_forward_length" in lkey:
            arch["ffn_intermediate"] = val
        elif "head_count_kv" in lkey:
            arch["kv_heads"] = val
        elif "head_count" in lkey and "kv" not in lkey:
            arch["attention_heads"] = val
        elif "head_dim" in lkey:
            arch["head_dim"] = val
        elif "architecture" in lkey and isinstance(val, str):
            arch["arch_name"] = val
        elif "activation" in lkey and isinstance(val, str):
            arch["activation"] = val.lower()

    # Derived: head_dim if not explicit
    if "head_dim" not in arch and "hidden_size" in arch and "attention_heads" in arch:
        if arch["attention_heads"] > 0:
            arch["head_dim"] = arch["hidden_size"] // arch["attention_heads"]

    return arch


def check_r1(arch: dict) -> RequirementResult:
    """R1: ReLU or structural sparse activation."""
    activation = arch.get("activation", "unknown").lower()

    if activation in COMPLIANT_ACTIVATIONS:
        passed = True
        evidence = f"Activation function detected: {activation}"
        note = "Fully compliant. Static sparsity map viable."
    elif activation in NON_COMPLIANT_ACTIVATIONS:
        passed = False
        evidence = f"Activation function detected: {activation}"
        note = (f"{activation.upper()} is smooth — static sparsity map provides limited benefit. "
                f"Consider relufication (~3-5% of training cost achieves ~95% sparsity).")
    else:
        passed = False
        evidence = f"Activation not detected in GGUF metadata (value: '{activation}')"
        note = ("Could not determine activation function. "
                "Inspect model card. Manual verification needed.")

    return RequirementResult(
        requirement="R1",
        name="ReLU or Structural Sparse Activation",
        passed=passed,
        measured_value=None,
        target_value=None,
        unit="activation_type",
        evidence=evidence,
        note=note,
    )


def check_r2(arch: dict) -> RequirementResult:
    """R2: Bounded KV cache — MQA or GQA <= 2 heads."""
    kv_heads = arch.get("kv_heads", None)
    head_dim  = arch.get("head_dim", 64)
    num_layers = arch.get("num_layers", 32)

    if kv_heads is None:
        return RequirementResult(
            requirement="R2",
            name="Bounded KV: MQA or GQA <= 2 Heads",
            passed=False,
            measured_value=None,
            target_value=MAX_KV_MB_AT_4K,
            unit="MB at 4K context",
            evidence="KV head count not found in GGUF metadata",
            note="Manually verify attention architecture from model card.",
        )

    context_4k = 4096
    kv_bytes = num_layers * kv_heads * head_dim * context_4k * 4  # L * H * d * C * 4bytes
    kv_mb = kv_bytes / (1024 * 1024)

    passed = kv_mb <= MAX_KV_MB_AT_4K and kv_heads <= 2

    if kv_heads == 1:
        evidence = f"MQA detected (1 KV head). KV cache at 4K context: {kv_mb:.1f} MB"
        note = "Fully compliant. KV cache fits in L3 for standard workloads."
    elif kv_heads <= 2:
        evidence = f"GQA with {kv_heads} KV heads. KV cache at 4K context: {kv_mb:.1f} MB"
        note = f"Compliant. KV cache {kv_mb:.0f} MB at 4K — near L3 boundary."
    else:
        evidence = f"{kv_heads} KV heads. KV cache at 4K context: {kv_mb:.1f} MB (target: <= {MAX_KV_MB_AT_4K} MB)"
        note = (f"Non-compliant. KV cache {kv_mb:.0f} MB at 4K context requires DRAM reads. "
                f"MQA would reduce this to {kv_mb/kv_heads:.0f} MB.")

    return RequirementResult(
        requirement="R2",
        name="Bounded KV: MQA or GQA <= 2 Heads",
        passed=passed,
        measured_value=round(kv_mb, 2),
        target_value=MAX_KV_MB_AT_4K,
        unit="MB at 4K context",
        evidence=evidence,
        note=note,
    )


def check_r3(arch: dict) -> RequirementResult:
    """R3: Explicit canonical + delta weight parameterization."""
    # R3 cannot be checked from GGUF metadata alone — it requires knowledge
    # of how the model was trained. No current public model satisfies R3.
    return RequirementResult(
        requirement="R3",
        name="Explicit Canonical + Delta Weight Parameterization",
        passed=False,
        measured_value=None,
        target_value=0.70,
        unit="aliasing ratio in core zone",
        evidence="R3 cannot be verified from GGUF metadata — requires training configuration",
        note=("No current public model is trained with explicit canonical + delta structure. "
              "R3 is novel architecture described in the AIOS paper. "
              "See GitHub Issue #6 for research collaboration on this requirement."),
    )


def check_r4(arch: dict) -> RequirementResult:
    """R4: Cache-aligned intermediate dimensions."""
    ffn_intermediate = arch.get("ffn_intermediate", None)
    hidden_size = arch.get("hidden_size", None)

    if ffn_intermediate is None or hidden_size is None:
        return RequirementResult(
            requirement="R4",
            name="Cache-Aligned Intermediate Dimensions",
            passed=False,
            measured_value=None,
            target_value=MAX_ACTIVATION_KB_PER_LAYER,
            unit="KB per layer at batch=1",
            evidence="FFN intermediate or hidden size not found in GGUF metadata",
            note="Manually verify from model card.",
        )

    # Per-layer activation footprint at batch=1, single token
    # Approximation: attention sublayer + FFN sublayer
    attn_bytes = (hidden_size + hidden_size) * 2       # simplified
    ffn_bytes  = (ffn_intermediate * 2 + hidden_size) * 2  # gate+up+down
    total_kb   = (attn_bytes + ffn_bytes) / 1024

    passed = total_kb <= MAX_ACTIVATION_KB_PER_LAYER

    evidence = (f"Per-layer activation footprint (batch=1, single token): "
                f"~{total_kb:.0f} KB (target: <= {MAX_ACTIVATION_KB_PER_LAYER} KB)")
    if passed:
        note = "Compliant. Activations stay in L2 cache during single-token generation."
    else:
        note = (f"Non-compliant. {total_kb:.0f} KB exceeds L2 budget. "
                f"Chunked prefill mitigates this for generation but prefill still spills. "
                f"Reduce FFN intermediate from {ffn_intermediate} to <= "
                f"{int((MAX_ACTIVATION_KB_PER_LAYER * 1024 - attn_bytes) / 4)}"
                f" for compliance.")

    return RequirementResult(
        requirement="R4",
        name="Cache-Aligned Intermediate Dimensions",
        passed=passed,
        measured_value=round(total_kb, 1),
        target_value=float(MAX_ACTIVATION_KB_PER_LAYER),
        unit="KB per layer at batch=1",
        evidence=evidence,
        note=note,
    )


def check_r5(model_path: Path) -> RequirementResult:
    """R5: Frequency-ordered weight layout with JSON manifest."""
    # Check for AIOS frequency manifest alongside model file
    manifest_candidates = [
        model_path.parent / "aios_manifest.json",
        model_path.with_suffix(".aios.json"),
        model_path.parent / "aios" / "manifest.json",
    ]

    for candidate in manifest_candidates:
        if candidate.exists():
            try:
                with open(candidate) as f:
                    manifest = json.load(f)
                if manifest.get("aios_contract_version") and manifest.get("frequency_ordered"):
                    return RequirementResult(
                        requirement="R5",
                        name="Frequency-Ordered Weight Layout",
                        passed=True,
                        measured_value=None,
                        target_value=None,
                        unit="manifest_present",
                        evidence=f"AIOS manifest found: {candidate}",
                        note="Compliant. Profiler initialization will be instantaneous.",
                    )
            except Exception:
                pass

    return RequirementResult(
        requirement="R5",
        name="Frequency-Ordered Weight Layout",
        passed=False,
        measured_value=None,
        target_value=None,
        unit="manifest_present",
        evidence="No AIOS frequency manifest found alongside model file",
        note=("Run the AIOS profiler to generate a manifest: "
              "python profiler/manifest.py --model <path>. "
              "The manifest is generated once and stored with the model."),
    )


def determine_tier(results: list) -> tuple:
    """Determine compliance tier from requirement results."""
    met = {r.requirement: r.passed for r in results}

    if met["R1"] and met["R2"] and met["R3"] and met["R4"] and met["R5"]:
        return "AIOS Native+", "All five requirements satisfied."
    elif met["R1"] and met["R2"] and met["R3"]:
        return "AIOS Native", "R1 + R2 + R3 satisfied. R4 and R5 add profiler and cache efficiency."
    elif met["R1"] and met["R2"]:
        return "AIOS Optimized", "R1 + R2 satisfied. Projected 70-80% MB/token reduction."
    else:
        r1_note = "" if met["R1"] else " Add ReLU activation (R1) for sparsity map."
        r2_note = "" if met["R2"] else " Add MQA/GQA<=2 (R2) to eliminate KV DRAM traffic."
        return "AIOS Compatible", f"Residency and aliasing apply.{r1_note}{r2_note}"


def main():
    parser = argparse.ArgumentParser(
        description="AIOS Model Contract compliance checker"
    )
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--output", default="results/compliance.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model not found: {model_path}")
        sys.exit(1)

    model_size_gb = model_path.stat().st_size / (1024**3)

    print(f"\nAIOS Model Contract Compliance Check")
    print(f"{'='*50}")
    print(f"Model: {model_path.name} ({model_size_gb:.2f} GB)\n")

    print("Parsing architecture...")
    raw_meta = parse_gguf_metadata(model_path)
    arch = extract_architecture(raw_meta)

    if args.verbose:
        for k, v in arch.items():
            print(f"  {k}: {v}")
    print()

    print("Checking requirements...")
    results = [
        check_r1(arch),
        check_r2(arch),
        check_r3(arch),
        check_r4(arch),
        check_r5(model_path),
    ]

    tier, tier_explanation = determine_tier(results)
    met_dict = {r.requirement: r.passed for r in results}

    notes = []
    if not met_dict["R1"]:
        notes.append("R1 (ReLU): relufication achieves ~95% sparsity at ~3-5% training cost — see Issue #1")
    if not met_dict["R2"]:
        notes.append("R2 (MQA): MQA reduces KV cache 16x vs 8-head GQA — target for new model designs")
    if not met_dict["R3"]:
        notes.append("R3 (aliasing): novel architecture — research collaboration invited, see Issue #6")
    if not met_dict["R4"] and met_dict["R1"]:
        notes.append("R4 (cache-aligned dims): current activation dimensions exceed L2 budget — chunked prefill mitigates")

    report = ComplianceReport(
        model_path=str(model_path.absolute()),
        model_size_gb=round(model_size_gb, 3),
        architecture=arch,
        requirements=[asdict(r) for r in results],
        tier=tier,
        tier_explanation=tier_explanation,
        requirements_met=met_dict,
        notes=notes,
    )

    # Print results
    print(f"\n{'='*50}")
    print(f"COMPLIANCE RESULTS")
    print(f"{'='*50}")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.requirement}: {r.name}")
        if args.verbose:
            print(f"       {r.evidence}")
        if not r.passed:
            print(f"       → {r.note}")

    print(f"\nCompliance tier: {tier}")
    print(f"  {tier_explanation}")
    for note in notes:
        print(f"  • {note}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"To add this model to the catalog: open a PR to models/catalog.md")


if __name__ == "__main__":
    main()
