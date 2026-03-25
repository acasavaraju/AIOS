#!/usr/bin/env python3
"""
AIOS Validation — Phase 1: Baseline MB/token Measurement

Measures the primary AIOS metric (MB/token from DRAM) plus supporting
metrics on any GGUF model using llama.cpp + hardware perf counters.

Usage:
    python baseline.py --model path/to/model.gguf --runs 5 --output results/baseline.json

Tracks: GitHub Issue #2
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np


# ── Hardware counter names by architecture ─────────────────────────────────────
# These are the most common names. Your CPU may differ — use --list-counters
# to discover available uncore/memory controller events.
DRAM_READ_COUNTERS = {
    "x86_intel": [
        "uncore_imc/data_reads/",
        "uncore_imc_0/data_reads/",
        "intel_uncore_imc/data_reads/",
    ],
    "x86_amd": [
        "amd_df/mem_read_requests/",
        "amd_umc/data_fill/",
    ],
    "arm": [
        "armv8_pmuv3/STALL_BACKEND_MEM/",
        "arm_cmn/rnid_rxdat_flits/",
    ],
}

LLC_MISS_COUNTERS = {
    "x86_intel": ["LLC-load-misses", "LLC-store-misses"],
    "x86_amd":   ["LLC-load-misses", "LLC-store-misses"],
    "arm":       ["LLC-load-misses"],
}


@dataclass
class HardwareProfile:
    cpu_model: str
    cpu_cores: int
    ram_gb: float
    l3_cache_mb: float
    architecture: str
    os: str
    kernel: str


@dataclass
class RunResult:
    run_index: int
    mb_per_token: Optional[float]    # None if perf counters unavailable
    tokens_per_sec: float
    total_tokens: int
    dram_reads_gb: Optional[float]
    llc_miss_rate: Optional[float]
    duration_sec: float
    perf_available: bool


@dataclass
class BaselineReport:
    model_path: str
    model_size_gb: float
    hardware: HardwareProfile
    runs: list
    mb_per_token_mean: Optional[float]
    mb_per_token_stddev: Optional[float]
    mb_per_token_cv: Optional[float]      # coefficient of variation
    tokens_per_sec_mean: float
    tokens_per_sec_stddev: float
    pass_criterion_met: bool              # CV < 5%
    perf_available: bool
    notes: list


def detect_hardware() -> HardwareProfile:
    """Detect CPU, RAM, cache configuration."""
    import psutil

    cpu_model = "Unknown"
    l3_mb = 0.0
    arch = platform.machine().lower()

    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":")[1].strip()
                        break
        except Exception:
            pass

        # L3 cache from sysfs
        cache_paths = list(Path("/sys/devices/system/cpu/cpu0/cache").glob("index*/"))
        for cp in cache_paths:
            try:
                level = (cp / "level").read_text().strip()
                if level == "3":
                    size_str = (cp / "size").read_text().strip()
                    if size_str.endswith("K"):
                        l3_mb = int(size_str[:-1]) / 1024
                    elif size_str.endswith("M"):
                        l3_mb = float(size_str[:-1])
                    break
            except Exception:
                pass

    elif platform.system() == "Darwin":
        try:
            cpu_model = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True).strip()
            l3_raw = subprocess.check_output(
                ["sysctl", "-n", "hw.l3cachesize"], text=True).strip()
            l3_mb = int(l3_raw) / (1024 * 1024)
        except Exception:
            pass

    ram_bytes = psutil.virtual_memory().total
    cores = psutil.cpu_count(logical=False) or psutil.cpu_count()

    # Classify architecture
    if "x86" in arch or "amd64" in arch or "i686" in arch:
        arch_class = "x86_intel"  # refine below
        try:
            vendor = ""
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "vendor_id" in line:
                        vendor = line.split(":")[1].strip().lower()
                        break
            if "amd" in vendor:
                arch_class = "x86_amd"
        except Exception:
            pass
    elif "arm" in arch or "aarch64" in arch:
        arch_class = "arm"
    else:
        arch_class = "unknown"

    return HardwareProfile(
        cpu_model=cpu_model,
        cpu_cores=cores,
        ram_gb=round(ram_bytes / (1024**3), 1),
        l3_cache_mb=l3_mb,
        architecture=arch_class,
        os=platform.system(),
        kernel=platform.release(),
    )


def find_llama_cpp() -> Optional[str]:
    """Locate llama.cpp main binary."""
    candidates = [
        "llama-cli",
        "llama.cpp/build/bin/llama-cli",
        "llama.cpp/build/bin/main",
        os.path.expanduser("~/llama.cpp/build/bin/llama-cli"),
        "/usr/local/bin/llama-cli",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    # Try PATH
    try:
        result = subprocess.run(["which", "llama-cli"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def find_perf_counter(arch_class: str) -> Optional[str]:
    """Find a working DRAM read counter for this CPU."""
    candidates = DRAM_READ_COUNTERS.get(arch_class, [])
    for counter in candidates:
        result = subprocess.run(
            ["perf", "stat", "-e", counter, "sleep", "0"],
            capture_output=True, text=True
        )
        if result.returncode == 0 and "not supported" not in result.stderr:
            return counter
    return None


def measure_run(
    llama_bin: str,
    model_path: str,
    tokens: int,
    perf_counter: Optional[str],
    run_idx: int,
    verbose: bool = False,
) -> RunResult:
    """Run a single inference measurement pass."""

    prompt = (
        "Analyze the following passage and provide a detailed summary: "
        "The development of large language models has fundamentally changed "
        "how we interact with artificial intelligence systems. These models, "
        "trained on vast corpora of text data, demonstrate remarkable capabilities "
        "across a wide range of tasks including reasoning, code generation, and "
        "creative writing. However, their computational requirements have historically "
        "limited deployment to cloud infrastructure with specialized GPU hardware. "
        "Recent advances in quantization and inference optimization are beginning "
        "to change this constraint significantly."
    )

    llama_cmd = [
        llama_bin,
        "--model", model_path,
        "--prompt", prompt,
        "--n-predict", str(tokens),
        "--threads", str(max(1, os.cpu_count() // 2)),
        "--ctx-size", "2048",
        "--no-display-prompt",
        "--log-disable",
    ]

    start = time.perf_counter()

    if perf_counter:
        perf_cmd = [
            "perf", "stat",
            "-e", perf_counter,
            "-e", "LLC-load-misses",
            "-e", "LLC-loads",
            "--",
        ] + llama_cmd

        result = subprocess.run(perf_cmd, capture_output=True, text=True)
        duration = time.perf_counter() - start
        perf_available = True
        perf_output = result.stderr

        # Parse perf output
        dram_bytes = None
        llc_misses = None
        llc_loads = None

        for line in perf_output.splitlines():
            line = line.strip()
            if perf_counter.split("/")[-1].rstrip("/") in line or "data_reads" in line:
                try:
                    val = int(line.split()[0].replace(",", ""))
                    # perf reports in cache lines (64 bytes) for some counters
                    dram_bytes = val * 64
                except (ValueError, IndexError):
                    pass
            if "LLC-load-misses" in line:
                try:
                    llc_misses = int(line.split()[0].replace(",", ""))
                except (ValueError, IndexError):
                    pass
            if "LLC-loads" in line and "misses" not in line:
                try:
                    llc_loads = int(line.split()[0].replace(",", ""))
                except (ValueError, IndexError):
                    pass

        llc_miss_rate = None
        if llc_misses is not None and llc_loads and llc_loads > 0:
            llc_miss_rate = llc_misses / llc_loads
    else:
        result = subprocess.run(llama_cmd, capture_output=True, text=True)
        duration = time.perf_counter() - start
        perf_available = False
        dram_bytes = None
        llc_miss_rate = None

    # Parse llama.cpp output for token count and speed
    output = result.stdout + result.stderr
    actual_tokens = tokens  # fallback
    tok_per_sec = None

    for line in output.splitlines():
        if "eval time" in line.lower() or "tokens per second" in line.lower():
            try:
                parts = line.split()
                for i, p in enumerate(parts):
                    if "tok/s" in p or ("per" in p and i + 1 < len(parts)):
                        tok_per_sec = float(parts[i - 1].replace(",", ""))
                        break
            except (ValueError, IndexError):
                pass

    if tok_per_sec is None and duration > 0:
        tok_per_sec = actual_tokens / duration

    mb_per_token = None
    if dram_bytes is not None and actual_tokens > 0:
        mb_per_token = (dram_bytes / (1024 * 1024)) / actual_tokens

    if verbose:
        print(f"  Run {run_idx+1}: {tok_per_sec:.1f} tok/s, "
              f"MB/token: {mb_per_token:.1f if mb_per_token else 'N/A'}")

    return RunResult(
        run_index=run_idx,
        mb_per_token=mb_per_token,
        tokens_per_sec=tok_per_sec or 0.0,
        total_tokens=actual_tokens,
        dram_reads_gb=(dram_bytes / (1024**3)) if dram_bytes else None,
        llc_miss_rate=llc_miss_rate,
        duration_sec=duration,
        perf_available=perf_available,
    )


def list_available_counters():
    """Print available perf counters for DRAM measurement."""
    print("Checking available memory controller counters...\n")
    result = subprocess.run(
        ["perf", "list", "uncore"],
        capture_output=True, text=True
    )
    lines = [l for l in (result.stdout + result.stderr).splitlines()
             if any(k in l.lower() for k in ["imc", "umc", "mem", "dram"])]
    if lines:
        print("Found:\n" + "\n".join(lines[:20]))
    else:
        print("No uncore memory counters found. Check perf permissions:\n"
              "  sudo sh -c \"echo 0 > /proc/sys/kernel/perf_event_paranoid\"")


def main():
    parser = argparse.ArgumentParser(
        description="AIOS Phase 1 baseline MB/token measurement"
    )
    parser.add_argument("--model", required=False, help="Path to GGUF model file")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs (default: 5)")
    parser.add_argument("--tokens", type=int, default=200, help="Tokens to generate per run")
    parser.add_argument("--output", default="results/baseline.json", help="Output JSON path")
    parser.add_argument("--llama-bin", default=None, help="Path to llama-cli binary")
    parser.add_argument("--list-counters", action="store_true",
                        help="List available perf counters and exit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.list_counters:
        list_available_counters()
        return

    if not args.model:
        parser.error("--model is required")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model file not found: {model_path}")
        sys.exit(1)

    model_size_gb = model_path.stat().st_size / (1024**3)

    print(f"\nAIOS Baseline Measurement")
    print(f"{'='*50}")
    print(f"Model:  {model_path.name} ({model_size_gb:.2f} GB)")
    print(f"Runs:   {args.runs}")
    print(f"Tokens: {args.tokens} per run\n")

    # Detect hardware
    hw = detect_hardware()
    print(f"Hardware: {hw.cpu_model}")
    print(f"  Cores: {hw.cpu_cores}, RAM: {hw.ram_gb} GB, L3: {hw.l3_cache_mb:.0f} MB\n")

    # Find llama.cpp
    llama_bin = args.llama_bin or find_llama_cpp()
    if not llama_bin:
        print("ERROR: llama-cli not found. Install llama.cpp and ensure it's in PATH.")
        print("  git clone https://github.com/ggerganov/llama.cpp")
        print("  cd llama.cpp && cmake -B build && cmake --build build -j")
        sys.exit(1)
    print(f"Using: {llama_bin}")

    # Find perf counter
    perf_counter = find_perf_counter(hw.architecture)
    if perf_counter:
        print(f"Perf counter: {perf_counter} (MB/token measurement active)")
    else:
        print("WARNING: No DRAM read counter found. MB/token will not be measured.")
        print("  Tokens/sec and qualitative metrics will still be collected.")
        print("  For MB/token: check perf permissions and run --list-counters\n")

    # Warm-up run (discarded)
    print("\nWarm-up run (discarded)...")
    measure_run(llama_bin, str(model_path), args.tokens, perf_counter, -1)

    # Measurement runs
    print(f"\nMeasurement runs:")
    run_results = []
    for i in range(args.runs):
        r = measure_run(llama_bin, str(model_path), args.tokens,
                        perf_counter, i, verbose=True)
        run_results.append(r)

    # Aggregate
    tps_vals = [r.tokens_per_sec for r in run_results if r.tokens_per_sec > 0]
    mbt_vals = [r.mb_per_token for r in run_results if r.mb_per_token is not None]

    tps_mean = float(np.mean(tps_vals)) if tps_vals else 0.0
    tps_std  = float(np.std(tps_vals))  if tps_vals else 0.0
    mbt_mean = float(np.mean(mbt_vals)) if mbt_vals else None
    mbt_std  = float(np.std(mbt_vals))  if mbt_vals else None
    mbt_cv   = (mbt_std / mbt_mean) if (mbt_mean and mbt_mean > 0) else None
    tps_cv   = (tps_std / tps_mean) if tps_mean > 0 else None

    # Pass criterion: CV < 5% (use tokens/sec if MB/token unavailable)
    cv_to_check = mbt_cv if mbt_cv is not None else tps_cv
    pass_criterion = (cv_to_check is not None and cv_to_check < 0.05)

    notes = []
    if not perf_counter:
        notes.append("MB/token not measured — perf counters unavailable")
    if not pass_criterion and cv_to_check is not None:
        notes.append(f"CV = {cv_to_check*100:.1f}% exceeds 5% threshold — do not proceed to optimization measurement")
    if pass_criterion:
        notes.append("Pass criterion met — baseline is stable")

    report = BaselineReport(
        model_path=str(model_path.absolute()),
        model_size_gb=round(model_size_gb, 3),
        hardware=hw,
        runs=[asdict(r) for r in run_results],
        mb_per_token_mean=round(mbt_mean, 3) if mbt_mean else None,
        mb_per_token_stddev=round(mbt_std, 3) if mbt_std else None,
        mb_per_token_cv=round(mbt_cv, 4) if mbt_cv else None,
        tokens_per_sec_mean=round(tps_mean, 2),
        tokens_per_sec_stddev=round(tps_std, 2),
        pass_criterion_met=pass_criterion,
        perf_available=perf_counter is not None,
        notes=notes,
    )

    # Print summary
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    if mbt_mean:
        print(f"MB/token (DRAM):  {mbt_mean:.1f} ± {mbt_std:.1f} MB  (CV: {mbt_cv*100:.1f}%)")
    print(f"Tokens/sec:       {tps_mean:.1f} ± {tps_std:.1f}")
    print(f"Pass criterion:   {'PASS' if pass_criterion else 'FAIL'} (CV < 5%)")
    for note in notes:
        print(f"Note: {note}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"\nPost results to: https://github.com/[org]/aios/issues/2")


if __name__ == "__main__":
    main()
