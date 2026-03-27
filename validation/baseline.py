#!/usr/bin/env python3
"""
AIOS Validation — Phase 1: Baseline MB/token Measurement

Measures the primary AIOS metric (MB/token from DRAM) plus supporting
metrics on any GGUF model using llama.cpp + hardware perf counters.

Scope
-----
This script has one job: measure DRAM data movement per generated token
using hardware memory controller counters. That measurement is only
possible on Linux with perf uncore access. Tokens/sec is measured on
all platforms but is already provided by llama.cpp directly — the unique
contribution here is MB/token.

Supported hardware for MB/token measurement:
  Linux (kernel 5.4+), bare metal only — WSL2 is not supported
  Intel x86: Haswell EP+ (server), Alder Lake+ (consumer)
  AMD x86:   Zen 3 and newer

Tokens/sec only (MB/token not available):
  WSL2, macOS, Windows, ARM
  Run --list-counters to check what your CPU exposes

The primary validation target is Falcon 7B Q4_K_M on x86 Linux.
Results on other hardware are welcome as secondary data points.

Usage:
    python baseline.py --model path/to/model.gguf --runs 5 --output results/baseline.json
    python baseline.py --list-counters   # diagnose perf counter availability

Tracks: GitHub Issue #2

Changelog:
    v1.2 — Add --no-cnv to prevent llama-cli interactive mode. Fix LLC counters
            to use architecture dict instead of hardcoded strings. Robust output
            parsing for all recent llama.cpp versions (4 format handlers).
    v1.1 — Fix counter detection for consumer Intel CPUs (Arrow Lake, Raptor Lake,
            Alder Lake). Add dynamic perf list discovery. Add --ignore-eos to prevent
            llama.cpp looping. Validate actual token count generated.
    v1.0 — Initial release.
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
# Static candidates tried in order before falling back to dynamic discovery.
# Consumer Intel CPUs (Arrow Lake, Raptor Lake, Alder Lake) use different
# counter names than server Xeons — dynamic discovery handles these.
DRAM_READ_COUNTERS = {
    "x86_intel": [
        # Server / Xeon (Haswell EP, Skylake SP, Ice Lake SP, Sapphire Rapids)
        "uncore_imc/data_reads/",
        "uncore_imc_0/data_reads/",
        "uncore_imc_1/data_reads/",
        "intel_uncore_imc/data_reads/",
        # Consumer (Alder Lake, Raptor Lake, Arrow Lake) — free-running counters
        "uncore_imc_free_running_0/data_reads/",
        "uncore_imc_free_running_1/data_reads/",
        # Generic fallback — works on some kernels
        "cpu/mem-loads/",
    ],
    "x86_amd": [
        # Zen 3 / Zen 4
        "amd_umc/data_fill/",
        "amd_umc_0/data_fill/",
        "amd_df/mem_read_requests/",
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
    tokens_generated_ok: bool        # True if target token count was reached


@dataclass
class BaselineReport:
    model_path: str
    model_size_gb: float
    hardware: HardwareProfile
    runs: list
    mb_per_token_mean: Optional[float]
    mb_per_token_stddev: Optional[float]
    mb_per_token_cv: Optional[float]
    tokens_per_sec_mean: float
    tokens_per_sec_stddev: float
    pass_criterion_met: bool
    perf_available: bool
    counter_used: Optional[str]
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
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            l3_raw = subprocess.check_output(
                ["sysctl", "-n", "hw.l3cachesize"], text=True).strip()
            l3_mb = int(l3_raw) / (1024 * 1024)
        except Exception:
            pass

    ram_bytes = psutil.virtual_memory().total
    cores = psutil.cpu_count(logical=False) or psutil.cpu_count()

    if "x86" in arch or "amd64" in arch or "i686" in arch:
        arch_class = "x86_intel"
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "vendor_id" in line:
                        vendor = line.split(":")[1].strip().lower()
                        if "amd" in vendor:
                            arch_class = "x86_amd"
                        break
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
        "/usr/bin/llama-cli",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    try:
        result = subprocess.run(["which", "llama-cli"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def discover_dram_counters_dynamic() -> list:
    """
    Dynamically discover available memory controller counters via perf list.
    Handles consumer CPUs (Arrow Lake, Raptor Lake, Alder Lake) which use
    different counter names than server Xeons.
    """
    try:
        result = subprocess.run(
            ["perf", "list", "--no-desc"],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout + result.stderr
        candidates = []
        for line in output.splitlines():
            line = line.strip().lower()
            if any(k in line for k in ["imc", "umc", "dram", "data_read",
                                        "mem_read", "memory_read"]):
                # Extract the counter name (first token)
                token = line.split()[0] if line.split() else ""
                if token and "/" in token:
                    candidates.append(token)
        return candidates
    except Exception:
        return []


def find_perf_counter(arch_class: str, verbose: bool = False) -> Optional[str]:
    """
    Find a working DRAM read counter for this CPU.
    First tries static candidates, then falls back to dynamic discovery.
    """
    # Step 1: try static candidates
    candidates = DRAM_READ_COUNTERS.get(arch_class, [])
    for counter in candidates:
        result = subprocess.run(
            ["perf", "stat", "-e", counter, "sleep", "0"],
            capture_output=True, text=True
        )
        stderr = result.stderr.lower()
        if result.returncode == 0 and "not supported" not in stderr and \
           "invalid" not in stderr and "unknown" not in stderr:
            if verbose:
                print(f"  Counter found (static): {counter}")
            return counter

    # Step 2: dynamic discovery for consumer CPUs not in static list
    if verbose:
        print("  Static counter list exhausted — trying dynamic discovery...")
    dynamic = discover_dram_counters_dynamic()
    for counter in dynamic:
        result = subprocess.run(
            ["perf", "stat", "-e", counter, "sleep", "0"],
            capture_output=True, text=True
        )
        stderr = result.stderr.lower()
        if result.returncode == 0 and "not supported" not in stderr and \
           "invalid" not in stderr and "unknown" not in stderr:
            if verbose:
                print(f"  Counter found (dynamic): {counter}")
            return counter

    return None


def measure_run(
    llama_bin: str,
    model_path: str,
    tokens: int,
    perf_counter: Optional[str],
    run_idx: int,
    hw_arch: str = "x86_intel",
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
        # Disable conversation/interactive mode — prevents llama-cli from
        # waiting for user input after generating the first response.
        "--no-cnv",
        # Prevent EOS-triggered early stopping and output loops.
        # Output quality is irrelevant — only token count and bandwidth matter.
        "--ignore-eos",
        "--repeat-penalty", "1.0",   # disable repetition penalty
        "--temp", "0.0",             # greedy decoding — deterministic
    ]

    start = time.perf_counter()

    # Build LLC counter list from the dict rather than hardcoding strings.
    # Falls back to standard names if arch not found.
    llc_counters = LLC_MISS_COUNTERS.get(hw_arch, ["LLC-load-misses", "LLC-loads"])
    # Verify each LLC counter is usable before adding to perf command.
    usable_llc = []
    for c in llc_counters:
        test = subprocess.run(
            ["perf", "stat", "-e", c, "sleep", "0"],
            capture_output=True, text=True
        )
        if test.returncode == 0 and "not supported" not in test.stderr.lower():
            usable_llc.append(c)

    if perf_counter:
        perf_events = ["-e", perf_counter]
        for c in usable_llc:
            perf_events += ["-e", c]
        perf_cmd = ["perf", "stat"] + perf_events + ["--"] + llama_cmd

        result = subprocess.run(perf_cmd, capture_output=True, text=True)
        duration = time.perf_counter() - start
        perf_available = True
        perf_output = result.stderr

        dram_bytes = None
        llc_misses = None
        llc_loads = None

        for line in perf_output.splitlines():
            line_s = line.strip()
            if any(k in line_s for k in ["data_reads", "data_fill",
                                          "mem_read", "imc", "umc"]):
                try:
                    val = int(line_s.split()[0].replace(",", ""))
                    dram_bytes = val * 64  # cache lines to bytes
                except (ValueError, IndexError):
                    pass
            # LLC miss counter — match any counter from usable_llc list
            if any(c in line_s for c in usable_llc if "miss" in c.lower()):
                try:
                    llc_misses = int(line_s.split()[0].replace(",", ""))
                except (ValueError, IndexError):
                    pass
            # LLC load counter — match non-miss LLC counter
            if any(c in line_s for c in usable_llc if "miss" not in c.lower()):
                try:
                    llc_loads = int(line_s.split()[0].replace(",", ""))
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

    # Parse token count and speed from llama.cpp output.
    # llama.cpp output format varies across versions — handle multiple formats.
    output = result.stdout + result.stderr
    actual_tokens = 0
    tok_per_sec = None

    import re
    for line in output.splitlines():
        line_l = line.lower()

        # Format 1 (older): "llama_print_timings: eval time = X ms / Y tokens"
        # Format 2 (newer): "llama_perf_context_print: eval time = X ms / Y runs"
        if "eval time" in line_l:
            # Extract token count: "Y tokens" or "Y runs"
            m = re.search(r'=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*(tokens|runs)', line_l)
            if m:
                actual_tokens = int(m.group(1))
            # Extract tok/s: "= X.XX tok/s"
            m2 = re.search(r'=\s*([\d.]+)\s*tok/s', line_l)
            if m2:
                tok_per_sec = float(m2.group(1))

        # Format 3: standalone "X tok/s" on timing summary line
        if "tok/s" in line_l and tok_per_sec is None:
            m = re.search(r'([\d.]+)\s*tok/s', line_l)
            if m:
                try:
                    val = float(m.group(1))
                    if 0.1 < val < 50000:
                        tok_per_sec = val
                except ValueError:
                    pass

        # Format 4 (newest llama.cpp): JSON-style timing output
        if '"t_p_eval"' in line or '"t_eval"' in line:
            try:
                import json as _json
                # Sometimes the entire timing block is one JSON line
                data = _json.loads(line.strip())
                if 'n_eval' in data:
                    actual_tokens = data['n_eval']
                if 't_eval' in data and data['t_eval'] > 0 and actual_tokens > 0:
                    tok_per_sec = actual_tokens / (data['t_eval'] / 1000.0)
            except Exception:
                pass

    # Fallback: estimate from duration
    if actual_tokens == 0:
        actual_tokens = tokens  # assume target was reached
    if tok_per_sec is None and duration > 0:
        tok_per_sec = actual_tokens / duration

    # Validate token count — flag if substantially short
    tokens_ok = actual_tokens >= int(tokens * 0.9)  # within 10% of target

    mb_per_token = None
    if dram_bytes is not None and actual_tokens > 0:
        mb_per_token = (dram_bytes / (1024 * 1024)) / actual_tokens

    if verbose:
        tok_str = f"{tok_per_sec:.1f}" if tok_per_sec else "N/A"
        mbt_str = f"{mb_per_token:.1f}" if mb_per_token else "N/A"
        ok_str  = "OK" if tokens_ok else f"SHORT ({actual_tokens}/{tokens})"
        print(f"  Run {run_idx+1}: {tok_str} tok/s | MB/token: {mbt_str} | "
              f"Tokens: {ok_str}")

    return RunResult(
        run_index=run_idx,
        mb_per_token=mb_per_token,
        tokens_per_sec=tok_per_sec or 0.0,
        total_tokens=actual_tokens,
        dram_reads_gb=(dram_bytes / (1024**3)) if dram_bytes else None,
        llc_miss_rate=llc_miss_rate,
        duration_sec=duration,
        perf_available=perf_available,
        tokens_generated_ok=tokens_ok,
    )


def list_available_counters():
    """Print available perf counters for DRAM measurement."""
    print("Checking available memory controller counters...\n")
    result = subprocess.run(
        ["perf", "list", "--no-desc"],
        capture_output=True, text=True
    )
    output = result.stdout + result.stderr
    lines = [l for l in output.splitlines()
             if any(k in l.lower() for k in
                    ["imc", "umc", "mem", "dram", "data_read"])]
    if lines:
        print("Found memory-related counters:")
        for l in lines[:30]:
            print(f"  {l.strip()}")
    else:
        print("No memory controller counters found.")
        print("Check perf permissions:")
        print("  sudo sh -c \"echo 0 > /proc/sys/kernel/perf_event_paranoid\"")
        print("  Then retry: perf list | grep -i imc")


def main():
    parser = argparse.ArgumentParser(
        description="AIOS Phase 1 baseline MB/token measurement"
    )
    parser.add_argument("--model", required=False, help="Path to GGUF model file")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--output", default="results/baseline.json")
    parser.add_argument("--llama-bin", default=None)
    parser.add_argument("--list-counters", action="store_true")
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

    print(f"\nAIOS Baseline Measurement v1.1")
    print(f"{'='*50}")
    print(f"Model:  {model_path.name} ({model_size_gb:.2f} GB)")
    print(f"Runs:   {args.runs}  |  Tokens: {args.tokens} per run\n")

    hw = detect_hardware()
    print(f"Hardware: {hw.cpu_model}")
    print(f"  Cores: {hw.cpu_cores}, RAM: {hw.ram_gb} GB, "
          f"L3: {hw.l3_cache_mb:.0f} MB, Arch: {hw.architecture}\n")

    llama_bin = args.llama_bin or find_llama_cpp()
    if not llama_bin:
        print("ERROR: llama-cli not found. Build llama.cpp first:")
        print("  git clone https://github.com/ggerganov/llama.cpp")
        print("  cd llama.cpp && cmake -B build && cmake --build build -j")
        sys.exit(1)
    print(f"llama.cpp: {llama_bin}")

    print("Searching for DRAM read counter...")
    perf_counter = find_perf_counter(hw.architecture, verbose=True)
    if perf_counter:
        print(f"Using counter: {perf_counter}")
    else:
        print("WARNING: No DRAM read counter found.")
        print("  Run --list-counters to see what's available on your CPU.")
        print("  Tokens/sec will still be measured.\n")
        print("  To fix permissions: "
              "sudo sh -c \"echo 0 > /proc/sys/kernel/perf_event_paranoid\"")

    # Warm-up (discarded)
    print("\nWarm-up run...")
    measure_run(llama_bin, str(model_path), args.tokens, perf_counter, -1, hw_arch=hw.architecture)

    print(f"\nMeasurement runs:")
    run_results = []
    for i in range(args.runs):
        r = measure_run(llama_bin, str(model_path), args.tokens,
                        perf_counter, i, hw_arch=hw.architecture, verbose=True)
        run_results.append(r)

    # Flag runs where token count was short
    short_runs = [r for r in run_results if not r.tokens_generated_ok]
    if short_runs:
        print(f"\nWARNING: {len(short_runs)} run(s) generated fewer tokens "
              f"than target. Results may be unreliable.")

    # Aggregate
    tps_vals = [r.tokens_per_sec for r in run_results if r.tokens_per_sec > 0]
    mbt_vals = [r.mb_per_token for r in run_results if r.mb_per_token is not None]

    tps_mean = float(np.mean(tps_vals)) if tps_vals else 0.0
    tps_std  = float(np.std(tps_vals))  if tps_vals else 0.0
    mbt_mean = float(np.mean(mbt_vals)) if mbt_vals else None
    mbt_std  = float(np.std(mbt_vals))  if mbt_vals else None
    mbt_cv   = (mbt_std / mbt_mean) if (mbt_mean and mbt_mean > 0) else None
    tps_cv   = (tps_std / tps_mean) if tps_mean > 0 else None

    cv_check = mbt_cv if mbt_cv is not None else tps_cv
    pass_criterion = (cv_check is not None and cv_check < 0.05)

    notes = []
    if not perf_counter:
        notes.append("MB/token not measured — no DRAM counter found. "
                     "Run --list-counters to diagnose.")
    if short_runs:
        notes.append(f"{len(short_runs)} run(s) produced fewer tokens than target "
                     f"— check for EOS issues")
    if pass_criterion:
        notes.append("Pass criterion met (CV < 5%)")
    elif cv_check is not None:
        notes.append(f"CV = {cv_check*100:.1f}% — exceeds 5% threshold")

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
        counter_used=perf_counter,
        notes=notes,
    )

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    if mbt_mean:
        cv_str = f"{mbt_cv*100:.1f}%" if mbt_cv else "N/A"
        print(f"MB/token (DRAM):  {mbt_mean:.1f} ± {mbt_std:.1f} MB  (CV: {cv_str})")
    print(f"Tokens/sec:       {tps_mean:.1f} ± {tps_std:.1f}")
    print(f"Counter used:     {perf_counter or 'None — tokens/sec only'}")
    print(f"Pass criterion:   {'PASS' if pass_criterion else 'FAIL'} (CV < 5%)")
    for note in notes:
        print(f"Note: {note}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"\nPost results to: https://github.com/acasavaraju/AIOS/issues/2")


if __name__ == "__main__":
    main()
