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
import re
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

AIOS_VERSION = "v1.4"
RAW_COUNTER_MARKERS = ("0..255", "0..15", ",edge", ",umask")
PERF_STAT_SEPARATOR = ";"
SCALED_SIZE_UNITS = {
    "B": 1,
    "KiB": 1024,
    "MiB": 1024 ** 2,
    "GiB": 1024 ** 3,
}

DRAM_READ_COUNTERS = {
    "x86_intel": [
        # Server / Xeon (Haswell EP, Skylake SP, Ice Lake SP, Sapphire Rapids)
        "uncore_imc/data_reads/",
        "uncore_imc_0/data_reads/",
        "uncore_imc_1/data_reads/",
        "intel_uncore_imc/data_reads/",
        # Consumer (Alder Lake, Raptor Lake, Arrow Lake) — free-running counters
        "uncore_imc_free_running_0/data_read/",
        "uncore_imc_free_running_1/data_read/",
        # Generic fallback — works on some kernels
        "cpu/mem-loads/",
        "mem-loads",
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
    "x86_intel": ["llc-load-misses","llc-store-misses","LLC-load-misses", "LLC-store-misses"],
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


def find_llama_cpp(filename: str = "llama-cli") -> Optional[str]:
    """Locate llama.cpp main binary."""

    candidates = [
        f"llama.cpp/build/bin/{filename}",
        "llama.cpp/build/bin/main",
        f"/usr/local/bin/{filename}",
        f"/usr/bin/{filename}",
        os.path.expanduser(F"~/llama.cpp/build/bin/{filename}"),
        f"{filename}", # must be careful putting this first, could be a gpu binary 
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    try:
        # the following code could potentially return any llama.cpp on PATH 
        # (other versions not specifically built here)

        result = subprocess.run(["which", filename], capture_output=True, text=True)
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
            ["perf", "list", "--no-desc", "-j"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        events = json.loads(result.stdout)
    except Exception:
        return []

    candidates = []
    seen = set()
    keywords = ("imc", "umc", "dram", "data_read", "mem_read", "memory_read", "mem-load")

    for event in events:
        fields = [
            str(event.get("Unit", "")),
            str(event.get("Topic", "")),
            str(event.get("EventName", "")),
            str(event.get("EventAlias", "")),
            str(event.get("MetricName", "")),
        ]
        haystack = " ".join(fields).lower()
        if not any(keyword in haystack for keyword in keywords):
            continue

        event_name = str(event.get("EventAlias") or event.get("EventName") or "").strip()
        if not event_name or any(marker in event_name for marker in RAW_COUNTER_MARKERS):
            continue

        if event_name not in seen:
            seen.add(event_name)
            candidates.append(event_name)

    return candidates


def probe_working(candidates: list[str]) -> list[str]:
    """Return candidates that `perf stat` accepts on this machine."""
    working = []

    for c in candidates:
        test = subprocess.run(
            ["perf", "stat", "-e", c.strip(), "sleep", "0"],
            capture_output=True, text=True
        )
        stderr = test.stderr.lower()
        if test.returncode == 0 and all(term not in stderr for term in ("not supported", "invalid", "unknown")):
            working.append(c)

    return working


def counter_preference_key(counter: str) -> tuple[int, str]:
    """Prefer true uncore DRAM counters over generic core memory events."""
    normalized = normalize_perf_event_name(counter)
    if "uncore_imc_free_running" in normalized:
        return (0, normalized)
    if "uncore_imc" in normalized or "intel_uncore_imc" in normalized:
        return (1, normalized)
    if "umc" in normalized or "amd_df" in normalized:
        return (2, normalized)
    if "data_read" in normalized or "data_reads" in normalized:
        return (3, normalized)
    if "mem-loads" in normalized:
        return (4, normalized)
    return (5, normalized)



def find_perf_counter(arch_class: str, verbose: bool = False) -> Optional[str]:
    """
    Find a working DRAM read counter for this CPU.
    First tries static candidates, then falls back to dynamic discovery.
    """
    # Step 1: try static candidates
    candidates = DRAM_READ_COUNTERS.get(arch_class, [])

    working = probe_working(candidates)
    if working:
        working.sort(key=counter_preference_key)
        if verbose:
            print(f"  Counter found (static): {working[0]}")
        return working[0]

    if verbose:
        print("  Static counter list exhausted — trying dynamic discovery...")

    dynamic = discover_dram_counters_dynamic()
    working = probe_working(dynamic)
    if working:
        working.sort(key=counter_preference_key)
        if verbose:
            print(f"  Counter found (dynamic): {working[0]}")
        return working[0]

    return None


def find_usable_llc_counters(hw_arch: str) -> list[str]:
    """Return the subset of LLC counters accepted by `perf stat`."""
    llc_candidates = LLC_MISS_COUNTERS.get(hw_arch, ["LLC-load-misses", "LLC-loads"])
    return probe_working(llc_candidates)


def normalize_perf_event_name(name: str) -> str:
    """Normalize event names for matching across aliases and output formats."""
    return name.strip().strip("/").lower()


def perf_event_leaf(name: str) -> str:
    """Return the leaf event token from a perf event name."""
    normalized = normalize_perf_event_name(name)
    if "/" in normalized:
        return normalized.rsplit("/", 1)[-1]
    return normalized


def perf_event_matches(event_name: str, requested_name: str) -> bool:
    """Match a perf output event row against a requested counter or alias."""
    event_norm = normalize_perf_event_name(event_name)
    request_norm = normalize_perf_event_name(requested_name)
    if not event_norm or not request_norm:
        return False
    if event_norm == request_norm:
        return True
    if event_norm.endswith("/" + request_norm):
        return True
    return perf_event_leaf(event_norm) == perf_event_leaf(request_norm)


def parse_perf_stat_number(raw_value: str) -> Optional[float]:
    """Parse a perf stat numeric value that may contain commas or decimals."""
    value = raw_value.strip().replace(",", "")
    if not value or value.startswith("<"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def perf_value_to_bytes(value: float, unit: str, event_name: str) -> Optional[int]:
    """Convert a perf stat event value into bytes when the unit is known."""
    unit = unit.strip()
    if unit in SCALED_SIZE_UNITS:
        return int(value * SCALED_SIZE_UNITS[unit])

    if unit in ("", "counts", "count"):
        event_leaf = perf_event_leaf(event_name)
        if any(marker in event_leaf for marker in ("mem-loads", "data_read", "data_reads", "data_fill", "mem_read", "rdcas")):
            return int(value * 64)

    return None


def iter_perf_csv_fields(perf_output: str):
    """Yield raw perf CSV fields as (raw_value, unit, event_name)."""
    for line in perf_output.splitlines():
        line = line.strip()
        if not line or PERF_STAT_SEPARATOR not in line:
            continue

        parts = line.split(PERF_STAT_SEPARATOR)
        if len(parts) < 3:
            continue

        yield parts[0].strip(), parts[1].strip(), parts[2].strip()


def iter_perf_text_fields(perf_output: str):
    """Yield perf text rows as (raw_value, unit, event_name)."""
    for line in perf_output.splitlines():
        tokens = line.strip().split()
        if len(tokens) < 2:
            continue

        raw_value = tokens[0]
        if raw_value.startswith("<"):
            unit = tokens[1] if len(tokens) > 1 else ""
            event_name = tokens[2] if len(tokens) > 2 else ""
            yield raw_value, unit, event_name
            continue

        if parse_perf_stat_number(raw_value) is None:
            continue

        unit = ""
        event_name = ""
        if len(tokens) >= 3 and tokens[1] in SCALED_SIZE_UNITS:
            unit = tokens[1]
            event_name = tokens[2]
        else:
            event_name = tokens[1]

        if event_name:
            yield raw_value, unit, event_name


def parse_perf_csv_rows(perf_output: str) -> list[tuple[float, str, str]]:
    """Parse `perf stat -x ';'` output rows into (value, unit, event_name)."""
    rows = []
    for raw_value, unit, event_name in iter_perf_csv_fields(perf_output):
        value = parse_perf_stat_number(raw_value)

        if value is None or not event_name or event_name.startswith("<"):
            continue

        rows.append((value, unit, event_name))

    return rows


def iter_perf_fields(perf_output: str):
    """Yield perf rows from either CSV or default human-readable output."""
    if PERF_STAT_SEPARATOR in perf_output:
        yield from iter_perf_csv_fields(perf_output)
    else:
        yield from iter_perf_text_fields(perf_output)


def perf_output_has_usable_event(perf_output: str, requested_name: str) -> bool:
    """Return True when perf output contains a usable row for the requested event."""
    for raw_value, _, event_name in iter_perf_fields(perf_output):
        if perf_event_matches(event_name, requested_name) and not raw_value.startswith("<"):
            return True
    return False


def event_set_is_compatible(events: list[str]) -> bool:
    """Check whether a set of perf events can be measured together."""
    command = ["perf", "stat", "-x", PERF_STAT_SEPARATOR]
    for event in events:
        command.extend(["-e", event])
    command.extend(["sleep", "0"])

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return False

    return all(perf_output_has_usable_event(result.stderr, event) for event in events)


def filter_compatible_llc_counters(perf_counter: Optional[str], llc_counters: list[str]) -> list[str]:
    """Keep only LLC counters that can coexist with the selected DRAM counter."""
    if not perf_counter:
        return llc_counters

    compatible = []
    for counter in llc_counters:
        candidate_set = [perf_counter] + compatible + [counter]
        if event_set_is_compatible(candidate_set):
            compatible.append(counter)
    return compatible


def parse_perf_output(perf_output: str, perf_counter: Optional[str], usable_llc: list[str]) -> tuple[Optional[int], Optional[float]]:
    """Extract DRAM bytes and LLC miss rate from perf stderr output."""
    dram_bytes_total = 0
    dram_bytes_found = False
    llc_misses_total = 0
    llc_misses_found = False
    llc_loads_total = 0
    llc_loads_found = False

    for raw_value, unit, event_name in iter_perf_fields(perf_output):
        value = parse_perf_stat_number(raw_value)
        if value is None or not event_name or raw_value.startswith("<"):
            continue

        if perf_counter and perf_event_matches(event_name, perf_counter):
            bytes_value = perf_value_to_bytes(value, unit, event_name)
            if bytes_value is not None:
                dram_bytes_total += bytes_value
                dram_bytes_found = True

        for counter in usable_llc:
            if not perf_event_matches(event_name, counter):
                continue
            if "miss" in counter.lower():
                llc_misses_total += int(value)
                llc_misses_found = True
            else:
                llc_loads_total += int(value)
                llc_loads_found = True

    llc_miss_rate = None
    if llc_misses_found and llc_loads_found and llc_loads_total > 0:
        llc_miss_rate = llc_misses_total / llc_loads_total

    dram_bytes = dram_bytes_total if dram_bytes_found else None
    return dram_bytes, llc_miss_rate


def parse_llama_output(output: str, requested_tokens: int, duration: float, command_succeeded: bool) -> tuple[int, float]:
    """Extract token count and throughput from llama.cpp output."""
    actual_tokens = 0
    tok_per_sec = None

    for line in output.splitlines():
        line_l = line.lower()

        if "eval time" in line_l:
            match = re.search(r'=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*(tokens|runs)', line_l)
            if match:
                actual_tokens = int(match.group(1))

            speed_match = re.search(r'=\s*([\d.]+)\s*tok/s', line_l)
            if speed_match:
                tok_per_sec = float(speed_match.group(1))

        if "tok/s" in line_l and tok_per_sec is None:
            match = re.search(r'([\d.]+)\s*tok/s', line_l)
            if match:
                try:
                    candidate = float(match.group(1))
                except ValueError:
                    candidate = None
                if candidate is not None and 0.1 < candidate < 50000:
                    tok_per_sec = candidate

        if '"t_p_eval"' in line or '"t_eval"' in line:
            try:
                data = json.loads(line.strip())
            except Exception:
                data = None
            if data:
                if "n_eval" in data:
                    actual_tokens = data["n_eval"]
                if data.get("t_eval", 0) > 0 and actual_tokens > 0:
                    tok_per_sec = actual_tokens / (data["t_eval"] / 1000.0)

    if actual_tokens == 0 and command_succeeded:
        actual_tokens = requested_tokens
    if tok_per_sec is None and duration > 0 and actual_tokens > 0:
        tok_per_sec = actual_tokens / duration

    return actual_tokens, tok_per_sec or 0.0


def build_llama_command(llama_bin: str, model_path: str, tokens: int) -> list[str]:
    """Build the llama.cpp invocation used for all runs."""
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

    return [
        llama_bin,
        "--model", model_path,
        "--prompt", prompt,
        "--n-predict", str(tokens),
        "--threads", str(max(1, os.cpu_count() // 2)),
        "--ctx-size", "2048",
        "--no-display-prompt",
        "--log-disable",
        "--single-turn",
        "--simple-io",
        "--ignore-eos",
        "--repeat-penalty", "1.0",
        "--temp", "0.0",
    ]


def build_measurement_command(llama_cmd: list[str], perf_counter: Optional[str], usable_llc: list[str]) -> tuple[list[str], bool]:
    """Wrap the llama command in `perf stat` when a DRAM counter is available."""
    if not perf_counter:
        return llama_cmd, False

    perf_events = ["-e", perf_counter]
    for counter in usable_llc:
        perf_events.extend(["-e", counter])

    return ["perf", "stat"] + perf_events + ["--"] + llama_cmd, True


def round_optional(value: Optional[float], digits: int) -> Optional[float]:
    """Round optional floats without treating zero as missing."""
    return round(value, digits) if value is not None else None


def summarize_runs(
    run_results: list[RunResult],
    perf_counter: Optional[str],
    llc_requested: list[str],
    llc_used: list[str],
) -> tuple[float, float, Optional[float], Optional[float], Optional[float], Optional[float], list[str]]:
    """Compute aggregate statistics and notes from the run list."""
    short_runs = [r for r in run_results if not r.tokens_generated_ok]
    tps_vals = [r.tokens_per_sec for r in run_results if r.tokens_per_sec > 0]
    mbt_vals = [r.mb_per_token for r in run_results if r.mb_per_token is not None]
    runs_with_dram = [r for r in run_results if r.perf_available]

    tps_mean = float(np.mean(tps_vals)) if tps_vals else 0.0
    tps_std = float(np.std(tps_vals)) if tps_vals else 0.0
    mbt_mean = float(np.mean(mbt_vals)) if mbt_vals else None
    mbt_std = float(np.std(mbt_vals)) if mbt_vals else None
    mbt_cv = (mbt_std / mbt_mean) if (mbt_mean is not None and mbt_mean > 0) else None
    tps_cv = (tps_std / tps_mean) if tps_mean > 0 else None

    notes = []
    if short_runs:
        notes.append(f"{len(short_runs)} run(s) produced fewer tokens than target — check for EOS issues")
    if not perf_counter:
        notes.append("MB/token not measured — no DRAM counter found. Run --list-counters to diagnose.")
    elif not runs_with_dram:
        notes.append(
            f"MB/token not measured — counter `{perf_counter}` was selected, but perf produced no usable DRAM data rows."
        )
    elif len(runs_with_dram) < len(run_results):
        notes.append(
            f"DRAM data was captured in {len(runs_with_dram)}/{len(run_results)} run(s) using `{perf_counter}`."
        )
    if llc_requested and len(llc_used) < len(llc_requested):
        notes.append(
            f"Skipped {len(llc_requested) - len(llc_used)} LLC counter(s) because they conflict with `{perf_counter}` on this machine."
        )

    return tps_mean, tps_std, mbt_mean, mbt_std, mbt_cv, tps_cv, notes


def measure_run(
    llama_bin: str,
    model_path: str,
    tokens: int,
    perf_counter: Optional[str],
    run_idx: int,
    hw_arch: str = "x86_intel",
    usable_llc: Optional[list[str]] = None,
    verbose: bool = False,
) -> RunResult:
    """Run a single inference measurement pass."""
    llama_cmd = build_llama_command(llama_bin, model_path, tokens)
    start = time.perf_counter()
    usable_llc = usable_llc if usable_llc is not None else find_usable_llc_counters(hw_arch)
    command, perf_requested = build_measurement_command(llama_cmd, perf_counter, usable_llc)

    result = subprocess.run(command, capture_output=True, text=True)
    duration = time.perf_counter() - start

    if perf_requested:
        dram_bytes, llc_miss_rate = parse_perf_output(result.stderr, perf_counter, usable_llc)
    else:
        dram_bytes, llc_miss_rate = None, None

    perf_available = perf_requested and dram_bytes is not None

    output = result.stdout + result.stderr
    actual_tokens, tok_per_sec = parse_llama_output(
        output,
        tokens,
        duration,
        command_succeeded=(result.returncode == 0),
    )

    # Validate token count — flag if substantially short
    tokens_ok = actual_tokens >= int(tokens * 0.9)  # within 10% of target

    mb_per_token = None
    if dram_bytes is not None and actual_tokens > 0:
        mb_per_token = (dram_bytes / (1024 * 1024)) / actual_tokens

    if verbose:
        tok_str = f"{tok_per_sec:.1f}" if tok_per_sec else "N/A"
        mbt_str = f"{mb_per_token:.1f}" if mb_per_token is not None else "N/A"
        status_str = "OK" if result.returncode == 0 else f"ERR rc={result.returncode}"
        ok_str  = "OK" if tokens_ok else f"SHORT ({actual_tokens}/{tokens})"
        perf_str = "DRAM OK" if perf_available else ("NO DRAM DATA" if perf_requested else "TOK/S ONLY")
        print(f"  Run {run_idx+1}: {tok_str} tok/s | MB/token: {mbt_str} | "
              f"Tokens: {ok_str} | Status: {status_str} | Perf: {perf_str}")

    return RunResult(
        run_index=run_idx,
        mb_per_token=mb_per_token,
        tokens_per_sec=tok_per_sec or 0.0,
        total_tokens=actual_tokens,
        dram_reads_gb=(dram_bytes / (1024**3)) if dram_bytes is not None else None,
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
                    ["imc", "umc", "mem", "dram", "data_read", "mem-load"])]
    if lines:
        print("Found memory-related counters:")
        for m in lines[:30]:
            print(f"  {m.strip()}")
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

    print(f"\nAIOS Baseline Measurement {AIOS_VERSION}")
    print(f"{'='*50}")
    print(f"Model:  {model_path.name} ({model_size_gb:.2f} GB)")
    print(f"Runs:   {args.runs}  |  Tokens: {args.tokens} per run\n")

    hw = detect_hardware()
    print(f"Hardware: {hw.cpu_model}")
    print(f"   Cores: {hw.cpu_cores}\tRAM: {hw.ram_gb} GB, "
          f"      L3: {hw.l3_cache_mb:.0f} MB\tArch: {hw.architecture}\n")

    llama_bin = args.llama_bin or find_llama_cpp()
    if not llama_bin:
        print("ERROR: llama-cli not found. Build llama.cpp first:")
        print("  git clone https://github.com/ggerganov/llama.cpp")
        print("  cd llama.cpp && cmake -B build && cmake --build build -j")
        sys.exit(1)
    print(f"llama.cpp: {llama_bin}")

    print("Searching for DRAM read counter...")
    perf_counter = find_perf_counter(hw.architecture, verbose=True)
    llc_candidates = find_usable_llc_counters(hw.architecture)
    usable_llc = filter_compatible_llc_counters(perf_counter, llc_candidates)
    if perf_counter:
        print(f"Using counter: {perf_counter}")
        if llc_candidates and len(usable_llc) < len(llc_candidates):
            print(
                f"Skipping {len(llc_candidates) - len(usable_llc)} incompatible LLC counter(s) "
                "to preserve DRAM measurement."
            )
    else:
        print("WARNING: No DRAM read counter found.")
        print("  Run --list-counters to see what's available on your CPU.")
        print("  Tokens/sec will still be measured.\n")
        print("  To fix permissions: "
              "sudo sh -c \"echo 0 > /proc/sys/kernel/perf_event_paranoid\"")

    # Warm-up (discarded)
    print("\nWarm-up run...")
    measure_run(
        llama_bin,
        str(model_path),
        args.tokens,
        perf_counter,
        -1,
        hw_arch=hw.architecture,
        usable_llc=usable_llc,
    )

    print(f"\nMeasurement runs:")
    run_results = []
    for i in range(args.runs):
        r = measure_run(
            llama_bin,
            str(model_path),
            args.tokens,
            perf_counter,
            i,
            hw_arch=hw.architecture,
            usable_llc=usable_llc,
            verbose=True,
        )
        run_results.append(r)

    # Flag runs where token count was short
    short_runs = [r for r in run_results if not r.tokens_generated_ok]
    if short_runs:
        print(f"\nWARNING: {len(short_runs)} run(s) generated fewer tokens "
              f"than target. Results may be unreliable.")

    tps_mean, tps_std, mbt_mean, mbt_std, mbt_cv, tps_cv, notes = summarize_runs(
        run_results,
        perf_counter,
        llc_candidates,
        usable_llc,
    )

    cv_check = mbt_cv if mbt_cv is not None else tps_cv
    pass_criterion = (cv_check is not None and cv_check < 0.05)

    if pass_criterion:
        notes.append("Pass criterion met (CV < 5%)")
    elif cv_check is not None:
        notes.append(f"CV = {cv_check*100:.1f}% — exceeds 5% threshold")

    report = BaselineReport(
        model_path=str(model_path.absolute()),
        model_size_gb=round(model_size_gb, 3),
        hardware=hw,
        runs=[asdict(r) for r in run_results],
        mb_per_token_mean=round_optional(mbt_mean, 3),
        mb_per_token_stddev=round_optional(mbt_std, 3),
        mb_per_token_cv=round_optional(mbt_cv, 4),
        tokens_per_sec_mean=round(tps_mean, 2),
        tokens_per_sec_stddev=round(tps_std, 2),
        pass_criterion_met=pass_criterion,
        perf_available=any(r.perf_available for r in run_results),
        counter_used=perf_counter,
        notes=notes,
    )

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    if mbt_mean is not None:
        cv_str = f"{mbt_cv*100:.1f}%" if mbt_cv is not None else "N/A"
        print(f"MB/token (DRAM):  {mbt_mean:.1f} ± {mbt_std:.1f} MB  (CV: {cv_str})")
    print(f"Tokens/sec:       {tps_mean:.1f} ± {tps_std:.1f}")
    print(f"Counter used:     {perf_counter or 'None — tokens/sec only'}")
    print(f"Pass criterion:   {'PASS' if pass_criterion else 'FAIL'} (CV < 5%)")
    for note in notes:
        print(f"Note: {note}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"baseline": asdict(report)}, f, indent=2, )
    print(f"\nSaved: {out_path}")
    print(f"\nPost results to: https://github.com/acasavaraju/AIOS/issues/2")


if __name__ == "__main__":
    main()
