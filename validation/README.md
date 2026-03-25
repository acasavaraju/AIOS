# AIOS Validation Suite

This directory contains the tools to validate quantitative claims in the AIOS paper.
Every claim labeled `[V:#N]` in the paper has a corresponding GitHub issue with a
protocol. Running these scripts and posting results to those issues is the highest-impact
contribution you can make.

---

## Setup

```bash
cd validation
pip install -r requirements.txt
```

**For MB/token measurement (Linux only):**
The primary metric requires hardware memory controller counters via `perf`.

```bash
# Check perf is available and counters are accessible
perf stat -e uncore_imc/data_reads/ sleep 0 2>&1
# If you see "Permission denied": sudo sh -c "echo 0 > /proc/sys/kernel/perf_event_paranoid"
# If the counter is missing: your CPU may use different counter names — see baseline.py --list-counters
```

**For macOS (Apple Silicon):**
Hardware memory counters require `sudo` and the Instruments toolchain.
See `baseline.py --platform macos` for the alternative measurement path.

---

## Scripts

### 1. `baseline.py` — Phase 1: Measure your baseline `[Issue #2]`

Measures MB/token from DRAM, tokens/sec, and variance across runs on any GGUF model.
**This is the prerequisite for all other validation.** Run this first.

```bash
python baseline.py \
    --model path/to/model.gguf \
    --runs 5 \
    --tokens 200 \
    --output results/baseline.json
```

Pass criterion: < 5% coefficient of variation across runs.

Output includes: MB/token mean and stddev, tokens/sec, LLC miss rate, hardware config.

---

### 2. `headroom.py` — Phase 2: Measure AIOS optimization potential `[Issue #2]`

Analyzes a GGUF model to determine how much AIOS can achieve on it.
Reports: aliasable block fraction, sparsity fraction, re-read ratio, estimated tier.

```bash
python headroom.py \
    --model path/to/model.gguf \
    --calibration data/calibration_prompts.jsonl \
    --output results/headroom.json
```

Interprets results against the Phase 2 thresholds from the paper:
- Weight aliasing: > 20% of blocks aliasable at similarity ≥ 0.95 in middle layers
- Sparsity: > 30% of FFN neurons near-zero at ε = 0.001 with < 0.5% perplexity impact
- Re-reads: > 40% of DRAM reads are re-reads of already-accessed blocks

---

### 3. `compliance.py` — Model Contract compliance check `[Issues #5, #6, #7]`

Checks a model against all five AIOS Model Contract requirements (R1–R5).
Reports compliance tier (Compatible / Optimized / Native / Native+).

```bash
python compliance.py \
    --model path/to/model.gguf \
    --calibration data/calibration_prompts.jsonl \
    --output results/compliance.json
```

---

## Reporting Results

Post results to the relevant GitHub issue. Include:

```
Hardware: [CPU model, core count, RAM GB, L3 cache size]
OS: [Linux/macOS/Windows, kernel version]
Model: [name, quantization level, parameter count]
AIOS version: [git commit hash]
Results: [attach JSON output]
```

Negative results are as valuable as positive ones.
If a projection is wrong, that is important information — say so clearly.

---

## Calibration Data

`data/calibration_prompts.jsonl` contains 500 diverse prompts for calibration passes.
Format: one JSON object per line with a `"text"` field.

To use your own calibration data:
```bash
python headroom.py --model model.gguf --calibration my_prompts.jsonl
```

Custom calibration data should be representative of your actual inference workload.
