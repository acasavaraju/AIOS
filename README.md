# AIOS
### A CPU-Native Inference Architecture for Large Language Models

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-aios__paper.md-green)](paper/aios_paper.md)
[![Model Contract](https://img.shields.io/badge/Spec-Model%20Contract%20v1.0-orange)](spec/model_contract.md)
[![Validation Status](https://img.shields.io/badge/Validation-In%20Progress-yellow)](../../issues)
[![SSRN](https://img.shields.io/badge/SSRN-6467298-blue)](https://ssrn.com/abstract=6467298)

---

## Current State

**AIOS is a published framework and open specification. It is not yet a working implementation.**

| Component | Status |
|-----------|--------|
| Paper (SSRN 6467298) | Published |
| Model Contract spec | Complete |
| Validation tooling (`baseline.py`, `headroom.py`, `compliance.py`) | Runnable |
| Runtime C ABI (`aios.h`) | Specified — not implemented |
| Profiler (Python) | Stubbed — not implemented |
| Empirical results | None — all projections are analytical `[A]` or prior work `[P]` |

The performance projections in this README are derived analytically from published architecture parameters and prior published results. No inference has run under AIOS. **This is where contributions are needed.**

---

## Reality Check

- AIOS is a specification and validation framework, not a working runtime yet
- No model has run under the AIOS runtime — the runtime is not implemented
- All performance numbers are analytical `[A]` or literature-backed `[P]`
- The first milestone is measuring baseline MB/token on a stock GGUF model — not optimizing it

If you are looking for a production-ready inference system, this is not it yet.

If you want to help validate or build a CPU-native inference layer, start with [Issue #2](../../issues/2).

---

## Who This Is For

AIOS is relevant if you are working on any of the following:

**Inference engineers** building or maintaining CPU inference pipelines who have hit the memory bandwidth wall — models that are too slow on CPU because the weight access pattern is unmanaged, the KV cache grows unbounded, or activations spill to DRAM during long-context generation.

**ML systems researchers** interested in the co-design space between model architecture and inference runtime — specifically activation sparsity exploitation, inter-layer weight redundancy, and KV cache budget management at the systems level.

**Model architects** evaluating design choices (activation function, attention head count, intermediate dimensions) for deployment targets where GPU infrastructure is constrained — regulated environments, edge deployments, air-gapped systems, or cost-sensitive production workloads.

**Systems engineers** who want to build the memory management layer that does not currently exist for CPU inference — the equivalent of what the CUDA runtime does for GPU VRAM, but for CPU cache hierarchy.

**Validation contributors** with access to bare-metal Linux hardware and GGUF models who can run `perf stat` with uncore memory controller counters. Every open issue in the validation tracker is an experiment waiting to be run.

AIOS is **not** relevant if your primary workload requires sub-100ms latency (GPU inference is the right choice), if you are looking for a drop-in inference engine (AIOS is a layer that sits beneath one), or if you need a working implementation today (see Current State above).

---

## What AIOS Is

AIOS is a **memory residency controller** — a layer between inference engines (llama.cpp, Ollama, vLLM) and hardware that manages how weight data moves from DRAM to CPU. It is not an inference engine and does not replace one.

**The primary metric is MB/token** — total bytes read from DRAM (not LLC) per generated token, measured via hardware memory controller counters (e.g., `uncore_imc/data_reads/` on Intel x86). This is distinct from tokens/sec, which is a function of both MB/token and arithmetic throughput. Reducing DRAM reads directly reduces memory bandwidth pressure — the dominant bottleneck for 7B+ model inference on CPU.

AIOS decomposes the inference bandwidth problem into four independently addressable resource dimensions:

```
MB/token  ≈  Weight reads  +  KV cache reads  +  Activation spill
                  |                  |                   |
             Aliasing +         MQA/GQA            Chunked prefill
             sparsity map     bounds KV size       keeps acts in L3
```

Throughput (tokens/sec) is additionally affected by attention compute and arithmetic throughput — neither of which are memory bandwidth terms.

Existing approaches address at most one dimension. GGUF quantization reduces weight size but not access patterns. PowerInfer addresses sparsity alone. KV cache quantization (TurboQuant, H2O) addresses KV reads alone. AIOS is designed to coordinate all three bandwidth dimensions through a unified offline profiler and deterministic runtime. Today, only the measurement and specification layers exist.

**Core design principle:** All complexity belongs in the profiler, not the runtime. The profiler runs once per model per hardware configuration and produces static artifacts. At inference time, the runtime does exactly four things: resolve a pointer, apply a delta, skip a sparse row, issue a static prefetch. No dynamic allocation in the hot path. No data-dependent policy decisions in the hot path.

This design is fully specified. It is not yet implemented.

---

## The Architecture Gap

CPU inference is slow not because CPUs are unsuited to it. On modern server CPUs, on-chip cache bandwidth is significantly higher than DRAM bandwidth. The bottleneck for 7B+ CPU inference is data movement from DRAM, not arithmetic throughput.

Three gaps explain the current state:

**Gap 1 — Inference engines were never redesigned for CPU.**
llama.cpp, vLLM, and other engines were built for GPU memory access patterns. Weight layouts, prefetch strategies, and memory allocation patterns reflect HBM/VRAM assumptions. The CPU path is functional but not architecture-native.

**Gap 2 — Model architectures reflect GPU assumptions.**
SiLU and GELU activations produce near-zero outputs for only 20–35% of FFN neurons vs 90–95% true zeros for ReLU `[P]`. GQA with 8 KV heads produces a 512 MB KV cache at 4K context vs 32 MB for MQA `[A]`. These choices were made for GPU computational efficiency, not CPU cache hierarchy.

**Gap 3 — The memory management layer does not exist for CPU.**
On GPU, the driver and CUDA runtime manage VRAM residency, memory pinning, and prefetch scheduling. No equivalent exists for CPU inference. The OS page cache is heuristic and workload-unaware. No model-aware layer explicitly owns weight residency policy, KV budget enforcement, or access scheduling. AIOS is that layer.

---

## The AIOS Model Contract

The Model Contract specifies five architectural requirements that models can satisfy to expose progressively larger AIOS optimization surfaces. The runtime operates on any GGUF model at AIOS Compatible tier. Contract compliance unlocks higher tiers.

| Req | Specification | Rationale | Key Assumption |
|-----|--------------|-----------|----------------|
| **R1** | ReLU or ReLU² activation in FFN layers | Static sparsity maps require true zeros — SiLU/GELU produce near-zeros only, limiting reliable skip rates | Calibration set representative of inference distribution; sparsity ≥ 70% conservative mode |
| **R2** | MQA (1 KV head) or GQA ≤ 2 groups | KV cache ≤ 64 MB at 4K context — L3-resident on server CPUs, zero DRAM KV reads for standard workloads | L3 ≥ 32 MB; context ≤ 4K tokens for full KV benefit |
| **R3** | Explicit canonical + delta weights in layers 40–60% of depth, trained with delta regularization | Designed aliasing targets 70–80% deduplication vs 45–55% emergent | **Novel architecture — no existing model satisfies R3. Extrapolated from ALBERT cross-layer sharing. Highest-risk claim.** |
| **R4** | Per-layer activation footprint ≤ 200 KB at batch=1 (constrains FFN intermediate dimensions) | Activations stay in L2 during single-token generation — eliminates activation DRAM reads | 512 KB L2 per core; batch=1; single-token generation phase (not prefill) |
| **R5** | Frequency-ordered checkpoint + JSON access-frequency manifest | Profiler initialization < 60s vs 5–15 min without manifest | Manifest generated once by profiler and stored with checkpoint |

Full specification with validation protocols: [`spec/model_contract.md`](spec/model_contract.md)

---

## Performance Projections

**Claim notation:**

| Tag | Meaning |
|-----|---------|
| `[A]` | Analytical — arithmetic from published architecture parameters |
| `[P]` | Prior published work — supported by cited experimental results |
| `[V:#N]` | Empirical validation pending — tracked as GitHub issue #N |

Native and Native+ tiers represent architecture research targets, not currently existing models.

**Projections by tier:**

| Model | Tier | MB/token reduction | Key assumptions |
|-------|------|--------------------|-----------------|
| Mistral 7B Q4_K_M | Compatible | 50–58% `[A]` | SiLU limits sparsity to 20–35%; aliasing 45–55% in middle layers; GQA-8 KV tiered |
| Falcon 7B Relufied Q4_K_M | Optimized | 80–85% `[A][V:#2]` | 95% FFN sparsity `[P]`; MQA gives 32 MB KV at 4K `[A]`; aliasing 45–55% `[A]` |
| Purpose-built (R1+R2+R3) | Native | 80–88% `[V:#6]` | R3 unvalidated novel architecture; no trained model exists |
| Purpose-built (R1–R5) | Native+ | 85–92% `[V:#7]` | Depends on R3 validation |

**Assumptions underlying all projections:**
- Workload is memory-bandwidth-bound (holds for 7B+ models on server CPUs at batch=1)
- Single-tenant inference, batch size 1, autoregressive generation phase
- Q4_K_M quantization applied before profiling
- Target hardware: x86 server with L3 ≥ 32 MB, DDR4/DDR5, Linux with perf access
- No swap activity during inference (hard requirement)

MB/token reduction does not translate 1:1 to tokens/sec improvement when arithmetic throughput becomes a co-bottleneck. This is less likely at 7B+ scale on CPU but should be measured, not assumed.

---

## Reference Implementation: Falcon 7B Relufied

Falcon 7B is the primary validation target because it satisfies R1 (via relufication) and R2 (native MQA) — the two highest-impact requirements — at the lowest path cost from an existing model.

**Architecture parameters relevant to AIOS projections:**

```
Layers:                       32
Hidden dim:                   4,544
KV heads:                     1  (MQA — 71 query heads share one KV projection)
Head dim:                     64
FFN intermediate:             ~18,176

KV cache at 4K context:
  32 layers × 1 head × 64 dim × 4096 tokens × 4 bytes = 32 MB  [A]

FFN active neurons after 95% sparsity:
  18,176 × 0.05 ≈ 909 per layer  [P]

FFN weight data read per layer after sparsity:
  909 × 4,544 × 4 bytes ≈ 16.5 MB vs ~990 MB baseline  [A]
  Across 32 layers: ~528 MB vs ~31.7 GB  [A]
```

**Relufication protocol:** Replace GELU with ReLU in FFN layers, fine-tune on RefinedWeb. Phase 1: 30B tokens. Phase 2: 20B tokens. Total compute: ~3–5% of original pretraining `[P]`. Expected sparsity: ~95% `[P]`. Perplexity impact: negligible on standard benchmarks `[P]`. Full replication protocol: [Issue #1](../../issues/1).

**Why not Llama 3.1 8B?**
SiLU activation (20–35% near-zero, not true zeros), GQA-8 (512 MB KV at 4K context). Still a valid AIOS Compatible tier target and useful as a comparison baseline. The sparsity and KV benefits are substantially weaker.

---

## Integration Model

**ABI integration (recommended — not yet implemented):**
```c
#include "aios.h"

aios_context_t *ctx;
aios_config_t config = { .manifest_path = "aios_artifacts/manifest.json", ... };
aios_init(&ctx, &config);  // loads profiler artifacts, starts prefetch thread

// In inference hot path — called per weight block access:
const void *data; uint32_t len;
aios_resolve_block(ctx, logical_block_id, &data, &len);

// Per FFN row — sparsity map check:
if (aios_is_active(ctx, layer, row)) {
    // compute this row
}
```

**LD_PRELOAD (zero-modification — not yet implemented):**
```bash
LD_PRELOAD=/path/to/libaios_preload.so ollama run falcon
```

The ABI is fully specified in `runtime/aios.h`. The LD_PRELOAD interception approach intercepts `malloc()`/`mmap()` calls from the inference engine and redirects weight block allocation through the AIOS residency system — no source modification required. See `runtime/README.md` for implementation guide and recommended build order.

---

## Hardware Requirements

**For MB/token measurement (Phase 1 — Issue #2):**
```
OS:      Linux kernel 5.4+  (uncore PMU access via perf)
CPU:     Intel Haswell+ or AMD Zen+  (uncore_imc or amd_umc counters)
RAM:     ≥ 16 GB  (model is ~4 GB; must hold entirely in RAM, swap = 0)
perf:    sudo sh -c "echo 0 > /proc/sys/kernel/perf_event_paranoid"

# Verify counter availability:
perf list | grep -i "imc\|umc"
perf stat -e uncore_imc/data_reads/ sleep 0
```

WSL2 does not expose uncore PMU counters — tokens/sec only. Bare metal Linux or a cloud VM required for the primary metric (AWS c6i.2xlarge ~$0.34/hr is sufficient).

**For profiler development and compliance checking:**
```
Python 3.9+, any OS
pip install numpy scipy psutil huggingface_hub
Any GGUF model file
```

---

## Quickstart

```bash
git clone https://github.com/[your-github-username]/aios && cd aios
pip install -r validation/requirements.txt

# Download reference model (~4 GB)
huggingface-cli download maddes8cht/tiiuae-falcon-7b-gguf \
    tiiuae-falcon-7b-Q4_K_M.gguf --local-dir models/

# Expected output: JSON result files only.
# No performance improvement yet — runtime is not implemented.

# What can AIOS optimize on this model?
python validation/headroom.py \
    --model models/tiiuae-falcon-7b-Q4_K_M.gguf \
    --calibration validation/data/calibration_prompts.jsonl \
    --output results/headroom.json --verbose

# Model Contract compliance check
python validation/compliance.py \
    --model models/tiiuae-falcon-7b-Q4_K_M.gguf \
    --output results/compliance.json --verbose

# Baseline measurement (Linux + perf required for MB/token)
python validation/baseline.py \
    --model models/tiiuae-falcon-7b-Q4_K_M.gguf \
    --runs 5 --tokens 200 \
    --output results/baseline.json
```

---

## Validation Tracker

| Issue | Claim | Basis | Status |
|-------|-------|-------|--------|
| [#1](../../issues/1) | Falcon 7B relufied: 95% FFN sparsity, perplexity parity | `[P]` Mirzadeh et al. 2023 | Replication pending |
| [#2](../../issues/2) | Falcon 7B + AIOS: 80–85% MB/token reduction | `[A]` derived from #1 | Validation pending |
| [#3](../../issues/3) | LD_PRELOAD interception overhead < 0.5% | `[A]` | Pending implementation |
| [#4](../../issues/4) | Per-ISA-tier MB/token and tokens/sec ranges | `[A]` | Pending Tier 2/3 kernels |
| [#5](../../issues/5) | AIOS Optimized tier: 70–80% reduction | `[V]` depends on #1, #2 | Pending |
| [#6](../../issues/6) | R3 canonical+delta architecture: ≥ 70% aliasing ratio | Novel — no prior work | Pending training experiments |
| [#7](../../issues/7) | AIOS Native+: 85–92% reduction | `[V]` depends on R1–R5 | Pending |
| [#8](../../issues/8) | Chunked prefill: 40–60% latency reduction | `[V]` | Pending implementation |
| [#9](../../issues/9) | Falcon MQA: zero KV DRAM reads at ≤ 4K context | `[A]` | Pending measurement |
| [#10](../../issues/10) | No activation DRAM spill during generation (R4 model) | `[V]` | Pending R4 model |
| [#11](../../issues/11) | Profiler init < 60s with R5 manifest | `[V]` | Pending R5 implementation |

Post results in the issue — including negative results and results that contradict projections. Both are valuable.

---

## How to Falsify AIOS

The fastest way to contribute is to challenge the core assumptions. If any of these fail, AIOS assumptions need revision.

1. **Measure MB/token** on a baseline GGUF model with stock llama.cpp ([Issue #2](../../issues/2)) — establishes whether the bandwidth bottleneck is as large as projected
2. **Test weight aliasing** — verify that cross-layer cosine similarity in practice meets the thresholds assumed ([Issue #2](../../issues/2) headroom analysis)
3. **Verify KV residency behavior** — measure whether KV cache actually stays L3-resident at different context lengths with MQA models ([Issue #9](../../issues/9))
4. **Replicate relufication** — confirm 95% sparsity and perplexity parity on Falcon 7B ([Issue #1](../../issues/1))
5. **Measure interception overhead** — verify LD_PRELOAD overhead is below 0.5% once implemented ([Issue #3](../../issues/3))

Post results in the relevant issue. Negative results are as valuable as confirmations.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Priority order:

**1. Run Issue #2 baseline** — measure MB/token on any GGUF model on Linux hardware. Even tokens/sec-only (WSL2 / no perf counters) is useful. One real number closes the most critical open question.

**2. Replicate Falcon 7B relufication (Issue #1)** — fine-tune Falcon 7B GELU→ReLU per the protocol in the issue. Produces the first validated AIOS Model Catalog entry.

**3. Implement `runtime/pointer_table.c`** — the most impactful single runtime component. Interface fully specified in `runtime/aios.h`. This enables the residency and prefetch systems to function. Start here before any other runtime work.

**4. R3 training experiments (Issue #6)** — train a model with explicit canonical + delta parameterization in middle-zone layers. Novel architecture. No existing model satisfies R3. See issue for experiment protocol.

**5. Model catalog entries** — run `compliance.py` on any GGUF model, open a PR to `models/catalog.md`. Any tier is useful.

---

## Repository Structure

```
aios/
├── paper/aios_paper.md          ← Full paper — cite this (SSRN 6467298)
├── spec/model_contract.md       ← Model Contract v1.0 — standalone testable spec
├── validation/
│   ├── baseline.py              ← Phase 1: MB/token + tokens/sec measurement
│   ├── headroom.py              ← Phase 2: aliasing/sparsity/re-read analysis
│   ├── compliance.py            ← Model Contract R1–R5 compliance checker
│   └── data/calibration_prompts.jsonl
├── profiler/
│   └── run.py                   ← Stubbed — see profiler/README.md for what to build
├── runtime/
│   ├── aios.h                   ← Complete public ABI — start here for runtime impl
│   └── README.md                ← Component guide and recommended build order
└── models/catalog.md            ← Validated model entries
```

---

## Citing This Work

```bibtex
@misc{casavaraju2026aios,
  title  = {AIOS: A CPU-Native Inference Architecture for Large Language Models},
  author = {Casavaraju, Anand},
  year   = {2026},
  url    = {https://ssrn.com/abstract=6467298}
}
```

## License

Apache 2.0. See [`LICENSE`](LICENSE).
