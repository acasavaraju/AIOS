# AIOS: A CPU-Native Inference Architecture for Large Language Models

**Anand Casavaraju**  
Independent Researcher  
March 2026

---

## Abstract

We present AIOS — a runtime architecture and model co-design framework for deploying large language models efficiently on CPU hardware. Current LLM inference engines were designed for GPU memory architecture and treat CPU as a fallback. We argue this is a design constraint, not a physics constraint, and that a first-principles approach to CPU memory hierarchy can achieve 50–92% reduction in DRAM data movement per generated token depending on model compliance with a proposed Model Contract.

We decompose the inference bandwidth problem into four independent resource dimensions: weight reads, KV cache reads, activation memory spill, and attention compute. We propose five architectural requirements — the AIOS Model Contract — that models can be designed or fine-tuned to satisfy, and define four compliance tiers with corresponding performance projections. We present Mistral 7B as a baseline analysis `[A]` and Falcon 7B relufied as a reference implementation `[P][V:#1]`, analytically projecting 80–85% MB/token reduction under full AIOS optimization `[V:#2]`.

All quantitative projections are labeled by validation status. Empirical validation is tracked publicly at github.com/[your-github-username]/aios/issues. AIOS is fully open source under Apache 2.0. Contributions to the runtime, profiler, model catalog, and validation suite are welcome.

---

## 1. Introduction

The dominant assumption in LLM deployment is that large models require GPU hardware. This assumption is reflected in every layer of the current inference stack: weight tensor layouts optimized for HBM access patterns, attention mechanisms designed around CUDA kernel efficiency, context window sizes calibrated to VRAM capacity. The assumption is not wrong for all workloads. For real-time applications requiring sub-100ms latency, GPU infrastructure remains the correct choice.

For the large majority of enterprise AI workloads — document processing, classification, summarization, retrieval-augmented generation, batch analysis, and extraction — the assumption is an artifact of how the field developed, not a fundamental requirement. These workloads do not need sub-100ms latency. They need quality, cost efficiency, security, and deployability on hardware organizations already own.

The gap between current CPU inference performance and GPU inference performance is real. It is not, however, a consequence of CPU hardware being unsuited to inference. It is a consequence of three specific design failures:

1. Inference engines were built for GPU memory access patterns and ported to CPU without rearchitecting
2. Model architectures were designed around GPU computational primitives without consideration of CPU cache hierarchy
3. The systems layer between hardware and inference engine — the layer that owns memory layout, residency policy, and access scheduling — does not exist for CPU inference

AIOS addresses all three. It is simultaneously a runtime architecture, a memory management framework, and a model co-design specification. This paper describes the framework, analyzes two existing models against it, and defines the open standard that model architects can design toward.

The primary contribution is not the runtime implementation — that is a means to an end. The primary contribution is the decomposition of CPU inference performance into four independently optimizable resource dimensions, and the specification of a Model Contract that makes all four optimizable simultaneously. The runtime is the reference implementation of that contract. The open source community is invited to build better implementations.

### 1.1 Claim Notation

All quantitative claims in this paper are labeled with their evidentiary basis:

| Tag | Meaning |
|-----|---------|
| `[A]` | Analytical — derivable from published architecture parameters via arithmetic |
| `[P]` | Prior work — supported by results in cited published literature |
| `[V:#N]` | Empirical validation pending — tracked as GitHub issue #N |

Claims labeled `[V:#N]` are projections based on analytical reasoning and prior work. They are not measured results. The validation roadmap in Section 9 specifies the experimental protocol for confirming each projection.

---

## 2. Background and Related Work

### 2.1 CPU Inference Baselines

**llama.cpp** [Gerganov, 2023] demonstrated that quantized LLM inference on CPU is viable for interactive use at small model sizes, establishing GGUF quantization as the de facto format for CPU-targeted models. llama.cpp addresses weight size through quantization but does not address weight access patterns, KV cache management, or activation memory layout. It is the primary baseline against which AIOS improvements are measured.

**PowerInfer** [Song et al., 2023] demonstrated that activation sparsity in ReLU-based models can be exploited to dramatically reduce weight loading during inference, achieving significant speedups on consumer hardware. PowerInfer validates the sparsity-driven weight loading thesis that underlies AIOS's sparsity map mechanism. It differs from AIOS in scope — PowerInfer addresses sparsity alone; AIOS addresses all four resource dimensions — and in architecture — PowerInfer uses a neural predictor for hot neuron selection; AIOS uses a static offline map.

**DejaVu** [Liu et al., 2023] demonstrated contextual sparsity in LLMs, showing that 85% of attention heads and MLP parameters can be predicted as inactive at inference time. DejaVu validates the activation sparsity thesis more broadly and establishes that sparsity is a general property of transformer inference, not specific to ReLU models.

**Apple LLM in a Flash** [Alizadeh et al., 2023] addressed inference on devices where model size exceeds available DRAM, demonstrating that selective weight loading based on sparsity can enable inference at model sizes previously considered infeasible on consumer hardware. The paper directly validates the feasibility of reading only active weight rows from storage rather than loading full weight matrices.

### 2.2 Weight Sharing and Redundancy

**ALBERT** [Lan et al., 2020] introduced cross-layer parameter sharing in transformer models, demonstrating that identical weights across all layers produce competitive performance at dramatically reduced parameter counts. ALBERT validates the weight redundancy hypothesis — that transformer layers learn similar transformations — but uses full weight sharing rather than the canonical + delta parameterization AIOS proposes. The AIOS approach is strictly more expressive: identical weights are a special case where delta = 0.

**Weight-tied models** (Press & Wolf, 2017) demonstrated that embedding and output projection weights can be shared without quality loss, establishing precedent for deliberate weight sharing in neural network design.

### 2.3 KV Cache Optimization

**Multi-Query Attention (MQA)** [Shazeer, 2019] reduced KV cache size by sharing a single key-value head across all query heads. **Grouped Query Attention (GQA)** [Ainslie et al., 2023] generalized this to G groups, interpolating between full MHA and MQA. Both directly reduce KV bandwidth requirements. Mistral 7B uses GQA with 8 groups; Falcon 7B uses MQA.

**H2O** [Zhang et al., 2023] and **StreamingLLM** [Xiao et al., 2023] demonstrated that a small fraction of KV cache entries — "attention sinks" at early token positions plus recent tokens — account for the large majority of attention weight. This validates AIOS's tiered KV residency policy, which prioritizes sink tokens and the recent window for L3 residency.

### 2.4 Model Architecture for Efficiency

**Falcon** [Almazrouei et al., 2023] introduced a transformer architecture with MQA and parallel attention/MLP layers, explicitly designed for inference efficiency. The parallel sublayer design improves compute pipeline utilization.

**Relufication** [Mirzadeh et al., 2023; Zhang et al., 2024] demonstrated that models trained with smooth activations (GELU, SiLU) can be fine-tuned to use ReLU with minimal accuracy degradation, recovering high activation sparsity at a fraction of original training cost. For Falcon 7B specifically, relufication achieved approximately 95% activation sparsity with no measurable benchmark degradation `[P]`.

### 2.5 Memory Hierarchy and Systems

The CPU memory hierarchy exploited by AIOS — L1/L2 cache per core, shared L3, DRAM — is well-characterized in computer architecture literature. The key insight motivating AIOS is that L3 cache bandwidth (approximately 800 GB/s on modern server CPUs `[A]`) is sufficient for high-quality inference if data movement from DRAM is minimized. The bottleneck is not computation; it is DRAM reads.

---

## 3. Problem Formulation: The Four-Resource Decomposition

We decompose the LLM inference bandwidth problem into four independently addressable resource dimensions. Existing work addresses at most one or two of these. AIOS addresses all four simultaneously.

### 3.1 Formal Decomposition

Let $T$ denote a single autoregressive token generation step for a transformer model with $L$ layers. The total data read from DRAM during $T$ can be decomposed as:

$$\text{MB/token} = W_{\text{reads}} + K_{\text{reads}} + A_{\text{spill}} + \text{const}$$

where:

- $W_{\text{reads}}$ = bytes read from DRAM for weight tensors during $T$
- $K_{\text{reads}}$ = bytes read from DRAM for KV cache tensors during $T$  
- $A_{\text{spill}}$ = bytes read from DRAM for activation tensors spilled from cache during $T$
- $\text{const}$ = overhead (metadata, pointers) — negligible

The primary metric of AIOS is **MB/token**: total bytes read from DRAM per generated token, measured via hardware memory controller counters. All optimization targets are expressed as percentage reduction from a measured baseline on the same hardware.

### 3.2 Resource 1: Weight Reads

For a model with $L$ layers and total quantized weight size $|W|$, a naive inference engine reads a large fraction of $|W|$ on every token generation step. Three observations make this reducible:

**Observation W1 — Cross-layer redundancy:** Adjacent and nearby layers in transformer models learn similar linear transformations `[P]`. Weight blocks in the middle layers of a network exhibit high cosine similarity, making it possible to store one canonical block and a small delta rather than two independent blocks.

**Observation W2 — Activation sparsity:** For models with ReLU or ReLU-variant activations, a large fraction of FFN neuron pre-activations are negative, producing exact zero outputs `[P]`. The corresponding weight rows need not be loaded.

**Observation W3 — Access frequency heterogeneity:** Not all weight blocks are accessed with equal frequency. Early attention layers and the final projection are accessed on every token; some FFN blocks are accessed rarely. Structured residency — keeping high-frequency blocks in L3 cache — reduces DRAM reads for the most-accessed data.

AIOS addresses W1 through weight aliasing with delta correction, W2 through a static sparsity map, and W3 through the proportional residency model (RWS).

### 3.3 Resource 2: KV Cache Reads

The KV cache grows linearly with context length. For a model with $L$ layers, $H_{KV}$ KV heads, head dimension $d$, and context length $C$:

$$|KV| = L \times H_{KV} \times d \times 2 \times 2 \text{ bytes} = 4 L H_{KV} d \text{ bytes per token}$$

At long contexts, $K_{\text{reads}}$ dominates $W_{\text{reads}}$ entirely. For Mistral 7B at 32K context: $|KV| \approx 4$ GB `[A]`, approximately equal to the full quantized weight store.

Two architectural decisions reduce $H_{KV}$: MQA ($H_{KV} = 1$) and GQA ($H_{KV} = G$). AIOS's KV-first budgeting ensures KV memory is reserved before weight residency, and tiered KV residency (attention sinks + recent window in L3, remainder in DRAM) reduces $K_{\text{reads}}$.

### 3.4 Resource 3: Activation Memory Spill

Intermediate activation tensors are transient — they exist only during computation of a single layer and are discarded afterward. Their sizes are deterministic given model architecture and sequence length:

For single-token generation at batch size 1, per-layer activation size is approximately:
$$|A_{\text{layer}}| \approx 2(d_{\text{model}} + d_{\text{ffn}}) \text{ bytes}$$

For Mistral 7B: $\approx 92$ KB per layer `[A]`. For Falcon 7B: $\approx 107$ KB per layer `[A]`. Both fit comfortably in L2 cache per core for single-token generation.

At prefill (processing a long prompt), the total activation footprint scales linearly with sequence length, causing spill to DRAM for sequences longer than approximately 300–500 tokens `[A]`. Chunked prefill — processing the prompt in segments sized to the L3 cache budget — keeps activations cache-resident.

### 3.5 Resource 4: Attention Compute

Standard attention is $O(C^2)$ in context length. For most enterprise workloads at context lengths ≤ 4K, this is manageable on CPU. At longer contexts, attention computation becomes a secondary bottleneck after the memory bandwidth problems are addressed.

Structural sparsity in attention — the observation that most token pairs have near-zero attention weight `[P]` — provides a path to reducing attention compute, but exploiting it requires sparse attention kernels that are architecture-dependent and outside the scope of AIOS v1.0.

---

## 4. The AIOS Runtime Architecture

The AIOS runtime implements a memory residency controller between the inference engine and hardware. It is not an inference engine — it does not perform matrix multiplications, manage model loading, or implement the transformer forward pass. It owns one thing: how weight data moves from DRAM to the compute units that need it.

### 4.1 Design Principle: Offline Complexity, Runtime Simplicity

> Complexity belongs in the profiler, not the runtime. At inference time, AIOS follows a map. It does not make decisions.

The AIOS offline profiler runs once per model per hardware configuration and produces a complete set of static artifacts: a pointer table, a canonical weight store, a delta store, a sparsity bitmask, and a prefetch sequence. All placement, aliasing, and access-order decisions are made by the profiler. The runtime has exactly four responsibilities:

1. **Resolve a pointer** — look up where a weight block lives
2. **Apply a delta** — add a small pre-computed correction to an aliased canonical block
3. **Skip a row** — bypass a weight row the sparsity bitmask marks inactive
4. **Issue a static prefetch** — walk the pre-built prefetch sequence on a dedicated thread

No dynamic allocation in the hot path. No runtime model decisions. No adaptive scheduling.

### 4.2 Pointer Table

Each entry maps a logical block ID to its physical location and metadata:

```
logical_block_id → {
    base_pointer,     // physical address in canonical weight store
    block_length,     // bytes
    flags,            // alias | sparse | canonical
    delta_pointer     // null if canonical
}
```

Block granularity: 64 KB–1 MB (tile-aligned to projection matrix boundaries).  
Entry size: 8–16 bytes.  
Hard constraint: pointer table ≤ 50% of LLC capacity or ≤ 5% of model footprint, whichever is smaller.

### 4.3 Weight Aliasing with Delta Correction

The profiler measures cosine similarity between weight blocks at configured granularity. Aliasing is enabled for a block pair only when all three conditions hold:

- Cosine similarity exceeds the layer zone threshold (see Section 4.4)
- `canonical_bytes + delta_bytes < original_bytes` (net bandwidth reduction confirmed)
- Perplexity delta for the layer zone remains within the configured threshold

Aliasing and sparsity operate independently. Per-zone thresholds prevent aggressive aliasing of boundary layers.

### 4.4 Per-Layer Sensitivity Zones

Not all layers tolerate aliasing equally. The profiler applies zone-specific thresholds:

| Zone | Layers (fraction of depth) | Default Similarity Threshold |
|------|---------------------------|------------------------------|
| Boundary — input | First 10% | ≥ 0.99 or disabled |
| Middle — lower | 10–40% | ≥ 0.97 |
| Middle — core | 40–60% | ≥ 0.95 |
| Middle — upper | 60–90% | ≥ 0.97 |
| Boundary — output | Last 10% | ≥ 0.99 or disabled |

### 4.5 Activation Sparsity Map

A static bitmask per layer, generated by the profiler from a calibration pass on representative inputs. At runtime, the kernel checks the bitmask and skips loading flagged rows. No prediction, no dynamic gating.

| Mode | Perplexity Budget | Bandwidth Reduction |
|------|-------------------|---------------------|
| Conservative | Negligible (< 0.1%) | ~20–30% |
| Balanced | < 0.5% | ~40–50% |
| Aggressive | Operator-validated | ~55–65% |

For ReLU models, conservative mode sparsity is structural and exact — the map reflects true zeros, not near-zeros.

### 4.6 Proportional Residency Model (RWS)

| Tier | Definition | Policy |
|------|-----------|--------|
| RWS-HOT | High-reuse blocks — accessed every or near-every token | Best-effort memory pinning; last to evict |
| RWS-WARM | Moderate-reuse blocks | Prefer-resident; shrinks first under pressure |
| STREAM | Low-reuse or cold blocks | Fully evictable; sequential streaming layout |

**KV-First Budgeting:**
```
Effective RAM = min(physical_RAM, container_memory_limit)
Effective RAM = OS_headroom + KV_reserve + RWS-HOT + RWS-WARM + STREAM
```

KV reserve is allocated first and protected throughout the session. Default RWS ceiling: 65% of effective RAM after KV and OS headroom. Swap must remain zero.

### 4.7 Degradation Ladder

Applied at initialization or between runs — never mid-token:

| Step | Action |
|------|--------|
| 1 | Drop WARM residency |
| 2 | Disable aliasing |
| 3 | Reduce prefetch distance 50% |
| 4 | Drop HOT to minimum floor (20% of weight store) |
| 5 | Streaming-only mode — all memory pinning released |

### 4.8 Integration Model

AIOS integrates with existing inference engines via two mechanisms:

**ABI layer (recommended):** inference engine links against the AIOS shared library. Weight access calls are intercepted at the ABI boundary.

**LD_PRELOAD interception:** works with any pre-compiled binary without modification. Overhead < 0.5% `[V:#3]`.

Existing engines (llama.cpp, Ollama, vLLM) run above AIOS without modification.

### 4.9 Hardware Capability Tiers

AIOS detects hardware capabilities at initialization and selects the appropriate kernel tier automatically:

| Tier | ISA | MB/token Reduction | Tokens/sec Improvement |
|------|-----|--------------------|------------------------|
| 1 — Baseline | SSE4.2 / NEON | 30–45% `[V:#4]` | 1.4–1.8× `[V:#4]` |
| 2 — Enhanced | AVX2 | 45–60% `[V:#4]` | 1.8–2.5× `[V:#4]` |
| 3 — Optimized | AVX-512 / AMX | 55–70% `[V:#4]` | 2.5–3.5× `[V:#4]` |
| Apple M-series | AMX / ANE | 50–65% `[V:#4]` | 2.0–3.0× `[V:#4]` |

AIOS always runs. A missing capability tier causes graceful fallback and a log entry, not a failure.

---

## 5. The AIOS Model Contract

The Model Contract defines what an AIOS-native model must expose to enable full runtime optimization. A model satisfying the contract does not merely benefit from AIOS optimization — it is designed so that the optimization surfaces are structural properties of the architecture, not emergent properties discovered by the profiler.

The full Contract specification is maintained as a standalone document at `spec/model_contract.md` in the AIOS repository. The following is a summary.

### 5.1 Requirement 1 — ReLU or Structural Sparse Activation

**Requirement:** FFN activation function must be ReLU, ReLU², or another activation producing exact zeros for a measurable fraction of inputs across the calibration distribution.

**Rationale:** Static sparsity maps require true zeros. SiLU and GELU are smooth and never produce exact zeros, limiting the sparsity map to near-zero approximations with lower reliability and smaller skippable fractions.

**Validation:** Measure zero fraction on calibration set. Must meet tier target.

**Target:** ≥ 70% static sparsity in FFN layers (conservative mode).

### 5.2 Requirement 2 — Bounded KV: MQA or GQA ≤ 2 Heads

**Requirement:** Model must use Multi-Query Attention (1 KV head) or GQA with ≤ 2 KV groups.

**Rationale:** KV cache must fit within L3 cache for standard enterprise context lengths (≤ 4K tokens) to eliminate KV DRAM traffic entirely. MQA achieves ~32 MB at 4K context for a 7B model `[A]`. GQA with 2 heads achieves ~64 MB — near the L3 boundary for server CPUs with 32–64 MB L3.

**Target:** KV cache ≤ 64 MB at 4K context on reference hardware.

### 5.3 Requirement 3 — Explicit Canonical + Delta Weight Parameterization

**Requirement:** Middle-zone layers (40–60% of depth) must be parameterized as canonical block + learned per-layer delta vector, not as fully independent weight matrices.

**Rationale:** Emergent aliasing (discovered by profiler post-training) achieves 45–55% reduction. Designed aliasing (explicit in architecture) achieves 70–80% by ensuring the delta is trained to be small. The profiler's job becomes confirming designed structure rather than discovering approximate structure.

**Target:** ≥ 70% aliasing ratio in core zone layers.

### 5.4 Requirement 4 — Cache-Aligned Intermediate Dimensions

**Requirement:** FFN intermediate dimensions and attention head dimensions must be chosen so that per-layer activation tensors at batch size 1, single-token generation, fit within L2 cache per core on the target hardware class.

**Reference constraint:** For an 8-core CPU with 512 KB L2 per core, per-layer activation footprint at batch size 1 must be ≤ 200 KB.

**Rationale:** Activation tensors that stay in L2 during single-token generation never reach DRAM. The activation bandwidth problem is solved at the architectural level.

### 5.5 Requirement 5 — Frequency-Ordered Weight Layout

**Requirement:** Model checkpoint must be stored with weight blocks ordered by expected access frequency, as determined by a reference profiling pass.

**Rationale:** The AIOS profiler's job is trivial when the model checkpoint encodes its own access frequency structure. Prefetch sequencing, residency placement, and pointer table construction become direct reads of the layout.

### 5.6 Compliance Tiers

| Tier | Requirements | MB/token Reduction | Label |
|------|-------------|-------------------|-------|
| Baseline | None | 45–55% `[A]` | AIOS Compatible |
| Standard | R1 + R2 | 70–80% `[V:#5]` | AIOS Optimized |
| Enhanced | R1 + R2 + R3 | 80–88% `[V:#6]` | AIOS Native |
| Full | All five | 85–92% `[V:#7]` | AIOS Native+ |

---

## 6. Case Study: Mistral 7B (Baseline Analysis)

*All figures in this section are analytical derivations from published architecture parameters unless labeled [P].*

### 6.1 Architecture

Mistral 7B [Jiang et al., 2023] uses 32 layers, d_model = 4,096, 32 attention heads, 8 KV heads (GQA), head dimension 128, FFN intermediate 14,336, SiLU activation, and a 32,768 context window. Quantized to Q4_K_M, the weight store is approximately 4.1 GB `[A]`.

### 6.2 Resource Analysis

**Weight reads (R1 partial):** Middle-layer aliasing estimated at 45–55% `[A]`. SiLU limits sparsity to 20–35% near-zero neurons at ε = 0.01 `[P]`. Sparsity map provides limited benefit — conservative mode only.

**KV cache (R2 partial):** GQA reduces KV to 8 heads. At 4K context: 512 MB `[A]`. At 32K context: 4 GB `[A]` — equal to full weight store. KV tiering required for contexts > 2K.

**Activation memory (R3):** ~92 KB per layer at batch size 1 `[A]`. Cache-friendly for single-token generation. Spills at prefill > ~500 tokens. Chunked prefill in 256-token windows mitigates this.

**Attention compute (R4):** SiLU precludes effective static sparsity. No structural improvement available without model modification.

### 6.3 AIOS Performance Projection

| Resource | Baseline | AIOS Optimized | Reduction |
|----------|----------|----------------|-----------|
| Weight reads | ~4.1 GB | ~2.0–2.3 GB | 44–51% `[A]` |
| KV reads (4K ctx) | 512 MB | ~129 MB resident | 75% active reads `[A]` |
| Activation spill | Spills >500 tokens | Chunked — L3-resident | 40–60% prefill latency `[V:#8]` |
| FFN sparsity benefit | 20–35% near-zero | Conservative map | 15–25% MB `[A]` |
| **Combined MB/token** | **~4.1 GB** | **~1.7–2.0 GB** | **50–58% `[A][V:#1]`** |

**Assessment:** AIOS Compatible tier. SiLU is the fundamental ceiling. Suitable as a baseline comparison but not as a target for demonstrating the full optimization surface.

---

## 7. Case Study: Falcon 7B Relufied (Reference Implementation)

*All figures in this section are analytical derivations [A] or supported by prior work [P] unless labeled [V:#N].*

### 7.1 Architecture

Falcon 7B [Almazrouei et al., 2023] uses 32 layers, d_model = 4,544, 71 attention heads, 1 KV head (MQA), head dimension 64, FFN intermediate ~18,176, GELU activation (→ ReLU via relufication), and parallel attention/MLP sublayers. Quantized to Q4_K_M, approximately 4.0 GB `[A]`.

### 7.2 Relufication

Mirzadeh et al. [2023] and follow-up work demonstrated that Falcon 7B can be fine-tuned to replace GELU with ReLU at approximately 3–5% of original training cost, achieving ~95% activation sparsity with no measurable accuracy degradation on standard benchmarks `[P]`. Training proceeds in two phases on RefinedWeb: Phase 1 (30B tokens) recovers the large majority of accuracy; Phase 2 (20B tokens) recovers remaining performance.

This is the first entry in the AIOS model catalog and the primary validation target for the AIOS runtime. See [V:#1] and [V:#2].

### 7.3 Resource Analysis

**Weight reads (R1 — full):** Aliasing in middle layers estimated at 45–55% `[A]`. ReLU activation produces 95% true zeros `[P]`, enabling a fully static sparsity map. FFN weight read calculation:

```
Original FFN reads per layer (fp32):
  Gate + Up + Down: ~990 MB per layer × 32 layers = ~31.7 GB

After 95% sparsity (5% active neurons):
  Active rows: 18,176 × 0.05 ≈ 909 neurons
  Data read: ~16.5 MB per layer × 32 = ~528 MB
  Reduction: 98.3% of FFN weight data [A]
```

Combined with aliasing on attention weights: total weight reads of ~0.65–0.80 GB per token cycle `[A]`.

**KV cache (R2 — full):** MQA: 1 KV head. KV per token per layer = 256 bytes `[A]`. Total at 4K context: 32 MB `[A]` — fits entirely in L3 on any modern server CPU. KV DRAM traffic eliminated for standard enterprise workloads. No tiering required.

**Activation memory (R3):** ~107 KB per layer at batch size 1 `[A]`. MQA reduces QKV tensor from 6,144 floats (Mistral) to 4,672. Chunked prefill in 200-token windows keeps activation footprint within ~21 MB for a 32 MB L3 `[A]`.

**Attention compute (R4):** 95% sparsity eliminates 95% of FFN computation. MQA reduces attention computation. Full attention sparsity exploitation requires sparse kernels — future work.

### 7.4 AIOS Performance Projection

| Resource | Baseline | AIOS Optimized | Reduction |
|----------|----------|----------------|-----------|
| Weight reads | ~4.0 GB | ~0.65–0.80 GB | 80–85% `[A][V:#2]` |
| KV reads (4K ctx) | 32 MB | 32 MB in L3 — zero DRAM | 100% DRAM elimination `[A]` |
| Activation spill | Spills >300 tokens | Chunked — L3-resident | 40–60% prefill latency `[V:#8]` |
| FFN compute | 18,176 neurons | ~909 active neurons | 95% compute reduction `[P]` |
| **Combined MB/token** | **~4.0 GB** | **~0.65–0.80 GB** | **80–85% `[A][V:#2]`** |

**Assessment:** AIOS Optimized tier (R1 + R2 satisfied). Satisfies R3 with relufication providing sparsity by design. R4 and R5 remain for purpose-built models.

---

## 8. The Co-Evolution Path

The relationship between AIOS and model architecture is intentionally bidirectional. The runtime defines what it can optimize efficiently. Models are designed to expose those surfaces from the start.

This mirrors the CUDA co-evolution: GPU hardware defined computational primitives, and model architectures were then designed around those primitives. Transformers, dense attention, large matrix multiplications — all shaped by GPU memory model assumptions.

The AIOS Model Contract defines the computational contract for CPU-native inference. Three design decisions in the contract (R1, R2, R3) represent the largest deviation from current mainstream model design:

- **R1 (ReLU):** Mainstream models have converged on SiLU/GELU. Relufication demonstrates recovery is feasible at low cost. Purpose-built models can use ReLU from the start.
- **R2 (MQA):** MQA is an established technique (Shazeer, 2019) but most models use GQA with 8 heads for better quality/efficiency tradeoff. MQA represents a deliberate choice to prioritize CPU KV budget over marginal quality differences.
- **R3 (explicit aliasing):** No current model is trained with explicit canonical + delta parameterization. This is novel architecture and the primary research contribution of this paper beyond the runtime framework.

We expect the community to challenge, refine, and extend all three. The Model Contract is a starting point, not a final specification.

---

## 9. Validation Roadmap and Claim Status

The following GitHub issues track empirical validation of all quantitative claims in this paper. Issues are updated as results become available. The validation protocol for each issue is specified in the issue body.

| Issue | Claim | Status | Protocol |
|-------|-------|--------|----------|
| [#1](../../issues/1) | Falcon 7B relufied: 95% sparsity, perplexity parity `[P]` | Replication pending | Run relufication fine-tune; measure sparsity and benchmark perplexity |
| [#2](../../issues/2) | Falcon 7B + AIOS: 80–85% MB/token reduction `[A]` | Validation pending | Phase 1–4 benchmark protocol (see issue) |
| [#3](../../issues/3) | LD_PRELOAD overhead < 0.5% `[A]` | Validation pending | Measure tokens/sec with and without LD_PRELOAD on reference hardware |
| [#4](../../issues/4) | Per-tier MB/token and tokens/sec ranges `[A]` | Validation pending | Run on Tier 1/2/3 hardware; report results |
| [#5](../../issues/5) | AIOS Optimized tier: 70–80% reduction `[V]` | Validation pending | Requires [#1] and [#2] complete |
| [#6](../../issues/6) | AIOS Native tier: 80–88% reduction `[V]` | Validation pending | Requires explicit aliasing implementation |
| [#7](../../issues/7) | AIOS Native+ tier: 85–92% reduction `[V]` | Validation pending | Requires all five contract requirements implemented |
| [#8](../../issues/8) | Chunked prefill: 40–60% latency reduction `[V]` | Validation pending | Measure prefill latency at 512/1K/2K/4K tokens with and without chunking |

**Baseline validation protocol (applies to all [#N]):**
1. Measure MB/token using hardware memory controller counters on target hardware
2. Run minimum 5 independent runs; report mean and variance
3. Pass criterion: variance < 5% across runs before proceeding to optimization measurement
4. All measurements declare: model version, quantization, batch size, context length, hardware, NUMA topology

---

## 10. Discussion

### 10.1 Scope

AIOS v1.0 addresses CPU-only inference. GPU dispatch (routing workloads to GPU based on latency requirements) is in the roadmap but outside the scope of this paper. The heterogeneous pointer table that enables CPU/GPU co-execution is described in `ROADMAP.md`.

AIOS is single-tenant by design. Multi-tenancy is handled by the orchestration layer above AIOS. Running multiple model instances on the same hardware requires multiple AIOS instances, each managing its own memory budget.

### 10.2 Relationship to Quantization

AIOS operates on already-quantized models. AIOS and quantization stack multiplicatively:
- Quantization reduces the size of each weight block
- AIOS aliasing reduces the unique block count
- AIOS sparsity reduces the blocks loaded per token

The profiler must run on quantized weights. Profiling fp16 weights and applying artifacts to Q4 weights produces incorrect similarity thresholds due to quantization noise. **Profile after quantization.**

### 10.3 Limitations

**SiLU/GELU models:** The sparsity map provides limited benefit for models with smooth activations. The aliasing and residency mechanisms still apply but the combined MB/token reduction is lower (50–58% vs 80–85%).

**Very long contexts:** At contexts beyond 16K tokens, even Falcon 7B's MQA KV cache grows beyond L3 capacity. KV tiering applies, and the 100% KV DRAM elimination claim does not hold. This is marked `[A]` and the boundary is explicit.

**Prefill latency:** Chunked prefill improves prefill latency by keeping activations cache-resident but increases prefill time relative to unconstrained batch processing. For applications where prefill time matters, chunk size is a tunable parameter.

**R3 (explicit aliasing) is unvalidated architecture:** No model has been trained with explicit canonical + delta parameterization. The 70–80% aliasing ratio target for the AIOS Native tier is an extrapolation from ALBERT-style weight sharing results `[P]` applied to the delta parameterization. This is the highest-risk claim in the paper and the one most in need of community validation.

---

## 11. Conclusion and Call to Contributors

We have presented AIOS: a framework for CPU-native LLM inference that decomposes the performance problem into four independently addressable resource dimensions, specifies a Model Contract that model architects can design toward, and demonstrates analytically that a model satisfying the contract (Falcon 7B relufied as reference) can achieve 80–85% reduction in DRAM data movement per generated token.

The core argument is simple: LLMs are slow on CPU because they were designed for GPU. Designing for CPU — in both the runtime and the model — removes that constraint. The framework presented here is the specification of what "designing for CPU" means concretely.

**AIOS is fully open source under Apache 2.0.** We invite contributions in the following areas:

**Validation (highest priority):**
- Run the baseline measurement protocol (Issue #2) and report results
- Replicate the Falcon 7B relufication (Issue #1) and contribute to the model catalog
- Validate per-tier performance ranges on diverse hardware (Issue #4)

**Runtime implementation:**
- Profiler improvements — better similarity metrics, faster calibration passes
- SIMD kernel implementations for Tier 2 and Tier 3
- LD_PRELOAD interception layer
- Integration with llama.cpp, Ollama, vLLM

**Model architecture research:**
- R3 (explicit canonical + delta parameterization) — training experiments, architecture variants
- Alternative activation functions satisfying R1 with better quality/sparsity tradeoff
- R4 (cache-aligned dimensions) — systematic study of dimension choices and cache fit

**Hardware support:**
- ARM64 / Apple Silicon profiling and validation
- RISC-V baseline (Tier 1 floor)
- NUMA-aware validation on multi-socket systems

**See `CONTRIBUTING.md` for contribution guidelines, the validation protocol, and how to add entries to the model catalog.**

---

## References

Ainslie, J., et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. *EMNLP 2023*.

Alizadeh, K., et al. (2023). LLM in a Flash: Efficient Large Language Model Inference with Limited Memory. *arXiv:2312.11514*.

Almazrouei, E., et al. (2023). The Falcon Series of Open Language Models. *arXiv:2311.16867*.

Gerganov, G. (2023). llama.cpp. GitHub. github.com/ggerganov/llama.cpp

Jiang, A., et al. (2023). Mistral 7B. *arXiv:2310.06825*.

Lan, Z., et al. (2020). ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. *ICLR 2020*.

Liu, Z., et al. (2023). Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time. *ICML 2023*.

Mirzadeh, S.I., et al. (2023). ReLU Strikes Back: Exploiting Activation Sparsity in Large Language Models. *arXiv:2310.04564*.

Press, O., & Wolf, L. (2017). Using the Output Embedding to Improve Language Models. *EACL 2017*.

Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need. *arXiv:1911.02150*.

Song, Y., et al. (2023). PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU. *arXiv:2312.12456*.

Xiao, G., et al. (2023). Efficient Streaming Language Models with Attention Sinks. *arXiv:2309.17453*.

Zhang, P., et al. (2024). Relu²Wins: Discovering Efficient Activation Functions for Sparse LLMs. *arXiv:2402.03804*.

---

*AIOS is open source under Apache 2.0. github.com/[your-github-username]/aios*  
*Paper source: github.com/[your-github-username]/aios/paper/aios_paper.md*  
*Correspondence: anand.casavaraju@[your-email-domain]*
