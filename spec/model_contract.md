# AIOS Model Contract
## Specification v1.0

*This is a standalone specification. It can be read and implemented independently of the AIOS runtime or the full paper.*

---

## Purpose

The AIOS Model Contract defines five architectural requirements that a model must satisfy to enable full CPU-native inference optimization under AIOS. Models satisfying the contract expose optimization surfaces that are structural properties of the architecture — not emergent properties that a runtime must approximate.

A model satisfying all five requirements is designated **AIOS Native+** and is projected to achieve 85–92% reduction in DRAM data movement per generated token relative to an unoptimized baseline on the same hardware.

This specification is maintained independently of the AIOS runtime. Any inference system can implement against this contract. Any model can be validated against it using the validation suite in `validation/`.

---

## Claim Notation

| Tag | Meaning |
|-----|---------|
| `[A]` | Analytical — derivable from architecture parameters |
| `[P]` | Supported by prior published work |
| `[V:#N]` | Empirical validation pending — see GitHub issue #N |

---

## The Five Requirements

---

### R1 — Activation Function: Structural Sparsity

**Statement:** The FFN activation function must produce exact zero outputs for a measurable and stable fraction of inputs across the calibration distribution.

**Compliant activations:**
- ReLU: `max(0, x)` — exact zero for all negative inputs
- ReLU²: `max(0, x)²` — exact zero for all negative inputs, smoother gradient
- LeakyReLU with threshold: zero below configurable negative threshold

**Non-compliant activations:**
- SiLU / Swish: `x × sigmoid(x)` — smooth, asymptotically approaches zero but never reaches it
- GELU: smooth approximation — never exact zero
- Any activation without a true zero region

**Why this matters:** AIOS's sparsity map is a static bitmask generated at profiling time. For the map to reliably skip weight rows, those rows must produce exact zeros — not near-zeros that vary with input. A map built on near-zeros has unreliable skip rates and requires conservative epsilon thresholds that reduce the benefit.

**Validation:**
```
Protocol:
  1. Run calibration pass on held-out set (minimum 1,000 diverse prompts)
  2. For each FFN layer, measure fraction of neuron pre-activations that are exactly zero
  3. Report per-layer zero fraction and mean across all FFN layers
  
Pass criterion: Mean zero fraction ≥ 0.70 across all FFN layers in conservative profiling mode
```

**Compliance tier contribution:** R1 is required for Standard, Native, and Native+ tiers.

---

### R2 — KV Compression: Bounded Cache Budget

**Statement:** The model must use an attention mechanism that bounds KV cache size to ≤ 64 MB at 4,096-token context on the reference hardware class (server CPU with 32–64 MB L3 cache).

**Compliant configurations:**
- Multi-Query Attention (MQA): 1 KV head — achieves ~32 MB at 4K context for a 7B model `[A]`
- GQA with ≤ 2 KV groups — achieves ~64 MB at 4K context for a 7B model `[A]`

**Non-compliant configurations:**
- GQA with 8 KV groups: ~256 MB at 4K context — requires KV tiering
- Standard MHA: ~512 MB at 4K context — exceeds L3 capacity entirely

**KV cache size formula:**
```
KV_bytes = L × H_kv × d_head × C × 4
where:
  L      = number of layers
  H_kv   = number of KV heads
  d_head = head dimension
  C      = context length (tokens)
  4      = 2 (K and V) × 2 bytes (fp16)
```

**Why this matters:** When the KV cache fits in L3, all KV reads during autoregressive generation are served from cache — zero DRAM traffic. For Falcon 7B with MQA: 32 layers × 1 head × 64 dim × 4,096 tokens × 4 bytes = 32 MB `[A]`. This fits in L3 on any modern server CPU, eliminating KV DRAM reads entirely for standard enterprise workloads.

**Validation:**
```
Protocol:
  1. Calculate KV cache size at 4K context using the formula above
  2. Verify on target hardware: run inference at 4K context and measure KV DRAM traffic
     via hardware memory controller counters
  
Pass criterion: KV cache ≤ 64 MB at 4K context [A]; KV DRAM reads → 0 for ≤ 4K context [V:#9]
```

**Compliance tier contribution:** R2 is required for Standard, Native, and Native+ tiers.

---

### R3 — Weight Parameterization: Explicit Canonical + Delta

**Statement:** Layers in the core zone (40–60% of model depth, measured by layer index) must be parameterized as a canonical weight block shared across the zone, plus a learned per-layer delta vector. The delta must be trained to be small — not merely a decomposition of a fully independent weight.

**Architecture specification:**

For a model with $L$ layers, the core zone is layers $\lfloor 0.4L \rfloor$ through $\lfloor 0.6L \rfloor$.

Each core zone layer $i$ stores:
```
W_i = W_canonical + δ_i
where:
  W_canonical  = shared canonical weight matrix (stored once per zone)
  δ_i          = per-layer delta (small; L2 norm << L2 norm of W_canonical)
```

The delta is learned, not computed post-hoc. Training loss should include a delta regularization term:
```
L_total = L_task + λ × Σ_i ||δ_i||²
```
where λ is a hyperparameter controlling delta magnitude. Recommended starting value: λ = 1e-4.

**Why this matters:** The AIOS profiler discovers weight similarity post-training and approximates aliasing. This achieves 45–55% reduction in practice `[A]`. Explicit canonical + delta parameterization targets 70–80% reduction by design — the architecture is trained to make deltas small, not merely similar. The canonical block lives permanently in RWS-HOT. Delta vectors are small enough to co-reside in RWS-HOT or WARM.

**What this is not:** This is not identical weight sharing (ALBERT). ALBERT forces W_i = W_j exactly. R3 allows learned per-layer deviation. It is strictly more expressive — ALBERT is the special case where all δ_i = 0.

**Validation:**
```
Protocol:
  1. Run AIOS profiler on model; measure per-zone aliasing ratio
  2. Confirm delta L2 norm is < 10% of canonical L2 norm in core zone
  3. Measure perplexity delta vs fully independent weight baseline
  
Pass criterion: ≥ 70% aliasing ratio in core zone; perplexity delta < 1.0 on standard benchmarks [V:#6]
```

**Compliance tier contribution:** R3 is required for Native and Native+ tiers.

---

### R4 — Activation Dimensions: L2 Cache Alignment

**Statement:** FFN intermediate dimensions and attention head dimensions must be chosen such that per-layer activation tensors at batch size 1, single-token generation, fit within L2 cache per core on the reference hardware class.

**Reference hardware class:** CPU with 256 KB – 1 MB L2 cache per core (typical modern server or consumer CPU).

**Constraint derivation:**
```
Per-layer activation footprint at batch=1, single token:
  = (d_model + d_ffn_intermediate + d_model) × 2 bytes   [attention sublayer]
  + (d_ffn_intermediate × 2 + d_model) × 2 bytes         [FFN sublayer]
  ≈ (2 × d_model + 4 × d_ffn_intermediate) × 2 bytes

Target: ≤ 200 KB (conservative; allows co-residency with weight working set in L2)

Constraint on d_ffn_intermediate:
  d_ffn_intermediate ≤ (200,000 bytes / 2 - 2 × d_model) / 4
  For d_model = 4,544: d_ffn_intermediate ≤ ~24,000
```

**Current model compliance:**
- Mistral 7B: d_ffn = 14,336 → ~92 KB per layer `[A]` ✓ Compliant
- Falcon 7B: d_ffn = 18,176 → ~107 KB per layer `[A]` ✓ Compliant
- Llama 3.1 70B: d_ffn = 28,672 → significantly larger — requires checking on target hardware

**Why this matters:** Activation tensors that remain in L2 during single-token generation never reach DRAM. The activation memory problem ($A_{\text{spill}}$) in the four-resource decomposition disappears at the architectural level for the generation phase. Prefill still requires chunking, but the dominant per-token cost is resolved by design.

**Validation:**
```
Protocol:
  1. Calculate per-layer activation footprint using the formula above
  2. Verify on target hardware: measure L2 miss rate during single-token generation
  
Pass criterion: Per-layer activation footprint ≤ 200 KB; no activation DRAM spill during
generation phase (steady-state L2 miss rate does not increase during generation) [V:#10]
```

**Compliance tier contribution:** R4 is required for Native+ tier only.

---

### R5 — Weight Layout: Frequency-Ordered Checkpoint

**Statement:** The model checkpoint must store weight blocks in descending order of expected access frequency, as determined by a reference profiling pass on representative inputs.

**Format specification:**

The checkpoint must include a frequency manifest alongside weight data:
```json
{
  "aios_contract_version": "1.0",
  "frequency_ordered": true,
  "layout": [
    {
      "block_id": "layer_0.attention.q_proj",
      "access_frequency": "high",
      "rws_tier": "HOT",
      "offset_bytes": 0,
      "size_bytes": 33554432
    },
    ...
  ]
}
```

**Access frequency categories:**
- `high` (RWS-HOT): Accessed on every or near-every token. Typically: first 2 layers (all weights), last 2 layers (all weights), LM head projection.
- `medium` (RWS-WARM): Accessed frequently but not every token. Typically: attention Q/O projections in middle layers.
- `low` (STREAM): Accessed rarely. Typically: FFN weights in middle layers for ReLU models (dominated by sparsity map).

**Why this matters:** The AIOS profiler constructs the pointer table, residency assignments, and prefetch sequence from scratch on an unlabeled checkpoint. This takes time and may produce suboptimal assignments. A frequency-ordered checkpoint with a manifest makes this instantaneous and deterministic — the profiler's output is pre-encoded in the checkpoint.

For models in the AIOS model catalog, the frequency manifest is generated by running the reference profiler on standard hardware and including the output in the model release.

**Validation:**
```
Protocol:
  1. Verify checkpoint includes frequency_ordered manifest
  2. Verify AIOS profiler initialization time < 60 seconds on reference hardware
     (vs typical 5–15 minutes without manifest)
  
Pass criterion: Manifest present; profiler init time < 60 seconds [V:#11]
```

**Compliance tier contribution:** R5 is required for Native+ tier only.

---

## Compliance Tiers Summary

| Tier | Requirements | Projected MB/token Reduction | Label |
|------|-------------|------------------------------|-------|
| Baseline | None — existing model | 45–55% `[A]` | AIOS Compatible |
| Standard | R1 + R2 | 70–80% `[V:#5]` | AIOS Optimized |
| Enhanced | R1 + R2 + R3 | 80–88% `[V:#6]` | AIOS Native |
| Full | R1 + R2 + R3 + R4 + R5 | 85–92% `[V:#7]` | AIOS Native+ |

---

## Compliance Validation

The AIOS repository provides automated compliance validation in `validation/compliance.py`:

```bash
# Run compliance check on a model checkpoint
python validation/compliance.py \
  --model path/to/model.gguf \
  --hardware-profile server-32mb-l3 \
  --calibration-data path/to/calibration_prompts.jsonl \
  --output compliance_report.json
```

Output includes:
- Per-requirement pass/fail with measured values
- Achieved compliance tier
- Projected MB/token range for target hardware
- Recommended AIOS configuration for the model

---

## Adding a Model to the Catalog

To add a validated model to `models/catalog.md`:

1. Run the compliance validation suite and obtain a compliance report
2. Run the AIOS baseline benchmark protocol (see `validation/README.md`)
3. Open a pull request adding an entry to `models/catalog.md` with:
   - Model name, base architecture, compliance tier
   - Compliance report (JSON) attached to the PR
   - Measured MB/token baseline and AIOS-optimized values (if available)
   - Hardware configuration used for validation

Models can be added at any compliance tier. AIOS Compatible tier entries are welcome — showing what AIOS achieves on existing unmodified models is useful data.

---

## Versioning

This specification is versioned independently of the AIOS runtime. Changes to requirements or tiers increment the minor version. Breaking changes (invalidating existing compliance certifications) increment the major version. 

Current version: **1.0**  
Models should record the contract version they were validated against.

---

*AIOS Model Contract v1.0 | Apache 2.0 | github.com/[your-github-username]/aios/spec/model_contract.md*
