# AIOS Roadmap

## v1.0 — Foundation (Current)
- [ ] Runtime: pointer table, aliasing, sparsity map, RWS residency, static prefetch (Tier 1 / SSE4.2 / NEON)
- [ ] Profiler: similarity analysis, sparsity calibration, manifest builder
- [ ] Validation: baseline measurement, headroom analysis, compliance check
- [ ] Paper published
- [ ] Model Contract v1.0 published
- [ ] Falcon 7B relufied: relufication fine-tune and catalog entry (Issue #1)
- [ ] Falcon 7B + AIOS: integrated benchmark (Issue #2)

## v1.1 — Hardware Tiers
- [ ] Tier 2 SIMD kernels (AVX2)
- [ ] Tier 3 SIMD kernels (AVX-512 / AMX)
- [ ] Apple Silicon (AMX / ANE) support
- [ ] ARM64 baseline validation
- [ ] Per-tier benchmark results (Issue #4)

## v1.2 — Integration
- [ ] LD_PRELOAD interception layer (Issue #3)
- [ ] llama.cpp ABI integration
- [ ] Ollama compatibility validation
- [ ] Chunked prefill implementation and benchmark (Issue #8)

## v2.0 — Model Contract Native
- [ ] R3 (explicit canonical + delta) training experiments
- [ ] First AIOS Native tier model in catalog
- [ ] Contract compliance automation in profiler
- [ ] Model Contract v1.1 (incorporating R3 findings)

## v3.0 — Heterogeneous Dispatch
- [ ] GPU dispatch layer — heterogeneous pointer table with GPU slots
- [ ] Workload routing by latency requirement
- [ ] CPU + GPU co-execution validation

## Ongoing
- [ ] Model catalog growth — community-submitted entries
- [ ] Validation issue closure as results come in
- [ ] Paper updates as empirical results replace analytical projections
