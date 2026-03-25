# Contributing to AIOS

AIOS is an open research project. Contributions at all levels are welcome — from running a validation experiment on your hardware to implementing a new SIMD kernel to training an R3-compliant model architecture.

## What's Most Needed Right Now

In priority order:

### 1. Validation Experiments (highest impact)
The paper makes analytical projections. We need measured results. If you have a CPU-based machine and can run experiments:
- Pick any open issue in the [validation tracker](../../issues)
- Follow the protocol in the issue body
- Post your results — even partial, even negative

**Negative results are valuable.** If a projection is wrong, knowing that is more important than confirming it.

### 2. Falcon 7B Relufication (Issue #1)
The reference implementation requires running the relufication fine-tune described in the paper. This needs compute — approximately 50B tokens of training on RefinedWeb. If you have access to training infrastructure:
- The fine-tune procedure is specified in Issue #1
- The result, if it matches prior work, becomes the first AIOS Model Catalog entry

### 3. Runtime Implementation
The runtime is a reference implementation in C. It is correct, not optimized. Contributions:
- SIMD kernels for Tier 2 (AVX2) and Tier 3 (AVX-512 / AMX)
- LD_PRELOAD interception layer
- ARM64 / Apple Silicon support
- llama.cpp integration

### 4. R3 Architecture Research
The explicit canonical + delta parameterization (Requirement 3 of the Model Contract) is unvalidated novel architecture. Training experiments and quality measurements are needed. This is the most open-ended research area.

---

## Contribution Guidelines

### For validation contributions
- Follow the protocol specified in the relevant GitHub issue exactly
- Report: hardware spec (CPU, RAM, L3 cache size), model version, quantization level, exact command used, full output
- If results differ from projections, explain what you measured and how — don't edit the projection until discussed

### For code contributions
- Fork the repo, work in a branch, open a PR against `main`
- Runtime (C): correctness first. Include a test against the reference output
- Profiler (Python): include timing measurements — the profiler runs offline but should be fast
- All contributions must pass existing tests (`make test`)

### For model catalog contributions
- Run `validation/compliance.py` and include the JSON report in the PR
- Include hardware configuration used for validation
- Any compliance tier is welcome — AIOS Compatible through Native+

### For architecture research (R3)
- Open an issue first to describe the experiment
- Share training config, dataset, and evaluation results
- Negative results (delta parameterization doesn't train well) are publishable findings — open an issue

---

## Code of Conduct

- Be direct about what is measured vs projected
- Disagree constructively — the projections may be wrong and that is fine
- Credit prior work in comments and documentation

---

## Questions

Open a GitHub Discussion. Don't open issues for questions — keep issues for tracked validation claims and bug reports.
