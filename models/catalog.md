# AIOS Model Catalog

Models validated against the AIOS Model Contract. All entries include a compliance report and measured or projected performance data.

Entries are added via pull request. See [CONTRIBUTING.md](../CONTRIBUTING.md) for the submission process.

---

## How to Read This Table

- **Tier:** AIOS compliance tier (Compatible / Optimized / Native / Native+)
- **MB/token reduction:** measured values in **bold**, analytical projections in *italics*
- **Validated:** whether the entry has measured AIOS results (vs analytical projection)
- **Issue:** tracking issue for empirical validation

---

## Catalog

| Model | Base | Activation | KV Heads | Tier | MB/token Reduction | Validated | Issue |
|-------|------|-----------|----------|------|--------------------|-----------|-------|
| Mistral 7B Q4_K_M | Mistral 7B | SiLU | 8 (GQA) | Compatible | *50–58%* | No | [#1](../../issues/1) |
| Falcon 7B Relufied Q4_K_M | Falcon 7B | ReLU | 1 (MQA) | Optimized | *80–85%* | No | [#2](../../issues/2) |

---

## Paper

Casavaraju, A. (2026). AIOS: A CPU-Native Inference Architecture for Large Language Models.
SSRN Working Paper. https://ssrn.com/abstract=6467298

---

## Submitting a Model

1. Run compliance check:
```bash
python validation/compliance.py \
  --model path/to/model.gguf \
  --output compliance_report.json
```

2. Run baseline benchmark (optional but preferred):
```bash
./validation/baseline.sh --model path/to/model.gguf --runs 5 --output baseline.json
```

3. Open a PR adding a row to this table. Attach compliance_report.json and baseline.json.

Any model at any tier is welcome. AIOS Compatible entries showing what AIOS achieves on existing models without modification are valuable data.
