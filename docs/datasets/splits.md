# Dataset splits

SomBench ships official `train` / `validation` / `test` splits. MOONATLAS records the split of **every**
processed sample in its provenance and treats them differently in every downstream claim:

| Split | Meaning | May it support a performance claim? |
|-------|---------|-------------------------------------|
| `test` | Held out from training | Yes — and only against the upstream test configuration |
| `validation` | Used during model selection upstream | **No.** In-sample |
| `train` | Used to fit the model | **No.** In-sample |
| `external` | Not part of a benchmark split | No |

Rules:

- Predictions on `train` or `validation` samples are labelled **in-sample** wherever they appear.
- Performance claims quote the official test metrics from the model cards; MOONATLAS's own recomputation is
  reported alongside them, never instead of them (see `docs/models/reproduction-checks.md`).
- Curated examples in a published catalogue prefer held-out samples; when they are not held out, the split is
  shown with the object.
