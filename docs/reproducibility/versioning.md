# Versioning

`processingVersion` is `science-<major>.<minor>.<patch>` and identifies the outputs, not the code release.

| Change | Bump |
|--------|------|
| Meaning, type or unit of a published field changes; a field is removed; ids change | **major** |
| A field is added; a new task, dataset or object kind appears | **minor** |
| A bug fix that changes values without changing their meaning | **patch** |
| Documentation, tests, refactors with identical output | none |

`schemaVersion` changes only when the contract's shape changes (currently `2`).

Rules:

- A schema change without a version bump is rejected in review.
- Every object carries the `processingVersion` that produced it, so mixed builds are detectable.
- The exported descriptor `schemas/contract-v2.json` is regenerated in the same pull request
  (`python -m moonatlas_science.contracts > schemas/contract-v2.json`); `tests/test_contract.py` fails on drift.
