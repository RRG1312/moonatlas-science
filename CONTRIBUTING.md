# Contributing to MOONATLAS Science

Thank you for considering a contribution. This repository publishes scientific outputs, so review is about
**correctness, provenance and reproducibility** more than style.

Read [README § scientific integrity rules](README.md#scientific-integrity-rules) first. Most rejected changes
fail there, not in the code.

## Before you write code

Open an issue first if your change:

- alters any published value, or the wording attached to one;
- adds or updates an upstream dataset, model or checkpoint;
- changes the data contract or `schemas/`;
- changes projection, footprint or coverage logic.

Typos, obvious bugs, tests and documentation can go straight to a pull request.

## Local setup

Python 3.11 or 3.12.

```bash
python -m venv .venv
. .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                            # fast suite: no downloads, no GPU
ruff check .
```

Working on inference or normalization also needs the upstream stack and samples:

```bash
python scripts/setup_environment.py
moonatlas-science fetch-models
moonatlas-science fetch-data
moonatlas-science smoke
```

Downloads land in `data/` (git-ignored; relocate with `MOONATLAS_DATA_ROOT`). **Never commit checkpoints,
upstream datasets, builds, artifacts or local editor/assistant configuration.**

## The two test levels

| Suite | Command | Needs | Runs in CI |
|-------|---------|-------|------------|
| Fast | `pytest` | The clone | Yes, on every PR |
| Upstream | `pytest -m upstream` | Downloaded models and/or samples | No |

Anything that needs a download must be marked `@pytest.mark.upstream` or skip itself
(`pytest.importorskip`). A clean clone must always be able to run `pytest`.

## Testing expectations

Every behavioural change needs a test that fails without it:

- **Geometry / projection:** a known coordinate, corner or footprint reproduced to a stated tolerance
  (`tests/test_projection.py`, `tests/test_geo.py`).
- **Normalization:** raw input → contract output, asserting the *value* is preserved.
- **Contract:** `tests/test_contract.py` — the descriptor, the validator and the example build must agree.
- **Integrity:** `tests/test_integrity_rules.py` — the permanent rules stay true of published data and docs.
- **Determinism:** the same input produces the same bytes twice (`tests/test_packaging.py`).

## Adding a scientific dataset

1. Record the source in `moonatlas_science/config.py` with its official repository, a **pinned revision** and a
   checksum where the host publishes one; add its licence and citation to `sources.py` and `NOTICE`.
2. Add acquisition to `steps/download_data.py`. Do not commit the data.
3. Write the normalization module: keep raw values untouched and document every transformation.
4. Add a projection/footprint test proving coordinates come from metadata, not from assumptions.
5. Document methodology and limitations under `docs/`.
6. If MOONATLAS transforms the data, the modification notice in `NOTICE` must say so.

## Adding a model task

As above, plus: use the official task configuration and a strict state-dict load, keep the raw artifact, quote
the metric the upstream card publishes, and add the task to the reproduction checks when the card provides a
comparable test configuration. Do not retrain, distil or fine-tune inside this repository without discussing it
first — the value of this project is that it runs *published* checkpoints.

## Updating the contract or schemas

- The contract descriptor is `src/moonatlas_science/contracts.py`; `moonatlas_science/validate.py` enforces it.
- Regenerate the export in the same pull request:
  `python -m moonatlas_science.contracts > schemas/contract-v2.json`.
- Bump `processingVersion` per `docs/reproducibility/versioning.md`; a schema change without a bump is rejected.
- If geometry or example outputs change, regenerate `examples/conformance/`:
  `python scripts/build_conformance.py`.

## Provenance requirements

Every published value must carry `task`, `sourceDataset`, `sourceTile`, `datasetSplit`, `model`, `checkpoint`,
`modelRevision`, `referenceSource` and `processingVersion`. A value that cannot carry provenance does not belong
in the contract — put it in a report artifact instead.

Keep the origin hierarchy strict (`docs/provenance/origins.md`): source observation · reference annotation ·
model prediction · MOONATLAS derived metric. A derived metric is labelled as derived and carries a versioned
formula id.

## Deterministic output requirements

- Same inputs and pinned revisions → byte-identical output.
- No wall-clock timestamps, no locale-dependent formatting, no unordered iteration reaching an output.
- A change that makes output non-deterministic is rejected even when the science is right.

## Pull request process

1. Branch from `main`: `feat/…`, `fix/…`, `docs/…`, `test/…`, `chore/…`.
2. One concern per pull request; never mix a science change with a refactor.
3. Fill in the template honestly — especially *does this change raw model output?*
4. CI (`pytest` + `ruff` on Python 3.11 and 3.12) must pass.
5. A merge does not publish anything by itself: maintainers cut versioned releases separately
   (`docs/architecture/releasing.md`), and applications choose when to adopt them.
6. Commits use `type(scope): description`, imperative mood. Please do not add AI co-authorship trailers.

## Review expectations

A maintainer checks: value preservation, complete provenance, determinism, wording against the integrity rules,
licence and attribution for anything new, and whether the change is testable with this repository
alone. Reviews are technical, not personal — see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Good first contributions

- Tighten the footprint containment check from a bounding box to the exact quadrilateral
  (`docs/coverage/semantics.md` explains the gap and what evidence the change needs).
- Extend the conformance fixtures with polar stereographic cases.
- Add batch-report coverage for a task that only has a smoke-level check.
