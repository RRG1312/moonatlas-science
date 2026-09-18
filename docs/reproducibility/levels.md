# Reproducibility levels

| Level | Proves | Command | Needs |
|-------|--------|---------|-------|
| **1 · Clone test** | The package, the contract, the examples and the deterministic transformations are intact | `pytest` and `moonatlas-science validate --root examples/science-smoke --scope published` | The clone only |
| **2 · Smoke inference** | Your environment reproduces published behaviour on one held-out sample per task | `python scripts/setup_environment.py` then `moonatlas-science smoke` | ~6.3 GB of checkpoints plus a few samples; GPU recommended |
| **3 · Full reproduction** | The supported outputs rebuild from pinned upstream sources | `fetch-data`, then `infer`/`normalize` per task, `mosaic`, `catalog`, `manifest`, `validate` | Checkpoints plus the task datasets; GPU strongly recommended |

Level 1 is what CI runs on every pull request. Levels 2 and 3 need downloads and hardware, so they are run
deliberately; tests that depend on them are marked `upstream` and excluded by default (`pytest -m upstream`).

This repository does not publish runtime or hardware estimates it has not measured on the machine you are
using. `moonatlas-science smoke` prints the duration of each step, which is the honest way to find out.
