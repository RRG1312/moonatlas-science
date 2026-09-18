# Pipeline architecture

Four stages. Each runs as its own process, writes artifacts to disk, and fails loudly instead of substituting
anything.

```
acquisition → inference → normalization → validation/packaging
```

## 1 · Acquisition

`moonatlas-science fetch-models` and `fetch-data` download the backbone, the three task checkpoints, their
TerraTorch configs and the requested SomBench samples, all at the revisions pinned in
`moonatlas_science/config.py`. Checkpoints are verified against the sha256 published by the hosting repository.
Everything lands under `data/` (git-ignored, relocatable with `MOONATLAS_DATA_ROOT`).

A missing or mismatching asset aborts the run. There is no fallback path, and fixtures never stand in for real
samples.

## 2 · Inference

`moonatlas_science/{crater,ice,imp}_inference.py` construct the model through the upstream package and its
official task configuration (`official.py`), load the checkpoint **strictly** (an unexpected or missing key is
an error), run the sample, and write a raw artifact per run under `data/artifacts/raw/<task>/<sample>/`:

- the model's own output arrays (`.npz`), unmodified;
- a `run.json` recording the checkpoint, its revision, the split, the loading result and timings.

Nothing else reads the model. Every later stage reads the artifact, so normalization can be re-run and
reviewed without a GPU.

## 3 · Normalization

`*_normalize.py` turn raw outputs into the published contract (`provenance/data-contract.md`):

- geometry comes from source metadata: the tile's projection and grid, recovered and verified, never assumed
  (`geo.py`, `projection.py`);
- values are preserved — boxes become coordinates and diameters, the regression array stays unclipped, the
  mask stays the model's argmax;
- rasters are encoded losslessly (`encoding.py`) and display rasters are generated separately;
- `mosaic.py` composes one ice mosaic per pole in the native polar stereographic grid, rejecting overlaps;
- `catalog.py` + `registry.py` select curated discoveries and assign public ids from the append-only registry;
- `manifest.py` writes `sources.json`, `manifest.json` and `processing-manifest.json` (sha256 of every file).

## 4 · Validation and packaging

`validate.py` is authoritative and runs over the finished build. `steps/build_dataset.py` publishes a validated
build as a dataset, `steps/package_artifact.py` packs it into a byte-reproducible archive with a checksum
manifest. Neither recomputes science: they copy, re-encode losslessly, or check.

## Why separate processes

Model frameworks do not reliably reinitialise inside one process, and a crash in one task must not lose
another's artifacts. `moonatlas-science smoke` runs the whole chain, one subprocess per step, stopping at the
first failure.
