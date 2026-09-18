# Examples

## science-smoke

A small, real MOONATLAS Science build: one held-out test sample per task, produced by the pipeline in this
repository and kept byte-for-byte as it was built (`processing-manifest.json` holds the checksum of every file).

| Task | Sample | Split |
|------|--------|-------|
| WAC crater detection | `M1174144247CE_r5360_c480` | test |
| Polar ice prospectivity | `patch_0001_0002_S_80S` | test |
| IMP segmentation | `M1126915118RE.ech.cog__target_1980__idx_1980_p0` | test |

```bash
moonatlas-science validate --root examples/science-smoke --scope published
```

The directory holds exactly the files the build produced — nothing else may be added to it, or the validator
rejects it. `--scope published` is needed because the raw inference artifacts stay on the machine that ran the inference;
every other check applies.

### Licence

The model outputs come from the Apache-2.0 NASA-IBM LFM checkpoints run by MOONATLAS; the inputs, previews and
reference values are derived from the SomBench datasets under **CC BY 4.0**, modified by georeferencing,
normalization and re-encoding. Attribution and the modification notice are in [NOTICE](../NOTICE); `sources.json` lists
every source with its pinned revision. See [docs/provenance/redistribution.md](../docs/provenance/redistribution.md).

### Generated text

Discovery descriptions are generated text and part of the checksummed output of `science-0.3.0`. One of them still
uses the project's former name; like any output change, it updates only with a new `processingVersion`.

## conformance

`conformance/geometry.json` holds expected values computed by this pipeline from the example build: pixel centres
from each object's own projection and grid, footprint containment cases and raster encodings. Any consumer that
reimplements the geometry — in any language — should reproduce them. Regenerate after a geometry change with
`python scripts/build_conformance.py`; `tests/test_conformance.py` fails when they drift.
