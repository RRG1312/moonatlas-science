# Changelog

All notable changes to MOONATLAS Science. Versions are `processingVersion` values
(`science-<major>.<minor>.<patch>`), which identify the *outputs*; see
`docs/reproducibility/versioning.md`.

## [Unreleased]

### science-0.3.1 — MOONATLAS re-issue, no new inference

A metadata release after the project was renamed MOONMIND → MOONATLAS. No inference was re-run: the example build was
re-normalized, catalogued and manifested from the same raw crater, ice and IMP artifacts as `science-0.3.0`.

- `examples/science-smoke` rebuilt by the pipeline (`normalize` × 3, `mosaic`, `catalog`, `manifest --scope smoke`,
  `validate`), then `schemas/contract-v2.json` and `examples/conformance/geometry.json` regenerated from it.
- Value-level comparison with the `science-0.3.0` example: **0 scientific differences** — crater predictions,
  ice raw values, IMP masks, coordinates and reference values are identical. All rasters are byte-identical.
- What changed: `processingVersion` (8 fields), `generatedAt`, one generated ice description (the former name), and
  the six checksums of the files that changed. Every checksum was verified against the file it describes.
- `craters.json` is now serialized compactly. The `science-0.3.0` example had been built before the pipeline switched
  to compact output, so it was not byte-reproducible from this repository; the decoded values are identical.
- Conformance pixel centres, containment cases and raster encodings are unchanged.

## [science-0.3.0] — 2026-09-18 — first public release

First public release of the pipeline that produced `science-0.3.0`. The repository history starts here; everything
below already existed and was exercised by the build this release describes.

### Pipeline
- Inference orchestration for the three published NASA-IBM LFM task checkpoints through the official TerraTorch
  path, with strict state-dict loading and raw artifacts preserved per run.
- Georeferencing with projection recovery for SomBench WAC tiles (45 transverse Mercator zones plus the two
  polar caps), verified to < 1e-6° against tile metadata.
- Normalization for crater detection, polar ice prospectivity and IMP segmentation into the data contract,
  per-pole ice mosaics in the native polar stereographic grid, and lossless raw raster encodings.
- Catalogue construction: MOONATLAS derived scores with versioned formula ids, curated discovery selection and
  the append-only discovery registry.
- Validation covering ids, coordinates, provenance against the pinned configuration, raw artifacts, geolocation
  round-trips, raster encodings, mosaic placement and manifest checksums.
- Deterministic dataset publishing and byte-reproducible artifact packaging.

### Reference build
`science-0.3.0`: 1,000 analyzed WAC tiles (80,454 detections at confidence ≥ 0.5), 156 polar ice patches in 2
mosaics, 130 IMP observations, 97 curated discoveries, 0 synthetic objects.

### Added for the public release
- `moonatlas-science` command over the existing steps, and an importable `moonatlas_science` package.
- `contracts.py` plus `schemas/contract-v2.json`: the data contract as data, checked against the validator.
- `--scope published` for validating a dataset on a machine that did not run the inference.
- `examples/science-smoke` (redistributable derived samples, CC BY 4.0) and `examples/conformance`.
- Scientific-integrity tests, contract tests and conformance tests that run on a clean clone.
- Documentation of methodology, provenance, coverage semantics, reproducibility levels and limitations.

### Known gaps
- The validator's footprint containment check uses the bounding box of a footprint rather than its
  quadrilateral (`docs/coverage/semantics.md`); it is conservative, never stricter than the definition.
- SomBench NAC crater detection is not used yet.
