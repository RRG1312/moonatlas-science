# What this repository redistributes, and on what basis

Every file in the repository, by class. Anything not listed here is MOONATLAS-owned code or documentation under
Apache-2.0. Checkpoints, bulk SomBench data, original reference catalogues and original instrument products are
**never** committed; they are downloaded from their official sources at pinned revisions.

| Files | Origin | Licence | Modified by MOONATLAS? | Redistribution basis | Attribution |
|-------|--------|---------|------------------------|----------------------|-------------|
| `src/**`, `scripts/**`, `tests/**` | MOONATLAS | Apache-2.0 | — | Own work | `LICENSE`, `NOTICE` |
| `docs/**`, `README.md`, community files | MOONATLAS | Apache-2.0 | — | Own work | `LICENSE` |
| `schemas/contract-v2.json` | Generated from `src/moonatlas_science/contracts.py` | Apache-2.0 | — | Own work | — |
| `configs/discovery-registry.json` | MOONATLAS identity keys built from public SomBench sample names and checkpoint file names | Apache-2.0 | — | Own work; contains names, no upstream content | `NOTICE` (SomBench, LFM) |
| `notebooks/moonatlas-inference.ipynb` | MOONATLAS, no stored outputs | Apache-2.0 | — | Own work | — |
| `examples/conformance/geometry.json` | Generated from the example build by `scripts/build_conformance.py` | Apache-2.0 (coordinates computed by MOONATLAS) | — | Own computation over the example | `NOTICE` |
| `examples/science-smoke/*.json` — model outputs (crater detections, prospectivity statistics, IMP masks' areas and scores) | MOONATLAS inference with the Apache-2.0 LFM checkpoints on SomBench samples | Outputs of Apache-2.0 models; inputs CC BY 4.0 | Georeferenced and normalized | CC BY 4.0 (inputs) with attribution and modification notice | `NOTICE`, `examples/science-smoke/sources.json` |
| `examples/science-smoke/*.json` — reference values (Robbins-derived boxes, Coyan-derived prospectivity statistics, Hargitai-derived mask areas) | SomBench datasets | CC BY 4.0, as distributed by SomBench | Georeferenced and normalized | SomBench's CC BY 4.0 grant; the original catalogues are not redistributed | `NOTICE` (SomBench + the three references) |
| `examples/science-smoke/overlays/ice/*` (8 WebP) | Model prediction and SomBench reference prospectivity rasters for one patch and its pole mosaic | CC BY 4.0 (reference) / model output | Re-encoded losslessly (raw) or clipped for display | CC BY 4.0 with modification notice | `NOTICE` |
| `examples/science-smoke/overlays/imp/*` (2 WebP) | Predicted mask and SomBench reference mask for one NAC tile | CC BY 4.0 (reference) / model output | Re-encoded | CC BY 4.0 with modification notice | `NOTICE` |
| `examples/science-smoke/previews/craters/*`, `previews/imp/*` (2 WebP) | Display previews of one SomBench WAC tile and one SomBench NAC tile (LROC imagery as distributed by SomBench) | CC BY 4.0, as distributed by SomBench | Contrast-stretched and re-encoded for viewing | CC BY 4.0 with modification notice; data products courtesy of LROC | `NOTICE` |
| `examples/science-smoke/sources.json`, `manifest.json`, `processing-manifest.json` | Pipeline metadata | Apache-2.0 | — | Own work | — |

What is deliberately **not** here:

- Model checkpoints and configs (Apache-2.0, several GB): downloaded at pinned revisions.
- SomBench tiles, input layers and label files in their original form: downloaded at pinned revisions.
- The original Robbins, Coyan et al. and Hargitai et al. products: only the derivatives distributed inside SomBench
  under CC BY 4.0 are used, and the originals are cited.
- LROC, LOLA and Diviner products in their original form.
- Any basemap or product imagery: diagnostic figures take an optional local basemap (`MOONATLAS_BASEMAP`) that is
  never committed.
