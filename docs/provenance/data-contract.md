# The science data contract

Version `2` (`schemaVersion`), produced by `processingVersion` `science-<major>.<minor>.<patch>`. The
machine-readable descriptor is `schemas/contract-v2.json`, generated from `moonatlas_science/contracts.py`;
`moonatlas_science/validate.py` is the authoritative checker.

## Files in a build

| File | Holds |
|------|-------|
| `manifest.json` | Build identity: processing and schema version, data mode, generation time, counts |
| `processing-manifest.json` | sha256 and size of every published file, and the samples each task processed |
| `sources.json` | Every upstream source: licence, citation, pinned revision, whether MOONATLAS modified it |
| `global-stats.json` | Counts consumers publish verbatim |
| `crater-regions.json` | One analyzed WAC tile per entry: footprint, projection, grid, provenance, detection count |
| `craters.json` | Detections per region as compact tuples |
| `crater-tiles/<region>.json` | Per-tile predicted and reference boxes |
| `ice-sectors.json` | One analyzed polar patch: raw statistics, rasters, anchor, provenance |
| `ice-mosaics.json` | One mosaic per pole in its native polar stereographic grid |
| `imp-regions.json` | One analyzed NAC observation: mask, area, tile share, uncalibrated score, provenance |
| `imp-observation-groups.json` | Repeat observations of one SomBench target (navigation only) |
| `discoveries.json` | The curated subset with the selection rule that produced each entry |
| `previews/…`, `overlays/…` | Display rasters and masks |

## Provenance on every published model output

`task`, `sourceDataset`, `sourceTile`, `datasetSplit`, `model`, `checkpoint`, `modelRevision`,
`referenceSource`, `processingVersion`. A build whose provenance disagrees with the pinned configuration fails
validation.

## Identifiers

`wac-…`, `ice-…`, `imp-…` for analyzed objects; `MM-CR-…`, `MM-ICE-…`, `MM-IMP-…` for curated discoveries.
Discovery ids come from the append-only registry (`configs/discovery-registry.json`): one scientific identity
key maps to one public id forever, ids are never reused, and rebuilding must not renumber anything.

## Value rasters

A raster publishes `rawPath` + `rawEncoding` (lossless, decodable back to the model's values) and, optionally,
a `displayPath` + `displayRange` for viewing. Numbers are read from raw rasters only.
