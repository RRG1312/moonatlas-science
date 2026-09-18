"""The science data contract: what a normalized build contains and what every published value must carry.

This module is a *descriptor*, not a second validator. `moonatlas_science.validate` remains the authoritative
check, and `tests/test_contract.py` asserts that the two agree — a build that this descriptor accepts but the
validator rejects is a bug in the descriptor.

The descriptor exists so that a consumer (or a contributor) can read the contract as data:

    python -m moonatlas_science.contracts > schemas/contract-v2.json

Versioning (docs/reproducibility/versioning.md): adding a field is a minor `processingVersion` bump, changing
the meaning, type or unit of a field — or removing one — is a major bump.
"""

from __future__ import annotations

import json
import sys

from . import config

#: Files every normalized build contains, and what each one holds.
FILES = {
    "manifest.json": "build identity: processingVersion, schemaVersion, dataMode, generatedAt, counts",
    "processing-manifest.json": "sha256 and size of every published file, plus the samples each task processed",
    "sources.json": "every upstream source with licence, citation, pinned revision and whether MOONATLAS modified it",
    "global-stats.json": "counts published verbatim by consumers; never recomputed downstream",
    "crater-regions.json": "one analyzed WAC tile per entry: footprint, projection, grid, provenance, detection count",
    "craters.json": "crater detections per region, as compact tuples expanded by the consumer",
    "crater-tiles/<region>.json": "per-tile detail: predicted and reference boxes for the inspector",
    "ice-sectors.json": "one analyzed polar patch per entry: raw statistics, rasters, anchor, provenance",
    "ice-mosaics.json": "one seamless mosaic per pole in its native polar stereographic grid",
    "imp-regions.json": "one analyzed NAC observation per entry: mask, predicted area, uncalibrated score, provenance",
    "imp-observation-groups.json": "repeat observations of one SomBench target, grouped for navigation only",
    "discoveries.json": "the curated subset, with the selection rule that produced each entry",
    "previews/…, overlays/…": "display rasters and masks; never read back as measurements",
}

#: Provenance fields attached to every published model output (`moonatlas_science.build.provenance`).
PROVENANCE_FIELDS = (
    "task",
    "sourceDataset",
    "sourceTile",
    "datasetSplit",
    "model",
    "checkpoint",
    "modelRevision",
    "referenceSource",
    "processingVersion",
)

#: Dataset splits, and which of them may ever be presented as independent evidence of model performance.
SPLITS = {"test": "held out", "validation": "in-sample", "train": "in-sample", "external": "not part of a benchmark split"}
HELD_OUT_SPLIT = "test"

#: Identifier prefixes per object kind. Ids are stable across rebuilds (the discovery registry is append-only).
ID_PREFIXES = {
    "crater-region": "wac-",
    "ice-sector": "ice-",
    "imp-region": "imp-",
    "crater-discovery": "MM-CR-",
    "ice-discovery": "MM-ICE-",
    "imp-discovery": "MM-IMP-",
}
#: Any object carrying one of these markers is synthetic and must never appear in a published build.
SYNTHETIC_MARKERS = ("mock-", "MOCK", "mock")

#: Origin of every value. They are never mixed, and a consumer must be able to tell them apart.
ORIGINS = {
    "source_observation": "an upstream image, input layer or tile metadata",
    "reference_annotation": "published expert or catalogue work, shown for comparison only",
    "model_prediction": "MOONATLAS's own inference with a published NASA-IBM LFM checkpoint",
    "derived_metric": "a MOONATLAS computation over the above, always labelled as derived",
}

#: Raster levels. A number may only come from a raw or normalized value, never from a display raster.
RASTER_LEVELS = {
    "raw": "the model's own output, unclipped and unsmoothed, in its own units",
    "display": "a clipped, colour-mapped representation for viewing",
}


def describe() -> dict:
    """The contract as plain data, for consumers, tests and `schemas/contract-v2.json`."""
    return {
        "schemaVersion": config.SCHEMA_VERSION,
        "processingVersion": config.PROCESSING_VERSION,
        "files": FILES,
        "provenanceFields": list(PROVENANCE_FIELDS),
        "splits": SPLITS,
        "heldOutSplit": HELD_OUT_SPLIT,
        "idPrefixes": ID_PREFIXES,
        "syntheticMarkers": list(SYNTHETIC_MARKERS),
        "origins": ORIGINS,
        "rasterLevels": RASTER_LEVELS,
    }


if __name__ == "__main__":
    json.dump(describe(), sys.stdout, indent=2)
    sys.stdout.write("\n")
