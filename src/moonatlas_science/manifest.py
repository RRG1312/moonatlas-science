"""manifest.json (read by the app) and processing-manifest.json (reproducibility anchor)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from . import artifacts, build, config

PROCESSING_MANIFEST = "processing-manifest.json"
DATA_MODE_BY_SCOPE = {"smoke": "real-smoke", "batch-preview": "batch-preview", "complete-preview": "complete-preview",
                      "production": "real"}
TASK_FILES = {
    "crater-detection": "crater-regions.json",
    "ice-prospectivity": "ice-sectors.json",
    "imp-segmentation": "imp-regions.json",
}


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _samples(task: str, items: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        sample = item["provenance"]["sourceTile"].removesuffix(".tif").removesuffix("_img")
        run, _ = artifacts.read_run(task, sample)
        rows.append({"id": item["id"], "sample": sample, "split": item["provenance"]["datasetSplit"],
                     "device": run["device"], "inferredAt": run["createdAt"]})
    return rows


# Every file the web app loads; a task that was not processed in this build is published as an empty collection.
CONTRACT_FILES = {"crater-regions.json": [], "craters.json": {}, "ice-sectors.json": [], "ice-mosaics.json": [],
                  "imp-regions.json": [], "imp-observation-groups.json": []}


def write(scope: str, generated_at: datetime | None = None) -> dict:
    """`generated_at` pins the timestamp for repackaging an existing build, so the same science yields the same bytes."""
    now = generated_at or datetime.now(timezone.utc)
    for name, empty in CONTRACT_FILES.items():
        if not (config.BUILD_DIR / name).exists():
            build.write_json(name, empty)
    build.write_json(
        "manifest.json",
        {
            "catalogVersion": now.strftime("%Y.%m"),
            "processingVersion": config.PROCESSING_VERSION,
            "generatedAt": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "dataMode": DATA_MODE_BY_SCOPE[scope],
            "schemaVersion": config.SCHEMA_VERSION,
        },
    )
    regions = build.read_json("crater-regions.json", [])
    craters = build.read_json("craters.json", {})
    sectors = build.read_json("ice-sectors.json", [])
    mosaics = build.read_json("ice-mosaics.json", [])
    imp = build.read_json("imp-regions.json", [])
    everything = [*regions, *sectors, *imp]

    models = []
    environment = {}
    for task, model in config.TASK_MODELS.items():
        items = build.read_json(TASK_FILES[task], [])
        if not items:
            continue
        sample = items[0]["provenance"]["sourceTile"].removesuffix(".tif").removesuffix("_img")
        run, _ = artifacts.read_run(task, sample)
        environment = {"versions": run["versions"], "upstreamCode": run["upstreamCode"]}
        models.append({
            "sourceId": model.source_id,
            "repo": model.checkpoint.repo_id,
            "revision": model.checkpoint.revision,
            "checkpoint": model.checkpoint.filename,
            "checkpointSha256": model.checkpoint.sha256,
            "config": model.config.filename,
            "configSha256": run["upstreamConfigSha256"],
            "strictLoad": not run["missingKeys"] and not run["unexpectedKeys"],
        })

    files = sorted(p for p in config.BUILD_DIR.rglob("*") if p.is_file() and p.name != PROCESSING_MANIFEST)
    document = {
        "processingVersion": config.PROCESSING_VERSION,
        "schemaVersion": config.SCHEMA_VERSION,
        "scope": scope,
        "generatedAt": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "upstream": {**environment.get("upstreamCode", config.UPSTREAM_CODE), "backbone": {
            "repo": config.BACKBONE_REPO, "revision": config.REVISIONS[config.BACKBONE_REPO]}},
        "environment": environment.get("versions", {}),
        "models": models,
        "datasets": [
            {"sourceId": "sombench-wac-crater-detection", "repo": config.WAC_DATASET_REPO, "revision": config.REVISIONS[config.WAC_DATASET_REPO]},
            {"sourceId": "sombench-ice-prospectivity-regression", "repo": config.ICE_DATASET_REPO, "revision": config.REVISIONS[config.ICE_DATASET_REPO]},
            {"sourceId": "sombench-imp-segmentation", "repo": config.IMP_DATASET_REPO, "revision": config.REVISIONS[config.IMP_DATASET_REPO]},
        ],
        "counts": {
            "craterRegions": len(regions),
            "craterDetections": sum(len(v) for v in craters.values()),
            "iceSectors": len(sectors),
            "iceMosaics": len(mosaics),
            "impRegions": len(imp),
            "impCandidates": sum(1 for r in imp if r["prediction"]["areaM2"] > 0),
            "bySplit": {split: sum(1 for i in everything if i["provenance"]["datasetSplit"] == split)
                        for split in ("train", "validation", "test", "external")},
        },
        "samples": {
            task: _samples(task, build.read_json(file, [])) for task, file in TASK_FILES.items()
        },
        "checksums": {p.relative_to(config.BUILD_DIR).as_posix(): _sha256(p) for p in files},
    }
    build.write_json(PROCESSING_MANIFEST, document)
    return document


def read() -> dict:
    return json.loads((config.BUILD_DIR / PROCESSING_MANIFEST).read_text(encoding="utf-8"))
