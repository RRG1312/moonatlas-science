"""Writing the normalized MOONATLAS dataset (docs/provenance/data-contract.md, contract v2): JSON, value rasters, masks, previews.

Raster encodings live in moonatlas_science/encoding.py (shared with the fixture generator and mirrored by
lib/data/valueImage.ts). Previews are display-stretched 8-bit images for viewing only, never analyzed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import CRS

from . import config, encoding
from .scoring import ice_p90_bounded_score, percentile_rank  # noqa: F401  (re-exported for normalizers)
from .sources import SOURCES

COORDINATE_DECIMALS = 6  # 1e-6° ≈ 3 cm on the Moon; enough for 1 m NAC pixels
PROJECTION_KINDS = {"stere": "polar-stereographic", "tmerc": "transverse-mercator"}


def coordinate(lat: float, lon: float) -> dict:
    lon = round(((lon + 180.0) % 360.0) - 180.0, COORDINATE_DECIMALS)
    if lon >= 180.0:
        lon = -180.0
    return {"latitude": round(float(lat), COORDINATE_DECIMALS), "longitude": lon}


def out_path(relative: str) -> Path:
    path = config.BUILD_DIR / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(relative: str, data, compact: bool = False) -> str:
    # LF on every OS: checksums in processing-manifest.json must match the git-normalized (eol=lf) checkout.
    # compact: large wire files (craters.json) without indentation — less to download and parse.
    text = json.dumps(data, allow_nan=False, separators=(",", ":")) if compact else json.dumps(data, indent=2, allow_nan=False)
    out_path(relative).write_text(text, encoding="utf-8", newline="\n")
    return relative


def read_json(relative: str, default=None):
    path = config.BUILD_DIR / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def upsert(relative: str, items: list[dict]) -> list[dict]:
    """Replaces entries with the same id, keeping others: steps can run per task and per sample."""
    existing = {item["id"]: item for item in read_json(relative, [])}
    existing.update({item["id"]: item for item in items})
    merged = sorted(existing.values(), key=lambda item: item["id"])
    write_json(relative, merged)
    return merged


def projection_record(proj4: str) -> dict:
    """Structured projection metadata (lib/validation/schemas.ts › ProjectionSchema) from a PROJ definition."""
    params = CRS.from_proj4(proj4).to_dict()
    kind = PROJECTION_KINDS.get(params.get("proj"))
    if kind is None or params.get("units", "m") != "m" or "R" not in params:
        raise ValueError(f"unsupported projection {proj4}")
    return {
        "kind": kind,
        "latitudeOfOriginDeg": params.get("lat_0", 0),
        "centralMeridianDeg": params.get("lon_0", 0),
        "scaleFactor": params.get("k", params.get("k_0", 1)),
        "falseEastingM": float(params.get("x_0", 0)),
        "falseNorthingM": float(params.get("y_0", 0)),
        "radiusM": float(params["R"]),
    }


def grid_record(tile) -> dict:
    return {
        "xMinM": round(tile.x_min, 3),
        "yMaxM": round(tile.y_max, 3),
        "pixelSizeXM": round(tile.pixel_size_x, 6),
        "pixelSizeYM": round(tile.pixel_size_y, 6),
        "widthPx": tile.width,
        "heightPx": tile.height,
    }


def value_raster(values: np.ndarray, covered: np.ndarray, stem: str) -> dict:
    """Writes the raw (16-bit, model output preserved) and display (clipped 8-bit) encodings of one raster."""
    raw, raw_encoding = encoding.raw_image(values, covered)
    raw_path = f"overlays/ice/{stem}-raw.webp"
    display_path = f"overlays/ice/{stem}.webp"
    encoding.save_lossless(raw, out_path(raw_path))
    encoding.save_lossless(encoding.display_image(values, covered), out_path(display_path))
    return {"rawPath": raw_path, "rawEncoding": raw_encoding, "displayPath": display_path, "displayRange": [0, 1]}


def mask_webp(mask: np.ndarray, relative: str) -> str:
    encoding.save_lossless(encoding.mask_image(mask), out_path(relative))
    return relative


def display_stretch(band: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    finite = band[np.isfinite(band)]
    lo, hi = np.percentile(finite, [low, high])
    scaled = (np.nan_to_num(band, nan=lo) - lo) / max(hi - lo, 1e-12)
    return (np.clip(scaled, 0, 1) * 255).round().astype(np.uint8)


def preview_webp(band: np.ndarray, relative: str) -> str:
    Image.fromarray(display_stretch(band), "L").save(out_path(relative), "WEBP", quality=88, method=6)
    return relative


def provenance(task: str, run: dict, source_dataset: str, source_tile: str, model: str, reference: str) -> dict:
    split = {"val": "validation"}.get(run["split"], run["split"])
    return {
        "task": task,
        "sourceDataset": source_dataset,
        "sourceTile": source_tile,
        "datasetSplit": split,
        "model": model,
        "checkpoint": run["checkpoint"],
        "modelRevision": run["modelRevision"],
        "referenceSource": reference,
        "processingVersion": config.PROCESSING_VERSION,
    }


def write_sources() -> None:
    revisions = {
        "lfm-backbone": config.REVISIONS[config.BACKBONE_REPO],
        "lfm-crater-detection": config.REVISIONS[config.CRATER_REPO],
        "lfm-ice-prospectivity": config.REVISIONS[config.ICE_REPO],
        "lfm-imp-segmentation": config.REVISIONS[config.IMP_REPO],
        "sombench-wac-crater-detection": config.REVISIONS[config.WAC_DATASET_REPO],
        "sombench-ice-prospectivity-regression": config.REVISIONS[config.ICE_DATASET_REPO],
        "sombench-imp-segmentation": config.REVISIONS[config.IMP_DATASET_REPO],
    }
    write_json("sources.json", [{**source, "revision": revisions.get(source["id"])} for source in SOURCES])
