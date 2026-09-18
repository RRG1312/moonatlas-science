"""Raw WAC detections → normalized crater regions, detections and tile details."""

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio

from . import artifacts, build, config
from .geo import wac_tile_projection

TASK = "crater-detection"
TILE_PX = 512


def match_boxes(predicted: np.ndarray, scores: np.ndarray, reference: np.ndarray, threshold: float):
    """Greedy one-to-one matching by descending confidence (COCO style). Returns [(pred, ref, iou)]."""
    if len(predicted) == 0 or len(reference) == 0:
        return []
    # torchvision provides the IoU used upstream; imported here so the contract and geometry parts of this
    # module stay importable (and testable) without the inference stack installed.
    import torch
    from torchvision.ops import box_iou

    ious = box_iou(torch.from_numpy(predicted), torch.from_numpy(reference)).numpy()
    taken: set[int] = set()
    matches = []
    for p in np.argsort(-scores, kind="stable"):
        candidates = [(ious[p, r], r) for r in range(len(reference)) if r not in taken and ious[p, r] >= threshold]
        if candidates:
            iou, r = max(candidates)
            taken.add(r)
            matches.append((int(p), int(r), float(iou)))
    return matches


def region_id(sample: str) -> str:
    product, row, col = sample.split("_")
    return f"wac-{product.lower()}-{row}-{col}"


def normalize(samples: list[str]) -> list[dict]:
    metadata = pd.read_parquet(config.WAC_DIR / "metadata.parquet")
    metadata["stem"] = metadata["WAC_VIS_TILE"].map(lambda p: p.rsplit("/", 1)[-1].removesuffix(".nc"))
    regions, craters = [], {}
    for sample in samples:
        run, arrays = artifacts.read_run(TASK, sample)
        rows = metadata[metadata["stem"] == sample]
        if len(rows) != 1:
            raise ValueError(f"{sample}: expected one metadata row, found {len(rows)}")
        meta = rows.iloc[0].to_dict()
        if {"val": "validation"}.get(meta["DATASET"], meta["DATASET"]) != {"val": "validation"}.get(run["split"], run["split"]):
            raise ValueError(f"{sample}: split mismatch between metadata ({meta['DATASET']}) and run ({run['split']})")
        tile = wac_tile_projection(meta, TILE_PX, TILE_PX)

        keep = arrays["scores"] >= config.CRATER_DISPLAY_CONFIDENCE
        boxes, scores = arrays["boxes_xyxy"][keep], arrays["scores"][keep]
        reference = arrays["reference_boxes_xyxy"]
        matches = match_boxes(boxes, scores, reference, config.CRATER_MATCH_IOU)
        matched = {p: iou for p, _, iou in matches}

        rid = region_id(sample)
        center_lon, center_lat = tile.center()
        k = tile.scale_factor(center_lon, center_lat)
        area_km2 = (TILE_PX * tile.pixel_size_x / k / 1000) * (TILE_PX * tile.pixel_size_y / k / 1000)

        tuples = []
        for i, ((x1, y1, x2, y2), score) in enumerate(zip(boxes, scores, strict=True)):
            lon, lat = tile.pixel_to_lonlat((x1 + x2) / 2, (y1 + y2) / 2)
            ground_x, ground_y = tile.pixel_ground_size_m(lon, lat)
            diameter_km = ((x2 - x1) * ground_x + (y2 - y1) * ground_y) / 2 / 1000
            truncated = int(x1 <= 0 or y1 <= 0 or x2 >= TILE_PX or y2 >= TILE_PX)
            point = build.coordinate(lat, lon)
            tuples.append([point["latitude"], point["longitude"], round(float(diameter_km), 3),
                           round(float(score), 4), truncated, int(i in matched)])
        craters[rid] = tuples

        with rasterio.open(config.WAC_DIR / "images_tiff" / f"{sample}.tif") as source:
            band_643 = source.read(4)  # bands 415, 566, 604, 643, 689 nm
        image_path = build.preview_webp(band_643, f"previews/craters/{rid}.webp")
        detail_path = build.write_json(
            f"crater-tiles/{rid}.json",
            {
                "regionId": rid,
                "imageSizePx": TILE_PX,
                "prediction": [[round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2),
                                round(float(y2 - y1), 2), round(float(s), 4)] for (x1, y1, x2, y2), s in zip(boxes, scores, strict=True)],
                "reference": [[round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2), round(float(y2 - y1), 2)]
                              for x1, y1, x2, y2 in reference],
                "matching": {"method": "greedy-by-confidence", "iouThreshold": config.CRATER_MATCH_IOU,
                             "matches": [[p, r, round(iou, 4)] for p, r, iou in matches]},
            },
        )
        regions.append(
            {
                "id": rid,
                **build.coordinate(center_lat, center_lon),
                "scale": "WAC",
                "source": {
                    "productId": meta["PRODUCT_ID"],
                    "footprint": [build.coordinate(lat, lon) for lon, lat in tile.corners()],
                    "sizeKm": round(TILE_PX * tile.pixel_size_x / k / 1000, 3),
                    "resolutionM": round(tile.pixel_size_x / k, 3),
                    "incidenceAngleDeg": round(float(meta["INCIDENCE_ANGLE"]), 2),
                    "imagePath": image_path,
                    "projection": build.projection_record(tile.proj4),
                    "grid": build.grid_record(tile),
                },
                "prediction": {
                    "detectionCount": len(tuples),
                    "densityPer1000Km2": round(len(tuples) / area_km2 * 1000, 3),
                    "confidenceThreshold": config.CRATER_DISPLAY_CONFIDENCE,
                },
                "reference": {"craterCount": int(len(reference))},
                "derived": {"matchedPredictions": len(matches), "score": None},
                "detailPath": detail_path,
                "provenance": build.provenance(TASK, run, "sombench-wac-crater-detection", f"{sample}.tif",
                                               "lfm-crater-detection", "robbins-2019"),
            }
        )

    merged_regions = build.upsert("crater-regions.json", regions)
    densities = [r["prediction"]["densityPer1000Km2"] for r in merged_regions]
    for region in merged_regions:
        region["derived"]["score"] = {
            "value": build.percentile_rank(region["prediction"]["densityPer1000Km2"], densities),
            "formula": "crater-region-density-rank-v1",
        }
    build.write_json("crater-regions.json", merged_regions)
    all_craters = {**build.read_json("craters.json", {}), **craters}
    build.write_json("craters.json", dict(sorted(all_craters.items())), compact=True)
    return regions
