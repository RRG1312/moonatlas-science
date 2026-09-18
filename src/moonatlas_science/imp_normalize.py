"""Raw IMP segmentation → normalized IMP regions with prediction and reference masks."""

from __future__ import annotations

import re

import numpy as np
import rasterio

from . import artifacts, build, config
from .geo import raster_tile

TASK = "imp-segmentation"
REFERENCE_NODATA = 255


def region_id(sample: str) -> str:
    match = re.fullmatch(r"(M\d+[LR]E)\.ech\.cog__target_(\d+)__idx_(\d+)_p0", sample)
    if not match:
        raise ValueError(f"unexpected IMP sample name {sample}")
    product, target, index = match.groups()
    return f"imp-{product.lower()}-t{target}-i{index}"


def normalize(samples: list[str]) -> list[dict]:
    regions = []
    for sample in samples:
        run, arrays = artifacts.read_run(TASK, sample)
        with rasterio.open(config.IMP_DIR / "all" / f"{sample}_img.tif") as source:
            image, tile, nodata = source.read(1), raster_tile(source), source.nodata
        with rasterio.open(config.IMP_DIR / "all" / f"{sample}_mask.tif") as source:
            reference, reference_tile = source.read(1), raster_tile(source)
        if reference_tile != tile:
            raise ValueError(f"{sample}: reference mask is not on the image grid")

        valid = np.isfinite(image) & (image != nodata if nodata is not None else True)
        predicted = (arrays["mask"] == 1) & valid
        annotated = reference != REFERENCE_NODATA
        reference_mask = (reference == 1) & annotated

        center_lon, center_lat = tile.center()
        ground_x, ground_y = tile.pixel_ground_size_m(center_lon, center_lat)
        pixel_area = ground_x * ground_y
        union = (predicted | reference_mask)[annotated].sum()
        rid = region_id(sample)

        centroid = None
        if predicted.any():
            rows, cols = np.nonzero(predicted)
            lon, lat = tile.pixel_to_lonlat(cols.mean() + 0.5, rows.mean() + 0.5)
            centroid = build.coordinate(lat, lon)

        regions.append(
            {
                "id": rid,
                **build.coordinate(center_lat, center_lon),
                "source": {
                    "nacProductId": sample.split(".")[0],
                    "sizeM": round(tile.width * ground_x, 2),
                    "resolutionM": round(ground_x, 4),
                    "footprint": [build.coordinate(lat, lon) for lon, lat in tile.corners()],
                    "imagePath": build.preview_webp(np.where(valid, image, np.nan), f"previews/imp/{rid}.webp"),
                    "projection": build.projection_record(tile.proj4),
                    "grid": build.grid_record(tile),
                },
                "prediction": {
                    "pixelCount": int(predicted.sum()),
                    "areaM2": round(float(predicted.sum() * pixel_area), 1),
                    "pixelFraction": round(float(predicted.sum() / valid.sum()), 4),
                    "meanSoftmaxScore": round(float(arrays["softmax_imp"][predicted].mean()), 4) if predicted.any() else None,
                    "centroid": centroid,
                    "maskPath": build.mask_webp(predicted, f"overlays/imp/{rid}.webp"),
                },
                "reference": {
                    "pixelCount": int(reference_mask.sum()),
                    "areaM2": round(float(reference_mask.sum() * pixel_area), 1),
                    "maskPath": build.mask_webp(reference_mask, f"overlays/imp/{rid}-reference.webp"),
                },
                "derived": {
                    "iou": round(float((predicted & reference_mask)[annotated].sum() / union), 4) if union else None,
                    "score": None,
                },
                "provenance": build.provenance(TASK, run, "sombench-imp-segmentation", f"{sample}_img.tif",
                                               "lfm-imp-segmentation", "hargitai-2025"),
            }
        )

    merged = build.upsert("imp-regions.json", regions)
    areas = [r["prediction"]["areaM2"] for r in merged if r["prediction"]["areaM2"] > 0]
    for region in merged:
        area = region["prediction"]["areaM2"]
        region["derived"]["score"] = (
            {"value": build.percentile_rank(area, areas), "formula": "imp-area-rank-v1"} if area > 0 else None
        )
    build.write_json("imp-regions.json", merged)
    return regions
