"""Per-pole ice mosaics composed from the raw model outputs in their native polar stereographic grid.

Rules (docs/provenance/data-contract.md › IcePolarMosaic): every patch must share one CRS and pixel size and sit on a common
pixel grid; overlapping patches are rejected (SomBench patches tile the polar grid, so an overlap means an
input error, not something to average away). Uncovered cells stay transparent. Values come from the raw
inference artifacts (prediction) and the upstream PRO layer (reference), never from re-decoded images.
"""

from __future__ import annotations

import numpy as np

from . import artifacts, build
from .ice_normalize import coverage, load_patch

GRID_TOLERANCE_PX = 1e-6


def build_mosaics() -> list[dict]:
    sectors = build.read_json("ice-sectors.json", [])
    if not sectors:
        raise ValueError("no normalized ice sectors; run process_ice.py first")
    mosaics = []
    for pole in ("south", "north"):
        members = sorted((s for s in sectors if s["pole"] == pole), key=lambda s: s["id"])
        if not members:
            continue  # no coverage is reported as absence, never synthesized
        patches = {}
        for sector in members:
            sample = sector["source"]["patchId"]
            reference, layers, tile, _ = load_patch(sample)
            _, arrays = artifacts.read_run("ice-prospectivity", sample)
            patches[sector["id"]] = (tile, arrays["prediction"].astype(np.float64), coverage(layers), reference)
        first = next(iter(patches.values()))[0]
        for sid, (tile, *_rest) in patches.items():
            if (tile.proj4, tile.pixel_size_x, tile.pixel_size_y) != (first.proj4, first.pixel_size_x, first.pixel_size_y):
                raise ValueError(f"{sid}: CRS or pixel size differs from the other {pole} patches")
        x_min = min(t.x_min for t, *_ in patches.values())
        y_max = max(t.y_max for t, *_ in patches.values())
        size = first.pixel_size_x
        width = round(max((t.x_min - x_min) / size + t.width for t, *_ in patches.values()))
        height = round(max((y_max - t.y_max) / size + t.height for t, *_ in patches.values()))

        values = {"prediction": np.zeros((height, width)), "reference": np.zeros((height, width))}
        covered = {"prediction": np.zeros((height, width), bool), "reference": np.zeros((height, width), bool)}
        placed = []
        for sid, (tile, prediction, prediction_covered, reference) in patches.items():
            col, row = (tile.x_min - x_min) / size, (y_max - tile.y_max) / size
            if abs(col - round(col)) > GRID_TOLERANCE_PX or abs(row - round(row)) > GRID_TOLERANCE_PX:
                raise ValueError(f"{sid}: patch is not aligned to the {pole} pixel grid")
            col, row = round(col), round(row)
            window = np.s_[row:row + tile.height, col:col + tile.width]
            for layer, data, mask in (("prediction", prediction, prediction_covered),
                                      ("reference", np.nan_to_num(reference), np.isfinite(reference))):
                if (covered[layer][window] & mask).any():
                    raise ValueError(f"{sid}: overlaps another {pole} patch ({layer})")
                values[layer][window] = np.where(mask, data, values[layer][window])
                covered[layer][window] |= mask
            placed.append({"sectorId": sid, "col": col, "row": row, "widthPx": tile.width, "heightPx": tile.height})

        tag = pole[0]
        grid = build.grid_record(first)
        grid.update({"xMinM": round(x_min, 3), "yMaxM": round(y_max, 3), "widthPx": width, "heightPx": height})
        mosaics.append(
            {
                "id": f"ice-mosaic-{tag}",
                "pole": pole,
                "projection": build.projection_record(first.proj4),
                "grid": grid,
                "prediction": build.value_raster(values["prediction"], covered["prediction"], f"ice-mosaic-{tag}"),
                "reference": build.value_raster(values["reference"], covered["reference"], f"ice-mosaic-{tag}-reference"),
                "patches": placed,
                "overlapRule": "reject",
            }
        )
    build.write_json("ice-mosaics.json", mosaics)
    return mosaics
