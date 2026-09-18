"""Raw dense ice prospectivity → normalized polar sectors (contract v2).

The model is an unconstrained regression: raw values are preserved (16-bit raw raster, raw statistics, raw peak).
Clipping to the [0, 1] target scale happens only in the display raster and in the bounded MOONATLAS score. The
discovery anchor (edge-safe maximum, moonatlas_science/anchor.py) is a MOONATLAS-derived locator, published separately.
"""

from __future__ import annotations

import numpy as np
import rasterio

from . import anchor as anchors
from . import artifacts, build, config
from .geo import raster_tile

TASK = "ice-prospectivity"


def pole_of_name(sample: str) -> str:
    # patch_{row}_{col}_S_80S (south) | patch_{row}_{col}_80N (north)
    if sample.endswith("_S_80S"):
        return "south"
    if sample.endswith("_80N") and "_S_" not in sample:
        return "north"
    raise ValueError(f"{sample}: patch name does not identify a pole")


def sector_id(sample: str) -> str:
    parts = sample.split("_")
    return f"ice-{pole_of_name(sample)[0]}-{parts[1]}-{parts[2]}"


def check_hemisphere(sample: str, hemisphere_tag: str, projection: dict) -> str:
    """The pole from the patch name, the GeoTIFF hemisphere tag and the CRS; any disagreement fails the build."""
    by_tag = {"S": "south", "N": "north"}.get(hemisphere_tag)
    by_crs = (("south" if projection["latitudeOfOriginDeg"] < 0 else "north")
              if projection["kind"] == "polar-stereographic" else None)
    poles = {pole_of_name(sample), by_tag, by_crs}
    if len(poles) != 1 or None in poles:
        raise ValueError(f"{sample}: pole disagrees between name, hemisphere tag {hemisphere_tag!r} and CRS {projection}")
    return poles.pop()


def read_layer(sample: str, layer: str):
    with rasterio.open(config.ICE_DIR / f"{sample}_{layer}.tif") as source:
        return source.read().astype(np.float32), raster_tile(source), source.tags()


def load_patch(sample: str):
    """All layers of a patch, checking they share one CRS and grid."""
    reference, tile, tags = read_layer(sample, config.ICE_LABEL_LAYER)
    layers = {}
    for layer in config.ICE_INPUT_LAYERS:
        values, layer_tile, _ = read_layer(sample, layer)
        if layer_tile != tile:
            raise ValueError(f"{sample}: layer {layer} is not on the same grid as {config.ICE_LABEL_LAYER}")
        layers[layer] = values
    return reference[0], layers, tile, tags


def coverage(layers: dict[str, np.ndarray]) -> np.ndarray:
    """Pixels where every input layer is finite; elsewhere upstream replaced no-data with 0, so no output is published."""
    return np.all([np.isfinite(v).all(axis=0) for v in layers.values()], axis=0)


def raw_stats(values: np.ndarray) -> dict:
    return {
        "min": round(float(values.min()), 6),
        "max": round(float(values.max()), 6),
        "mean": round(float(values.mean()), 6),
        "p90": round(float(np.percentile(values, 90)), 6),
    }


def pixel_record(tile, pixel: dict) -> dict:
    """Pixel-center coordinate, raw value, grid position and distance to the valid edge."""
    lon, lat = tile.pixel_to_lonlat(pixel["col"] + 0.5, pixel["row"] + 0.5)
    return {**build.coordinate(lat, lon), "value": round(pixel["value"], 6), "col": pixel["col"], "row": pixel["row"],
            "edgeDistancePx": pixel["edgeDistancePx"]}


def normalize(samples: list[str]) -> list[dict]:
    sectors = []
    for sample in samples:
        run, arrays = artifacts.read_run(TASK, sample)
        raw = arrays["prediction"].astype(np.float64)
        reference, layers, tile, tags = load_patch(sample)
        projection = build.projection_record(tile.proj4)
        pole = check_hemisphere(sample, tags.get("hemisphere", ""), projection)

        covered = coverage(layers)
        reference_covered = np.isfinite(reference)
        both = covered & reference_covered
        if raw.shape != reference.shape or not both.any():
            raise ValueError(f"{sample}: prediction and reference grids do not overlap")

        peak, anchor = anchors.peak_and_anchor(raw, covered)
        center_lon, center_lat = tile.center()
        k = tile.scale_factor(center_lon, center_lat)
        sid = sector_id(sample)

        slope, tmax, lpsr = layers["SLOPE"][0], layers["TMAX"][0], layers["LPSR"][0]
        error = raw[both] - reference[both]
        prediction_stats = raw_stats(raw[covered])
        sectors.append(
            {
                "id": sid,
                **build.coordinate(center_lat, center_lon),
                "pole": pole,
                "source": {
                    "patchId": sample,
                    "footprintGrid": [[build.coordinate(lat, lon) for lon, lat in row] for row in tile.grid(9)],
                    "sizeKm": round(tile.width * tile.pixel_size_x / k / 1000, 3),
                    "resolutionM": round(tile.pixel_size_x / k, 3),
                    "psrFraction": round(float(lpsr[covered].mean()), 4),
                    # Only layers whose units are consistent with the data (degrees 0–30, kelvin 83–324, binary PSR);
                    # DICE, LPSR_DEN and LPSR_DIS units are undocumented upstream and are not published.
                    "atAnchor": {} if anchor is None else {
                        "slopeDeg": round(float(slope[anchor["row"], anchor["col"]]), 2),
                        "maxTemperatureK": round(float(tmax[anchor["row"], anchor["col"]]), 1),
                        "permanentShadow": bool(lpsr[anchor["row"], anchor["col"]] >= 0.5),
                    },
                    "projection": projection,
                    "grid": build.grid_record(tile),
                },
                "prediction": {
                    "raw": prediction_stats,
                    "display": {
                        "coveredPixels": int(covered.sum()),
                        "clippedPixels": int(((raw < 0) | (raw > 1))[covered].sum()),
                    },
                    "rawPeak": pixel_record(tile, peak),
                    "heatmap": build.value_raster(raw, covered, sid),
                },
                "reference": {
                    "raw": raw_stats(reference[reference_covered].astype(np.float64)),
                    "heatmap": build.value_raster(np.nan_to_num(reference).astype(np.float64), reference_covered, f"{sid}-reference"),
                },
                "derived": {
                    "rmse": round(float(np.sqrt(np.mean(error**2))), 6),
                    "mae": round(float(np.mean(np.abs(error))), 6),
                    "score": {"value": build.ice_p90_bounded_score(prediction_stats["p90"]), "formula": "ice-p90-bounded-v2"},
                    "discoveryAnchor": None if anchor is None else {
                        **pixel_record(tile, anchor), "method": anchors.METHOD, "edgeMarginPx": anchors.EDGE_MARGIN_PX},
                },
                "provenance": build.provenance(TASK, run, "sombench-ice-prospectivity-regression", sample,
                                               "lfm-ice-prospectivity", "coyan-2025"),
            }
        )
    build.upsert("ice-sectors.json", sectors)
    return sectors
