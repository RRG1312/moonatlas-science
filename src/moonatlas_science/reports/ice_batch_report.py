"""Review an ice batch build, per pole: mosaic numerical validation, edge effects, anchors, diagnostics, statistics.

Checks run against the source GeoTIFFs through rasterio/pyproj, independently of the pipeline's own projection code:
  1. SOURCE PIXEL → projected → lunar coordinate → MOSAIC PIXEL reproduces the raw prediction and reference values,
     and MOSAIC PIXEL → lunar coordinate lands on the source pixel center.
  2. Coverage: mosaic pixels outside patches are transparent; inside, coverage equals the source coverage.
  3. Grid-wide forward/inverse round trip (all longitudes, including across ±180°).
  4. Hemisphere: patch names, sector poles, latitude signs and the mosaic projection agree.
  5. Seams between adjacent patches (reference and input layers must be continuous if placement is right).
  6. Orientation against independent imagery: Diviner TMAX vs the LROC polar color mosaic under the 8 dihedral
     transforms of each patch (the identity must correlate best).
  7. Edge effects by distance to the valid edge, and discovery-anchor sensitivity to the edge margin (0/4/8/16 px).
Outputs (development only, never shipped): data/artifacts/sanity/<build name>/{report.txt, report.json,
mosaic-debug-<pole>.png, geography-check-<pole>.png}. Exits 1 when a numerical or consistency check fails.

Usage: MOONATLAS_BUILD_DIR=data/build/science-0.2.0-heldout-test python -m moonatlas_science.reports.ice_batch_report
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from pyproj import CRS, Transformer

from moonatlas_science import artifacts, config
from moonatlas_science.ice_normalize import coverage, load_patch, pole_of_name

from .. import anchor as anchors
from .. import encoding
from .. import projection as proj

SAMPLES_PER_PATCH = 64
SEED = 20260917
POSITION_TOLERANCE_PX = 1e-6
GEO_TOLERANCE_M = 1e-3
ROUND_TRIP_TOLERANCE_M = 1e-3
ANCHOR_MARGINS = (0, 4, 8, 16)
EDGE_RINGS = ((0, 4), (4, 8), (8, 12), (12, 16), (16, 24), (24, 10_000))
BASEMAP = config.DATA_ROOT / "raw" / "moonkit" / "lroc_color_poles_8k.tif"  # scripts/textures/build_textures.py
LUNAR_GEOGRAPHIC = CRS.from_proj4("+proj=longlat +R=1737400 +no_defs")
# lib/design/colormaps.ts › ICE_INSPECTOR_COLORMAP
COLORMAP = [(0, "#07161D"), (0.25, "#0C2C3A"), (0.5, "#155A73"), (0.75, "#2F95B3"), (0.9, "#6CC6DE"), (1, "#C8EEF8")]
DIHEDRAL = {
    "identity": lambda a: a, "rot90": lambda a: np.rot90(a, 1), "rot180": lambda a: np.rot90(a, 2),
    "rot270": lambda a: np.rot90(a, 3), "flip-lr": np.fliplr, "flip-ud": np.flipud,
    "transpose": lambda a: a.T, "anti-transpose": lambda a: np.rot90(a, 2).T,
}


def load_json(name: str):
    return json.loads((config.BUILD_DIR / name).read_text(encoding="utf-8"))


def decode(raster: dict) -> tuple[np.ndarray, np.ndarray]:
    rgba = np.asarray(Image.open(config.BUILD_DIR / raster["rawPath"]).convert("RGBA"))
    return encoding.decode_raw(rgba, raster["rawEncoding"])


def colorize(values: np.ndarray) -> np.ndarray:
    stops = [s for s, _ in COLORMAP]
    rgb = [[int(c[i:i + 2], 16) for i in (1, 3, 5)] for _, c in COLORMAP]
    v = np.clip(values, 0, 1)
    return np.dstack([np.interp(v, stops, [c[k] for c in rgb]) for k in range(3)]).astype(np.uint8)


def great_circle_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    # Haversine: arccos of the spherical law of cosines bottoms out at ~2.6 cm on the Moon in float64.
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * config.LUNAR_RADIUS_M * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


class Basemap:
    """LROC polar color mosaic (simple cylindrical, 8-bit gray), sampled bilinearly at lunar coordinates."""

    def __init__(self, path: Path):
        Image.MAX_IMAGE_PIXELS = None
        self.pixels = np.asarray(Image.open(path).convert("L"))
        self.height, self.width = self.pixels.shape

    def sample(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        col = (np.asarray(lon) + 180) / 360 * self.width - 0.5
        row = np.clip((90 - np.asarray(lat)) / 180 * self.height - 0.5, 0, self.height - 1.001)
        c0, r0 = np.floor(col).astype(int), np.floor(row).astype(int)
        fc, fr = col - np.floor(col), row - r0
        c1 = (c0 + 1) % self.width
        c0 %= self.width
        px = self.pixels.astype(np.float64, copy=False)
        top = px[r0, c0] * (1 - fc) + px[r0, c1] * fc
        bottom = px[r0 + 1, c0] * (1 - fc) + px[r0 + 1, c1] * fc
        return top * (1 - fr) + bottom * fr


def block_mean(a: np.ndarray, block: int) -> np.ndarray:
    h, w = a.shape[0] // block, a.shape[1] // block
    return a[:h * block, :w * block].reshape(h, block, w, block).mean(axis=(1, 3))


def high_pass(a: np.ndarray, radius: int = 4) -> np.ndarray:
    padded = np.pad(a, radius, mode="reflect")
    window = 2 * radius + 1
    integral = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    local = (integral[window:, window:] - integral[:-window, window:] - integral[window:, :-window]
             + integral[:-window, :-window]) / window**2
    return a - local


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a @ b) / math.sqrt((a @ a) * (b @ b)))


def main() -> int:
    mosaics = load_json("ice-mosaics.json")
    sectors = {s["id"]: s for s in load_json("ice-sectors.json")}
    basemap = Basemap(BASEMAP) if BASEMAP.exists() else None
    out = config.SANITY_DIR / config.BUILD_DIR.name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    report = {
        "build": config.BUILD_DIR.name,
        "tolerances": {"positionPx": POSITION_TOLERANCE_PX, "geoM": GEO_TOLERANCE_M, "roundTripM": ROUND_TRIP_TOLERANCE_M,
                       "value": "rawEncoding.scale / 2"},
        "anchor": {"method": anchors.METHOD, "edgeMarginPx": anchors.EDGE_MARGIN_PX},
        "poles": {},
        "problems": [],
    }
    if len({m["pole"] for m in mosaics}) != len(mosaics):
        report["problems"].append("more than one mosaic per pole")
    for mosaic in mosaics:
        pole_report, problems = review_pole(mosaic, sectors, basemap, rng, out)
        report["poles"][mosaic["pole"]] = pole_report
        report["problems"] += [f"{mosaic['pole']}: {p}" for p in problems]
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    (out / "report.txt").write_text(_text(report), encoding="utf-8", newline="\n")
    print(_text(report))
    print(f"written to {out}")
    return 1 if report["problems"] else 0


def review_pole(mosaic: dict, sectors: dict, basemap: Basemap | None, rng, out: Path) -> tuple[dict, list[str]]:
    pole = mosaic["pole"]
    p, g = mosaic["projection"], mosaic["grid"]
    prediction, prediction_covered = decode(mosaic["prediction"])
    reference, reference_covered = decode(mosaic["reference"])
    half_step = {"prediction": mosaic["prediction"]["rawEncoding"]["scale"] / 2 + 1e-9,
                 "reference": mosaic["reference"]["rawEncoding"]["scale"] / 2 + 1e-9}
    problems: list[str] = []
    if (p["latitudeOfOriginDeg"] < 0) != (pole == "south"):
        problems.append(f"mosaic projection latitude of origin {p['latitudeOfOriginDeg']} disagrees with pole {pole}")
    patch_rows, all_values, per_patch = [], [], {}
    in_patch = np.zeros(prediction_covered.shape, bool)
    mosaic_crs = None
    rings = {ring: [0, 0, 0, 0.0, 0] for ring in EDGE_RINGS}  # pixels, >1, <0, Σ|pred − ref|, reference-valid pixels

    for placed in mosaic["patches"]:
        sector = sectors[placed["sectorId"]]
        sample = sector["source"]["patchId"]
        if pole_of_name(sample) != pole or sector["pole"] != pole or (sector["latitude"] < 0) != (pole == "south"):
            problems.append(f"{sample}: hemisphere disagrees between name, sector and mosaic")
        _, arrays = artifacts.read_run("ice-prospectivity", sample)
        raw = arrays["prediction"].astype(np.float64)
        reference_source, layers, _, _ = load_patch(sample)
        covered = coverage(layers)
        with rasterio.open(config.ICE_DIR / f"{sample}_PRO.tif") as src:
            transform, crs = src.transform, CRS.from_wkt(src.crs.to_wkt())
        mosaic_crs = mosaic_crs or crs
        to_geo = Transformer.from_crs(crs, LUNAR_GEOGRAPHIC, always_xy=True)
        window = np.s_[placed["row"]:placed["row"] + placed["heightPx"], placed["col"]:placed["col"] + placed["widthPx"]]
        in_patch[window] = True

        if not np.array_equal(prediction_covered[window], covered):
            problems.append(f"{sample}: mosaic prediction coverage differs from the source coverage")
        full_error = float(np.abs(prediction[window][covered] - raw[covered]).max())
        if full_error > half_step["prediction"]:
            problems.append(f"{sample}: mosaic prediction differs from the raw artifact by {full_error:.3g}")

        rows, cols = np.nonzero(covered)
        pick = rng.choice(len(rows), size=min(SAMPLES_PER_PATCH, len(rows)), replace=False)
        position_errors, geo_errors, value_errors, ref_errors = [], [], [], []
        for r, c in zip(rows[pick], cols[pick]):
            x, y = transform * (c + 0.5, r + 0.5)
            lon, lat = to_geo.transform(x, y)
            mx, my = proj.forward(p, lat, lon)
            mcol, mrow = (mx - g["xMinM"]) / g["pixelSizeXM"], (g["yMaxM"] - my) / g["pixelSizeYM"]
            position_errors.append(math.hypot(mcol - (placed["col"] + c + 0.5), mrow - (placed["row"] + r + 0.5)))
            mc, mr = int(math.floor(mcol)), int(math.floor(mrow))
            value_errors.append(abs(prediction[mr, mc] - raw[r, c]) if prediction_covered[mr, mc] else math.inf)
            if np.isfinite(reference_source[r, c]):
                ref_errors.append(abs(reference[mr, mc] - reference_source[r, c]) if reference_covered[mr, mc] else math.inf)
            back_lat, back_lon = proj.pixel_to_latlon(p, g, mc + 0.5, mr + 0.5)
            geo_errors.append(float(great_circle_m(lat, lon, back_lat, back_lon)))
        row = {
            "patch": sample, "sectorId": placed["sectorId"], "split": sector["provenance"]["datasetSplit"],
            "col": placed["col"], "row": placed["row"], "samples": len(pick),
            "maxPositionErrorPx": max(position_errors), "maxGeoErrorM": max(geo_errors),
            "maxValueError": max(value_errors), "maxReferenceError": max(ref_errors, default=0.0),
            "coveredPixels": int(covered.sum()),
        }
        for key, limit in (("maxPositionErrorPx", POSITION_TOLERANCE_PX), ("maxGeoErrorM", GEO_TOLERANCE_M),
                           ("maxValueError", half_step["prediction"]), ("maxReferenceError", half_step["reference"])):
            if not row[key] <= limit:
                problems.append(f"{sample}: {key} {row[key]:.3g} exceeds {limit:.3g}")
        values = raw[covered]
        row.update(_stats(values))

        distance = anchors.edge_distance(covered)
        valid_ref = covered & np.isfinite(reference_source)
        for lo, hi in EDGE_RINGS:
            ring = covered & (distance >= lo) & (distance < hi)
            ring_ref = valid_ref & (distance >= lo) & (distance < hi)
            t = rings[(lo, hi)]
            t[0] += int(ring.sum())
            t[1] += int((raw[ring] > 1).sum())
            t[2] += int((raw[ring] < 0).sum())
            t[3] += float(np.abs(raw[ring_ref] - reference_source[ring_ref]).sum())
            t[4] += int(ring_ref.sum())
        row["anchorSensitivity"] = _anchor_sensitivity(raw, covered, distance, transform, to_geo)
        published_peak, published_anchor = sector["prediction"]["rawPeak"], sector["derived"]["discoveryAnchor"]
        expected_peak, expected_anchor = anchors.peak_and_anchor(raw, covered)
        for label, record, truth in (("raw peak", published_peak, expected_peak), ("anchor", published_anchor, expected_anchor)):
            key = lambda r: None if r is None else (r["row"], r["col"], r["edgeDistancePx"])  # noqa: E731
            if key(record) != key(truth):
                problems.append(f"{sample}: published {label} differs from the recomputed one")
        row["rawPeakEdgeDistancePx"] = published_peak["edgeDistancePx"]
        row["anchorEdgeDistancePx"] = published_anchor["edgeDistancePx"] if published_anchor else None
        patch_rows.append(row)
        all_values.append(values)
        per_patch[sample] = (placed, raw, covered, reference_source, layers["TMAX"][0], transform, to_geo, sector)

    if prediction_covered[~in_patch].any() or reference_covered[~in_patch].any():
        problems.append("mosaic has covered pixels outside every patch")
    rects = [(q["col"], q["row"], q["col"] + q["widthPx"], q["row"] + q["heightPx"]) for q in mosaic["patches"]]
    overlaps = [(a, b) for i, a in enumerate(rects) for b in rects[i + 1:]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]]
    if overlaps:
        problems.append(f"overlapping patch rectangles: {overlaps}")

    round_trip = _round_trip(p, g, mosaic_crs, rng)
    if round_trip["maxErrorM"] > ROUND_TRIP_TOLERANCE_M:
        problems.append(f"grid round trip error {round_trip['maxErrorM']:.3g} m")
    orientation = _orientation(per_patch, basemap) if basemap else {}
    values = np.concatenate(all_values)
    total = {k: sum(t[i] for t in rings.values()) for k, i in (("pixels", 0), ("above1", 1), ("below0", 2), ("valid", 4))}
    mae = sum(t[3] for t in rings.values()) / total["valid"]
    edge_rings = [{
        "edgeDistancePx": [lo, None if hi >= 10_000 else hi - 1],
        "pixelShare": round(t[0] / total["pixels"], 4),
        "above1Enrichment": round((t[1] / max(1, total["above1"])) / (t[0] / total["pixels"]), 2) if t[0] else None,
        "below0Enrichment": round((t[2] / max(1, total["below0"])) / (t[0] / total["pixels"]), 2) if t[0] else None,
        "maeVsReferenceRatio": round((t[3] / t[4]) / mae, 2) if t[4] else None,
    } for (lo, hi), t in rings.items()]
    cells = 9 * 9
    pole_report = {
        "mosaic": {
            "id": mosaic["id"], "widthPx": g["widthPx"], "heightPx": g["heightPx"], "pixelSizeM": g["pixelSizeXM"],
            "extentKm": [round(g["widthPx"] * g["pixelSizeXM"] / 1000, 2), round(g["heightPx"] * g["pixelSizeYM"] / 1000, 2)],
            "projection": p, "patches": len(mosaic["patches"]), "gridCells": cells,
            "emptyCells": cells - len(mosaic["patches"]), "coveredPixels": int(prediction_covered.sum()),
            "coveredFractionOfGrid": round(float(prediction_covered.mean()), 4), "overlaps": len(overlaps),
            "polePixel": [round((p["falseEastingM"] - g["xMinM"]) / g["pixelSizeXM"], 3),
                          round((g["yMaxM"] - p["falseNorthingM"]) / g["pixelSizeYM"], 3)],
            "files": {f"{layer}{kind}": (config.BUILD_DIR / mosaic[layer][key]).stat().st_size
                      for layer in ("prediction", "reference") for kind, key in (("Raw", "rawPath"), ("Display", "displayPath"))},
        },
        "patches": patch_rows,
        "roundTrip": round_trip,
        "seams": _seams(per_patch),
        "orientation": orientation,
        "edgeRings": edge_rings,
        "anchorSensitivity": _sensitivity_summary(patch_rows),
        "distribution": _stats(values) | {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                                          "p99": float(np.percentile(values, 99))},
    }
    _debug_image(mosaic, prediction, prediction_covered, in_patch, per_patch, mosaic_crs, basemap, out / f"mosaic-debug-{pole}.png")
    if basemap:
        _geography_image(per_patch, basemap, orientation, out / f"geography-check-{pole}.png")
    return pole_report, problems


def _stats(values: np.ndarray) -> dict:
    return {
        "count": int(values.size), "min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()),
        "p90": float(np.percentile(values, 90)), "below0": int((values < 0).sum()), "above1": int((values > 1).sum()),
        "clippedPercent": round(100 * float(((values < 0) | (values > 1)).mean()), 3),
    }


def _anchor_sensitivity(raw, covered, distance, transform, to_geo) -> dict:
    picks = {}
    for margin in ANCHOR_MARGINS:
        eligible = covered & (distance >= margin)
        if not eligible.any():
            picks[margin] = None  # no pixel this far from the valid edge
            continue
        r, c = anchors.argmax_where(raw, eligible)
        lon, lat = to_geo.transform(*(transform * (c + 0.5, r + 0.5)))
        picks[margin] = (r, c, float(raw[r, c]), lat, lon, int(distance[r, c]))
    r0, c0, v0, lat0, lon0, _ = picks[0]
    return {str(m): None if pick is None else {
        "value": round(pick[2], 6), "deltaValue": round(pick[2] - v0, 6), "shiftPx": round(math.hypot(pick[0] - r0, pick[1] - c0), 2),
        "shiftKm": round(float(great_circle_m(lat0, lon0, pick[3], pick[4])) / 1000, 3), "edgeDistancePx": pick[5]}
        for m, pick in picks.items()}


def _sensitivity_summary(rows: list[dict]) -> dict:
    summary = {}
    for margin in ANCHOR_MARGINS[1:]:
        cells = [r["anchorSensitivity"][str(margin)] for r in rows if r["anchorSensitivity"][str(margin)] is not None]
        moved = [c for c in cells if c["shiftPx"] > 0]
        summary[str(margin)] = {
            "moved": len(moved), "patches": len(cells), "withoutEligiblePixel": len(rows) - len(cells),
            "maxAbsDeltaValue": max(abs(c["deltaValue"]) for c in cells),
            "medianAbsDeltaValueOfMoved": float(np.median([abs(c["deltaValue"]) for c in moved])) if moved else 0.0,
            "maxShiftKm": max(c["shiftKm"] for c in cells),
            "medianShiftKmOfMoved": float(np.median([c["shiftKm"] for c in moved])) if moved else 0.0,
        }
    return summary


def _round_trip(p, g, crs, rng) -> dict:
    """Random continuous mosaic points: our inverse → pyproj forward must return the same projected point."""
    to_projected = Transformer.from_crs(LUNAR_GEOGRAPHIC, crs, always_xy=True)
    cols, rows = rng.uniform(0, g["widthPx"], 20000), rng.uniform(0, g["heightPx"], 20000)
    errors, longitudes, latitudes = [], [], []
    for c, r in zip(cols, rows):
        lat, lon = proj.pixel_to_latlon(p, g, c, r)
        x, y = to_projected.transform(lon, lat)
        errors.append(math.hypot(x - (g["xMinM"] + c * g["pixelSizeXM"]), y - (g["yMaxM"] - r * g["pixelSizeYM"])))
        longitudes.append(lon)
        latitudes.append(lat)
    longitudes, latitudes = np.array(longitudes), np.array(latitudes)
    return {"points": len(errors), "maxErrorM": max(errors), "longitudeRange": [float(longitudes.min()), float(longitudes.max())],
            "latitudeRange": [float(latitudes.min()), float(latitudes.max())],
            "pointsWithin1DegOfAntimeridian": int((np.abs(longitudes) > 179).sum())}


def _seams(patches: dict) -> list[dict]:
    """Adjacent patches: mean |difference| across the shared edge vs between the last two interior rows/columns."""
    by_cell = {(q[0]["row"] // 256, q[0]["col"] // 256): (name, q) for name, q in patches.items()}
    seams = []
    for (r, c), (name, (_, raw, _, ref, tmax, transform, _, _)) in sorted(by_cell.items()):
        for dr, dc in ((0, 1), (1, 0)):
            if (r + dr, c + dc) not in by_cell:
                continue
            other_name, (_, raw2, _, ref2, tmax2, transform2, _, _) = by_cell[(r + dr, c + dc)]
            expected = transform * ((256, 0) if dc else (0, 256))
            actual = transform2 * (0, 0)
            row = {"patches": [name, other_name], "direction": "east" if dc else "south",
                   "edgeGapM": math.hypot(expected[0] - actual[0], expected[1] - actual[1])}
            for label, a, b in (("reference", ref, ref2), ("TMAX", tmax, tmax2), ("prediction", raw, raw2)):
                edge_a, inner_a, edge_b = (a[:, -1], a[:, -2], b[:, 0]) if dc else (a[-1, :], a[-2, :], b[0, :])
                ok = np.isfinite(edge_a) & np.isfinite(edge_b) & np.isfinite(inner_a)
                row[label] = ({"acrossSeam": float(np.abs(edge_a[ok] - edge_b[ok]).mean()),
                               "interior": float(np.abs(edge_a[ok] - inner_a[ok]).mean())} if ok.any() else None)
            seams.append(row)
    return seams


def _patch_basemap(basemap: Basemap, transform, to_geo) -> np.ndarray:
    cols, rows = np.meshgrid(np.arange(256) + 0.5, np.arange(256) + 0.5)
    xs, ys = transform * (cols, rows)
    lon, lat = to_geo.transform(xs, ys)
    return basemap.sample(np.asarray(lat), np.asarray(lon))


def _orientation(patches: dict, basemap: Basemap) -> dict:
    """Pearson r between TMAX and basemap brightness (4 px blocks) for the 8 dihedral transforms of TMAX."""
    result = {}
    for name, (_, _, _, _, tmax, transform, to_geo, _) in patches.items():
        image = block_mean(_patch_basemap(basemap, transform, to_geo), 4)
        temperature = block_mean(np.nan_to_num(tmax, nan=float(np.nanmean(tmax))), 4)
        scores = {}
        for label, f in DIHEDRAL.items():
            t = np.ascontiguousarray(f(temperature))
            scores[label] = {"r": round(correlation(t, image), 3), "rHighPass": round(correlation(high_pass(t), high_pass(image)), 3)}
        result[name] = {"scores": scores, "bestHighPass": max(scores, key=lambda k: scores[k]["rHighPass"])}
    return result


def _debug_image(mosaic, values, covered, in_patch, patches, crs, basemap, path: Path) -> None:
    p, g, pole = mosaic["projection"], mosaic["grid"], mosaic["pole"]
    sign, hemi = (-1, "S") if pole == "south" else (1, "N")
    margin, h, w = 80, g["heightPx"], g["widthPx"]
    canvas = np.zeros((h, w, 3), np.uint8) + 14
    if basemap:
        to_geo = Transformer.from_crs(crs, LUNAR_GEOGRAPHIC, always_xy=True)
        cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        lon, lat = to_geo.transform(g["xMinM"] + cols * g["pixelSizeXM"], g["yMaxM"] - rows * g["pixelSizeYM"])
        canvas[:] = np.clip(basemap.sample(np.asarray(lat), np.asarray(lon)) * 0.55, 0, 255)[..., None].astype(np.uint8)
    color = colorize(values)
    canvas[covered] = color[covered]
    stripes = (np.add.outer(np.arange(h), np.arange(w)) // 6) % 2 == 0
    canvas[in_patch & ~covered & stripes] = (200, 60, 60)
    image = Image.new("RGB", (w + 2 * margin, h + 2 * margin), (8, 10, 12))
    image.paste(Image.fromarray(canvas), (margin, margin))
    draw = ImageDraw.Draw(image)
    font, small = _font(26), _font(20)

    def px(lat, lon):
        x, y = proj.forward(p, lat, lon)
        return margin + (x - g["xMinM"]) / g["pixelSizeXM"], margin + (g["yMaxM"] - y) / g["pixelSizeYM"]

    for lat in range(80, 90):  # graticule: parallels every 1°, meridians every 30°
        points = [px(sign * lat, lon) for lon in np.linspace(-180, 180, 721)]
        draw.line(points, fill=(90, 140, 160) if lat % 5 == 0 else (55, 75, 85), width=2)
        x, y = px(sign * lat, 0)
        draw.text((x + 6, y - 24), f"{lat}°{hemi}", fill=(150, 190, 205), font=small)
    for lon in range(-180, 180, 30):
        draw.line([px(sign * 89.999, lon), px(sign * 80, lon)], fill=(55, 75, 85), width=2)
        x, y = px(sign * 80.6, lon)
        draw.text((x, y), f"{abs(lon)}°{'E' if lon > 0 else 'W' if lon < 0 else ''}", fill=(150, 190, 205), font=small)
    pole_xy = px(sign * 90, 0)
    draw.ellipse([pole_xy[0] - 8, pole_xy[1] - 8, pole_xy[0] + 8, pole_xy[1] + 8], outline=(255, 255, 255), width=3)
    draw.text((pole_xy[0] + 12, pole_xy[1] + 4), f"90°{hemi}", fill=(255, 255, 255), font=small)
    for row in range(9):
        for col in range(9):
            x0, y0 = margin + col * 256, margin + row * 256
            draw.rectangle([x0, y0, x0 + 256, y0 + 256], outline=(60, 60, 60), width=1)
    for placed, _, _, _, _, _, _, sector in patches.values():
        x0, y0 = margin + placed["col"], margin + placed["row"]
        draw.rectangle([x0, y0, x0 + placed["widthPx"] - 1, y0 + placed["heightPx"] - 1], outline=(255, 214, 90), width=4)
        label = sector["source"]["patchId"].removeprefix("patch_").replace("_S_80S", "").replace("_80N", "")
        draw.rectangle([x0 + 4, y0 + 4, x0 + 190, y0 + 64], fill=(8, 10, 12))
        draw.text((x0 + 10, y0 + 8), label, fill=(255, 214, 90), font=font)
        draw.text((x0 + 10, y0 + 38), sector["provenance"]["datasetSplit"].upper(), fill=(160, 220, 160), font=small)
        peak, anchor = sector["prediction"]["rawPeak"], sector["derived"]["discoveryAnchor"]
        x, y = margin + placed["col"] + peak["col"] + 0.5, margin + placed["row"] + peak["row"] + 0.5
        draw.line([x - 11, y, x + 11, y], fill=(255, 255, 255), width=3)
        draw.line([x, y - 11, x, y + 11], fill=(255, 255, 255), width=3)
        if anchor:
            x, y = margin + placed["col"] + anchor["col"] + 0.5, margin + placed["row"] + anchor["row"] + 0.5
            draw.ellipse([x - 12, y - 12, x + 12, y + 12], outline=(255, 120, 200), width=4)
    draw.rectangle([margin - 2, margin - 2, margin + w + 1, margin + h + 1], outline=(255, 255, 255), width=2)
    draw.text((margin, 18), f"{mosaic['id']} · {w}×{h} px · {g['pixelSizeXM']} m/px · polar stereographic lat0 "
              f"{p['latitudeOfOriginDeg']} k0 {p['scaleFactor']} · yellow = analyzed patch (row_col, split) · red stripes = "
              f"no-data · gray cells = no processed patch · + = raw model peak · pink ○ = MOONATLAS edge-safe anchor · "
              f"underlay = LROC polar color mosaic", fill=(220, 220, 220), font=small)
    image.save(path, optimize=True)


def _geography_image(patches: dict, basemap: Basemap, orientation: dict, path: Path) -> None:
    tile, label_h, columns = 256, 34, 4
    names = sorted(patches)
    rows = math.ceil(len(names) / columns)
    image = Image.new("RGB", (columns * (3 * tile + 16), rows * (tile + label_h + 16)), (8, 10, 12))
    draw = ImageDraw.Draw(image)
    font = _font(18)
    for i, name in enumerate(names):
        _, raw, covered, _, tmax, transform, to_geo, _ = patches[name]
        x0, y0 = (i % columns) * (3 * tile + 16), (i // columns) * (tile + label_h + 16)
        shade = _patch_basemap(basemap, transform, to_geo)
        lo, hi = np.percentile(shade, [2, 98])
        tmin, tmax_value = np.nanmin(tmax), np.nanmax(tmax)
        panels = [np.clip((shade - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8),
                  np.clip((np.nan_to_num(tmax, nan=tmin) - tmin) / max(tmax_value - tmin, 1e-9) * 255, 0, 255).astype(np.uint8)]
        for k, panel in enumerate(panels):
            image.paste(Image.fromarray(panel).convert("RGB"), (x0 + k * tile, y0 + label_h))
        colored = colorize(raw)
        colored[~covered] = (60, 20, 20)
        image.paste(Image.fromarray(colored), (x0 + 2 * tile, y0 + label_h))
        o = orientation.get(name, {})
        identity = o.get("scores", {}).get("identity", {})
        draw.text((x0 + 4, y0 + 6), f"{name.removeprefix('patch_')}  LROC | TMAX | prediction   r={identity.get('r')} "
                  f"hp={identity.get('rHighPass')} best={o.get('bestHighPass')}", fill=(230, 230, 230), font=font)
    image.save(path, optimize=True)


def _font(size: int):
    for candidate in ("C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _text(report: dict) -> str:
    lines = [f"ICE BATCH REVIEW · {report['build']} · anchor {report['anchor']['method']} "
             f"(edge margin {report['anchor']['edgeMarginPx']} px)", ""]
    for pole, r in report["poles"].items():
        m, d, rt = r["mosaic"], r["distribution"], r["roundTrip"]
        lines += [
            f"══ {pole.upper()} POLE ══",
            f"MOSAIC {m['id']}: {m['widthPx']}×{m['heightPx']} px at {m['pixelSizeM']} m/px ({m['extentKm'][0]} × "
            f"{m['extentKm'][1]} km) · lat0 {m['projection']['latitudeOfOriginDeg']} k0 {m['projection']['scaleFactor']}",
            f"  patches {m['patches']} of {m['gridCells']} grid cells · empty cells {m['emptyCells']} · overlaps {m['overlaps']}",
            f"  covered pixels {m['coveredPixels']:,} ({100 * m['coveredFractionOfGrid']:.2f}% of the grid) · pole at px {m['polePixel']}",
            "  files " + " · ".join(f"{k} {v / 1024:.0f} KB" for k, v in m["files"].items()),
            "",
            "PROJECTION VALIDATION (source GeoTIFF via rasterio/pyproj → published mosaic)",
            f"  {'patch':<22} {'split':<5} {'n':>3} {'pos err px':>11} {'geo err m':>10} {'value err':>10} {'ref err':>10}",
        ]
        for p in r["patches"]:
            lines.append(f"  {p['patch']:<22} {p['split']:<5} {p['samples']:>3} {p['maxPositionErrorPx']:>11.2e} "
                         f"{p['maxGeoErrorM']:>10.2e} {p['maxValueError']:>10.2e} {p['maxReferenceError']:>10.2e}")
        lines += [
            f"  grid round trip: {rt['points']} points, max {rt['maxErrorM']:.2e} m, latitudes {rt['latitudeRange'][0]:.2f}…"
            f"{rt['latitudeRange'][1]:.2f}, longitudes {rt['longitudeRange'][0]:.2f}…{rt['longitudeRange'][1]:.2f}, "
            f"{rt['pointsWithin1DegOfAntimeridian']} within 1° of ±180°",
            "",
            "SEAMS (mean |Δ| across the shared edge vs between the two last interior rows/columns)",
        ]
        for s in r["seams"] or [{"patches": ["none adjacent"], "direction": ""}]:
            if "edgeGapM" not in s:
                lines.append("  no adjacent patches in this batch")
                continue
            parts = " · ".join(f"{k} {s[k]['acrossSeam']:.4g} vs {s[k]['interior']:.4g}" for k in ("reference", "TMAX", "prediction") if s[k])
            lines.append(f"  {s['patches'][0]} → {s['direction']} (edge gap {s['edgeGapM']:.2e} m): {parts}")
        lines += ["", "ORIENTATION vs LROC polar mosaic (TMAX, Pearson r high-pass; identity must win)"]
        for name, o in r["orientation"].items():
            scores = o["scores"]
            others = max(v["rHighPass"] for k, v in scores.items() if k != "identity")
            lines.append(f"  {name:<22} identity r={scores['identity']['r']:+.3f} hp={scores['identity']['rHighPass']:+.3f} · "
                         f"best other hp={others:+.3f} · best={o['bestHighPass']}")
        lines += ["", "EDGE EFFECTS by distance to the valid edge (enrichment = share of values / share of pixels)",
                  f"  {'edge px':<9} {'pixels':>7} {'>1 enrich':>9} {'<0 enrich':>9} {'MAE vs ref':>10}"]
        for ring in r["edgeRings"]:
            lo, hi = ring["edgeDistancePx"]
            lines.append(f"  {lo:>2}-{'∞' if hi is None else hi:<6} {ring['pixelShare']:>7.3f} {ring['above1Enrichment']:>9} "
                         f"{ring['below0Enrichment']:>9} {ring['maeVsReferenceRatio']:>10}")
        lines += ["", "ANCHOR SENSITIVITY (edge-safe maximum vs raw peak)",
                  f"  {'patch':<22} {'raw peak':>8} {'edge':>4} | " + " | ".join(f"m={mg:<2} Δvalue  shift km edge" for mg in ANCHOR_MARGINS[1:])]
        for p in r["patches"]:
            a = p["anchorSensitivity"]
            cells = " | ".join(f"{a[str(mg)]['deltaValue']:>+10.4f} {a[str(mg)]['shiftKm']:>8.2f} {a[str(mg)]['edgeDistancePx']:>4}"
                               if a[str(mg)] else f"{'no interior pixel':>24}" for mg in ANCHOR_MARGINS[1:])
            lines.append(f"  {p['patch']:<22} {a['0']['value']:>8.4f} {a['0']['edgeDistancePx']:>4} | {cells}")
        for mg, s in r["anchorSensitivity"].items():
            lines.append(f"  margin {mg:>2}: moved {s['moved']}/{s['patches']} (no interior pixel: {s['withoutEligiblePixel']}) · "
                         f"max |Δvalue| {s['maxAbsDeltaValue']:.4f} · "
                         f"median shift of moved {s['medianShiftKmOfMoved']:.1f} km · max shift {s['maxShiftKm']:.1f} km")
        lines += [
            "",
            "PREDICTION DISTRIBUTION (raw, covered pixels; descriptive only, not a probability of ice)",
            f"  count {d['count']:,} · min {d['min']:.4f} · max {d['max']:.4f} · mean {d['mean']:.4f} · median {d['median']:.4f}",
            f"  P90 {d['p90']:.4f} · P95 {d['p95']:.4f} · P99 {d['p99']:.4f} · <0: {d['below0']:,} · >1: {d['above1']:,} · "
            f"clipped for display {d['clippedPercent']:.3f}%",
            f"  {'patch':<22} {'min':>8} {'max':>8} {'mean':>8} {'p90':>8} {'<0':>6} {'>1':>6} {'clip%':>7}",
        ]
        for p in r["patches"]:
            lines.append(f"  {p['patch']:<22} {p['min']:>8.4f} {p['max']:>8.4f} {p['mean']:>8.4f} {p['p90']:>8.4f} "
                         f"{p['below0']:>6} {p['above1']:>6} {p['clippedPercent']:>7.3f}")
        lines.append("")
    lines += ["PROBLEMS: " + ("none" if not report["problems"] else ""), *[f"  {x}" for x in report["problems"]], ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
