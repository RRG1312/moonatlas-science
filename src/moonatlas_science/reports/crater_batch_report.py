"""Review a WAC crater batch build: counts, distributions, upstream-configured mAP, geolocation, polar tiles, diagnostics.

Engineering validation only (never marketing figures):
  1. Detections from the raw artifacts (after the upstream score ≥ 0.05 + NMS postprocessing) and at the 0.5 deployment
     threshold; confidence and diameter distributions; reference matches and unmatched predictions.
  2. mAP / AP50 / AP75 recomputed over the whole batch with the upstream test configuration: the stored
     predict_step outputs go through the same apply_nms_batch as test_step, into torchmetrics MeanAveragePrecision
     (bbox, average="macro", max_detection_thresholds=[100, 300, 500], from WAC_config.yaml metric_kwargs).
  3. Geolocation: every published detection is recomputed from its tile-detail box through the published projection
     and grid (moonatlas_science/projection.py, independent of geo.py), and must sit inside the tile footprint.
  4. Orientation of each tile against the LROC color mosaic (WAC 643 nm vs basemap under the 8 dihedral transforms).
  5. Polar tiles (polar stereographic grids) listed explicitly, with a diagnostic figure per polar tile.
Outputs: data/artifacts/sanity/<build>/craters/{report.txt, report.json, coverage-map.png, polar-<tile>.png}.

Usage: MOONATLAS_BUILD_DIR=data/build/science-0.2.0-heldout-test python -m moonatlas_science.reports.crater_batch_report
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw

from moonatlas_science import artifacts, config

from .. import projection as proj
from ..ice_batch_report import DIHEDRAL, Basemap, _font, block_mean, correlation, great_circle_m, high_pass
from . import globe_backdrop

BASEMAP = config.DATA_ROOT / "raw" / "moonkit" / "lroc_color_poles_8k.tif"
# Detail boxes are rounded to 0.01 px (≤ 1.4 m at 100 m/px) and coordinates to 6 decimals (≤ 3 cm).
GEO_TOLERANCE_M = 2.0
DIAMETER_BINS_KM = [0, 1, 2, 3, 5, 10, 20, 50]


def load_json(name: str):
    return json.loads((config.BUILD_DIR / name).read_text(encoding="utf-8"))


def sample_of(region: dict) -> str:
    return region["provenance"]["sourceTile"].removesuffix(".tif")


def main() -> int:
    import torch
    from torchmetrics.detection import MeanAveragePrecision

    regions = load_json("crater-regions.json")
    craters = load_json("craters.json")
    out = config.SANITY_DIR / config.BUILD_DIR.name / "craters"
    out.mkdir(parents=True, exist_ok=True)
    basemap = Basemap(BASEMAP) if BASEMAP.exists() else None
    problems: list[str] = []
    # One metric per split: train/validation outputs are in-sample and never evaluation evidence.
    metrics = {split: MeanAveragePrecision(iou_type="bbox", average="macro", max_detection_thresholds=[100, 300, 500])
               for split in ("train", "validation", "test")}
    rows = []
    confidences, diameters, truncated = [], [], 0
    totals = {"afterUpstreamPostprocessing": 0, "atDeploymentThreshold": 0, "referenceBoxes": 0, "matched": 0}

    for region in sorted(regions, key=lambda r: r["id"]):
        sample = sample_of(region)
        _, arrays = artifacts.read_run("crater-detection", sample)
        detail = load_json(region["detailPath"])
        tuples = craters[region["id"]]
        threshold = region["prediction"]["confidenceThreshold"]
        metrics[region["provenance"]["datasetSplit"]].update(
            [{"boxes": torch.from_numpy(arrays["boxes_xyxy"]), "scores": torch.from_numpy(arrays["scores"]),
              "labels": torch.from_numpy(arrays["labels"])}],
            [{"boxes": torch.from_numpy(arrays["reference_boxes_xyxy"]),
              "labels": torch.ones(len(arrays["reference_boxes_xyxy"]), dtype=torch.long)}],
        )
        kept = int((arrays["scores"] >= threshold).sum())
        if kept != len(tuples) or kept != region["prediction"]["detectionCount"]:
            problems.append(f"{sample}: {kept} artifact detections ≥ {threshold} but {len(tuples)} published")
        p, g = region["source"]["projection"], region["source"]["grid"]
        worst = 0.0
        for (x, y, w, h, _), t in zip(detail["prediction"], tuples, strict=True):
            lat, lon = proj.pixel_to_latlon(p, g, x + w / 2, y + h / 2)
            worst = max(worst, float(great_circle_m(lat, lon, t[0], t[1])))
        if worst > GEO_TOLERANCE_M:
            problems.append(f"{sample}: detection coordinates differ from projection+grid by {worst:.2f} m")
        lats = [c["latitude"] for c in region["source"]["footprint"]]
        confidences += [t[3] for t in tuples]
        diameters += [t[2] for t in tuples]
        truncated += sum(t[4] for t in tuples)
        totals["afterUpstreamPostprocessing"] += len(arrays["scores"])
        totals["atDeploymentThreshold"] += kept
        totals["referenceBoxes"] += region["reference"]["craterCount"]
        totals["matched"] += region["derived"]["matchedPredictions"]
        orientation = _orientation(region, sample, basemap) if basemap else None
        rows.append({
            "regionId": region["id"], "sample": sample, "split": region["provenance"]["datasetSplit"],
            "projection": p["kind"], "centralMeridianDeg": p["centralMeridianDeg"],
            "latitudeOfOriginDeg": p["latitudeOfOriginDeg"], "center": [region["latitude"], region["longitude"]],
            "latRange": [min(lats), max(lats)], "detections": kept, "afterUpstreamPostprocessing": int(len(arrays["scores"])),
            "reference": region["reference"]["craterCount"], "matched": region["derived"]["matchedPredictions"],
            "maxCoordinateErrorM": worst, "orientation": orientation,
        })

    per_split = {}
    for split, metric in metrics.items():
        if any(r["split"] == split for r in rows):
            computed = metric.compute()
            per_split[split] = {k: round(float(computed[k]), 4) for k in ("map", "map_50", "map_75")}
    confidences, diameters = np.array(confidences), np.array(diameters)
    polar = [r for r in rows if r["projection"] == "polar-stereographic"]
    wins = [r for r in rows if r["orientation"] and r["orientation"]["best"] == "identity"]
    report = {
        "build": config.BUILD_DIR.name,
        "tiles": len(rows),
        "splits": sorted({r["split"] for r in rows}),
        "detections": totals | {"unmatchedAtDeploymentThreshold": totals["atDeploymentThreshold"] - totals["matched"],
                                "truncated": truncated},
        "metricsUpstreamConfigurationBySplit": per_split,
        "metricsNote": "test = held-out engineering check; train/validation are in-sample and never evidence of performance",
        "modelCardTestMetrics": {"map": 0.2581, "map_50": 0.6183},
        "confidence": {"quantiles": {q: round(float(np.quantile(confidences, q)), 4) for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)},
                       "histogram": _histogram(confidences, np.arange(0.5, 1.0001, 0.05))},
        "diameterKm": {"quantiles": {q: round(float(np.quantile(diameters, q)), 3) for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)},
                       "histogram": _histogram(diameters, DIAMETER_BINS_KM + [max(51.3, float(diameters.max()) + 0.01)])},
        "coverage": {
            "latitudeRange": [min(r["latRange"][0] for r in rows), max(r["latRange"][1] for r in rows)],
            "northern": sum(1 for r in rows if r["center"][0] >= 0), "southern": sum(1 for r in rows if r["center"][0] < 0),
            "nearside": sum(1 for r in rows if abs(r["center"][1]) <= 90), "farside": sum(1 for r in rows if abs(r["center"][1]) > 90),
            "polarTiles": [r["sample"] for r in polar],
        },
        "orientation": {"tilesChecked": sum(1 for r in rows if r["orientation"]), "identityBest": len(wins),
                        "conclusive": sum(1 for r in rows if r["orientation"] and max(r["orientation"]["identity"], r["orientation"]["bestOther"]) >= 0.1),
                        "conclusiveContradictions": sum(1 for r in rows if r["orientation"] and r["orientation"]["best"] != "identity"
                                                        and max(r["orientation"]["identity"], r["orientation"]["bestOther"]) >= 0.1),
                        "polarIdentityBest": sum(1 for r in polar if r["orientation"] and r["orientation"]["best"] == "identity")},
        "bySplit": {split: {"tiles": sum(1 for r in rows if r["split"] == split),
                            "detections": sum(r["detections"] for r in rows if r["split"] == split)}
                    for split in ("train", "validation", "test")},
        "scoreAnalysis": _score_analysis(craters),
        "crossTileRepeats": _cross_tile_repeats(craters),
        "tiles_": rows,
        "problems": problems,
    }
    _coverage_map(regions, craters, out / "coverage-map.png")
    for r in polar:
        _polar_figure(next(x for x in regions if x["id"] == r["regionId"]), craters, basemap, out / f"polar-{r['sample']}.png")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    (out / "report.txt").write_text(_text(report), encoding="utf-8", newline="\n")
    print(_text(report))
    print(f"written to {out}")
    return 1 if problems else 0


def _score_analysis(craters: dict) -> dict:
    """crater-diameter-rank-v1 over the processed catalogue, and what discovery curation does to it."""
    import bisect

    diameters = sorted(t[2] for tuples in craters.values() for t in tuples if not t[4])
    n = len(diameters)
    rank = lambda d: 100 * (bisect.bisect_left(diameters, d) + 0.5 * (bisect.bisect_right(diameters, d) - bisect.bisect_left(diameters, d))) / n  # noqa: E731
    discoveries = [d for d in load_json("discoveries.json") if d["type"] == "CRATER_CANDIDATE"]
    curated = [rank(next(m["value"] for m in d["metrics"] if m["key"] == "diameter")) for d in discoveries]
    quantile = lambda q: round(float(np.quantile(diameters, q)), 3)  # noqa: E731
    return {
        "population": n,
        "diameterKmAtPercentile": {"p50": quantile(0.5), "p90": quantile(0.9), "p99": quantile(0.99), "p999": quantile(0.999)},
        "detectionsAtOrAbove99": sum(1 for d in diameters if d >= quantile(0.99)),
        "curatedDiscoveries": len(curated),
        "curatedPercentileRange": [round(min(curated), 2), round(max(curated), 2)] if curated else None,
        "curatedDiameterKmRange": [min(m["value"] for d in discoveries for m in d["metrics"] if m["key"] == "diameter"),
                                   max(m["value"] for d in discoveries for m in d["metrics"] if m["key"] == "diameter")] if discoveries else None,
    }


def _cross_tile_repeats(craters: dict) -> dict:
    """Detections of (likely) the same crater in overlapping tiles: other-tile centers within max(1 km, 25% of the
    diameter) and diameters within 25%. An estimate of repeated observation, not a deduplication."""
    from collections import defaultdict

    cell = 0.25
    buckets = defaultdict(list)
    entries = [(region_id, t[0], t[1], t[2]) for region_id, tuples in craters.items() for t in tuples]
    for i, (_, lat, lon, _) in enumerate(entries):
        buckets[(int(math.floor(lat / cell)), int(math.floor(lon / cell)))].append(i)
    repeated = 0
    for i, (region_id, lat, lon, diameter) in enumerate(entries):
        key = (int(math.floor(lat / cell)), int(math.floor(lon / cell)))
        tolerance_m = max(1000.0, 250.0 * diameter)
        for dy in (-1, 0, 1):
            found = False
            for dx in (-1, 0, 1):
                for j in buckets.get((key[0] + dy, key[1] + dx), ()):
                    other_region, olat, olon, odiameter = entries[j]
                    if other_region == region_id or abs(odiameter - diameter) > 0.25 * max(diameter, odiameter):
                        continue
                    if float(great_circle_m(lat, lon, olat, olon)) <= tolerance_m:
                        found = True
                        break
                if found:
                    break
            if found:
                repeated += 1
                break
    return {"detections": len(entries), "withLikelyRepeatInAnotherTile": repeated,
            "share": round(repeated / max(1, len(entries)), 4)}


def _histogram(values: np.ndarray, edges) -> list[dict]:
    counts, edges = np.histogram(values, bins=edges)
    return [{"from": round(float(a), 3), "to": round(float(b), 3), "count": int(c)} for a, b, c in zip(edges[:-1], edges[1:], counts)]


def _tile_grid_latlon(region: dict) -> tuple[np.ndarray, np.ndarray]:
    p, g = region["source"]["projection"], region["source"]["grid"]
    step = 4
    centers = np.arange(step / 2, g["widthPx"], step)
    lat = np.empty((len(centers), len(centers)))
    lon = np.empty_like(lat)
    for i, row in enumerate(centers):
        for j, col in enumerate(centers):
            lat[i, j], lon[i, j] = proj.pixel_to_latlon(p, g, col, row)
    return lat, lon


def _orientation(region: dict, sample: str, basemap: Basemap) -> dict:
    """Pearson r (high-pass) of the 643 nm band vs the LROC color mosaic, 4 px blocks, 8 dihedral transforms."""
    with rasterio.open(config.WAC_DIR / "images_tiff" / f"{sample}.tif") as src:
        band = np.nan_to_num(src.read(4).astype(np.float64))
    lat, lon = _tile_grid_latlon(region)
    image = basemap.sample(lat, lon)
    wac = block_mean(band, 4)
    scores = {k: round(correlation(high_pass(np.ascontiguousarray(f(wac))), high_pass(image)), 3) for k, f in DIHEDRAL.items()}
    best = max(scores, key=scores.get)
    return {"identity": scores["identity"], "bestOther": max(v for k, v in scores.items() if k != "identity"), "best": best}


def _coverage_map(regions: list[dict], craters: dict, path: Path) -> None:
    image = globe_backdrop(2048, 0.55)
    draw = ImageDraw.Draw(image)
    font = _font(16)

    def xy(lat, lon):
        return (lon + 180) / 360 * image.width, (90 - lat) / 180 * image.height

    for region in regions:
        polar = region["source"]["projection"]["kind"] == "polar-stereographic"
        corners = [xy(c["latitude"], c["longitude"]) for c in region["source"]["footprint"]]
        if max(x for x, _ in corners) - min(x for x, _ in corners) > image.width / 2:  # footprint across ±180°
            continue
        draw.polygon(corners, outline=(255, 120, 200) if polar else (90, 220, 255), width=3 if polar else 2)
        for t in craters[region["id"]]:
            x, y = xy(t[0], t[1])
            draw.point((x, y), fill=(255, 230, 120))
    draw.text((10, 8), f"WAC held-out test tiles ({len(regions)}) · cyan = transverse Mercator tiles · pink = polar stereographic "
              f"tiles · yellow = predicted crater centers (confidence ≥ 0.5) · equirectangular, 0° at center", fill=(235, 235, 235),
              font=font)
    image.save(path, optimize=True)


def _polar_figure(region: dict, craters: dict, basemap: Basemap | None, path: Path) -> None:
    """Tile with predictions and references | LROC mosaic resampled into the tile grid | position on a polar map."""
    sample = sample_of(region)
    detail = load_json(region["detailPath"])
    p, g = region["source"]["projection"], region["source"]["grid"]
    size = 512
    with rasterio.open(config.WAC_DIR / "images_tiff" / f"{sample}.tif") as src:
        band = np.nan_to_num(src.read(4).astype(np.float64))
    lo, hi = np.percentile(band, [1, 99])
    tile = Image.fromarray(np.clip((band - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(tile)
    for x, y, w, h in detail["reference"]:
        draw.rectangle([x, y, x + w, y + h], outline=(255, 255, 255), width=1)
    for x, y, w, h, _ in detail["prediction"]:
        draw.ellipse([x, y, x + w, y + h], outline=(90, 220, 255), width=2)
    # North arrow from the projection: direction of increasing latitude at the tile center.
    lat0, lon0 = proj.pixel_to_latlon(p, g, 256, 256)
    xN, yN = proj.forward(p, min(lat0 + 0.05, 89.999), lon0)
    xC, yC = proj.forward(p, lat0, lon0)
    ang = math.atan2(-(yN - yC), xN - xC)
    draw.line([40, 60, 40 + 30 * math.cos(ang), 60 + 30 * math.sin(ang)], fill=(255, 90, 90), width=4)
    draw.text((46 + 30 * math.cos(ang), 50 + 30 * math.sin(ang)), "N", fill=(255, 90, 90), font=_font(20))

    panels = [tile]
    if basemap:
        lat, lon = _tile_grid_latlon(region)
        shade = basemap.sample(lat, lon)
        slo, shi = np.percentile(shade, [1, 99])
        panels.append(Image.fromarray(np.clip((shade - slo) / max(shi - slo, 1e-9) * 255, 0, 255).astype(np.uint8))
                      .resize((size, size), Image.NEAREST).convert("RGB"))
    # Polar context map: tile footprint and detections on a polar stereographic plane around the pole.
    south = p["latitudeOfOriginDeg"] < 0
    polar_p = proj.polar_stereographic("south" if south else "north")
    context = Image.new("RGB", (size, size), (14, 16, 18))
    cdraw = ImageDraw.Draw(context)
    tile_x, tile_y = proj.forward(polar_p, region["latitude"], region["longitude"])
    extent = max(1_000_000.0, 1.3 * math.hypot(tile_x - polar_p["falseEastingM"], tile_y - polar_p["falseNorthingM"]))

    def cxy(lat, lon):
        x, y = proj.forward(polar_p, lat, lon)
        return (x - polar_p["falseEastingM"] + extent) / (2 * extent) * size, (extent - (y - polar_p["falseNorthingM"])) / (2 * extent) * size

    sign = -1 if south else 1
    for parallel in range(30, 90, 5):
        cdraw.line([cxy(sign * parallel, lon) for lon in range(-180, 181, 3)], fill=(60, 80, 90), width=1)
    for meridian in range(-180, 180, 30):
        cdraw.line([cxy(sign * 89.9, meridian), cxy(sign * 30, meridian)], fill=(45, 60, 70), width=1)
        x, y = cxy(sign * (90 - 0.8 * (90 - abs(region['latitude']))), meridian)
        cdraw.text((x - 10, y - 6), f"{meridian}", fill=(120, 150, 160), font=_font(12))
    cdraw.polygon([cxy(c["latitude"], c["longitude"]) for c in region["source"]["footprint"]], outline=(255, 120, 200), width=2)
    for t in craters[region["id"]]:
        x, y = cxy(t[0], t[1])
        cdraw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(255, 230, 120))
    x, y = cxy(sign * 90, 0)
    cdraw.text((x + 4, y), "pole", fill=(255, 255, 255), font=_font(14))
    panels.append(context)

    figure = Image.new("RGB", (len(panels) * (size + 12), size + 44), (8, 10, 12))
    for i, panel in enumerate(panels):
        figure.paste(panel, (i * (size + 12), 40))
    ImageDraw.Draw(figure).text(
        (6, 8), f"{sample} · {p['kind']} lat0 {p['latitudeOfOriginDeg']} k0 {p['scaleFactor']} · center "
        f"{region['latitude']:.3f}, {region['longitude']:.3f} · 643 nm with predictions (cyan) and Robbins references (white) | "
        f"LROC mosaic resampled into the tile grid | tile and detections on the polar map", fill=(230, 230, 230), font=_font(15))
    figure.save(path, optimize=True)


def _text(r: dict) -> str:
    d, c = r["detections"], r["coverage"]
    lines = [
        f"CRATER BATCH REVIEW · {r['build']} · {r['tiles']} tiles · splits {r['splits']}",
        "",
        f"DETECTIONS after upstream postprocessing (score ≥ 0.05 + NMS): {d['afterUpstreamPostprocessing']:,}",
        f"  at deployment threshold ≥ 0.5: {d['atDeploymentThreshold']:,} · truncated at tile edge: {d['truncated']:,}",
        f"  Robbins reference boxes (upstream-filtered): {d['referenceBoxes']:,} · matched (IoU ≥ 0.5, one-to-one): {d['matched']:,}"
        f" · unmatched predictions: {d['unmatchedAtDeploymentThreshold']:,}",
        "",
        "mAP with the upstream test configuration, per split (model card test: mAP "
        f"{r['modelCardTestMetrics']['map']} · AP50 {r['modelCardTestMetrics']['map_50']}):",
        *[f"  {split:<10} mAP {m['map']} · AP50 {m['map_50']} · AP75 {m['map_75']}"
          + ("   held-out engineering check" if split == "test" else "   IN-SAMPLE: not evidence of performance")
          for split, m in r["metricsUpstreamConfigurationBySplit"].items()],
        "",
        "CONFIDENCE (≥ 0.5) quantiles: " + " · ".join(f"q{int(float(q) * 100)} {v}" for q, v in r["confidence"]["quantiles"].items()),
        "  " + " · ".join(f"{h['from']:.2f}–{h['to']:.2f}: {h['count']}" for h in r["confidence"]["histogram"]),
        "DIAMETER km quantiles: " + " · ".join(f"q{int(float(q) * 100)} {v}" for q, v in r["diameterKm"]["quantiles"].items()),
        "  " + " · ".join(f"{h['from']:g}–{h['to']:g}: {h['count']}" for h in r["diameterKm"]["histogram"]),
        "",
        f"COVERAGE latitudes {c['latitudeRange'][0]:.2f}…{c['latitudeRange'][1]:.2f} · north {c['northern']} / south {c['southern']}"
        f" · nearside {c['nearside']} / farside {c['farside']}",
        f"  polar stereographic tiles: {len(c['polarTiles'])} {c['polarTiles']}",
        f"ORIENTATION vs LROC mosaic: identity best for {r['orientation']['identityBest']} of {r['orientation']['tilesChecked']} tiles "
        f"(polar: {r['orientation']['polarIdentityBest']} of {len(c['polarTiles'])}); conclusive (max r ≥ 0.1) "
        f"{r['orientation']['conclusive']}, contradictions {r['orientation']['conclusiveContradictions']}",
        "",
        f"  {'tile':<30} {'proj':<5} {'center':>20} {'det':>4} {'ref':>4} {'match':>5} {'coord err m':>11} {'orient id':>9} {'other':>6}",
    ]
    for t in r["tiles_"]:
        o = t["orientation"] or {}
        lines.append(f"  {t['sample']:<30} {'PS' if t['projection'] == 'polar-stereographic' else 'TM':<5} "
                     f"{t['center'][0]:>9.3f},{t['center'][1]:>9.3f} {t['detections']:>4} {t['reference']:>4} {t['matched']:>5} "
                     f"{t['maxCoordinateErrorM']:>11.2f} {o.get('identity', float('nan')):>9.3f} {o.get('bestOther', float('nan')):>6.3f}")
    sa, ct = r["scoreAnalysis"], r["crossTileRepeats"]
    lines[1:1] = [
        "BY SPLIT: " + " · ".join(f"{k} {v['tiles']} tiles / {v['detections']:,} detections" for k, v in r["bySplit"].items()),
        f"INTEREST SCORE (crater-diameter-rank-v1) over {sa['population']:,} non-truncated detections: diameter at P50 "
        f"{sa['diameterKmAtPercentile']['p50']} km · P90 {sa['diameterKmAtPercentile']['p90']} · P99 {sa['diameterKmAtPercentile']['p99']} · "
        f"P99.9 {sa['diameterKmAtPercentile']['p999']}; {sa['curatedDiscoveries']} curated discoveries span percentiles "
        f"{sa['curatedPercentileRange']} (diameters {sa['curatedDiameterKmRange']} km)",
        f"CROSS-TILE REPEATS (estimate): {ct['withLikelyRepeatInAnotherTile']:,} of {ct['detections']:,} detections "
        f"({100 * ct['share']:.1f}%) have a similar detection in another, overlapping tile",
    ]
    lines += ["", "PROBLEMS: " + ("none" if not r["problems"] else ""), *[f"  {p}" for p in r["problems"]], ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
