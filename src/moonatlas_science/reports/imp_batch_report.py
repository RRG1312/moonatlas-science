"""Review an IMP segmentation batch build: per-tile results, aggregate IoU/F1, geography and diagnostics.

Engineering validation only. meanSoftmaxScore is an UNCALIBRATED score, never a probability.
  1. Per tile: argmax pixels and area, tile share, mean class-1 softmax (uncalibrated), IoU vs the Hargitai et al. (2025)
     reference over annotated pixels (mask ≠ 255), centroid inside the true footprint.
  2. Aggregate class-1 IoU and F1 over all annotated pixels of the batch (one confusion matrix, as a dataset-level test
     metric accumulates); the published model-card figures remain the only performance claims.
  3. Geography: distinct footprints (several NAC observations can share one tile grid), the published IMP sites
     (imp_groups.py: repeat observations of one SomBench target) with their spans, and a
     diagnostic with site locators on the LROC mosaic plus every tile's NAC image with prediction and reference outlines.
Outputs: data/artifacts/sanity/<build>/imp/{report.txt, report.json, geography.png}.

Usage: MOONATLAS_BUILD_DIR=data/build/science-0.2.0-heldout-test python -m moonatlas_science.reports.imp_batch_report
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

from ..ice_batch_report import Basemap, _font
from ..imp_groups import group_observations, span_m

BASEMAP = config.DATA_ROOT / "raw" / "moonkit" / "lroc_color_poles_8k.tif"


def load_json(name: str):
    return json.loads((config.BUILD_DIR / name).read_text(encoding="utf-8"))


def sample_of(region: dict) -> str:
    return region["provenance"]["sourceTile"].removesuffix(".tif").removesuffix("_img")


def inside(point: dict, footprint: list[dict]) -> bool:
    lats = [c["latitude"] for c in footprint]
    lons = [c["longitude"] for c in footprint]
    return min(lats) <= point["latitude"] <= max(lats) and min(lons) <= point["longitude"] <= max(lons)


def main() -> int:
    regions = sorted(load_json("imp-regions.json"), key=lambda r: r["id"])
    out = config.SANITY_DIR / config.BUILD_DIR.name / "imp"
    out.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    rows = []
    confusion = {split: [0, 0, 0] for split in ("train", "validation", "test")}  # tp, fp, fn per split
    for region in regions:
        sample = sample_of(region)
        _, arrays = artifacts.read_run("imp-segmentation", sample)
        with rasterio.open(config.IMP_DIR / "all" / f"{sample}_mask.tif") as src:
            reference = src.read(1)
        predicted = arrays["mask"].astype(bool)
        annotated = reference != 255
        truth = (reference == 1) & annotated
        t, f_p, f_n = int((predicted & truth).sum()), int((predicted & ~truth & annotated).sum()), int((~predicted & truth).sum())
        counts = confusion[region["provenance"]["datasetSplit"]]
        counts[0], counts[1], counts[2] = counts[0] + t, counts[1] + f_p, counts[2] + f_n
        p = region["prediction"]
        if int(predicted.sum()) != p["pixelCount"]:
            problems.append(f"{sample}: published pixel count differs from the argmax mask")
        if p["centroid"] and not inside(p["centroid"], region["source"]["footprint"]):
            problems.append(f"{sample}: centroid outside the true footprint")
        g = region["source"]["grid"]
        rows.append({
            "id": region["id"], "sample": sample, "split": region["provenance"]["datasetSplit"],
            "nacProductId": region["source"]["nacProductId"],
            "center": [region["latitude"], region["longitude"]], "footprintKey": (round(g["xMinM"], 1), round(g["yMaxM"], 1),
                                                                                    region["source"]["projection"]["centralMeridianDeg"]),
            "sizeM": region["source"]["sizeM"], "resolutionM": region["source"]["resolutionM"],
            "pixelCount": p["pixelCount"], "areaM2": p["areaM2"], "tileShare": p["pixelFraction"],
            "meanSoftmaxScoreUncalibrated": p["meanSoftmaxScore"], "iou": region["derived"]["iou"],
            "referencePixels": region["reference"]["pixelCount"], "annotatedPixels": int(annotated.sum()),
        })

    footprints: dict = {}
    for r in rows:
        footprints.setdefault(str(r["footprintKey"]), []).append(r["id"])
    by_id = {r["id"]: r for r in regions}
    report = {
        "build": config.BUILD_DIR.name,
        "tiles": len(rows),
        "distinctFootprints": len(footprints),
        "sharedFootprints": {k: v for k, v in footprints.items() if len(v) > 1},
        "groups": [{"center": [round(g["latitude"], 4), round(g["longitude"], 4)], "target": g["sourceTarget"],
                    "tiles": g["observationIds"], "spanM": round(span_m([by_id[o] for o in g["observationIds"]]), 1)}
                   for g in group_observations(regions)],
        "aggregateBySplit": {split: {"tiles": sum(1 for r in rows if r["split"] == split),
                                     "iouClass1": round(tp / (tp + fp + fn), 4), "f1Class1": round(2 * tp / (2 * tp + fp + fn), 4)}
                             for split, (tp, fp, fn) in confusion.items() if tp + fp + fn},
        "aggregateNote": "annotated pixels per split; test = held-out descriptive check, train/validation in-sample (never evidence)",
        "modelCardTestMetrics": {"iouClass1": 0.5709, "f1Class1": 0.7268},
        "tiles_": [{k: v for k, v in r.items() if k != "footprintKey"} for r in rows],
        "problems": problems,
    }
    _geography(regions, report, out / "geography.png")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    (out / "report.txt").write_text(_text(report), encoding="utf-8", newline="\n")
    print(_text(report))
    print(f"written to {out}")
    return 1 if problems else 0


def _geography(regions: list[dict], report: dict, path: Path) -> None:
    basemap = Basemap(BASEMAP)
    font, small = _font(18), _font(14)
    # Panel 1: nearside overview (60°S–60°N, 90°W–90°E) with site locators.
    w, h = 1200, 800
    lats = np.linspace(60, -60, h)[:, None].repeat(w, 1)
    lons = np.linspace(-90, 90, w)[None, :].repeat(h, 0)
    overview = Image.fromarray(np.clip(basemap.sample(lats, lons), 0, 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(overview)
    for i, site in enumerate(report["groups"], start=1):
        lat, lon = site["center"]
        x, y = (lon + 90) / 180 * w, (60 - lat) / 120 * h
        draw.ellipse([x - 14, y - 14, x + 14, y + 14], outline=(255, 165, 61), width=3)
        if len(site["tiles"]) < 2:
            continue  # label only multi-observation groups: one label per group would overlap
        draw.text((x + 18, y - 22), f"site {i}: {abs(lat):.3f}°{'N' if lat >= 0 else 'S'} {abs(lon):.3f}°{'E' if lon >= 0 else 'W'} · "
                  f"{len(site['tiles'])} tiles within {site['spanM'] / 1000:.1f} km", fill=(255, 200, 120), font=font)
    draw.text((10, 8), "IMP held-out test tiles on the LROC mosaic (nearside, 60°S–60°N) · locators are enlarged UI marks; "
              "each true footprint is ~256 m", fill=(235, 235, 235), font=font)

    # Panel 2: every tile, NAC image with prediction (amber) and reference (white) outlines.
    tile, cols = 256, 5
    grid = Image.new("RGB", (cols * (tile + 10), math.ceil(len(regions) / cols) * (tile + 58)), (8, 10, 12))
    gdraw = ImageDraw.Draw(grid)
    by_id = {r["id"]: r for r in report["tiles_"]}
    for i, region in enumerate(regions):
        sample = sample_of(region)
        with rasterio.open(config.IMP_DIR / "all" / f"{sample}_img.tif") as src:
            image = src.read(1).astype(np.float64)
        with rasterio.open(config.IMP_DIR / "all" / f"{sample}_mask.tif") as src:
            reference = src.read(1)
        _, arrays = artifacts.read_run("imp-segmentation", sample)
        lo, hi = np.nanpercentile(image, [1, 99])
        rgb = np.dstack([np.clip((np.nan_to_num(image) - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)] * 3)
        for mask, color in ((arrays["mask"].astype(bool), (255, 165, 61)), (reference == 1, (255, 255, 255))):
            edge = mask & ~(np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1))
            rgb[edge] = color
        x0, y0 = (i % cols) * (tile + 10), (i // cols) * (tile + 58)
        grid.paste(Image.fromarray(rgb), (x0, y0 + 50))
        r = by_id[region["id"]]
        gdraw.text((x0 + 2, y0 + 2), f"{region['id'][4:]}", fill=(255, 200, 120), font=small)
        gdraw.text((x0 + 2, y0 + 18), f"{r['center'][0]:.4f}, {r['center'][1]:.4f} · IoU {r['iou']}", fill=(220, 220, 220), font=small)
        gdraw.text((x0 + 2, y0 + 34), f"{r['areaM2']:,.0f} m² · softmax {r['meanSoftmaxScoreUncalibrated']} (uncalibrated)",
                   fill=(180, 180, 180), font=small)

    figure = Image.new("RGB", (max(w, grid.width), h + grid.height + 20), (8, 10, 12))
    figure.paste(overview, (0, 0))
    figure.paste(grid, (0, h + 20))
    figure.save(path, optimize=True)


def _text(r: dict) -> str:
    lines = [
        f"IMP BATCH REVIEW · {r['build']} · {r['tiles']} tiles · {r['distinctFootprints']} distinct footprints",
        f"  shared footprints (same tile grid, different NAC observations): {r['sharedFootprints'] or 'none'}",
        f"  observation groups (SomBench target): {len(r['groups'])} · imaged more than once "
        f"{sum(1 for g in r['groups'] if len(g['tiles']) > 1)} · largest {max(len(g['tiles']) for g in r['groups'])} "
        f"observations · max span {max(g['spanM'] for g in r['groups']):.0f} m",
        "  repeat targets: " + " · ".join(f"t{g['target']} {g['center']} ({len(g['tiles'])} observations)"
                                          for g in r["groups"] if len(g["tiles"]) > 1),
        f"AGGREGATE class-1 per split (model card test IoU₁ {r['modelCardTestMetrics']['iouClass1']}, F1₁ "
        f"{r['modelCardTestMetrics']['f1Class1']}):",
        *[f"  {split:<10} {a['tiles']:>3} tiles · IoU {a['iouClass1']} · F1 {a['f1Class1']}"
          + ("   held-out descriptive check" if split == "test" else "   IN-SAMPLE: not evidence of performance")
          for split, a in r["aggregateBySplit"].items()],
        "",
        f"  {'id':<32} {'center':>21} {'size m':>7} {'pixels':>7} {'area m²':>9} {'share':>6} {'softmax*':>8} {'IoU':>6} {'ref px':>6}",
    ]
    for t in r["tiles_"]:
        lines.append(f"  {t['id']:<32} {t['center'][0]:>10.5f},{t['center'][1]:>10.5f} {t['sizeM']:>7.1f} {t['pixelCount']:>7} "
                     f"{t['areaM2']:>9,.1f} {t['tileShare']:>6.3f} {t['meanSoftmaxScoreUncalibrated']:>8} {t['iou']:>6} {t['referencePixels']:>6}")
    lines += ["  * mean class-1 softmax over predicted pixels: UNCALIBRATED, not a probability", "",
              "PROBLEMS: " + ("none" if not r["problems"] else ""), *[f"  {p}" for p in r["problems"]], ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
