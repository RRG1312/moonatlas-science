"""Dataset coverage inventory across train / validation / test for ICE, WAC craters and IMP. No inference.

Geography comes from source metadata and source rasters only:
  ICE    every patch listed in the Hugging Face repo at the pinned revision; its PRO GeoTIFF (projection, grid, valid
         pixels) is downloaded to place it in the 9×9 polar grid and to measure its valid coverage.
  WAC    metadata.parquet (exact tile projection recovered per tile, footprint from the projected bounds).
  IMP    every image GeoTIFF (projection, footprint); tiles are also clustered by proximity (a coverage view, not
         the published identity: imp_groups.py groups observations by upstream SomBench target).
Outputs: data/artifacts/coverage/{coverage-report.json, coverage-report.txt, ice-north-all-splits.png,
ice-south-all-splits.png, craters-all-splits-global.png, imp-all-splits-global.png}.

Usage: python -m moonatlas_science.reports.coverage_inventory
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw

from moonatlas_science import config, hub
from moonatlas_science.geo import raster_tile, wac_tile_projection

from ..ice_batch_report import _font, great_circle_m
from . import globe_backdrop

OUT = config.ARTIFACTS_DIR / "coverage"
BASEMAP = config.DATA_ROOT / "raw" / "moonkit" / "lroc_color_poles_8k.tif"
SPLITS = ("train", "validation", "test")
COLORS = {"train": (90, 170, 255), "validation": (255, 190, 70), "test": (120, 230, 140), "unlisted": (200, 90, 200),
          "missing": (70, 70, 70)}
COVERAGE_CELL_DEG = 0.5
WAC_OVERLAP_KM = 25.6  # half a tile: centers closer than this share most of their footprint
WAC_AREA_KM = 300.0    # tiles within this distance form one geographic area (e.g. one image strip)
IMP_SITE_KM = 5.0
# Published per-sample storage measured on the held-out build (science-0.3.0-heldout-test), KB.
STORAGE_KB = {"ice-patch": 287, "ice-mosaic-per-patch": 273, "wac-tile": 74, "imp-tile": 26}


def lunar_area_km2() -> float:
    return 4 * math.pi * (config.LUNAR_RADIUS_M / 1000) ** 2


def cell_area_km2(lat_index: np.ndarray) -> np.ndarray:
    r = config.LUNAR_RADIUS_M / 1000
    d = math.radians(COVERAGE_CELL_DEG)
    lat0 = np.radians(-90 + lat_index * COVERAGE_CELL_DEG)
    return r * r * d * (np.sin(lat0 + d) - np.sin(lat0))


def union_find(n: int, pairs) -> list[int]:
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return [find(i) for i in range(n)]


def close_pairs(points: list[tuple[float, float]], km: float):
    lat = np.array([p[0] for p in points])
    lon = np.array([p[1] for p in points])
    for i in range(len(points)):
        d = great_circle_m(lat[i], lon[i], lat[i + 1:], lon[i + 1:]) / 1000
        for j in np.nonzero(d < km)[0]:
            yield i, i + 1 + int(j)


# ─── ICE ──────────────────────────────────────────────────────────────────────


def ice_inventory() -> dict:
    from huggingface_hub import list_repo_files

    repo = config.ICE_DATASET_REPO
    files = list_repo_files(repo, repo_type="dataset", revision=config.REVISIONS[repo])
    patches = sorted({m.group(1) for f in files if (m := re.match(r"(patch_\d{4}_\d{4}(?:_S)?_80[NS])_", f))})
    split_of = {}
    for split, name in (("train", "train_filtered.txt"), ("validation", "val_filtered.txt"), ("test", "test_filtered.txt")):
        hub.fetch(repo, "dataset", name, config.ICE_DIR)
        for patch in (config.ICE_DIR / name).read_text(encoding="utf-8").split():
            if patch in split_of:
                raise ValueError(f"{patch} is listed in two splits")
            split_of[patch] = split
    result = {"repoPatches": len(patches), "listedPatches": len(split_of), "poles": {}}
    for pole, suffix in (("north", "_80N"), ("south", "_S_80S")):
        members = [p for p in patches if p.endswith(suffix) and (pole == "south" or "_S_" not in p)]
        cells, origin = {}, None
        for patch in members:
            hub.fetch(repo, "dataset", f"{patch}_PRO.tif", config.ICE_DIR)
            with rasterio.open(config.ICE_DIR / f"{patch}_PRO.tif") as src:
                tile, values = raster_tile(src), src.read(1)
            row, col = (int(v) for v in patch.split("_")[1:3])
            cell_origin = (tile.x_min - col * tile.width * tile.pixel_size_x, tile.y_max + row * tile.height * tile.pixel_size_y)
            origin = origin or cell_origin
            if max(abs(a - b) for a, b in zip(origin, cell_origin)) > 1e-3:
                raise ValueError(f"{patch}: grid position disagrees with its row/column name")
            lon, lat = tile.center()
            cells[(row, col)] = {"patch": patch, "split": split_of.get(patch, "unlisted"), "centerLat": round(lat, 4),
                                 "centerLon": round(lon, 4), "validFraction": round(float(np.isfinite(values).mean()), 4),
                                 "validPixels": int(np.isfinite(values).sum())}
        matrix = [[cells.get((r, c), {"split": "missing"})["split"] for c in range(9)] for r in range(9)]
        counts = Counter(v["split"] for v in cells.values())
        valid = defaultdict(int)
        for v in cells.values():
            valid[v["split"]] += v["validPixels"]
        result["poles"][pole] = {
            "cells": 81, "repoPatches": len(cells), "bySplit": {s: counts.get(s, 0) for s in (*SPLITS, "unlisted")},
            "missingFromRepo": 81 - len(cells), "unionOfSplits": sum(counts.get(s, 0) for s in SPLITS),
            "matrix": matrix,
            "unlisted": [v | {"row": r, "col": c} for (r, c), v in sorted(cells.items()) if v["split"] == "unlisted"],
            "validPixelsBySplit": dict(valid),
            "addedValidPixelsVsTest": valid["train"] + valid["validation"],
            "gridOrigin": [round(origin[0], 3), round(origin[1], 3)],
            "cells_": {f"{r},{c}": v for (r, c), v in sorted(cells.items())},
        }
        _ice_figure(pole, result["poles"][pole], OUT / f"ice-{pole}-all-splits.png")
    return result


def _ice_figure(pole: str, data: dict, path: Path) -> None:
    size, margin = 96, 70
    image = Image.new("RGB", (9 * size + 2 * margin + 300, 9 * size + 2 * margin), (10, 12, 14))
    draw = ImageDraw.Draw(image)
    font, small = _font(18), _font(13)
    for (key, cell) in data["cells_"].items():
        r, c = (int(v) for v in key.split(","))
        x0, y0 = margin + c * size, margin + r * size
        base = COLORS[cell["split"]]
        shade = tuple(int(v * (0.35 + 0.65 * cell["validFraction"])) for v in base)
        draw.rectangle([x0, y0, x0 + size - 2, y0 + size - 2], fill=shade)
        draw.text((x0 + 5, y0 + 5), f"{r},{c}", fill=(10, 10, 10), font=small)
        draw.text((x0 + 5, y0 + size - 22), f"{100 * cell['validFraction']:.0f}% valid", fill=(10, 10, 10), font=small)
    for r in range(9):
        for c in range(9):
            if f"{r},{c}" not in data["cells_"]:
                x0, y0 = margin + c * size, margin + r * size
                draw.rectangle([x0, y0, x0 + size - 2, y0 + size - 2], outline=COLORS["missing"], width=2)
    # 80° circle and pole in grid pixels: the grid spans 9 × 256 px at 238.56 m around the pole at the grid center.
    cell_m = 256 * 238.56  # the pole (false origin 500 km) is not at the grid center
    cx = margin + (500_000 - data["gridOrigin"][0]) / cell_m * size
    cy = margin + (data["gridOrigin"][1] - 500_000) / cell_m * size
    rho80 = 2 * 1737.4 * 0.994 * math.tan(math.radians(5)) / (238.56 * 256 / 1000) * size
    draw.ellipse([cx - rho80, cy - rho80, cx + rho80, cy + rho80], outline=(255, 255, 255), width=2)
    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(255, 255, 255))
    b = data["bySplit"]
    legend = [f"{pole.upper()} POLE · 9×9 grid", f"train {b['train']}", f"validation {b['validation']}", f"test {b['test']}",
              f"in repo, no split {b['unlisted']}", f"not in repo {data['missingFromRepo']}", f"union of splits {data['unionOfSplits']}/81",
              "", "shade = valid-pixel share", "white circle = 80° latitude"]
    for i, line in enumerate(legend):
        color = COLORS.get(line.split(" ")[0], (230, 230, 230)) if i in (1, 2, 3) else (230, 230, 230)
        if line.startswith("in repo"):
            color = COLORS["unlisted"]
        draw.text((9 * size + 2 * margin - 20, margin + i * 28), line, fill=color, font=font)
    draw.text((margin, 20), f"SomBench ice prospectivity patches by split · {pole} polar stereographic grid (row, col) · "
              f"source metadata and PRO rasters only, no inference", fill=(230, 230, 230), font=small)
    image.save(path, optimize=True)


# ─── WAC ──────────────────────────────────────────────────────────────────────


def wac_inventory() -> dict:
    metadata = pd.read_parquet(config.WAC_DIR / "metadata.parquet")
    tiles = []
    for _, row in metadata.iterrows():
        meta = row.to_dict()
        tile = wac_tile_projection(meta)
        lon, lat = tile.center()
        tiles.append({
            "stem": meta["WAC_VIS_TILE"].rsplit("/", 1)[-1].removesuffix(".nc"),
            "split": {"val": "validation"}.get(meta["DATASET"], meta["DATASET"]),
            "lat": lat, "lon": lon, "projection": "polar-stereographic" if "+proj=stere" in tile.proj4 else "transverse-mercator",
            "grid": tile.grid(12),
        })
    rows = int(180 / COVERAGE_CELL_DEG)
    covered = {s: np.zeros((rows, 2 * rows), bool) for s in SPLITS}
    for t in tiles:
        for line in t["grid"]:
            for lon, lat in line:
                i = min(rows - 1, int((lat + 90) / COVERAGE_CELL_DEG))
                j = min(2 * rows - 1, int((lon + 180) / COVERAGE_CELL_DEG))
                covered[t["split"]][i, j] = True
    areas = cell_area_km2(np.arange(rows))[:, None]
    area = lambda mask: float((mask * areas).sum())  # noqa: E731
    union = covered["train"] | covered["validation"] | covered["test"]
    points = [(t["lat"], t["lon"]) for t in tiles]
    overlap_groups = union_find(len(tiles), close_pairs(points, WAC_OVERLAP_KM))
    near_duplicates = union_find(len(tiles), close_pairs(points, 5.0))
    area_groups = union_find(len(tiles), close_pairs(points, WAC_AREA_KM))

    def split_stats(members):
        lats, lons = [t["lat"] for t in members], [t["lon"] for t in members]
        return {
            "tiles": len(members),
            "latRange": [round(min(lats), 2), round(max(lats), 2)], "lonRange": [round(min(lons), 2), round(max(lons), 2)],
            "latBands15": dict(Counter(f"{int(math.floor(v / 15) * 15)}" for v in lats)),
            "lonBands30": dict(Counter(f"{int(math.floor(v / 30) * 30)}" for v in lons)),
            "north": sum(v >= 0 for v in lats), "south": sum(v < 0 for v in lats),
            "nearside": sum(abs(v) <= 90 for v in lons), "farside": sum(abs(v) > 90 for v in lons),
            "projections": dict(Counter(t["projection"] for t in members)),
        }

    index = {id(t): i for i, t in enumerate(tiles)}
    result = {
        "tiles": len(tiles),
        "bySplit": {s: split_stats([t for t in tiles if t["split"] == s]) for s in SPLITS},
        "all": split_stats(tiles),
        "overlapGroups": {"thresholdKm": WAC_OVERLAP_KM, "count": len(set(overlap_groups)),
                          "tilesSharingFootprint": sum(1 for g, n in Counter(overlap_groups).items() if n > 1)},
        "nearIdenticalFootprints": {"thresholdKm": 5.0, "distinct": len(set(near_duplicates)),
                                    "groupsWithRepeats": sum(1 for n in Counter(near_duplicates).values() if n > 1),
                                    "largestGroup": max(Counter(near_duplicates).values())},
        "summedTileAreaKm2": round(len(tiles) * (512 * 0.1) ** 2),
        "geographicAreas": {"thresholdKm": WAC_AREA_KM, "count": len(set(area_groups)),
                            "bySplit": {s: len({area_groups[index[id(t)]] for t in tiles if t["split"] == s}) for s in SPLITS},
                            "areasWithoutTest": len({area_groups[i] for i, t in enumerate(tiles)}
                                                    - {area_groups[i] for i, t in enumerate(tiles) if t["split"] == "test"})},
        "coveredAreaKm2": {s: round(area(covered[s])) for s in SPLITS} | {"union": round(area(union))},
        "coveredFractionOfMoon": {s: round(area(covered[s]) / lunar_area_km2(), 4) for s in SPLITS}
        | {"union": round(area(union) / lunar_area_km2(), 4)},
        "addedAreaVsTestKm2": round(area(union & ~covered["test"])),
        "coverageGridDeg": COVERAGE_CELL_DEG,
    }
    _wac_figure(tiles, OUT / "craters-all-splits-global.png", result)
    return result


def _globe_canvas(width: int = 2400):
    return globe_backdrop(width, 0.5)


def _wac_figure(tiles: list[dict], path: Path, result: dict) -> None:
    image = _globe_canvas()
    draw = ImageDraw.Draw(image)
    w, h = image.size
    xy = lambda lat, lon: ((lon + 180) / 360 * w, (90 - lat) / 180 * h)  # noqa: E731
    for split in SPLITS:  # test drawn last so it stays visible
        for t in (t for t in tiles if t["split"] == split):
            g = t["grid"]
            ring = [g[0][0], g[0][-1], g[-1][-1], g[-1][0]]
            pts = [xy(lat, lon) for lon, lat in ring]
            if max(p[0] for p in pts) - min(p[0] for p in pts) > w / 2:
                continue
            draw.polygon(pts, outline=COLORS[split], width=2)
    font = _font(22)
    s = result["bySplit"]
    draw.rectangle([0, 0, w, 44], fill=(10, 12, 14))
    draw.text((12, 10), f"SomBench WAC crater tiles by split (metadata only) · train {s['train']['tiles']} (blue) · validation "
              f"{s['validation']['tiles']} (amber) · test {s['test']['tiles']} (green) · {result['geographicAreas']['count']} "
              f"areas ≤ {WAC_AREA_KM:.0f} km · union {100 * result['coveredFractionOfMoon']['union']:.2f}% of the Moon",
              fill=(235, 235, 235), font=font)
    image.save(path, optimize=True)


# ─── IMP ──────────────────────────────────────────────────────────────────────


def imp_inventory() -> dict:
    split_of = {}
    for split, name in (("train", "train.txt"), ("validation", "val.txt"), ("test", "test.txt")):
        hub.fetch(config.IMP_DATASET_REPO, "dataset", name, config.IMP_DIR)
        for sample in (config.IMP_DIR / name).read_text(encoding="utf-8").split():
            split_of[sample] = split
    tiles = []
    for sample, split in sorted(split_of.items()):
        hub.fetch(config.IMP_DATASET_REPO, "dataset", f"all/{sample}_img.tif", config.IMP_DIR)
        with rasterio.open(config.IMP_DIR / "all" / f"{sample}_img.tif") as src:
            tile = raster_tile(src)
        lon, lat = tile.center()
        target = re.search(r"target_(\d+)", sample).group(1)
        tiles.append({"sample": sample, "split": split, "lat": lat, "lon": lon, "target": target,
                      "product": sample.split(".")[0], "footprint": (round(tile.x_min, 1), round(tile.y_max, 1), tile.proj4)})
    groups = union_find(len(tiles), close_pairs([(t["lat"], t["lon"]) for t in tiles], IMP_SITE_KM))
    sites = defaultdict(list)
    for t, g in zip(tiles, groups):
        sites[g].append(t)
    footprints = Counter(t["footprint"] for t in tiles)
    site_rows = []
    for members in sites.values():
        site_rows.append({
            "center": [round(float(np.mean([t["lat"] for t in members])), 4), round(float(np.mean([t["lon"] for t in members])), 4)],
            "observations": len(members), "distinctFootprints": len({t["footprint"] for t in members}),
            "targets": sorted({t["target"] for t in members}), "splits": dict(Counter(t["split"] for t in members)),
        })
    site_rows.sort(key=lambda s: (-s["observations"], s["center"]))
    by_split_sites = {s: sum(1 for r in site_rows if s in r["splits"]) for s in SPLITS}
    result = {
        "tiles": len(tiles),
        "bySplit": {s: sum(t["split"] == s for t in tiles) for s in SPLITS},
        "distinctFootprints": len(footprints),
        "footprintsObservedMoreThanOnce": sum(1 for n in footprints.values() if n > 1),
        "targets": len({t["target"] for t in tiles}),
        "sites": {"thresholdKm": IMP_SITE_KM, "count": len(site_rows), "bySplitPresence": by_split_sites,
                  "sitesWithoutTest": sum(1 for r in site_rows if "test" not in r["splits"]),
                  "sitesWithTestOnly": sum(1 for r in site_rows if set(r["splits"]) == {"test"}),
                  "mixedSplitSites": sum(1 for r in site_rows if len(r["splits"]) > 1)},
        "latRange": [round(min(t["lat"] for t in tiles), 2), round(max(t["lat"] for t in tiles), 2)],
        "lonRange": [round(min(t["lon"] for t in tiles), 2), round(max(t["lon"] for t in tiles), 2)],
        "sites_": site_rows,
    }
    _imp_figure(site_rows, OUT / "imp-all-splits-global.png", result)
    return result


def _imp_figure(sites: list[dict], path: Path, result: dict) -> None:
    image = _globe_canvas()
    draw = ImageDraw.Draw(image)
    w, h = image.size
    font, small = _font(22), _font(15)
    for i, site in enumerate(sorted(sites, key=lambda s: s["observations"])):
        lat, lon = site["center"]
        x, y = (lon + 180) / 360 * w, (90 - lat) / 180 * h
        r = 6 + 3 * math.sqrt(site["observations"])
        dominant = max(site["splits"], key=site["splits"].get)
        draw.ellipse([x - r, y - r, x + r, y + r], outline=COLORS[dominant], width=3)
        if len(site["splits"]) > 1:
            draw.ellipse([x - r - 4, y - r - 4, x + r + 4, y + r + 4], outline=(255, 255, 255), width=1)
        if site["observations"] > 1:
            draw.text((x + r + 3, y - 8), str(site["observations"]), fill=(235, 235, 235), font=small)
    draw.rectangle([0, 0, w, 44], fill=(10, 12, 14))
    b, s = result["bySplit"], result["sites"]
    draw.text((12, 10), f"SomBench IMP tiles grouped into physical sites (≤ {IMP_SITE_KM:.0f} km) · {result['tiles']} tiles · "
              f"{s['count']} sites · train {b['train']} (blue) · validation {b['validation']} (amber) · test {b['test']} (green) · "
              f"number = observations · white ring = mixed splits", fill=(235, 235, 235), font=font)
    image.save(path, optimize=True)


# ─── Report ───────────────────────────────────────────────────────────────────


def decision(report: dict) -> dict:
    ice, wac, imp = report["ice"], report["craters"], report["imp"]
    ice_added = sum(p["bySplit"]["train"] + p["bySplit"]["validation"] for p in ice["poles"].values())
    wac_added = wac["bySplit"]["train"]["tiles"] + wac["bySplit"]["validation"]["tiles"]
    imp_added = imp["bySplit"]["train"] + imp["bySplit"]["validation"]
    return {
        "ice": {"additionalPatches": ice_added,
                "coverageGain": {p: f"test {d['bySplit']['test']} → union {d['unionOfSplits']} of 81 cells" for p, d in ice["poles"].items()},
                "storageMB": round(ice_added * (STORAGE_KB["ice-patch"] + STORAGE_KB["ice-mosaic-per-patch"]) / 1024, 1)},
        "craters": {"additionalTiles": wac_added,
                    "coverageGain": f"{wac['coveredAreaKm2']['test']:,} → {wac['coveredAreaKm2']['union']:,} km² "
                                    f"({100 * wac['coveredFractionOfMoon']['test']:.2f}% → {100 * wac['coveredFractionOfMoon']['union']:.2f}% of the Moon); "
                                    f"areas {wac['geographicAreas']['bySplit']['test']} → {wac['geographicAreas']['count']}",
                    "storageMB": round(wac_added * STORAGE_KB["wac-tile"] / 1024, 1)},
        "imp": {"additionalTiles": imp_added,
                "coverageGain": f"sites with test {imp['sites']['bySplitPresence']['test']} → all sites {imp['sites']['count']}",
                "storageMB": round(imp_added * STORAGE_KB["imp-tile"] / 1024, 1)},
    }


def text(report: dict) -> str:
    lines = ["DATASET COVERAGE INVENTORY (source metadata and rasters only, no inference)", ""]
    ice = report["ice"]
    lines.append(f"ICE · repo patches {ice['repoPatches']} · listed in split files {ice['listedPatches']}")
    for pole, d in ice["poles"].items():
        b = d["bySplit"]
        lines += [f"  {pole.upper()}: train {b['train']} · validation {b['validation']} · test {b['test']} · union {d['unionOfSplits']}/81"
                  f" · in repo without split {b['unlisted']} · not in repo {d['missingFromRepo']}",
                  "    matrix (T=train V=validation X=test U=unlisted .=missing):"]
        code = {"train": "T", "validation": "V", "test": "X", "unlisted": "U", "missing": "."}
        lines += ["      " + " ".join(code[v] for v in row) for row in d["matrix"]]
        for u in d["unlisted"]:
            lines.append(f"    unlisted {u['patch']} at ({u['row']},{u['col']}): {100 * u['validFraction']:.1f}% valid pixels")
        lines.append(f"    valid pixels: test {d['validPixelsBySplit'].get('test', 0):,} · added by train+validation {d['addedValidPixelsVsTest']:,}")
    w = report["craters"]
    lines += ["", f"CRATERS WAC · {w['tiles']} tiles"]
    for s in SPLITS:
        st = w["bySplit"][s]
        lines.append(f"  {s:<10} {st['tiles']:>4} tiles · lat {st['latRange']} · lon {st['lonRange']} · N/S {st['north']}/{st['south']} · "
                     f"near/far {st['nearside']}/{st['farside']} · {st['projections']} · areas {w['geographicAreas']['bySplit'][s]}")
    lines += [f"  geographic areas (≤ {WAC_AREA_KM:.0f} km): {w['geographicAreas']['count']} · areas without any test tile "
              f"{w['geographicAreas']['areasWithoutTest']}",
              f"  overlapping footprints (centers < {WAC_OVERLAP_KM} km): {w['overlapGroups']['count']} distinct footprints, "
              f"{w['overlapGroups']['tilesSharingFootprint']} groups with more than one tile",
              "  covered area: " + " · ".join(f"{k} {v:,} km² ({100 * w['coveredFractionOfMoon'][k]:.2f}%)" for k, v in w["coveredAreaKm2"].items()),
              f"  area added beyond test: {w['addedAreaVsTestKm2']:,} km² · summed tile area {w['summedTileAreaKm2']:,} km² vs union "
              f"{w['coveredAreaKm2']['union']:,} km² · near-identical footprints (< 5 km): {w['nearIdenticalFootprints']['distinct']} distinct, "
              f"{w['nearIdenticalFootprints']['groupsWithRepeats']} repeated, largest group {w['nearIdenticalFootprints']['largestGroup']}",
              "  latitude bands (15°, all splits): " + ", ".join(f"{k}: {v}" for k, v in sorted(w["all"]["latBands15"].items(), key=lambda kv: int(kv[0]))),
              "  longitude bands (30°, all splits): " + ", ".join(f"{k}: {v}" for k, v in sorted(w["all"]["lonBands30"].items(), key=lambda kv: int(kv[0])))]
    m = report["imp"]
    s = m["sites"]
    lines += ["", f"IMP · {m['tiles']} tiles · train {m['bySplit']['train']} · validation {m['bySplit']['validation']} · test {m['bySplit']['test']}",
              f"  distinct footprints {m['distinctFootprints']} ({m['footprintsObservedMoreThanOnce']} observed more than once) · "
              f"targets {m['targets']} · lat {m['latRange']} · lon {m['lonRange']}",
              f"  physical sites (≤ {IMP_SITE_KM:.0f} km): {s['count']} · with a test tile {s['bySplitPresence']['test']} · "
              f"without test {s['sitesWithoutTest']} · mixed splits {s['mixedSplitSites']}",
              "  largest sites: " + " · ".join(f"{r['center']} ×{r['observations']} {r['splits']}" for r in m["sites_"][:6])]
    lines += ["", "DECISION INPUTS (coverage value vs processing/storage cost)"]
    for task, d in report["decisionInputs"].items():
        lines.append(f"  {task}: {json.dumps(d, ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"ice": ice_inventory(), "craters": wac_inventory(), "imp": imp_inventory()}
    report["decisionInputs"] = decision(report)
    serializable = json.loads(json.dumps(report, default=str))
    for pole in serializable["ice"]["poles"].values():
        pole.pop("cells_", None)
    (OUT / "coverage-report.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8", newline="\n")
    (OUT / "coverage-report.txt").write_text(text(report), encoding="utf-8", newline="\n")
    print(text(report))
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
