"""Scientific validation of a normalized build (contract v2). Returns a list of problems; empty means valid.

Covers docs/provenance/data-contract.md › Validation Rules for real data: ids, provenance pinned to the released checkpoints and
datasets, projection metadata reproducing footprints, raw rasters decoding to the preserved model output,
display rasters derived only by clipping, raster sanity (shape, NaN/Inf, degenerate outputs), count consistency
and manifest checksums.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from . import anchor as anchors
from . import artifacts, config, encoding
from . import projection as proj
from .catalog import crater_identity, region_identity
from .imp_groups import group_observations
from .registry import DiscoveryRegistry, RegistryError

SPLITS = {"train", "validation", "test", "external"}
PINNED = {
    "crater-detection": ("lfm-crater-detection", "sombench-wac-crater-detection", "robbins-2019"),
    "ice-prospectivity": ("lfm-ice-prospectivity", "sombench-ice-prospectivity-regression", "coyan-2025"),
    "imp-segmentation": ("lfm-imp-segmentation", "sombench-imp-segmentation", "hargitai-2025"),
}
ID_PATTERNS = {
    "crater-detection": r"wac-[a-z0-9]+-r\d+-c\d+",
    "ice-prospectivity": r"ice-[ns]-\d{4}-\d{4}",
    "imp-segmentation": r"imp-[a-z0-9]+-t\d+-i\d+",
}
GEOLOCATION_TOLERANCE_DEG = 1e-5  # stored coordinates have 6 decimals


class Report:
    def __init__(self):
        self.problems: list[str] = []

    def check(self, condition, message: str) -> bool:
        if not condition:
            self.problems.append(message)
        return bool(condition)


def _load(root: Path, name: str, default):
    path = root / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def sample_of(item: dict) -> str:
    return item["provenance"]["sourceTile"].removesuffix(".tif").removesuffix("_img")


def is_synthetic(item: dict) -> bool:
    p = item["provenance"]
    return (item["id"].startswith("mock-") or p["sourceTile"].upper().startswith("MOCK")
            or p["processingVersion"].startswith("mock") or p["modelRevision"] == "mock")


def _coordinate(report: Report, where: str, point: dict):
    lat, lon = point.get("latitude"), point.get("longitude")
    report.check(isinstance(lat, (int, float)) and -90 <= lat <= 90, f"{where}: latitude {lat} out of range")
    report.check(isinstance(lon, (int, float)) and -180 <= lon < 180, f"{where}: longitude {lon} not in [-180, 180)")


def _angle_close(a: dict, lat: float, lon: float) -> bool:
    return (abs(a["latitude"] - lat) <= GEOLOCATION_TOLERANCE_DEG
            and abs((a["longitude"] - lon + 180) % 360 - 180) <= GEOLOCATION_TOLERANCE_DEG)


def _geolocation(report: Report, where: str, item_center: dict, projection: dict, grid: dict, corners: list[dict]):
    """The published projection + grid must reproduce the published center and footprint corners."""
    width, height = grid["widthPx"], grid["heightPx"]
    lat, lon = proj.pixel_to_latlon(projection, grid, width / 2, height / 2)
    report.check(_angle_close(item_center, lat, lon), f"{where}: projection/grid do not reproduce the center")
    for corner, (col, row) in zip(corners, [(0, 0), (width, 0), (width, height), (0, height)], strict=True):
        lat, lon = proj.pixel_to_latlon(projection, grid, col, row)
        report.check(_angle_close(corner, lat, lon), f"{where}: projection/grid do not reproduce footprint corner")


def _image(report: Report, root: Path, relative: str, shape: tuple[int, int] | None, where: str):
    path = root / relative
    if not report.check(path.exists(), f"{where}: missing asset {relative}"):
        return None
    with Image.open(path) as image:
        array = np.asarray(image if image.mode in ("RGB", "RGBA") else image.convert("RGBA"))
    if shape is not None:
        report.check(array.shape[:2] == shape, f"{where}: {relative} is {array.shape[:2]}, expected {shape}")
    return array


def _value_raster(report: Report, root: Path, raster: dict, shape, where: str):
    """Decodes raw and display rasters and checks display = clip(raw) and non-degenerate coverage.

    A null displayPath means the patch ships no display image of its own (production ICE patches read the pole mosaic
    window instead, see docs/provenance/data-contract.md); the raw raster is still checked.
    """
    raw_rgba = _image(report, root, raster["rawPath"], shape, where)
    if raw_rgba is None:
        return None
    if raster["displayPath"] is None:
        values, covered = encoding.decode_raw(raw_rgba, raster["rawEncoding"])
        if not report.check(covered.any(), f"{where}: raster has no covered pixels"):
            return None
        report.check(float(values[covered].std()) > 0, f"{where}: raster is constant (degenerate)")
        return values, covered
    display = _image(report, root, raster["displayPath"], shape, where)
    if display is None or raw_rgba.shape[:2] != display.shape[:2]:
        return None
    values, covered = encoding.decode_raw(raw_rgba, raster["rawEncoding"])
    display_alpha = display[..., 3] if display.shape[-1] == 4 else np.full(display.shape[:2], 255)
    report.check(set(np.unique(display_alpha)) <= {0, 255}, f"{where}: display alpha is not binary")
    report.check(((display_alpha == 255) == covered).all(), f"{where}: raw and display coverage differ")
    report.check((display[..., 0] == display[..., 1]).all() and (display[..., 1] == display[..., 2]).all(),
                 f"{where}: display raster is not gray")
    if not report.check(covered.any(), f"{where}: raster has no covered pixels"):
        return None
    expected = np.round(np.clip(values, 0, 1) * 255)
    tolerance_levels = 255 * raster["rawEncoding"]["scale"]
    report.check(np.all(np.abs(display[..., 0][covered] - expected[covered]) <= 1 + tolerance_levels),
                 f"{where}: display raster is not the clipped raw raster")
    report.check(float(values[covered].std()) > 0, f"{where}: raster is constant (degenerate)")
    return values, covered


def _provenance(report: Report, where: str, task: str, provenance: dict, sources: dict):
    model, dataset, reference = PINNED[task]
    task_model = config.TASK_MODELS[task]
    report.check(provenance.get("task") == task, f"{where}: task {provenance.get('task')} != {task}")
    report.check(provenance.get("model") == model and sources.get(model) == "model", f"{where}: model {provenance.get('model')}")
    report.check(provenance.get("sourceDataset") == dataset and sources.get(dataset) == "dataset",
                 f"{where}: dataset {provenance.get('sourceDataset')}")
    report.check(provenance.get("referenceSource") == reference and sources.get(reference) == "reference",
                 f"{where}: reference {provenance.get('referenceSource')}")
    report.check(provenance.get("datasetSplit") in SPLITS, f"{where}: split {provenance.get('datasetSplit')}")
    report.check(provenance.get("checkpoint") == task_model.checkpoint.filename, f"{where}: checkpoint {provenance.get('checkpoint')}")
    report.check(provenance.get("modelRevision") == task_model.checkpoint.revision, f"{where}: model revision is not the pinned one")
    report.check(provenance.get("processingVersion") == config.PROCESSING_VERSION, f"{where}: processing version")


def _raw_artifact(report: Report, task: str, sample: str, where: str):
    try:
        run, arrays = artifacts.read_run(task, sample)
    except FileNotFoundError:
        report.check(False, f"{where}: raw inference artifact missing for {sample}")
        return None
    report.check(not run["missingKeys"] and not run["unexpectedKeys"], f"{where}: checkpoint was not loaded strictly")
    report.check(run["checkpointSha256"] == config.TASK_MODELS[task].checkpoint.sha256, f"{where}: checkpoint sha256")
    for name, summary_ in run["outputs"].items():
        report.check(summary_["nonFinite"] == 0, f"{where}: {name} contains NaN/Inf")
    main = {"crater-detection": "raw_scores", "ice-prospectivity": "prediction", "imp-segmentation": "logits"}[task]
    values = arrays[main]
    if report.check(values.size > 0, f"{where}: {main} is empty"):
        report.check(float(values.std()) > 0, f"{where}: {main} is constant (degenerate output)")
    return arrays


def _within(point: dict, footprint: list[dict], margin: float) -> bool:
    lats = [p["latitude"] for p in footprint]
    lon_deg = np.degrees(np.unwrap(np.radians([p["longitude"] for p in footprint + [point]])))
    return (min(lats) - margin <= point["latitude"] <= max(lats) + margin
            and lon_deg[:-1].min() - margin <= lon_deg[-1] <= lon_deg[:-1].max() + margin)


def _check_craters(report: Report, root: Path, regions: list, craters: dict, raw: dict):
    report.check(sorted(craters) == sorted(r["id"] for r in regions), "craters.json keys differ from crater regions")
    for region in regions:
        where = f"crater-detection:{region['id']}"
        source, tuples = region["source"], craters.get(region["id"], [])
        _geolocation(report, where, region, source["projection"], source["grid"], source["footprint"])
        report.check(len(tuples) == region["prediction"]["detectionCount"], f"{where}: detection count mismatch")
        report.check(sum(t[5] for t in tuples) == region["derived"]["matchedPredictions"], f"{where}: matched count mismatch")
        threshold = region["prediction"]["confidenceThreshold"]
        arrays = raw.get(region["id"])
        if arrays is not None:
            report.check(len(tuples) == int((arrays["scores"] >= threshold).sum()), f"{where}: tuples differ from raw scores ≥ threshold")
        for i, (lat, lon, diameter, confidence, truncated, matched) in enumerate(tuples):
            _coordinate(report, f"{where}#{i}", {"latitude": lat, "longitude": lon})
            report.check(diameter > 0, f"{where}#{i}: diameter {diameter}")
            report.check(threshold <= confidence <= 1, f"{where}#{i}: confidence {confidence}")
            report.check(truncated in (0, 1) and matched in (0, 1), f"{where}#{i}: flags")
            report.check(_within({"latitude": lat, "longitude": lon}, source["footprint"], 0.2),
                         f"{where}#{i}: detection outside its tile footprint")
        detail = _load(root, region["detailPath"], None)
        if report.check(detail is not None, f"{where}: missing {region['detailPath']}"):
            size = detail["imageSizePx"]
            report.check(len(detail["prediction"]) == len(tuples), f"{where}: tile detail prediction count")
            report.check(len(detail["reference"]) == region["reference"]["craterCount"], f"{where}: reference count")
            report.check(len(detail["matching"]["matches"]) == region["derived"]["matchedPredictions"], f"{where}: matches count")
            for box in [*detail["prediction"], *detail["reference"]]:
                x, y, w, h = box[:4]
                report.check(w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= size + 1e-6 and y + h <= size + 1e-6,
                             f"{where}: box {box} outside the tile")
        _image(report, root, source["imagePath"], (512, 512), where)


def _check_ice(report: Report, root: Path, sectors: list, mosaics: list, raw: dict):
    decoded = {}
    for sector in sectors:
        where = f"ice-prospectivity:{sector['id']}"
        source, p, r = sector["source"], sector["prediction"], sector["reference"]
        grid = source["grid"]
        fg = source["footprintGrid"]
        _geolocation(report, where, sector, source["projection"], grid, [fg[0][0], fg[0][8], fg[8][8], fg[8][0]])
        report.check((source["projection"]["latitudeOfOriginDeg"] < 0) == (sector["pole"] == "south"), f"{where}: projection pole")
        report.check(abs(sector["latitude"]) >= 75 and (sector["latitude"] < 0) == (sector["pole"] == "south"),
                     f"{where}: sector is not in the polar extent of its pole")
        report.check(source["patchId"].endswith("_S_80S") == (sector["pole"] == "south")
                     and sector["id"].startswith(f"ice-{sector['pole'][0]}-"), f"{where}: patch name or id disagrees with the pole")
        anchor = sector["derived"]["discoveryAnchor"]
        _coordinate(report, f"{where}:rawPeak", p["rawPeak"])
        if anchor is not None:
            _coordinate(report, f"{where}:discoveryAnchor", anchor)
            report.check(anchor["method"] == anchors.METHOD and anchor["edgeMarginPx"] == anchors.EDGE_MARGIN_PX
                         and anchor["edgeDistancePx"] >= anchor["edgeMarginPx"], f"{where}: discovery anchor method or margin")
        shape = (grid["heightPx"], grid["widthPx"])
        result = _value_raster(report, root, p["heatmap"], shape, f"{where}:prediction")
        _value_raster(report, root, r["heatmap"], shape, f"{where}:reference")
        arrays = raw.get(sector["id"])
        if result is not None and arrays is not None:
            values, covered = result
            model = arrays["prediction"].astype(np.float64)
            error = float(np.abs(values[covered] - model[covered]).max())
            report.check(error <= p["heatmap"]["rawEncoding"]["scale"] / 2 + 1e-9,
                         f"{where}: raw raster differs from the model output by {error:.2e}")
            report.check(abs(float(model[covered].max()) - p["raw"]["max"]) <= 1e-6
                         and abs(float(model[covered].min()) - p["raw"]["min"]) <= 1e-6,
                         f"{where}: raw statistics differ from the model output")
            report.check(int(((model < 0) | (model > 1))[covered].sum()) == p["display"]["clippedPixels"], f"{where}: clipped pixel count")
            report.check(int(covered.sum()) == p["display"]["coveredPixels"], f"{where}: covered pixel count")
            report.check(abs(p["rawPeak"]["value"] - p["raw"]["max"]) <= 1e-6, f"{where}: raw peak is not the raw maximum")
            expected = anchors.peak_and_anchor(model, covered, anchors.EDGE_MARGIN_PX)
            report.check((anchor is None) == (expected[1] is None), f"{where}: discovery anchor presence differs from the model output")
            for label, record, truth in (("raw peak", p["rawPeak"], expected[0]), ("discovery anchor", anchor, expected[1])):
                if record is None or truth is None:
                    continue
                lat, lon = proj.pixel_to_latlon(source["projection"], grid, record["col"] + 0.5, record["row"] + 0.5)
                report.check((record["row"], record["col"], record["edgeDistancePx"]) == (truth["row"], truth["col"], truth["edgeDistancePx"])
                             and abs(record["value"] - truth["value"]) <= 1e-6, f"{where}: {label} differs from the model output")
                report.check(abs(lat - record["latitude"]) <= GEOLOCATION_TOLERANCE_DEG
                             and abs((lon - record["longitude"] + 180) % 360 - 180) <= GEOLOCATION_TOLERANCE_DEG,
                             f"{where}: {label} coordinate is not its pixel center")
            decoded[sector["id"]] = (values, covered)
        report.check(sector["derived"]["score"]["formula"] == "ice-p90-bounded-v2", f"{where}: score formula")
        report.check(sector["derived"]["rmse"] >= 0 and sector["derived"]["mae"] >= 0, f"{where}: error metrics")

    placed = [patch["sectorId"] for m in mosaics for patch in m["patches"]]
    report.check(sorted(placed) == sorted(s["id"] for s in sectors), "every ice sector must be in exactly one mosaic")
    for mosaic in mosaics:
        where = f"ice-mosaic:{mosaic['id']}"
        grid = mosaic["grid"]
        shape = (grid["heightPx"], grid["widthPx"])
        report.check((mosaic["projection"]["latitudeOfOriginDeg"] < 0) == (mosaic["pole"] == "south"), f"{where}: projection pole")
        result = _value_raster(report, root, mosaic["prediction"], shape, f"{where}:prediction")
        _value_raster(report, root, mosaic["reference"], shape, f"{where}:reference")
        for patch in mosaic["patches"]:
            member = next((s for s in sectors if s["id"] == patch["sectorId"]), None)
            if not report.check(member is not None and member["pole"] == mosaic["pole"], f"{where}: {patch['sectorId']} pole"):
                continue
            sg = member["source"]["grid"]
            report.check(member["source"]["projection"] == mosaic["projection"], f"{where}: {member['id']} projection differs")
            report.check(abs(sg["xMinM"] - (grid["xMinM"] + patch["col"] * grid["pixelSizeXM"])) < 0.01
                         and abs(sg["yMaxM"] - (grid["yMaxM"] - patch["row"] * grid["pixelSizeYM"])) < 0.01,
                         f"{where}: {member['id']} is not placed at its grid position")
            report.check(patch["col"] + patch["widthPx"] <= shape[1] and patch["row"] + patch["heightPx"] <= shape[0],
                         f"{where}: patch {patch['sectorId']} outside the mosaic")
            if result is not None and member["id"] in decoded:
                window = np.s_[patch["row"]:patch["row"] + patch["heightPx"], patch["col"]:patch["col"] + patch["widthPx"]]
                values, covered = decoded[member["id"]]
                mosaic_values, mosaic_covered = result
                scale = mosaic["prediction"]["rawEncoding"]["scale"] + member["prediction"]["heatmap"]["rawEncoding"]["scale"]
                report.check((mosaic_covered[window] == covered).all(), f"{where}: coverage differs from {member['id']}")
                report.check(float(np.abs(mosaic_values[window][covered] - values[covered]).max()) <= scale / 2 + 1e-9,
                             f"{where}: values differ from {member['id']}")


def _check_imp(report: Report, root: Path, imp: list, raw: dict):
    for region in imp:
        where = f"imp-segmentation:{region['id']}"
        source, p = region["source"], region["prediction"]
        _geolocation(report, where, region, source["projection"], source["grid"], source["footprint"])
        report.check(0 <= p["pixelFraction"] <= 1, f"{where}: pixel fraction")
        report.check(p["areaM2"] >= 0 and region["reference"]["areaM2"] >= 0, f"{where}: negative area")
        report.check(region["derived"]["iou"] is None or 0 <= region["derived"]["iou"] <= 1, f"{where}: IoU range")
        report.check(p["meanSoftmaxScore"] is None or 0 <= p["meanSoftmaxScore"] <= 1, f"{where}: softmax score range")
        if p["centroid"] is not None:
            _coordinate(report, f"{where}:centroid", p["centroid"])
            report.check(_within(p["centroid"], source["footprint"], 1e-4), f"{where}: centroid outside the tile")
        arrays = raw.get(region["id"])
        if arrays is not None:
            report.check(int(arrays["mask"].sum()) == p["pixelCount"], f"{where}: pixel count differs from the argmax mask")
        shape = (source["grid"]["heightPx"], source["grid"]["widthPx"])
        for layer in ("prediction", "reference"):
            mask = _image(report, root, region[layer]["maskPath"], shape, where)
            if mask is not None:
                alpha = mask[..., 3] if mask.shape[-1] == 4 else np.full(shape, 255)
                report.check(int((alpha == 255).sum()) == region[layer]["pixelCount"], f"{where}: {layer} mask pixel count")
        _image(report, root, source["imagePath"], shape, where)


def _check_imp_groups(report: Report, root: Path, imp: list) -> None:
    """imp-observation-groups.json must be exactly the deterministic grouping of the published IMP regions."""
    groups = _load(root, "imp-observation-groups.json", None)
    if report.check(groups is not None, "imp-observation-groups.json missing"):
        report.check(groups == group_observations(imp), "imp-observation-groups.json differs from imp-regions.json")


def _check_registry(report: Report, root: Path, discoveries: list, regions: list) -> None:
    """Every published discovery id must be the registry id of that object's scientific identity."""
    try:
        registry = DiscoveryRegistry.load()
    except RegistryError as error:
        report.check(False, f"discovery registry: {error}")
        return
    by_key = {e["key"]: e for e in registry.entries}
    regions_by_id = {r["id"]: r for r in regions}
    for d in discoveries:
        if d["type"] == "CRATER_CANDIDATE":
            region_id, index = d["objectId"].split("#")
            boxes = _load(root, regions_by_id[region_id]["detailPath"], {"prediction": []})["prediction"]
            key = crater_identity(d["provenance"], boxes[int(index)]) if int(index) < len(boxes) else None
        else:
            key = region_identity(d["provenance"])
        entry = by_key.get(key)
        report.check(entry is not None and entry["id"] == d["id"] and entry["type"] == d["type"]
                     and entry["firstCatalogued"] == d["cataloguedAt"],
                     f"{d['id']}: does not match the discovery registry entry for {key}")


def _check_manifest(report: Report, root: Path, regions: list, craters: dict, sectors: list, mosaics: list, imp: list):
    stats = _load(root, "global-stats.json", None)
    discoveries = _load(root, "discoveries.json", None)
    report.check(discoveries is not None and stats is not None, "discoveries.json or global-stats.json missing")
    if discoveries is not None:
        _check_registry(report, root, discoveries, regions)
    if stats is not None:
        report.check(stats["craterDetections"] == sum(len(v) for v in craters.values()), "global stats crater count")
        report.check(stats["polarPatchesAnalyzed"]["north"] + stats["polarPatchesAnalyzed"]["south"] == len(sectors), "global stats ice count")
        report.check(stats["impTilesAnalyzed"] == len(imp), "global stats IMP count")
    processing = _load(root, "processing-manifest.json", None)
    if not report.check(processing is not None, "processing-manifest.json missing"):
        return
    counts = processing["counts"]
    report.check(counts["craterRegions"] == len(regions), "manifest crater region count")
    report.check(counts["craterDetections"] == sum(len(v) for v in craters.values()), "manifest detection count")
    report.check(counts["iceSectors"] == len(sectors) and counts["iceMosaics"] == len(mosaics), "manifest ice counts")
    report.check(counts["impRegions"] == len(imp), "manifest IMP count")
    for relative, digest in processing["checksums"].items():
        path = root / relative
        if report.check(path.exists(), f"manifest lists missing file {relative}"):
            report.check(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"checksum mismatch for {relative}")
    listed = set(processing["checksums"])
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "processing-manifest.json"}
    report.check(actual == listed, f"files not covered by the manifest: {sorted(actual ^ listed)[:5]}")


def validate(root: Path | None = None, *, check_raw: bool = True) -> list[str]:
    """Every contract problem in a build. `check_raw=False` skips the cross-checks against the local raw
    inference artifacts (data/artifacts/raw), which exist only on the machine that ran the inference: use it to
    validate a *published* dataset, where everything else still applies."""
    root = root or config.BUILD_DIR
    report = Report()
    manifest = _load(root, "manifest.json", None)
    report.check(manifest is not None and manifest.get("dataMode") in ("real", "real-smoke", "batch-preview", "complete-preview"),
                 "manifest dataMode must be a real science mode")
    report.check(manifest is not None and manifest.get("schemaVersion") == config.SCHEMA_VERSION, "manifest schemaVersion")
    sources = {s["id"]: s["kind"] for s in _load(root, "sources.json", [])}
    report.check(bool(sources), "sources.json missing")

    regions = _load(root, "crater-regions.json", [])
    craters = _load(root, "craters.json", {})
    sectors = _load(root, "ice-sectors.json", [])
    mosaics = _load(root, "ice-mosaics.json", [])
    imp = _load(root, "imp-regions.json", [])
    everything = [*regions, *sectors, *imp]
    ids = [item["id"] for item in [*everything, *mosaics]]
    report.check(len(ids) == len(set(ids)), "duplicate ids")
    report.check(not any(is_synthetic(i) for i in everything), "synthetic objects in a real build")
    report.check(not any(i.startswith("mock-") for i in ids), "mock ids in a real build")

    raw = {}
    for task, items in (("crater-detection", regions), ("ice-prospectivity", sectors), ("imp-segmentation", imp)):
        for item in items:
            where = f"{task}:{item['id']}"
            report.check(re.fullmatch(ID_PATTERNS[task], item["id"]) is not None, f"{where}: id pattern")
            _coordinate(report, where, item)
            _provenance(report, where, task, item["provenance"], sources)
            raw[item["id"]] = _raw_artifact(report, task, sample_of(item), where) if check_raw else None

    _check_craters(report, root, regions, craters, raw)
    _check_ice(report, root, sectors, mosaics, raw)
    _check_imp(report, root, imp, raw)
    _check_imp_groups(report, root, imp)
    _check_manifest(report, root, regions, craters, sectors, mosaics, imp)
    return report.problems


def summary(root: Path | None = None) -> dict:
    root = root or config.BUILD_DIR
    regions = _load(root, "crater-regions.json", [])
    sectors = _load(root, "ice-sectors.json", [])
    imp = _load(root, "imp-regions.json", [])
    return {
        "dataMode": _load(root, "manifest.json", {}).get("dataMode"),
        "craterPredictions": sum(len(v) for v in _load(root, "craters.json", {}).values()),
        "craterRegions": len(regions),
        "icePatches": len(sectors),
        "iceMosaics": len(_load(root, "ice-mosaics.json", [])),
        "impRegions": len(imp),
        "discoveries": len(_load(root, "discoveries.json", [])),
        "syntheticObjects": sum(1 for item in [*regions, *sectors, *imp] if is_synthetic(item)),
    }
