"""Discoveries and global statistics built from normalized MOONATLAS files (docs/provenance/data-contract.md › Discovery System).

Shared by the science pipeline (real data) and the fixture generator, so selection rules, wording and counts
never drift. Every value comes from the normalized objects passed in; nothing is invented here.
"""

from __future__ import annotations

from typing import Callable

from .imp_groups import distance_m
from .registry import DiscoveryRegistry, RegistryError
from .sources import SOURCES

# Discovery curation (docs/provenance/data-contract.md › Discovery System): PRODUCT curation, not a scientific ranking. Each category is
# a transparent rule over model outputs; candidates are taken in the rule's order and skipped when closer than the
# spacing to anything already selected, so the selection spreads across the processed geography. Held-out test
# candidates of every category are curated first with the full quota (only they can be featured, and adding in-sample
# data must never displace them); in-sample candidates then fill the same quotas away from them. Being held out makes
# an object a model demonstration, not a more important place. Deterministic for a given dataset.
CRATER_PER_CATEGORY = 8         # per category, for test and again for in-sample
CRATER_MIN_CONFIDENCE = 0.9     # large boxes at low confidence are the likeliest false positives
CRATER_HIGH_CONFIDENCE = 0.99
CRATER_LEGIBLE_KM = 3.0         # >= 30 px across in the 100 m/px WAC tile shown by the inspector
CRATER_SPACING_KM = 120.0      # the widest spacing that still fills every held-out category
ICE_DISCOVERIES_PER_POLE = 8
ICE_SPACING_KM = 100.0
IMP_DISCOVERIES = 16            # at most one observation per IMP observation group
IMP_SPACING_KM = 50.0
UNMATCHED_NOTE = (" No Robbins (2019) reference crater matched this detection under the IoU ≥ 0.5 rule; that does not "
                  "make it a new or previously unknown crater.")
# A registered crater whose exact box is gone but a box this close (px, center and size) exists means the same
# detection moved slightly (e.g. non-deterministic inference): fail instead of silently issuing a new public id.
CRATER_DRIFT_PX = 2.0
# Predictions on samples the model was trained or tuned on are genuine outputs, never independent evidence.
IN_SAMPLE = {
    "train": " This is a training sample: the model saw this area during training, so the prediction is not independent "
             "evidence of model performance.",
    "validation": " This is a validation sample used during model development, so the prediction is not independent "
                  "evidence of model performance.",
}


def region_label(lat: float, lon: float) -> str:
    side = "NEARSIDE" if abs(lon) <= 90 else "FARSIDE"
    return f"{side} · {'NORTHERN' if lat >= 0 else 'SOUTHERN'} HEMISPHERE"


def crater_identity(prov: dict, box: list[float]) -> str:
    """Model detection identity: dataset, source tile, checkpoint and the predicted box in source pixels."""
    x, y, w, h = box[:4]
    return (f"crater-detection|{prov['sourceDataset']}|{prov['sourceTile']}|{prov['checkpoint']}|"
            f"box={round(x + w / 2)},{round(y + h / 2)},{round(w)},{round(h)}")


def region_identity(prov: dict) -> str:
    """Ice patch / IMP tile identity: the fixed source region. Values may change with a checkpoint; the URL does not."""
    return f"{prov['task']}|{prov['sourceDataset']}|{prov['sourceTile']}"


def _box_of(key: str) -> tuple[int, ...]:
    return tuple(int(v) for v in key.rsplit("box=", 1)[1].split(","))


def _check_crater_drift(registry: DiscoveryRegistry, regions: list[dict], crater_boxes: Callable[[dict], list]) -> None:
    for region in regions:
        prov = region["provenance"]
        current = {crater_identity(prov, box) for box in crater_boxes(region)}
        tile_prefix = crater_identity(prov, [0, 0, 0, 0]).rsplit("box=", 1)[0]
        for entry in registry.entries_with_prefix(tile_prefix):
            if entry["key"] in current:
                continue
            old = _box_of(entry["key"])
            near = [key for key in current if max(abs(a - b) for a, b in zip(_box_of(key), old)) <= CRATER_DRIFT_PX]
            if near:
                raise RegistryError(f"{entry['id']} ({entry['key']}) is missing but {near[0]} is within "
                                    f"{CRATER_DRIFT_PX} px: identity drift, investigate before rebuilding the catalog")


def _discovery(dtype, object_id, title, region, lat, lon, metrics, description, score, prov, curation):
    """A discovery without its public id; build_discoveries fills id and cataloguedAt from the registry."""
    return {
        "id": None,
        "type": dtype,
        "curation": curation,
        "objectId": object_id,
        "featured": prov["datasetSplit"] == "test",
        "title": title,
        "region": region,
        "latitude": lat,
        "longitude": lon,
        "metrics": metrics,
        "description": description,
        "score": score,
        "provenance": prov,
        "cataloguedAt": None,
    }


def spaced(candidates: list, coordinate: Callable, spacing_km: float, limit: int, taken: list = ()) -> list:
    """Greedy geographic spread: keep candidates in the given order unless closer than the spacing to a kept one
    (or to one already `taken`)."""
    kept: list = []
    for candidate in candidates:
        point = coordinate(candidate)
        if all(distance_m(point, coordinate(k)) >= spacing_km * 1000 for k in (*taken, *kept)):
            kept.append(candidate)
            if len(kept) == limit:
                break
    return kept


def curate(categories: list[tuple[str, list]], split: Callable, coordinate: Callable, spacing_km: float,
           limit: int) -> list[tuple[str, object]]:
    """(category, candidate) picks: every category's test candidates first, then its in-sample candidates, each pass
    capped at `limit` and spaced away from everything already picked."""
    picked: list[tuple[str, object]] = []
    for held_out in (True, False):
        for name, candidates in categories:
            pool = [c for c in candidates if (split(c) == "test") == held_out]
            picked += [(name, c) for c in spaced(pool, coordinate, spacing_km, limit, [c for _, c in picked])]
    return picked


def build_discoveries(regions: list[dict], craters: dict, sectors: list[dict], imp_regions: list[dict],
                      registry: DiscoveryRegistry, crater_boxes: Callable[[dict], list], date: str,
                      note: str = "", imp_groups: list[dict] | None = None) -> list[dict]:
    """Curate discoveries and give each the registry id of its scientific identity (new identities get new ids)."""
    _check_crater_drift(registry, regions, crater_boxes)
    selected: list[tuple[str, dict]] = []  # (scientific identity key, discovery without id)
    split_of = lambda obj: obj["provenance"]["datasetSplit"]  # noqa: E731
    by_id = {r["id"]: r for r in regions}
    point = lambda lat, lon: {"latitude": lat, "longitude": lon}  # noqa: E731

    # Craters: three transparent categories over confident, non-truncated detections, spread across the tiles.
    # tuple = [lat, lon, diameterKm, confidence, truncated, matched]; ties keep catalogue order (region id, index).
    confident = [(region_id, index, t) for region_id, tuples in sorted(craters.items()) for index, t in enumerate(tuples)
                 if not t[4] and t[3] >= CRATER_MIN_CONFIDENCE]
    by_confidence = lambda c: -c[2][3]  # noqa: E731
    legible = [c for c in confident if c[2][2] >= CRATER_LEGIBLE_KM]
    categories = [
        ("large", sorted(confident, key=lambda c: (-c[2][2], -c[2][3]))),
        ("high-confidence", sorted((c for c in legible if c[2][3] >= CRATER_HIGH_CONFIDENCE), key=by_confidence)),
        ("unmatched-high-confidence", sorted((c for c in legible if not c[2][5]), key=by_confidence)),
    ]
    crater_split = lambda c: by_id[c[0]]["provenance"]["datasetSplit"]  # noqa: E731
    for curation, (region_id, index, t) in curate(categories, crater_split, lambda c: point(c[2][0], c[2][1]),
                                                  CRATER_SPACING_KM, CRATER_PER_CATEGORY):
        lat, lon, diameter, confidence, _, matched = t
        prov = by_id[region_id]["provenance"]
        selected.append((crater_identity(prov, crater_boxes(by_id[region_id])[index]), _discovery(
            "CRATER_CANDIDATE", f"{region_id}#{index}", "CRATER CANDIDATE", region_label(lat, lon), lat, lon,
            [
                {"key": "confidence", "label": "CONFIDENCE", "value": confidence, "origin": "model_prediction"},
                {"key": "diameter", "label": "DIAMETER", "value": diameter, "unit": "km", "origin": "model_prediction"},
                {"key": "referenceMatch", "label": "REFERENCE MATCH", "value": bool(matched), "origin": "derived_metric"},
            ],
            f"{note}The NASA-IBM LFM crater detection model predicts a crater candidate about {diameter:.1f} km across "
            f"with confidence {confidence:.2f}. It is a model detection, not a reference catalog entry."
            + ("" if matched else UNMATCHED_NOTE),
            None, prov, curation,  # no interest score: a diameter rank of curated craters saturates (P99.88-P100)
        )))

    # Ice: highest raw P90 patches per pole, spread by discovery anchor.
    for pole in ("south", "north"):
        # Patches without an edge-safe anchor (thin slivers of coverage) have no defensible discovery location.
        ranked = sorted((s for s in sectors if s["pole"] == pole and s["derived"]["discoveryAnchor"] is not None),
                        key=lambda s: (-s["derived"]["score"]["value"], s["id"]))
        anchor_of = lambda s: s["derived"]["discoveryAnchor"]  # noqa: E731
        for _, s in curate([("highest-prospectivity", ranked)], split_of, anchor_of, ICE_SPACING_KM,
                           ICE_DISCOVERIES_PER_POLE):
            raw, peak, anchor = s["prediction"]["raw"], s["prediction"]["rawPeak"], s["derived"]["discoveryAnchor"]
            metrics = [
                {"key": "rawPeak", "label": "RAW PEAK PROSPECTIVITY", "value": peak["value"], "origin": "model_prediction"},
                {"key": "rawP90", "label": "RAW P90 PROSPECTIVITY", "value": raw["p90"], "origin": "model_prediction"},
            ]
            if "permanentShadow" in s["source"]["atAnchor"]:
                metrics.append({"key": "psr", "label": "PSR AT ANCHOR", "value": s["source"]["atAnchor"]["permanentShadow"],
                                "origin": "source_observation"})
            selected.append((region_identity(s["provenance"]), _discovery(
                "HIGH_PROSPECTIVITY_REGION", s["id"], "HIGH PROSPECTIVITY REGION", f"{s['pole'].upper()} POLAR REGION",
                anchor["latitude"], anchor["longitude"], metrics,
                f"{note}The NASA-IBM LFM ice prospectivity model predicts a raw peak prospectivity of {peak['value']:.3f} "
                f"(raw P90 {raw['p90']:.3f}) against a 0–1 target scale for this polar patch. It is an unconstrained "
                "regression prediction, not a measurement or probability of ice. The marked location is a MOONATLAS "
                "edge-safe anchor, not the model peak.",
                s["derived"]["score"], s["provenance"], "highest-prospectivity",
            )))

    # IMP: one observation per observation group (its largest predicted area), groups spread geographically.
    regions_by_id = {r["id"]: r for r in imp_regions}
    primaries = []
    for group in imp_groups if imp_groups is not None else [{"observationIds": [r["id"]]} for r in imp_regions]:
        best = next((regions_by_id[o] for o in group["observationIds"] if regions_by_id[o]["derived"]["score"]), None)
        if best:
            primaries.append(best)
    primaries.sort(key=lambda r: (-r["prediction"]["areaM2"], r["id"]))
    for _, r in curate([("largest-area", primaries)], split_of, lambda r: point(r["latitude"], r["longitude"]),
                       IMP_SPACING_KM, IMP_DISCOVERIES):
        metrics = [{"key": "area", "label": "PREDICTED AREA", "value": r["prediction"]["areaM2"], "unit": "m²",
                    "origin": "model_prediction"}]
        if r["derived"]["iou"] is not None:
            metrics.append({"key": "iou", "label": "IoU VS REFERENCE", "value": r["derived"]["iou"], "origin": "derived_metric"})
        selected.append((region_identity(r["provenance"]), _discovery(
            "IMP_CANDIDATE", r["id"], "IMP CANDIDATE", region_label(r["latitude"], r["longitude"]),
            r["latitude"], r["longitude"], metrics,
            f"{note}The NASA-IBM LFM segmentation model predicts an irregular mare patch candidate covering "
            f"{r['prediction']['areaM2']:,.0f} m² of this {r['source']['sizeM']:.0f} m NAC tile. It is a model segmentation "
            "of a volcanic feature candidate, not an active volcano.",
            r["derived"]["score"], r["provenance"], "largest-area",
        )))

    for _, item in selected:
        if item["provenance"]["datasetSplit"] in IN_SAMPLE:
            item["description"] += IN_SAMPLE[item["provenance"]["datasetSplit"]]

    keys = [key for key, _ in selected]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise RegistryError(f"two discoveries share one scientific identity: {duplicates}")
    entries = registry.assign({key: item["type"] for key, item in selected}, date)
    for key, item in selected:
        item["id"], item["cataloguedAt"] = entries[key]["id"], entries[key]["firstCatalogued"]
    return [item for _, item in selected]


def build_stats(regions: list[dict], craters: dict, sectors: list[dict], imp_regions: list[dict], imp_groups: list[dict],
                discoveries: list[dict], reference_boxes: Callable[[dict], int]) -> dict:
    objects = [*regions, *sectors, *imp_regions]
    crater_count = sum(len(v) for v in craters.values())
    splits = {"train": 0, "validation": 0, "test": 0, "external": 0}
    for obj in objects:
        splits[obj["provenance"]["datasetSplit"]] += 1
    scores = [o["derived"]["score"] for o in objects if o["derived"]["score"]]
    by_type = lambda t: sum(1 for d in discoveries if d["type"] == t)  # noqa: E731
    used = {o["provenance"][k] for o in objects for k in ("sourceDataset", "model")}
    return {
        "regionsAnalyzed": len(objects),
        "craterRegions": len(regions),
        "craterDetections": crater_count,  # model detections, not unique craters: overlapping tiles repeat them
        "polarPatchesAnalyzed": {
            "north": sum(1 for s in sectors if s["pole"] == "north"),
            "south": sum(1 for s in sectors if s["pole"] == "south"),
        },
        "impTilesAnalyzed": len(imp_regions),
        "impObservationGroups": len(imp_groups),
        "impCandidates": sum(1 for r in imp_regions if r["prediction"]["areaM2"] > 0),
        "modelPredictions": crater_count + len(sectors) + len(imp_regions),
        "highInterestRegions": sum(1 for s in scores if s["value"] >= 80),
        "discoveries": {
            "total": len(discoveries),
            "crater": by_type("CRATER_CANDIDATE"),
            "ice": by_type("HIGH_PROSPECTIVITY_REGION"),
            "imp": by_type("IMP_CANDIDATE"),
            "featured": sum(1 for d in discoveries if d["featured"]),
        },
        "predictionsBySplit": splits,
        "referenceAnnotations": {"craterBoxes": sum(reference_boxes(r) for r in regions), "iceMaps": len(sectors),
                                 "impMasks": len(imp_regions)},
        "datasets": sum(1 for s in SOURCES if s["kind"] == "dataset" and s["id"] in used),
        "models": sum(1 for s in SOURCES if s["kind"] == "model" and s["id"] in used),
    }
