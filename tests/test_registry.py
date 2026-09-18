"""Stable public discovery ids: the registry and catalog must never renumber, reuse or guess."""

import json
import random

import pytest

from moonatlas_science.catalog import build_discoveries
from moonatlas_science.registry import DiscoveryRegistry, RegistryError

DATE = "2026-09-17"


def prov(task, tile, split="test"):
    dataset = {"crater-detection": "sombench-wac-crater-detection", "ice-prospectivity": "sombench-ice-prospectivity-regression",
               "imp-segmentation": "sombench-imp-segmentation"}[task]
    return {"task": task, "sourceDataset": dataset, "sourceTile": tile, "datasetSplit": split, "model": "lfm",
            "checkpoint": "model.ckpt", "modelRevision": "0" * 40, "referenceSource": None,
            "processingVersion": "science-0.3.0"}


def sector(patch, score, lon=0.0):
    # Distinct anchors: curation spaces ice discoveries geographically.
    return {"id": f"ice-{patch}", "pole": "south", "latitude": -85, "longitude": lon,
            "source": {"atAnchor": {}},
            "prediction": {"raw": {"p90": score / 100}, "rawPeak": {"value": score / 100}},
            "derived": {"score": {"value": score, "formula": "ice-p90-bounded-v2"},
                        "discoveryAnchor": {"latitude": -82, "longitude": lon}},
            "provenance": prov("ice-prospectivity", patch)}


def imp(tile, area, lon=33.0):
    return {"id": f"imp-{tile}", "latitude": 4, "longitude": lon, "source": {"sizeM": 256},
            "prediction": {"areaM2": area}, "derived": {"iou": 0.5, "score": {"value": area, "formula": "imp-area-rank-v1"}},
            "provenance": prov("imp-segmentation", tile)}


def crater_tile(tile, boxes):
    """A WAC tile whose detections are [x, y, w, h, confidence]; tuples mirror craters.json (same order)."""
    region = {"id": f"wac-{tile.lower()}", "provenance": prov("crater-detection", f"{tile}.tif")}
    tuples = [[-50.0 + 3 * i, 10.0 + 3 * i, w / 10, conf, 0, 1] for i, (_, _, w, _, conf) in enumerate(boxes)]
    return region, tuples, [list(b) for b in boxes]


def catalog(registry, sectors=(), imps=(), tiles=()):
    regions = [t[0] for t in tiles]
    craters = {t[0]["id"]: t[1] for t in tiles}
    boxes = {t[0]["id"]: t[2] for t in tiles}
    return build_discoveries(regions, craters, list(sectors), list(imps), registry, lambda r: boxes[r["id"]], DATE)


def ids_by_object(discoveries):
    return {d["objectId"]: d["id"] for d in discoveries}


SECTORS = [sector(f"patch_{i:04d}_S_80S", 50 + i, lon=60.0 * i - 180) for i in range(6)]
IMPS = [imp(f"M{i}RE", 100 + i, lon=10.0 * i) for i in range(3)]
TILES = [crater_tile("M1CE_r1_c1", [(10, 10, 40, 40, 0.95), (200, 200, 60, 60, 0.92)]),
         crater_tile("M2CE_r2_c2", [(30, 50, 20, 22, 0.95)])]


def test_reordered_upstream_data_keeps_every_id():
    first = ids_by_object(catalog(DiscoveryRegistry(), SECTORS, IMPS, TILES))
    registry = DiscoveryRegistry()
    shuffled = SECTORS[:]
    random.Random(7).shuffle(shuffled)
    assert ids_by_object(catalog(registry, shuffled, IMPS[::-1], TILES[::-1])) == first


def test_new_objects_never_renumber_existing_discoveries():
    registry = DiscoveryRegistry()
    before = ids_by_object(catalog(registry, SECTORS[:3], IMPS[:1], TILES[:1]))
    after = ids_by_object(catalog(registry, SECTORS, IMPS, TILES))
    assert {k: after[k] for k in before} == before
    assert len(set(after.values())) == len(after)


def test_removed_objects_keep_their_id_reserved():
    registry = DiscoveryRegistry()
    catalog(registry, SECTORS[:2])
    removed_id = ids_by_object(catalog(registry, SECTORS[:2]))["ice-patch_0001_S_80S"]
    later = ids_by_object(catalog(registry, [SECTORS[0], SECTORS[5]]))
    assert removed_id not in later.values()
    assert ids_by_object(catalog(registry, SECTORS[:2]))["ice-patch_0001_S_80S"] == removed_id  # it comes back as itself


def test_rebuild_is_deterministic_and_the_saved_registry_is_byte_identical(tmp_path):
    path = tmp_path / "registry.json"
    registry = DiscoveryRegistry.load(path)
    first = catalog(registry, SECTORS, IMPS, TILES)
    registry.save(path)
    saved = path.read_bytes()
    registry = DiscoveryRegistry.load(path)
    second = catalog(registry, SECTORS, IMPS, TILES)
    registry.save(path)
    assert json.dumps(first) == json.dumps(second)
    assert path.read_bytes() == saved
    assert all(d["cataloguedAt"] == DATE for d in second)


def test_identity_collisions_fail_loudly():
    duplicate_patch = [SECTORS[0], {**SECTORS[1], "provenance": SECTORS[0]["provenance"]}]
    with pytest.raises(RegistryError, match="share one scientific identity"):
        catalog(DiscoveryRegistry(), duplicate_patch)
    entry = {"id": "MM-ICE-000001", "type": "HIGH_PROSPECTIVITY_REGION", "key": "k", "firstCatalogued": DATE}
    with pytest.raises(RegistryError, match="registered as both"):
        DiscoveryRegistry([entry, {**entry, "id": "MM-ICE-000002"}])
    with pytest.raises(RegistryError, match="registered twice"):
        DiscoveryRegistry([entry, {**entry, "key": "other"}])
    with pytest.raises(RegistryError, match="requested as"):
        DiscoveryRegistry([entry]).assign({"k": "IMP_CANDIDATE"}, DATE)


def test_corrupt_or_rewritten_registries_are_rejected(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RegistryError, match="not valid JSON"):
        DiscoveryRegistry.load(path)
    with pytest.raises(RegistryError, match="malformed"):
        DiscoveryRegistry([{"id": "MM-CR-000001", "type": "IMP_CANDIDATE", "key": "k", "firstCatalogued": DATE}])
    saved = tmp_path / "saved.json"
    registry = DiscoveryRegistry()
    catalog(registry, SECTORS[:2])
    registry.save(saved)
    tampered = DiscoveryRegistry([{**e, "key": e["key"] + "-edited"} for e in registry.entries])
    with pytest.raises(RegistryError, match="would change or disappear"):
        tampered.save(saved)
    with pytest.raises(RegistryError, match="would change or disappear"):
        DiscoveryRegistry(registry.entries[:1]).save(saved)


def test_crater_identity_uses_the_box_not_the_array_position():
    registry = DiscoveryRegistry()
    tile = crater_tile("M1CE_r1_c1", [(10, 10, 40, 40, 0.95), (200, 200, 60, 60, 0.92)])
    first = catalog(registry, tiles=[tile])[0]
    region, tuples, boxes = tile
    swapped = (region, tuples[::-1], boxes[::-1])  # same detections, other order
    again = next(d for d in catalog(registry, tiles=[swapped]) if d["id"] == first["id"])
    assert again["objectId"] != first["objectId"]


def test_crater_box_drift_fails_instead_of_issuing_a_new_id():
    registry = DiscoveryRegistry()
    catalog(registry, tiles=[crater_tile("M1CE_r1_c1", [(200, 200, 60, 60, 0.93)])])
    moved = crater_tile("M1CE_r1_c1", [(201.6, 200, 60, 60, 0.93)])
    with pytest.raises(RegistryError, match="identity drift"):
        catalog(registry, tiles=[moved])


def test_fixture_registries_use_a_separate_id_range():
    discoveries = catalog(DiscoveryRegistry(number_base=900000), SECTORS[:1])
    assert discoveries[0]["id"] == "MM-ICE-900001"


def test_curation_spreads_discoveries_and_keeps_one_imp_observation_per_group():
    crowded = [sector(f"patch_{i:04d}_S_80S", 90 - i, lon=0.5 * i) for i in range(5)]  # anchors a few km apart
    assert len(catalog(DiscoveryRegistry(), crowded)) == 1
    one_target = [imp("M1RE", 500, lon=33.0), imp("M2LE", 900, lon=33.0)]  # two NAC observations of one target
    groups = [{"observationIds": ["imp-M2LE", "imp-M1RE"]}]
    discoveries = build_discoveries([], {}, [], one_target, DiscoveryRegistry(), lambda r: [], DATE, imp_groups=groups)
    assert [d["objectId"] for d in discoveries] == ["imp-M2LE"]


def test_adding_in_sample_data_never_displaces_held_out_discoveries():
    held_out = sector("patch_0001_S_80S", 50, lon=0.0)
    trained = {**sector("patch_0002_S_80S", 90, lon=0.5), "provenance": prov("ice-prospectivity", "patch_0002_S_80S", split="train")}
    alone = catalog(DiscoveryRegistry(), [held_out])
    mixed = catalog(DiscoveryRegistry(), [trained, held_out])  # the stronger training patch sits a few km away
    assert [(d["objectId"], d["featured"]) for d in mixed] == [(d["objectId"], d["featured"]) for d in alone] == [("ice-patch_0001_S_80S", True)]


def test_in_sample_discoveries_say_they_are_not_independent_evidence():
    trained = {**SECTORS[0], "provenance": prov("ice-prospectivity", "patch_0000_S_80S", split="train")}
    (discovery,) = catalog(DiscoveryRegistry(), [trained])
    assert "training sample" in discovery["description"] and not discovery["featured"]


def test_crater_categories_are_labeled_and_unmatched_never_claims_a_new_crater(monkeypatch):
    monkeypatch.setattr("moonatlas_science.catalog.CRATER_PER_CATEGORY", 1)  # one pick per category, so the second crater is not "large"
    region = {"id": "wac-m9ce_r1_c1", "provenance": prov("crater-detection", "M9CE_r1_c1.tif")}
    # [lat, lon, diameterKm, confidence, truncated, matched]: a large matched one, a confident unmatched one far away
    tuples = [[-50.0, 10.0, 40.0, 0.95, 0, 1], [20.0, 120.0, 5.0, 0.97, 0, 0]]
    boxes = [[0, 0, 400, 400, 0.95], [900, 900, 50, 50, 0.97]]
    discoveries = build_discoveries([region], {region["id"]: tuples}, [], [], DiscoveryRegistry(), lambda r: boxes, DATE)
    by_curation = {d["curation"]: d for d in discoveries}
    assert set(by_curation) == {"large", "unmatched-high-confidence"}
    note = by_curation["unmatched-high-confidence"]["description"]
    assert "No Robbins (2019) reference crater matched" in note and "not make it a new or previously unknown crater" in note
    assert all(d["score"] is None for d in discoveries)
