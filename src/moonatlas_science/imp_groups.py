"""IMP observation groups: the repeat NAC observations of one SomBench target (docs/provenance/data-contract.md › IMPObservationGroup).

An **observation** is one NAC tile and its model output. SomBench frames each sample around a published IMP annotation
and names it `<PRODUCT>.ech.cog__target_<id>__idx_<id>_p0`; tiles that share the target id share the same 256 m ground
footprint and differ only in which NAC image they come from. That upstream id is the only identity here that does not
depend on MOONATLAS or on the model, so it defines the group.

What was rejected, and why (audit of the complete catalogue, 130 observations):
  * Centre distance (single linkage, 5 km) chained observations up to 6.6 km apart.
  * Tile-footprint overlap is not evidence of one physical feature: of 61 overlapping pairs with different targets, 18
    share less than 5% of their reference IMP area (median 0.16), i.e. they image neighbouring but distinct annotations.
  * The Hargitai et al. (2025) reference masks cannot arbitrate either: even two observations of the *same* target agree
    on only 0.49 of their reference area (median), matching the documented tens-of-metres annotation misalignment.
So a group is **repeat observations of one annotated target**, not a claim that a physical IMP feature is identical:
the product term stays neutral (observation group). Grouping never merges or alters segmentations.
Pure Python so the fixture generator and the pipeline share it.
"""

from __future__ import annotations

import itertools
import math
import re

HALF_TILE_DIAGONAL_M = 256 * math.sqrt(2) / 2
LUNAR_RADIUS_M = 1_737_400.0
TARGET_PATTERN = re.compile(r"target_(\d+)")


def distance_m(a: dict, b: dict) -> float:
    p1, p2 = math.radians(a["latitude"]), math.radians(b["latitude"])
    dl = math.radians(b["longitude"] - a["longitude"])
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * LUNAR_RADIUS_M * math.asin(math.sqrt(min(1.0, h)))


def mean_coordinate(points: list[dict]) -> dict:
    """Spherical mean (unit-vector average), safe across the antimeridian."""
    x = y = z = 0.0
    for p in points:
        lat, lon = math.radians(p["latitude"]), math.radians(p["longitude"])
        x += math.cos(lat) * math.cos(lon)
        y += math.cos(lat) * math.sin(lon)
        z += math.sin(lat)
    lon = math.degrees(math.atan2(y, x))
    lat = math.degrees(math.atan2(z, math.hypot(x, y)))
    lon = round(((lon + 180) % 360) - 180, 6)
    return {"latitude": round(lat, 6), "longitude": -180.0 if lon >= 180 else lon}


def target_id(region: dict) -> str:
    match = TARGET_PATTERN.search(region["provenance"]["sourceTile"])
    if not match:
        raise ValueError(f"{region['id']}: source tile {region['provenance']['sourceTile']} has no SomBench target id")
    return match.group(1)


def footprint(region: dict) -> tuple:
    """(projection definition, tile grid box): identical for every observation of one target."""
    g = region["source"]["grid"]
    x0, y1 = g["xMinM"], g["yMaxM"]
    box = (round(x0, 3), round(x0 + g["widthPx"] * abs(g["pixelSizeXM"]), 3),
           round(y1 - g["heightPx"] * abs(g["pixelSizeYM"]), 3), round(y1, 3))
    return tuple(sorted(region["source"]["projection"].items())), box


def span_m(members: list[dict]) -> float:
    return max((distance_m(a, b) for a, b in itertools.combinations(members, 2)), default=0.0)


def group_observations(regions: list[dict]) -> list[dict]:
    """Groups for IMP regions, deterministic for a given set of regions (input order does not matter)."""
    groups: dict[str, list[dict]] = {}
    for region in sorted(regions, key=lambda r: r["id"]):
        groups.setdefault(target_id(region), []).append(region)

    out = []
    for target, members in groups.items():
        footprints = {footprint(r) for r in members}
        if len(footprints) > 1:  # the upstream id would no longer mean "one framed target": stop instead of guessing
            raise ValueError(f"target {target} spans {len(footprints)} different tile footprints: {[r['id'] for r in members]}")
        center = mean_coordinate(members)
        observations = sorted(members, key=lambda r: (-r["prediction"]["areaM2"], r["id"]))
        splits = {"train": 0, "validation": 0, "test": 0, "external": 0}
        for r in members:
            splits[r["provenance"]["datasetSplit"]] += 1
        first = min(r["id"] for r in members)  # "imp-<product>-t<target>-i<index>" or "mock-imp-…" in fixtures
        mock = "mock-" if first.startswith("mock-") else ""
        out.append({
            "id": f"{mock}imp-group-t{target}",
            **center,
            "sourceTarget": target,
            "radiusM": round(max(distance_m(center, r) for r in members) + HALF_TILE_DIAGONAL_M, 1),
            "observationIds": [r["id"] for r in observations],
            "splits": splits,
        })
    return sorted(out, key=lambda g: g["id"])


if __name__ == "__main__":
    PROJ = {"kind": "transverse-mercator", "centralMeridianDeg": 33.0, "radiusM": LUNAR_RADIUS_M}

    def region(product, target, lat, lon, area, split="train", x_min=0.0):
        grid = {"xMinM": x_min, "yMaxM": 1000.0, "pixelSizeXM": 1.0, "pixelSizeYM": -1.0, "widthPx": 256, "heightPx": 256}
        return {"id": f"imp-{product}-t{target}-i{target}", "latitude": lat, "longitude": lon,
                "prediction": {"areaM2": area}, "provenance": {"datasetSplit": split,
                "sourceTile": f"{product.upper()}.ech.cog__target_{target}__idx_{target}_p0_img.tif"},
                "source": {"grid": grid, "projection": PROJ}}

    same = [region("m1le", 1266, 4.342, 33.7, 10), region("m2le", 1266, 4.342, 33.7, 20, "test")]
    neighbour = region("m3re", 1267, 4.3437, 33.7, 5, x_min=200.0)  # overlaps the first two, different annotation
    far = region("m4re", 90, -25.65, -27.62, 7)
    groups = group_observations([far, neighbour, *same[::-1]])
    assert [g["id"] for g in groups] == ["imp-group-t1266", "imp-group-t1267", "imp-group-t90"]
    assert groups[0]["observationIds"] == ["imp-m2le-t1266-i1266", "imp-m1le-t1266-i1266"]  # largest predicted area first
    assert groups[0]["splits"]["test"] == 1 and groups[0]["sourceTarget"] == "1266"
    assert span_m(same) == 0.0 and groups == group_observations([*same, neighbour, far])
    moved = {**same[1], "source": {**same[1]["source"], "grid": {**same[1]["source"]["grid"], "xMinM": 50.0}}}
    try:
        group_observations([same[0], moved])
        raise AssertionError("one target with two footprints must fail")
    except ValueError:
        pass
    across = mean_coordinate([{"latitude": 0, "longitude": 179.99}, {"latitude": 0, "longitude": -179.99}])
    assert abs(abs(across["longitude"]) - 180) < 1e-6
    print("ok")
