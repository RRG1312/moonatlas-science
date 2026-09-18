"""Regenerate examples/conformance/ from the bundled example build.

The fixtures let any consumer — in any language — check that its own geometry agrees with this pipeline:
projected pixel centres, footprint containment and the raster encoding. Run after changing geometry or the
example build, and commit the result:

    python scripts/build_conformance.py
"""

from __future__ import annotations

import json

from moonatlas_science import config, projection

ROOT = config.PROJECT_ROOT
EXAMPLE = ROOT / "examples" / "science-smoke"
OUT = ROOT / "examples" / "conformance"


def load(name: str):
    return json.loads((EXAMPLE / name).read_text(encoding="utf-8"))


def pixel_cases(item: dict, kind: str) -> list[dict]:
    """lat/lon of a few pixel centres, from the object's own published projection and grid."""
    source = item["source"]
    grid, proj = source["grid"], source["projection"]
    cases = []
    for col, row in ((0, 0), (grid["widthPx"] // 2, grid["heightPx"] // 2), (grid["widthPx"] - 1, grid["heightPx"] - 1)):
        x = grid["xMinM"] + (col + 0.5) * grid["pixelSizeXM"]
        y = grid["yMaxM"] - (row + 0.5) * grid["pixelSizeYM"]
        latitude, longitude = projection.inverse(proj, x, y)
        cases.append({"objectId": item["id"], "kind": kind, "col": col, "row": row,
                      "latitude": round(latitude, 9), "longitude": round(longitude, 9)})
    return cases


def containment_cases(item: dict, kind: str) -> list[dict]:
    """Points inside and outside a published footprint: coverage is a footprint test, never a distance.

    The cases are far enough from the edge that the footprint's exact quadrilateral and its bounding box agree,
    so a consumer implementing either test passes (docs/coverage/semantics.md records the difference).
    """
    footprint = item["source"]["footprint"]
    centre = {"latitude": item["latitude"], "longitude": item["longitude"]}
    span = max(abs(c["latitude"] - centre["latitude"]) for c in footprint)
    return [
        {"objectId": item["id"], "kind": kind, "point": centre, "inside": True},
        {"objectId": item["id"], "kind": kind,
         "point": {"latitude": round(centre["latitude"] + span * 4, 9), "longitude": centre["longitude"]},
         "inside": False},
        {"objectId": item["id"], "kind": kind,
         "point": {"latitude": centre["latitude"], "longitude": round(centre["longitude"] + span * 4, 9)},
         "inside": False},
    ]


def main() -> int:
    regions, imp, sectors = load("crater-regions.json"), load("imp-regions.json"), load("ice-sectors.json")
    document = {
        "generatedFrom": {"example": "examples/science-smoke", "processingVersion": load("manifest.json")["processingVersion"]},
        "note": "Expected values produced by moonatlas_science.projection from each object's own published "
                "projection and grid. A consumer that reimplements the geometry must reproduce them.",
        "pixelCentres": [c for item in regions for c in pixel_cases(item, "crater-region")]
                        + [c for item in imp for c in pixel_cases(item, "imp-region")],
        "footprintContainment": [c for item in regions for c in containment_cases(item, "crater-region")]
                                + [c for item in imp for c in containment_cases(item, "imp-region")],
        "rasterEncodings": [
            {"objectId": sector["id"], "raster": "prediction.heatmap", "encoding": sector["prediction"]["heatmap"]["rawEncoding"],
             "displayRange": sector["prediction"]["heatmap"]["displayRange"], "rawStats": sector["prediction"]["raw"]}
            for sector in sectors
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "geometry.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{OUT / 'geometry.json'}: {len(document['pixelCentres'])} pixel centres, "
          f"{len(document['footprintContainment'])} containment cases, {len(document['rasterEncodings'])} encodings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
