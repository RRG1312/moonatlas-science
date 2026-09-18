"""The published conformance fixtures must match this implementation, and the rules they encode.

A consumer in another language (the MOONATLAS web application is one) checks its own geometry against
examples/conformance/geometry.json. If this test fails, either the geometry changed — which needs a
processingVersion decision — or the fixtures need regenerating with `python scripts/build_conformance.py`.
"""

from __future__ import annotations

import json

import pytest

from moonatlas_science import config, projection
from moonatlas_science.validate import _within  # bounding-box containment, see docs/coverage/semantics.md

FIXTURE = config.PROJECT_ROOT / "examples" / "conformance" / "geometry.json"


@pytest.fixture(scope="module")
def fixtures():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def objects(smoke_dataset):
    items = {}
    for name in ("crater-regions.json", "imp-regions.json", "ice-sectors.json"):
        for item in json.loads((smoke_dataset / name).read_text(encoding="utf-8")):
            items[item["id"]] = item
    return items


def test_pixel_centres_match_the_published_projection(fixtures, objects):
    for case in fixtures["pixelCentres"]:
        source = objects[case["objectId"]]["source"]
        latitude, longitude = projection.pixel_to_latlon(
            source["projection"], source["grid"], case["col"] + 0.5, case["row"] + 0.5
        )
        assert latitude == pytest.approx(case["latitude"], abs=1e-9), case
        assert longitude == pytest.approx(case["longitude"], abs=1e-9), case


def test_footprint_containment_matches(fixtures, objects):
    for case in fixtures["footprintContainment"]:
        footprint = objects[case["objectId"]]["source"]["footprint"]
        assert bool(_within(case["point"], footprint, 0.0)) == case["inside"], case


def test_raster_encodings_are_published_with_their_range(fixtures, objects):
    for case in fixtures["rasterEncodings"]:
        raster = objects[case["objectId"]]["prediction"]["heatmap"]
        assert raster["rawEncoding"] == case["encoding"]
        assert raster["displayRange"] == case["displayRange"]
        stats = case["rawStats"]
        assert stats["min"] <= stats["mean"] <= stats["max"] and stats["p90"] <= stats["max"]


def test_the_fixture_was_generated_from_this_build(fixtures):
    assert fixtures["generatedFrom"]["processingVersion"] == config.PROCESSING_VERSION
