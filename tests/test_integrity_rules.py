"""The permanent scientific rules, as tests.

Each test pins one rule that a well-meaning change could quietly break. They run on a clean clone against
examples/science-smoke and the shipped documentation.

    prediction ≠ reference · model detection ≠ confirmed crater · unmatched ≠ new crater
    prospectivity ≠ probability · mean softmax is uncalibrated · IMP candidate ≠ active volcano
    proximity ≠ coverage · train/validation ≠ held-out evidence · raw ≠ display
    derived metric ≠ upstream model output · no coverage ≠ absence of a feature
"""

from __future__ import annotations

import json
import re

import pytest

from moonatlas_science import config, contracts
from moonatlas_science.validate import _within

DOCS = config.PROJECT_ROOT / "docs"
#: Phrases that may only ever appear as something the data explicitly denies ("not an active volcano").
CLAIMS = ("confirmed crater", "discovered crater", "new crater", "verified crater",
          "probability of ice", "ice probability", "active volcano", "measurement of ice")
NEGATIONS = ("not", "never", "no", "isn't", "rather than", "instead of")


def claims_unnegated(text: str) -> list[str]:
    """Occurrences of a forbidden phrase that are not denied by a negation just before it."""
    found = []
    lowered = text.lower()
    for phrase in CLAIMS:
        start = 0
        while (at := lowered.find(phrase, start)) != -1:
            context = lowered[max(0, at - 40):at]
            if not any(word in context for word in NEGATIONS):
                found.append(f"{phrase!r} in …{lowered[max(0, at - 40):at + len(phrase) + 10]}…")
            start = at + len(phrase)
    return found


@pytest.fixture(scope="module")
def data(smoke_dataset):
    return {name: json.loads((smoke_dataset / name).read_text(encoding="utf-8"))
            for name in ("crater-regions.json", "craters.json", "ice-sectors.json", "imp-regions.json",
                         "discoveries.json")}


def test_prediction_and_reference_are_separate_groups(data):
    """A reference annotation is never merged into, or presented as, a model prediction."""
    for name in ("crater-regions.json", "ice-sectors.json", "imp-regions.json"):
        for item in data[name]:
            assert "prediction" in item and "reference" in item, item["id"]
            assert item["prediction"] != item["reference"], item["id"]
    sector = data["ice-sectors.json"][0]
    assert sector["prediction"]["heatmap"]["rawPath"] != sector["reference"]["heatmap"]["rawPath"]
    imp = data["imp-regions.json"][0]
    assert imp["prediction"]["maskPath"] != imp["reference"]["maskPath"]


def test_published_text_never_claims_a_confirmed_or_new_crater(data):
    """A model detection is a candidate; an unmatched prediction is not a discovery of a new crater."""
    assert claims_unnegated(json.dumps(data)) == []
    for discovery in data["discoveries.json"]:
        if discovery["type"] == "CRATER_CANDIDATE":
            assert "candidate" in discovery["description"].lower()
            assert "not a reference catalog entry" in discovery["description"].lower()


def test_unmatched_predictions_are_labelled_as_unmatched_not_as_new(data):
    """`referenceMatch` is a MOONATLAS derived comparison, never a claim about the Moon."""
    for discovery in data["discoveries.json"]:
        for metric in discovery["metrics"]:
            if metric["key"] == "referenceMatch":
                assert metric["origin"] == "derived_metric"
                assert isinstance(metric["value"], bool)
    curations = {d["curation"] for d in data["discoveries.json"] if d["type"] == "CRATER_CANDIDATE"}
    assert not {c for c in curations if "new" in c or "discovery" in c}, curations


def test_ice_prospectivity_is_a_score_and_never_a_probability(data):
    """Raw values stay unclipped (the regression is unconstrained); only display rasters are clipped to 0–1."""
    sector = data["ice-sectors.json"][0]
    raw = sector["prediction"]["raw"]
    assert {"min", "mean", "p90", "max"} <= set(raw)
    assert sector["prediction"]["heatmap"]["displayRange"] == [0, 1]
    assert sector["prediction"]["heatmap"]["rawPath"] != sector["prediction"]["heatmap"]["displayPath"]
    assert claims_unnegated(json.dumps(sector)) == []
    for metric in (m for d in data["discoveries.json"] if d["type"] == "HIGH_PROSPECTIVITY_REGION" for m in d["metrics"]):
        assert "probability" not in metric["label"].lower()


def test_imp_score_is_uncalibrated_and_the_candidate_is_not_a_volcano(data):
    imp = data["imp-regions.json"][0]
    assert "meanSoftmaxScore" in imp["prediction"]
    assert claims_unnegated(json.dumps(data["discoveries.json"])) == []
    for discovery in data["discoveries.json"]:
        if discovery["type"] == "IMP_CANDIDATE":
            assert "candidate" in discovery["description"].lower()
    contract = json.dumps(contracts.describe()).lower()
    assert "uncalibrated" in contract


def test_coverage_comes_from_footprints_not_from_distance(data):
    """PROXIMITY IS NOT COVERAGE: a detection belongs to a tile because it is inside its published footprint."""
    region = data["crater-regions.json"][0]
    footprint = region["source"]["footprint"]
    assert len(footprint) == 4
    for detection in data["craters.json"][region["id"]]:
        point = {"latitude": detection[0], "longitude": detection[1]} if isinstance(detection, list) else detection
        assert _within(point, footprint, 0.01), detection
    outside = {"latitude": region["latitude"] + 3, "longitude": region["longitude"] + 3}
    assert not _within(outside, footprint, 0.0)
    imp = data["imp-regions.json"][0]
    assert len(imp["source"]["footprint"]) == 4
    assert not _within({"latitude": imp["latitude"] + 0.05, "longitude": imp["longitude"]}, imp["source"]["footprint"], 0.0)


def test_splits_are_published_and_only_test_is_held_out(data):
    for name in ("crater-regions.json", "ice-sectors.json", "imp-regions.json"):
        for item in data[name]:
            split = item["provenance"]["datasetSplit"]
            assert split in contracts.SPLITS, split
            assert contracts.SPLITS[split] == "held out" or "in-sample" in contracts.SPLITS[split] or split == "external"
    assert contracts.HELD_OUT_SPLIT == "test"
    assert contracts.SPLITS["train"] == "in-sample" and contracts.SPLITS["validation"] == "in-sample"


def test_raw_and_display_rasters_are_distinct_levels(data):
    sector = data["ice-sectors.json"][0]
    for raster in (sector["prediction"]["heatmap"], sector["reference"]["heatmap"]):
        assert raster["rawPath"].endswith("-raw.webp")
        assert raster["displayPath"] is None or raster["displayPath"] != raster["rawPath"]
        assert "rawEncoding" in raster
    assert set(contracts.RASTER_LEVELS) == {"raw", "display"}


def test_derived_metrics_declare_their_origin_and_formula(data):
    origins = {m["origin"] for d in data["discoveries.json"] for m in d["metrics"]}
    assert origins <= set(contracts.ORIGINS), origins
    for item in data["ice-sectors.json"] + data["imp-regions.json"]:
        score = item["derived"].get("score")
        if score is not None:
            assert score["formula"] and score["formula"][-1].isdigit(), score  # versioned formula id
            assert 0 <= score["value"] <= 100
    for discovery in data["discoveries.json"]:
        for metric in discovery["metrics"]:
            if metric["origin"] == "derived_metric":
                assert metric["key"] not in ("confidence", "diameter", "areaM2"), metric["key"]


def test_the_documentation_states_the_permanent_rules():
    """The rules live in the docs a contributor reads, not only in code review."""
    def plain(text: str) -> str:
        return re.sub(r"[*_`]", "", text.lower())  # markdown emphasis must not hide a rule

    readme = plain((config.PROJECT_ROOT / "README.md").read_text(encoding="utf-8"))
    coverage = plain((DOCS / "coverage" / "semantics.md").read_text(encoding="utf-8"))
    for phrase in ("proximity is not coverage", "not a probability", "uncalibrated", "in-sample"):
        assert phrase in readme, phrase
    assert "no coverage" in coverage and "absence" in coverage
