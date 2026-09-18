"""The data contract: the descriptor, the validator and a published build must agree.

Runs on a clean clone against examples/science-smoke, so a contributor sees a contract break immediately.
"""

from __future__ import annotations

import inspect
import json

import pytest

from moonatlas_science import build, config, contracts, summarize_dataset, validate_dataset

SCHEMA_FILE = "schemas/contract-v2.json"


@pytest.fixture(scope="module")
def dataset(smoke_dataset):
    return {name: json.loads((smoke_dataset / name).read_text(encoding="utf-8"))
            for name in ("manifest.json", "sources.json", "crater-regions.json", "ice-sectors.json",
                         "imp-regions.json", "discoveries.json", "global-stats.json")}


def test_the_descriptor_lists_the_provenance_fields_the_pipeline_writes():
    written = inspect.getsource(build.provenance)
    for field in contracts.PROVENANCE_FIELDS:
        assert f'"{field}"' in written, field


def test_every_published_object_carries_full_provenance(dataset):
    for name in ("crater-regions.json", "ice-sectors.json", "imp-regions.json", "discoveries.json"):
        for item in dataset[name]:
            missing = set(contracts.PROVENANCE_FIELDS) - set(item["provenance"])
            assert not missing, f"{name} {item['id']}: missing {sorted(missing)}"
            assert item["provenance"]["processingVersion"] == dataset["manifest.json"]["processingVersion"]
            assert item["provenance"]["datasetSplit"] in contracts.SPLITS


def test_ids_follow_the_declared_prefixes(dataset):
    for name, prefix in (("crater-regions.json", "crater-region"), ("ice-sectors.json", "ice-sector"),
                         ("imp-regions.json", "imp-region")):
        for item in dataset[name]:
            assert item["id"].startswith(contracts.ID_PREFIXES[prefix]), item["id"]
    for discovery in dataset["discoveries.json"]:
        kind = {"CRATER_CANDIDATE": "crater-discovery", "HIGH_PROSPECTIVITY_REGION": "ice-discovery",
                "IMP_CANDIDATE": "imp-discovery"}[discovery["type"]]
        assert discovery["id"].startswith(contracts.ID_PREFIXES[kind]), discovery["id"]


def test_no_synthetic_object_reaches_a_published_build(dataset):
    text = json.dumps(dataset)
    for marker in contracts.SYNTHETIC_MARKERS:
        assert marker not in text, marker


def test_every_contract_file_exists_in_the_example(smoke_dataset):
    for name in contracts.FILES:
        if "<" in name or "…" in name:  # per-object files and asset folders
            continue
        assert (smoke_dataset / name).exists(), name
    assert (smoke_dataset / "crater-tiles").is_dir()
    assert (smoke_dataset / "previews").is_dir() and (smoke_dataset / "overlays").is_dir()


def test_the_example_build_validates_as_a_published_dataset(smoke_dataset):
    problems = validate_dataset(smoke_dataset)  # full scope: raw artifacts are not shipped with a dataset
    assert all("raw inference artifact missing" in problem for problem in problems), problems
    from moonatlas_science import validate

    assert validate.validate(smoke_dataset, check_raw=False) == []


def test_the_example_summary_matches_its_own_manifest(smoke_dataset, dataset):
    facts = summarize_dataset(smoke_dataset)
    assert facts["dataMode"] == dataset["manifest.json"]["dataMode"]
    assert facts["craterRegions"] == len(dataset["crater-regions.json"])
    assert facts["icePatches"] == len(dataset["ice-sectors.json"])
    assert facts["impRegions"] == len(dataset["imp-regions.json"])
    assert facts["syntheticObjects"] == 0


def test_the_exported_schema_matches_the_descriptor():
    exported = json.loads((config.PROJECT_ROOT / SCHEMA_FILE).read_text(encoding="utf-8"))
    assert exported == contracts.describe(), f"regenerate with: python -m moonatlas_science.contracts > {SCHEMA_FILE}"
    assert exported["schemaVersion"] == config.SCHEMA_VERSION
    assert exported["processingVersion"] == config.PROCESSING_VERSION
