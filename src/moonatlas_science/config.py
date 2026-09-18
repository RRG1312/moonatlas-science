"""Pinned upstream resources, smoke samples and local paths for the offline science pipeline.

Everything that identifies *what* was run lives here, so a processing manifest can be traced back to
exact upstream code, weights and data. Revisions are Hugging Face commit SHAs / a GitHub commit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Bump when normalization, geolocation or asset encoding changes (docs/provenance/data-contract.md › Versioning).
PROCESSING_VERSION = "science-0.3.0"
SCHEMA_VERSION = "2"

def _default_root() -> Path:
    """The source checkout when the package runs from a clone, otherwise the working directory."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").exists() else Path.cwd()


# Everything the pipeline reads or writes hangs off these three, all overridable by environment variable.
PROJECT_ROOT = Path(os.environ.get("MOONATLAS_ROOT", _default_root()))
DATA_ROOT = Path(os.environ.get("MOONATLAS_DATA_ROOT", PROJECT_ROOT / "data"))
UPSTREAM_DIR = DATA_ROOT / "upstream"
MODELS_DIR = UPSTREAM_DIR / "models"
DATASETS_DIR = UPSTREAM_DIR / "datasets"
UPSTREAM_CODE_DIR = UPSTREAM_DIR / "lfm"
ARTIFACTS_DIR = DATA_ROOT / "artifacts"
RAW_DIR = ARTIFACTS_DIR / "raw"
SANITY_DIR = ARTIFACTS_DIR / "sanity"
STAGING_DIR = ARTIFACTS_DIR / "staging"
# Separate builds (e.g. a batch preview) must never mix with the smoke build: point MOONATLAS_BUILD_DIR elsewhere.
BUILD_DIR = Path(os.environ.get("MOONATLAS_BUILD_DIR", DATA_ROOT / "build" / PROCESSING_VERSION))
# Where a validated build is published as a dataset, one folder per data mode.
PUBLISHED_DATASETS_DIR = Path(os.environ.get("MOONATLAS_PUBLISH_DIR", PROJECT_ROOT / "datasets"))
# Optional equirectangular image used only as a backdrop in diagnostic figures. No measurement depends on it,
# and none is redistributed with this repository.
_basemap = os.environ.get("MOONATLAS_BASEMAP")
GLOBE_BASEMAP = Path(_basemap) if _basemap else None

UPSTREAM_CODE = {
    "url": "https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model",
    "commit": "56b09e146fde2dd006791dcc728f30cc12d6a6b7",
}


@dataclass(frozen=True)
class HubFile:
    repo_id: str
    repo_type: str  # "model" | "dataset"
    revision: str
    filename: str
    local_path: Path
    sha256: str | None = None  # LFS sha256 published by Hugging Face (large files only)


BACKBONE_REPO = "nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model"
CRATER_REPO = "nasa-ibm-ai4science/Crater-Detection-NASA-IBM-Lunar-Foundation-Model"
ICE_REPO = "nasa-ibm-ai4science/Ice-Prospectivity-NASA-IBM-Lunar-Foundation-Model"
IMP_REPO = "nasa-ibm-ai4science/IMP-Segmentation-NASA-IBM-Lunar-Foundation-Model"
WAC_DATASET_REPO = "nasa-ibm-ai4science/Sombench-WAC-Crater-Detection"
ICE_DATASET_REPO = "nasa-ibm-ai4science/Sombench-Ice-Prospectivity-Regression"
IMP_DATASET_REPO = "nasa-ibm-ai4science/Sombench-IMP-Segmentation"

REVISIONS = {
    BACKBONE_REPO: "f29ade2b0b7273babb1ff581723816f99cb00996",
    CRATER_REPO: "f09ebeae2099c28f09bf4060f56d3e327a6192de",
    ICE_REPO: "fb0510d0d676ae7bfae9acd88219b05449c769ee",
    IMP_REPO: "10a09ffae35ef5712c40c2a1fc41ec82d4881ea6",
    WAC_DATASET_REPO: "20f800becb64a0c7ad314652a683f97f0281fe67",
    ICE_DATASET_REPO: "78040503ceaa68107829f48114774112e8ee940d",
    IMP_DATASET_REPO: "1418851011f4b1d4fd8140a8d765b17fda86ebc0",
}


def _model_file(repo: str, filename: str, local: str, sha256: str | None = None) -> HubFile:
    return HubFile(repo, "model", REVISIONS[repo], filename, MODELS_DIR / local, sha256)


@dataclass(frozen=True)
class TaskModel:
    """One released task checkpoint and the TerraTorch config shipped next to it."""

    task: str
    source_id: str  # id in moonatlas_science/sources.py
    checkpoint: HubFile
    config: HubFile


BACKBONE_FILES = [
    _model_file(BACKBONE_REPO, "backbone/config.yaml", "backbone/config.yaml"),
    _model_file(BACKBONE_REPO, "backbone/checkpoint.pt", "backbone/checkpoint.pt",
                "5843824dfa56f89552b37b5295aa865ed28777c5f2b33286db2ef81b4cf08e5b"),
]

TASK_MODELS = {
    "crater-detection": TaskModel(
        "crater-detection",
        "lfm-crater-detection",
        _model_file(CRATER_REPO, "WAC_ni_lfm_ps8_lora_s46.ckpt", "crater/WAC_ni_lfm_ps8_lora_s46.ckpt",
                    "fab00e52e68237e84c10c5c392fbeb4fc5ea5a485b89c2513744d73b4190b1e0"),
        _model_file(CRATER_REPO, "WAC_config.yaml", "crater/WAC_config.yaml"),
    ),
    "ice-prospectivity": TaskModel(
        "ice-prospectivity",
        "lfm-ice-prospectivity",
        _model_file(ICE_REPO, "ni_lfm_ps8_all_modalities_s42.ckpt", "ice/ni_lfm_ps8_all_modalities_s42.ckpt",
                    "94a64dfe7e8c892dc687ef1eeea88044eb43b2b91263151532d7a68ac0d1bbee"),
        _model_file(ICE_REPO, "config.yaml", "ice/config.yaml"),
    ),
    "imp-segmentation": TaskModel(
        "imp-segmentation",
        "lfm-imp-segmentation",
        _model_file(IMP_REPO, "ni_lfm_ps8_frozen_s44.ckpt", "imp/ni_lfm_ps8_frozen_s44.ckpt",
                    "d6377b1ae33db8b4ac27454600630333d81469cb994937c8726465d29577fa24"),
        _model_file(IMP_REPO, "config.yaml", "imp/config.yaml"),
    ),
}

# Dataset folders exactly as the upstream configs expect them under `data/`.
WAC_DIR = DATASETS_DIR / "wac_craters_dataset"
ICE_DIR = DATASETS_DIR / "prospectivity_dataset"
IMP_DIR = DATASETS_DIR / "imp_dataset"

# One held-out test sample per task for the science smoke test (all from the official test splits).
SMOKE_SAMPLES = {
    "crater-detection": "M1174144247CE_r5360_c480",
    "ice-prospectivity": "patch_0001_0002_S_80S",
    "imp-segmentation": "M1126915118RE.ech.cog__target_1980__idx_1980_p0",
}

ICE_INPUT_LAYERS = ["ASP_SIN_COS", "CUR", "DICE", "LPSR", "LPSR_DEN", "LPSR_DIS", "SLOPE", "TMAX"]
ICE_LABEL_LAYER = "PRO"

# Deployment threshold used by upstream for displayed detections (lunar_object_detection_task.py:
# "Plot uses the DEPLOYMENT threshold (0.5), not the metric's score_threshold (0.05)").
CRATER_DISPLAY_CONFIDENCE = 0.5
# Greedy one-to-one matching of predicted to reference boxes (COCO AP@50 criterion).
CRATER_MATCH_IOU = 0.5

LUNAR_RADIUS_M = 1_737_400.0
