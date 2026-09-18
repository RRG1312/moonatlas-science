"""Pinned Hugging Face downloads with integrity checks (models by LFS sha256, everything by revision)."""

from __future__ import annotations

from pathlib import Path

from . import config
from .official import sha256_file

WAC_SPLIT_FILES = {"test": "test.json", "validation": "val.json", "train": "train.json"}


def fetch(repo_id: str, repo_type: str, filename: str, local_dir: Path, sha256: str | None = None) -> Path:
    """Downloads `filename` at the pinned revision into local_dir/filename (skips a verified local copy)."""
    from huggingface_hub import hf_hub_download

    target = local_dir / filename
    marker = target.with_name(target.name + ".sha256")
    if target.exists() and sha256 is None:
        return target
    if target.exists() and marker.exists() and marker.read_text().strip() == f"{sha256} {target.stat().st_size}":
        return target
    if not target.exists():
        print(f"  downloading {repo_id}/{filename}")
        hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=filename, revision=config.REVISIONS[repo_id],
                        local_dir=local_dir)
    if sha256 is not None:
        print(f"  verifying sha256 of {target.name}")
        actual = sha256_file(target)
        if actual != sha256:
            target.unlink()
            raise RuntimeError(f"{filename}: sha256 {actual} does not match the pinned {sha256}; file removed")
        marker.write_text(f"{sha256} {target.stat().st_size}")
    return target


def fetch_models(include_backbone: bool = True) -> list[Path]:
    files = [*config.BACKBONE_FILES] if include_backbone else [config.BACKBONE_FILES[0]]
    for model in config.TASK_MODELS.values():
        files += [model.config, model.checkpoint]
    paths = []
    for hub_file in files:
        local_dir = hub_file.local_path
        for _ in Path(hub_file.filename).parts:
            local_dir = local_dir.parent
        paths.append(fetch(hub_file.repo_id, "model", hub_file.filename, local_dir, hub_file.sha256))
    return paths


def fetch_crater_samples(samples: list[str]) -> None:
    import pandas as pd

    repo = config.WAC_DATASET_REPO
    fetch(repo, "dataset", "metadata.parquet", config.WAC_DIR)
    metadata = pd.read_parquet(config.WAC_DIR / "metadata.parquet")
    splits = {row["WAC_VIS_TILE"].rsplit("/", 1)[-1].removesuffix(".nc"): row["DATASET"] for _, row in metadata.iterrows()}
    for sample in samples:
        split = {"val": "validation"}.get(splits[sample], splits[sample])
        fetch(repo, "dataset", WAC_SPLIT_FILES[split], config.WAC_DIR)
        fetch(repo, "dataset", f"images_tiff/{sample}.tif", config.WAC_DIR)


def fetch_ice_samples(samples: list[str]) -> None:
    repo = config.ICE_DATASET_REPO
    for name in ("band_statistics.json", "train_filtered.txt", "val_filtered.txt", "test_filtered.txt"):
        fetch(repo, "dataset", name, config.ICE_DIR)
    for sample in samples:
        for layer in [*config.ICE_INPUT_LAYERS, config.ICE_LABEL_LAYER]:
            fetch(repo, "dataset", f"{sample}_{layer}.tif", config.ICE_DIR)


def fetch_imp_samples(samples: list[str]) -> None:
    repo = config.IMP_DATASET_REPO
    for name in ("train.txt", "val.txt", "test.txt"):
        fetch(repo, "dataset", name, config.IMP_DIR)
    for sample in samples:
        for kind in ("img", "mask"):
            fetch(repo, "dataset", f"all/{sample}_{kind}.tif", config.IMP_DIR)


FETCHERS = {
    "crater-detection": fetch_crater_samples,
    "ice-prospectivity": fetch_ice_samples,
    "imp-segmentation": fetch_imp_samples,
}
