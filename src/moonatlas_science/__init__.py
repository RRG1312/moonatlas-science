"""MOONATLAS Science: run published NASA-IBM Lunar Foundation Model checkpoints and normalize their outputs.

The public API is deliberately small. Everything else in the package is implementation detail that the
pipeline steps use, and may move between releases.

    from moonatlas_science import validate_dataset, summarize_dataset, PROCESSING_VERSION

    problems = validate_dataset(Path("examples/science-smoke"))

Pipeline steps run as modules or through the ``moonatlas-science`` command, each in its own process:

    python -m moonatlas_science.steps.run_crater_inference
    moonatlas-science smoke
"""

from __future__ import annotations

from pathlib import Path

from .config import PROCESSING_VERSION, SCHEMA_VERSION

__all__ = [
    "PROCESSING_VERSION",
    "SCHEMA_VERSION",
    "validate_dataset",
    "summarize_dataset",
    "fetch_models",
    "fetch_samples",
    "__version__",
]

__version__ = PROCESSING_VERSION.removeprefix("science-")


def validate_dataset(root: Path) -> list[str]:
    """Every contract, cross-file and provenance problem in a normalized build. Empty means valid."""
    from . import validate

    return validate.validate(Path(root))


def summarize_dataset(root: Path) -> dict:
    """Counts and data mode of a normalized build, read from the build itself."""
    from . import validate

    return validate.summary(Path(root))


def fetch_models(include_backbone: bool = True) -> list[Path]:
    """Download the pinned checkpoints and task configs, verifying published sha256 checksums."""
    from . import hub

    return hub.fetch_models(include_backbone=include_backbone)


def fetch_samples(task: str, samples: list[str]) -> None:
    """Download the named SomBench samples for one task at the pinned dataset revision."""
    from . import hub

    fetchers = {
        "crater-detection": hub.fetch_crater_samples,
        "ice-prospectivity": hub.fetch_ice_samples,
        "imp-segmentation": hub.fetch_imp_samples,
    }
    if task not in fetchers:
        raise ValueError(f"unknown task {task!r}: expected one of {', '.join(sorted(fetchers))}")
    fetchers[task](samples)
