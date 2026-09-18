"""Normalize raw IMP inference artifacts into the MOONATLAS build.

Usage: python -m moonatlas_science.steps.process_imp [--samples ID ...]   (default: the science smoke-test sample)
"""

from pathlib import Path

from moonatlas_science import imp_normalize
from moonatlas_science.args import task_samples

if __name__ == "__main__":
    for result in imp_normalize.normalize(task_samples(__doc__.splitlines()[0], "imp-segmentation")):
        print(result if isinstance(result, Path) else result["id"])
