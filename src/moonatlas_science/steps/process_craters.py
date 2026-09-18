"""Normalize raw crater inference artifacts into the MOONATLAS build.

Usage: python -m moonatlas_science.steps.process_craters [--samples ID ...]   (default: the science smoke-test sample)
"""

from pathlib import Path

from moonatlas_science import crater_normalize
from moonatlas_science.args import task_samples

if __name__ == "__main__":
    for result in crater_normalize.normalize(task_samples(__doc__.splitlines()[0], "crater-detection")):
        print(result if isinstance(result, Path) else result["id"])
