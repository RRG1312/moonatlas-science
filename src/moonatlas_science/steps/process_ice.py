"""Normalize raw ice inference artifacts into the MOONATLAS build.

Usage: python -m moonatlas_science.steps.process_ice [--samples ID ...]   (default: the science smoke-test sample)
"""

from pathlib import Path

from moonatlas_science import ice_normalize
from moonatlas_science.args import task_samples

if __name__ == "__main__":
    for result in ice_normalize.normalize(task_samples(__doc__.splitlines()[0], "ice-prospectivity")):
        print(result if isinstance(result, Path) else result["id"])
