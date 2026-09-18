"""Run IMP segmentation with the released NASA-IBM LFM checkpoint (raw artifacts only).

Usage: python -m moonatlas_science.steps.run_imp_inference [--samples ID ...]   (default: the science smoke-test sample)
"""

from pathlib import Path

from moonatlas_science import imp_inference
from moonatlas_science.args import task_samples

if __name__ == "__main__":
    for result in imp_inference.run(task_samples(__doc__.splitlines()[0], "imp-segmentation")):
        print(result if isinstance(result, Path) else result["id"])
