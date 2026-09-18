"""Run WAC crater detection with the released NASA-IBM LFM checkpoint (raw artifacts only).

Usage: python -m moonatlas_science.steps.run_crater_inference [--samples ID ...]   (default: the science smoke-test sample)
"""

from pathlib import Path

from moonatlas_science import crater_inference
from moonatlas_science.args import task_samples

if __name__ == "__main__":
    for result in crater_inference.run(task_samples(__doc__.splitlines()[0], "crater-detection")):
        print(result if isinstance(result, Path) else result["id"])
