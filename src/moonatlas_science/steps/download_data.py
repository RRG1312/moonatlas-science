"""Download SomBench split files, metadata and the requested samples at the pinned dataset revisions.

Usage:
  python -m moonatlas_science.steps.download_data                  # the three science smoke-test samples (~20 MB)
  python -m moonatlas_science.steps.download_data --task crater-detection --samples M1174144247CE_r5360_c480
"""

import argparse

from moonatlas_science import config, hub

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=list(config.TASK_MODELS))
    parser.add_argument("--samples", nargs="+")
    args = parser.parse_args()
    if bool(args.task) != bool(args.samples):
        parser.error("--task and --samples go together")
    requests = {args.task: args.samples} if args.task else {task: [s] for task, s in config.SMOKE_SAMPLES.items()}
    for task, samples in requests.items():
        print(f"{task}: {', '.join(samples)}")
        hub.FETCHERS[task](samples)
    print("ok")
