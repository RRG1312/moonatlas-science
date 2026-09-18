"""The ``moonatlas-science`` command: a thin dispatcher over the pipeline steps.

Each command maps to one proven step module and runs it in its own process, exactly as the pipeline does on
its own (model frameworks do not survive being re-initialised in one process). Nothing here reimplements a
step: `--help` of a step module remains authoritative.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from . import config

# command -> (module, help). Reports are reached through `moonatlas-science report <name>`.
COMMANDS = {
    "fetch-models": ("moonatlas_science.steps.download_models", "download the pinned checkpoints and task configs"),
    "fetch-data": ("moonatlas_science.steps.download_data", "download SomBench splits, metadata and samples"),
    "infer": (None, "run one task's inference (see --task)"),
    "normalize": (None, "normalize one task's raw artifacts into the build (see --task)"),
    "mosaic": ("moonatlas_science.steps.build_ice_mosaic", "compose the per-pole ice mosaics"),
    "catalog": ("moonatlas_science.steps.build_catalog", "build observation groups, discoveries and global stats"),
    "manifest": ("moonatlas_science.steps.build_manifest", "write sources, manifest and processing manifest"),
    "validate": ("moonatlas_science.steps.validate_data", "validate a normalized build against the data contract"),
    "smoke": ("moonatlas_science.steps.smoke_test", "one held-out sample per task, end to end"),
    "sanity": ("moonatlas_science.steps.sanity_report", "sanity figures and report for inferred samples"),
    "publish": ("moonatlas_science.steps.build_dataset", "publish a validated build as a dataset"),
    "package": ("moonatlas_science.steps.package_artifact", "pack a published dataset into a reproducible artifact"),
}
TASK_MODULES = {
    "infer": {
        "crater-detection": "moonatlas_science.steps.run_crater_inference",
        "ice-prospectivity": "moonatlas_science.steps.run_ice_inference",
        "imp-segmentation": "moonatlas_science.steps.run_imp_inference",
    },
    "normalize": {
        "crater-detection": "moonatlas_science.steps.process_craters",
        "ice-prospectivity": "moonatlas_science.steps.process_ice",
        "imp-segmentation": "moonatlas_science.steps.process_imp",
    },
}
REPORTS = ["coverage_inventory", "crater_batch_report", "ice_batch_report", "imp_batch_report", "discoveries_map"]


def _run(module: str, rest: list[str]) -> int:
    return subprocess.run([sys.executable, "-m", module, *rest]).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="moonatlas-science",
        description=__doc__.splitlines()[0],
        epilog="Every command forwards unknown arguments to the underlying step module.",
    )
    parser.add_argument("--version", action="version", version=config.PROCESSING_VERSION)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        step = sub.add_parser(name, help=help_text, add_help=False)
        if name in TASK_MODULES:
            step.add_argument("--task", required=True, choices=sorted(TASK_MODULES[name]))
    report = sub.add_parser("report", help="run a diagnostic report over a build", add_help=False)
    report.add_argument("name", choices=REPORTS)
    environment = sub.add_parser("env", help="show the resolved paths and pinned versions")

    args, rest = parser.parse_known_args(argv)
    if args.command == "env":
        del environment
        print(f"processing version : {config.PROCESSING_VERSION}")
        print(f"schema version     : {config.SCHEMA_VERSION}")
        print(f"project root       : {config.PROJECT_ROOT}")
        print(f"data root          : {config.DATA_ROOT}")
        print(f"build dir          : {config.BUILD_DIR}")
        print(f"published datasets : {config.PUBLISHED_DATASETS_DIR}")
        print(f"upstream code      : {config.UPSTREAM_CODE['url']} @ {config.UPSTREAM_CODE['commit']}")
        for repo, revision in config.REVISIONS.items():
            print(f"pinned             : {repo} @ {revision}")
        return 0
    if args.command == "report":
        return _run(f"moonatlas_science.reports.{args.name}", rest)
    if args.command in TASK_MODULES:
        return _run(TASK_MODULES[args.command][args.task], rest)
    module, _ = COMMANDS[args.command]
    assert module is not None
    return _run(module, rest)


if __name__ == "__main__":
    sys.exit(main())
