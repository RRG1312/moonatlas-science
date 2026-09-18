"""Shared command-line conventions for the pipeline entry scripts."""

from __future__ import annotations

import argparse

from . import config

TASKS = list(config.TASK_MODELS)


def task_samples(description: str, task: str) -> list[str]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--samples", nargs="+", help="sample ids (default: the science smoke-test sample)")
    args = parser.parse_args()
    return args.samples or [config.SMOKE_SAMPLES[task]]


def tasks_arg(description: str) -> list[str]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    return parser.parse_args().tasks
