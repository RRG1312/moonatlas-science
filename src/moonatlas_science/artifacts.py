"""Raw inference artifacts: enough to reproduce and debug every prediction (never shipped to Vercel)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config


def sample_dir(task: str, sample: str) -> Path:
    return config.RAW_DIR / task / sample


def array_summary(array: np.ndarray) -> dict:
    finite = np.isfinite(array) if np.issubdtype(array.dtype, np.floating) else np.ones(array.shape, bool)
    values = array[finite]
    summary = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
        "nonFinite": int(array.size - finite.sum()),
    }
    if values.size:
        summary.update(min=float(values.min()), max=float(values.max()), mean=float(values.mean()))
    if array.ndim >= 3:  # per channel of (…, C, H, W)
        channels = array.reshape(-1, *array.shape[-3:])[0]
        summary["channels"] = [
            {"mean": float(np.nanmean(c)), "std": float(np.nanstd(c)), "min": float(np.nanmin(c)), "max": float(np.nanmax(c))}
            for c in channels
        ]
    return summary


def write_run(task: str, sample: str, record: dict, model_input: np.ndarray, arrays: dict[str, np.ndarray]) -> Path:
    directory = sample_dir(task, sample)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "input.npy", model_input)
    np.savez_compressed(directory / "outputs.npz", **arrays)
    run = {
        **record,
        "task": task,
        "sample": sample,
        "processingVersion": config.PROCESSING_VERSION,
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": array_summary(model_input),
        "outputs": {name: array_summary(value) for name, value in arrays.items()},
    }
    (directory / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return directory


def read_run(task: str, sample: str) -> tuple[dict, dict[str, np.ndarray]]:
    directory = sample_dir(task, sample)
    run_path = directory / "run.json"
    if not run_path.exists():
        raise FileNotFoundError(f"no inference artifact for {task}/{sample}; run the inference step first")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    with np.load(directory / "outputs.npz") as data:
        arrays = {name: data[name] for name in data.files}
    return run, arrays
