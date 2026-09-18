"""Runs the released NASA-IBM task checkpoints through the official TerraTorch path.

The task (model, necks, heads, PEFT wrapping) and the datamodule (loading, no-data handling,
normalization) are instantiated from the TerraTorch YAML shipped next to each checkpoint, with
`terratorch.cli_tools.build_lightning_cli` — the same entry point as `terratorch predict/test`.
Nothing is re-implemented. Documented deviations from a plain `terratorch predict` run:

* The YAML is staged with absolute paths for its `data/`, `backbone/` and `custom_modules_path` roots
  (upstream uses relative symlinks), and the accelerator follows the available hardware.
* Weights are loaded strictly (every key must match) from the checkpoint's full `state_dict`,
  as `LightningModule.load_from_checkpoint` does. A mismatch fails the run.
* Model outputs are captured with a forward hook on `task.model` so the raw tensor is preserved
  alongside the task's own postprocessing.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata as metadata
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import config


class InferenceError(RuntimeError):
    """Any failure of real model inference. Never caught to substitute other data."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rewrite_paths(node: Any, roots: dict[str, Path]) -> Any:
    if isinstance(node, dict):
        return {key: _rewrite_paths(value, roots) for key, value in node.items()}
    if isinstance(node, list):
        return [_rewrite_paths(value, roots) for value in node]
    if isinstance(node, str):
        for prefix, root in roots.items():
            if node.startswith(prefix):
                return (root / node[len(prefix):]).as_posix()
    return node


def stage_config(task_model: config.TaskModel, datasets_root: Path, overrides: dict[str, Any]) -> tuple[Path, dict]:
    """Writes the upstream YAML with absolute roots and returns it with a record of every change."""
    source = task_model.config.local_path
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    roots = {"data/": datasets_root, "backbone/": config.MODELS_DIR / "backbone"}
    staged = _rewrite_paths(document, roots)
    staged["custom_modules_path"] = (config.UPSTREAM_CODE_DIR / "terratorch_integration").as_posix()
    for dotted, value in overrides.items():
        target = staged
        *parents, leaf = dotted.split(".")
        for key in parents:
            target = target[key]
        target[leaf] = value
    path = config.STAGING_DIR / task_model.task / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(staged, sort_keys=False), encoding="utf-8")
    record = {
        "upstreamConfig": task_model.config.filename,
        "upstreamConfigSha256": sha256_file(source),
        "pathRoots": {prefix: root.as_posix() for prefix, root in roots.items()},
        "overrides": overrides,
    }
    return path, record


@dataclass
class LoadedTask:
    task: Any
    datamodule: Any
    trainer: Any
    device: str
    record: dict = field(default_factory=dict)


def _versions() -> dict[str, str]:
    names = ["torch", "torchvision", "terratorch", "lightning", "timm", "peft", "rasterio", "pyproj", "ni_lfm"]
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def load_task(task_model: config.TaskModel, datasets_root: Path, data_overrides: dict[str, Any] | None = None) -> LoadedTask:
    import torch
    from lightning.pytorch.callbacks import BasePredictionWriter
    from terratorch.cli_tools import build_lightning_cli

    for hub_file in (task_model.checkpoint, task_model.config):
        if not hub_file.local_path.exists():
            raise InferenceError(f"missing {hub_file.local_path}; run download_models.py first")

    cuda = torch.cuda.is_available()
    if cuda:
        # Upstream configs set trainer.deterministic; cuBLAS needs this for deterministic kernels.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    overrides = {"trainer.accelerator": "gpu" if cuda else "cpu", "trainer.devices": 1, **(data_overrides or {})}
    staged, record = stage_config(task_model, datasets_root, overrides)

    started = time.perf_counter()
    cli = build_lightning_cli(["--config", str(staged), "--trainer.logger", "false"], run=False)
    task, datamodule, trainer = cli.model, cli.datamodule, cli.trainer
    trainer.callbacks = [c for c in trainer.callbacks if not isinstance(c, BasePredictionWriter)]

    checkpoint = torch.load(task_model.checkpoint.local_path, map_location="cpu", weights_only=False, mmap=True)
    state_dict = checkpoint["state_dict"]
    result = task.load_state_dict(state_dict, strict=True)  # raises on any missing or unexpected key
    device = "cuda" if cuda else "cpu"
    task.eval().to(device)

    record.update(
        {
            "checkpoint": task_model.checkpoint.filename,
            "checkpointSha256": task_model.checkpoint.sha256,
            "modelRepo": task_model.checkpoint.repo_id,
            "modelRevision": task_model.checkpoint.revision,
            "checkpointEpoch": checkpoint.get("epoch"),
            "checkpointGlobalStep": checkpoint.get("global_step"),
            "stateDictTensors": len(state_dict),
            "missingKeys": list(result.missing_keys),
            "unexpectedKeys": list(result.unexpected_keys),
            "taskClass": f"{type(task).__module__}.{type(task).__name__}",
            "datamoduleClass": f"{type(datamodule).__module__}.{type(datamodule).__name__}",
            "device": torch.cuda.get_device_name(0) if cuda else "cpu",
            "versions": _versions(),
            "upstreamCode": config.UPSTREAM_CODE,
            "loadSeconds": round(time.perf_counter() - started, 2),
        }
    )
    return LoadedTask(task, datamodule, trainer, device, record)


class OutputRecorder:
    """Forward hook that keeps a CPU copy of every `task.model` output (the raw model output)."""

    def __init__(self, module):
        self.outputs: list[Any] = []
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self.outputs.append(_to_cpu(output))

    def close(self):
        self._handle.remove()


def _to_cpu(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(v) for v in value)
    if hasattr(value, "output"):  # terratorch ModelOutput
        clone = copy.copy(value)
        clone.output = _to_cpu(value.output)
        return clone
    return value
