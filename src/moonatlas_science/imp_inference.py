"""IMP segmentation inference with the released frozen-encoder checkpoint (ni_lfm_ps8_frozen_s44.ckpt)."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from . import artifacts, config
from .official import InferenceError, OutputRecorder, load_task

TASK = "imp-segmentation"


def split_of(sample: str) -> str:
    for split, name in (("test", "test.txt"), ("validation", "val.txt"), ("train", "train.txt")):
        if sample in (config.IMP_DIR / name).read_text(encoding="utf-8").split():
            return split
    raise InferenceError(f"{sample} is not listed in any upstream split file")


def stage_inputs(sample: str) -> Path:
    """Predict directory holding only this tile's NAC image (never the reference mask)."""
    directory = config.STAGING_DIR / TASK / "predict" / sample
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    source = config.IMP_DIR / "all" / f"{sample}_img.tif"
    if not source.exists():
        raise InferenceError(f"missing {source.name}; run download_data.py first")
    shutil.copy2(source, directory / source.name)
    return directory


def run(samples: list[str]) -> list[Path]:
    import torch

    model = config.TASK_MODELS[TASK]
    loaded = load_task(model, config.DATASETS_DIR, {"data.init_args.num_workers": 0, "data.init_args.batch_size": 1})
    task, datamodule, trainer = loaded.task, loaded.datamodule, loaded.trainer
    trainer.logger = None
    written = []
    recorder = OutputRecorder(task.model)
    try:
        for sample in samples:
            split = split_of(sample)
            directory = stage_inputs(sample)
            datamodule.predict_root = str(directory)  # as LightningInferenceModel.inference_on_dir
            recorder.outputs.clear()
            started = time.perf_counter()
            predictions = trainer.predict(model=task, datamodule=datamodule, return_predictions=True)
            seconds = time.perf_counter() - started
            if len(recorder.outputs) != 1 or len(predictions) != 1:
                raise InferenceError(f"{sample}: expected one forward pass, got {len(recorder.outputs)}")
            raw = recorder.outputs[0]
            logits = (raw.output if hasattr(raw, "output") else raw).numpy().astype(np.float32).squeeze(0)
            if logits.shape != (2, 256, 256):
                raise InferenceError(f"{sample}: unexpected logits shape {logits.shape}")
            if not np.isfinite(logits).all():
                raise InferenceError(f"{sample}: non-finite logits")
            # Official hard prediction (terratorch to_segmentation_prediction): argmax over classes.
            mask = logits.argmax(axis=0).astype(np.uint8)
            logits_t = torch.from_numpy(logits)
            softmax_imp = torch.softmax(logits_t, dim=0)[1].numpy().astype(np.float32)
            datamodule.setup("predict")
            model_input = datamodule.aug(datamodule.collate_fn([datamodule.predict_dataset[0]]))["image"]
            record = {
                **loaded.record,
                "split": split,
                "sourceFile": f"all/{sample}_img.tif",
                "referenceFile": f"all/{sample}_mask.tif",
                "datasetRepo": config.IMP_DATASET_REPO,
                "datasetRevision": config.REVISIONS[config.IMP_DATASET_REPO],
                "inferenceSeconds": round(seconds, 3),
                "classes": ["background", "irregular mare patch"],
                "outputSemantics": (
                    "logits per class; mask = argmax (upstream postprocessing); softmax_imp is the class-1 softmax "
                    "score, uncalibrated and not a probability"
                ),
            }
            written.append(
                artifacts.write_run(
                    TASK,
                    sample,
                    record,
                    model_input.squeeze(0).numpy().astype(np.float32),
                    {"logits": logits, "mask": mask, "softmax_imp": softmax_imp},
                )
            )
    finally:
        recorder.close()
    return written
