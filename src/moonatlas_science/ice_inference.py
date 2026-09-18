"""Polar ice prospectivity inference with the released full fine-tune (ni_lfm_ps8_all_modalities_s42.ckpt)."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from . import artifacts, config
from .official import InferenceError, OutputRecorder, load_task

TASK = "ice-prospectivity"


def split_of(sample: str) -> str:
    for split, name in (("test", "test_filtered.txt"), ("validation", "val_filtered.txt"), ("train", "train_filtered.txt")):
        if sample in (config.ICE_DIR / name).read_text(encoding="utf-8").split():
            return split
    raise InferenceError(f"{sample} is not listed in any upstream split file")


def stage_inputs(sample: str) -> Path:
    """Predict directory holding only this patch's evidential layers (never the PRO label)."""
    directory = config.STAGING_DIR / TASK / "predict" / sample
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    for layer in config.ICE_INPUT_LAYERS:
        source = config.ICE_DIR / f"{sample}_{layer}.tif"
        if not source.exists():
            raise InferenceError(f"missing input layer {source.name}; run download_data.py first")
        shutil.copy2(source, directory / source.name)
    return directory


def run(samples: list[str]) -> list[Path]:
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
            # Same as terratorch LightningInferenceModel.inference_on_dir for multimodal datamodules.
            datamodule.predict_root = {modality: str(directory) for modality in datamodule.modalities}
            recorder.outputs.clear()
            started = time.perf_counter()
            predictions = trainer.predict(model=task, datamodule=datamodule, return_predictions=True)
            seconds = time.perf_counter() - started
            if len(recorder.outputs) != 1 or len(predictions) != 1:
                raise InferenceError(f"{sample}: expected one forward pass, got {len(recorder.outputs)}")
            raw = recorder.outputs[0]
            raw = (raw.output if hasattr(raw, "output") else raw).numpy().astype(np.float32)
            predicted, file_names = predictions[0]
            predicted = predicted.detach().cpu().numpy().astype(np.float32)
            if not np.array_equal(raw.squeeze(), predicted.squeeze()):
                raise InferenceError(f"{sample}: predict_step output differs from the recorded model output")
            if raw.squeeze().shape != (256, 256):
                raise InferenceError(f"{sample}: unexpected output shape {raw.shape}")
            batch_inputs = _normalized_inputs(datamodule, directory)
            record = {
                **loaded.record,
                "split": split,
                "sourceFiles": [f"{sample}_{layer}.tif" for layer in config.ICE_INPUT_LAYERS],
                "referenceFile": f"{sample}_{config.ICE_LABEL_LAYER}.tif",
                "datasetRepo": config.ICE_DATASET_REPO,
                "datasetRevision": config.REVISIONS[config.ICE_DATASET_REPO],
                "predictFileNames": _file_names(file_names),
                "inferenceSeconds": round(seconds, 3),
                "modalities": list(datamodule.modalities),
                "outputSemantics": "dense regression of the Coyan et al. (2025) prospectivity target; not a probability",
                "postprocessing": "none (regression output as returned by PixelwiseRegressionTask.predict_step)",
            }
            written.append(
                artifacts.write_run(TASK, sample, record, batch_inputs, {"prediction": raw.squeeze()})
            )
    finally:
        recorder.close()
    return written


def _file_names(names) -> list[str]:
    """Input files the datamodule actually read (dict per modality for multimodal datamodules)."""
    values = names.values() if isinstance(names, dict) else [names]
    return sorted(Path(str(v)).name for group in values for v in (group if isinstance(group, (list, tuple)) else [group]))


def _normalized_inputs(datamodule, directory: Path) -> np.ndarray:
    """The normalized input stack the model received: predict dataset sample + the datamodule's aug."""
    datamodule.setup("predict")
    batch = datamodule.aug(datamodule.collate_fn([datamodule.predict_dataset[0]]))
    image = batch["image"]
    layers = [image[m] for m in datamodule.modalities] if isinstance(image, dict) else [image]
    return np.concatenate([layer.squeeze(0).numpy() for layer in layers], axis=0).astype(np.float32)
