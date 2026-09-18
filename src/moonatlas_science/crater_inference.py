"""WAC crater detection inference with the released LoRA checkpoint (WAC_ni_lfm_ps8_lora_s46.ckpt)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import artifacts, config
from .official import InferenceError, OutputRecorder, load_task

TASK = "crater-detection"


def stage_annotations(samples: list[str], split: str) -> Path:
    """COCO file restricted to the requested tiles (upstream loads every listed tile at setup).

    Only the image list is reduced; per-sample loading and normalization are unchanged.
    """
    split_file = {"test": "test.json", "validation": "val.json", "train": "train.json"}[split]
    coco = json.loads((config.WAC_DIR / split_file).read_text(encoding="utf-8"))
    wanted = {s: None for s in samples}
    images = [im for im in coco["images"] if Path(im["file_name"]).stem in wanted]
    found = {Path(im["file_name"]).stem for im in images}
    missing = [s for s in samples if s not in found]
    if missing:
        raise InferenceError(f"tiles not in the {split} split: {missing}")
    ids = {im["id"] for im in images}
    subset = {**coco, "images": images, "annotations": [a for a in coco["annotations"] if a["image_id"] in ids]}
    directory = config.STAGING_DIR / TASK / "annotations"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "test.json").write_text(json.dumps(subset), encoding="utf-8")
    return directory


def split_of(samples: list[str]) -> str:
    """Split from metadata.parquet; one inference call handles one split."""
    import pandas as pd

    metadata = pd.read_parquet(config.WAC_DIR / "metadata.parquet")
    stems = metadata["WAC_VIS_TILE"].map(lambda p: p.rsplit("/", 1)[-1].removesuffix(".nc"))
    splits = {({"val": "validation"}.get(s, s)) for s in metadata.loc[stems.isin(samples), "DATASET"]}
    if len(splits) != 1 or stems.isin(samples).sum() != len(samples):
        raise InferenceError(f"samples must exist in metadata.parquet and share one split, got {sorted(splits)}")
    return splits.pop()


def run(samples: list[str]) -> list[Path]:
    import torch

    split = split_of(samples)
    annotations_dir = stage_annotations(samples, split)
    model = config.TASK_MODELS[TASK]
    loaded = load_task(
        model,
        config.DATASETS_DIR,
        {"data.init_args.annotations_dir": annotations_dir.as_posix(), "data.init_args.num_workers": 0},
    )
    task, datamodule = loaded.task, loaded.datamodule
    if loaded.record["missingKeys"] or loaded.record["unexpectedKeys"]:
        raise InferenceError("checkpoint did not load strictly")

    datamodule.setup("test")
    dataset = datamodule.test_dataset
    index = {s["stem"]: i for i, s in enumerate(dataset._samples)}
    written = []
    recorder = OutputRecorder(task.model)
    try:
        for sample in samples:
            if sample not in index:
                raise InferenceError(f"{sample} was dropped by the upstream dataset (no surviving annotations)")
            item = dataset[index[sample]]
            batch = datamodule.collate_fn([item])
            batch["image"] = batch["image"].to(loaded.device)
            recorder.outputs.clear()
            started = time.perf_counter()
            with torch.no_grad():
                # Body of terratorch ObjectDetectionTask.predict_step: model forward, then the task's
                # score filter (score_threshold) and NMS (iou_threshold).
                predictions = task.predict_step(batch, 0)
            seconds = time.perf_counter() - started
            if len(recorder.outputs) != 1:
                raise InferenceError(f"expected one model forward, recorded {len(recorder.outputs)}")
            raw = recorder.outputs[0]
            raw = raw.output if hasattr(raw, "output") else raw
            raw, final = raw[0], predictions[0]
            arrays = {
                "raw_boxes_xyxy": raw["boxes"].numpy().astype(np.float32),
                "raw_scores": raw["scores"].numpy().astype(np.float32),
                "raw_labels": raw["labels"].numpy().astype(np.int64),
                "boxes_xyxy": final["boxes"].detach().cpu().numpy().astype(np.float32),
                "scores": final["scores"].detach().cpu().numpy().astype(np.float32),
                "labels": final["labels"].detach().cpu().numpy().astype(np.int64),
                "reference_boxes_xyxy": item["boxes"].numpy().astype(np.float32),
            }
            if arrays["boxes_xyxy"].size and not np.isfinite(arrays["boxes_xyxy"]).all():
                raise InferenceError(f"{sample}: non-finite predicted boxes")
            record = {
                **loaded.record,
                "split": split,
                "sourceFile": f"images_tiff/{sample}.tif",
                "datasetRepo": config.WAC_DATASET_REPO,
                "datasetRevision": config.REVISIONS[config.WAC_DATASET_REPO],
                "inferenceSeconds": round(seconds, 3),
                "postprocessing": {
                    "scoreThreshold": float(task.score_threshold),
                    "nmsIouThreshold": float(task.iou_threshold),
                    "roiHeads": {
                        "scoreThresh": float(task.model.torchvision_model.roi_heads.score_thresh),
                        "nmsThresh": float(task.model.torchvision_model.roi_heads.nms_thresh),
                        "detectionsPerImg": int(task.model.torchvision_model.roi_heads.detections_per_img),
                    },
                    "boxFormat": "xyxy, pixels, origin at the upper-left edge of the 512 x 512 tile",
                },
                "referenceBoxes": "upstream dataset targets: Robbins (2019) boxes >= min_diameter_px not touching no-data",
            }
            model_input = item["image"].numpy().astype(np.float32)
            written.append(artifacts.write_run(TASK, sample, record, model_input, arrays))
    finally:
        recorder.close()
    return written
