"""Human-reviewable evidence for each inference: overlays of source, prediction and reference, plus a report.

A successful script is not proof. These figures show whether preprocessing, orientation and model output
make sense; the report lists the actual values (never placeholders).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from . import artifacts, build, config
from .geo import ProjectedTile, raster_tile, wac_tile_projection

CYAN, GREEN, AMBER = "#35d6ff", "#4cc38a", "#ffa53d"


def _fmt(lat: float, lon: float) -> str:
    return f"{abs(lat):.3f}°{'N' if lat >= 0 else 'S'} {abs(lon):.3f}°{'E' if lon >= 0 else 'W'}"


def _north_arrow(ax, tile: ProjectedTile, x: float = 0.07, y: float = 0.88):
    """Arrow toward true north at the tile center, computed through the projection (never assumed)."""
    lon, lat = tile.center()
    toward = 1.0 if lat < 89.99 else -1.0
    c0, r0 = tile.lonlat_to_pixel(lon, lat)
    c1, r1 = tile.lonlat_to_pixel(lon, lat + 0.01 * toward)
    dx, dy = toward * float(c1 - c0), toward * float(r1 - r0)  # pixel space, rows grow downward
    norm = np.hypot(dx, dy) or 1.0
    dx, dy = dx / norm, dy / norm
    style = {"arrowstyle": "->", "color": "white", "lw": 1.6}
    ax.annotate("", xy=(x + 0.07 * dx, y - 0.07 * dy), xytext=(x, y), xycoords="axes fraction", arrowprops=style)
    ax.text(x + 0.1 * dx, y - 0.1 * dy, "N", transform=ax.transAxes, color="white", ha="center", va="center",
            fontsize=9, fontweight="bold")


def _sun_arrow(ax, tile: ProjectedTile, sub_solar_lat: float, sub_solar_lon: float, x: float = 0.07, y: float = 0.66):
    """Arrow toward the sun: great-circle bearing to the sub-solar point, mapped through the projection.

    Shadows in the image must fall on the opposite side. (The metadata SUB_SOLAR_GROUND_AZIMUTH has an
    undocumented convention, so it is not used.)
    """
    from pyproj import Geod

    radius = float(tile.proj4.split("+R=")[1].split()[0])
    geod = Geod(a=radius, b=radius)
    lon, lat = tile.center()
    azimuth, _, _ = geod.inv(lon, lat, sub_solar_lon, sub_solar_lat)
    step_lon, step_lat, _ = geod.fwd(lon, lat, azimuth, 2000.0)
    c0, r0 = tile.lonlat_to_pixel(lon, lat)
    c1, r1 = tile.lonlat_to_pixel(step_lon, step_lat)
    dx, dy = float(c1 - c0), float(r1 - r0)
    norm = np.hypot(dx, dy) or 1.0
    dx, dy = dx / norm, dy / norm
    style = {"arrowstyle": "->", "color": AMBER, "lw": 1.6}
    ax.annotate("", xy=(x + 0.07 * dx, y - 0.07 * dy), xytext=(x, y), xycoords="axes fraction", arrowprops=style)
    ax.text(x + 0.11 * dx, y - 0.11 * dy, "SUN", transform=ax.transAxes, color=AMBER, ha="center", va="center", fontsize=8)
    return (azimuth + 360) % 360


def crater_figure(sample: str) -> tuple[Path, dict]:
    run, arrays = artifacts.read_run("crater-detection", sample)
    metadata = pd.read_parquet(config.WAC_DIR / "metadata.parquet")
    meta = metadata[metadata["WAC_VIS_TILE"].str.contains(sample, regex=False)].iloc[0].to_dict()
    tile = wac_tile_projection(meta)
    with rasterio.open(config.WAC_DIR / "images_tiff" / f"{sample}.tif") as source:
        image = build.display_stretch(source.read(4))
    scores, boxes = arrays["scores"], arrays["boxes_xyxy"]
    shown = scores >= config.CRATER_DISPLAY_CONFIDENCE
    from .crater_normalize import match_boxes

    matches = match_boxes(boxes[shown], scores[shown], arrays["reference_boxes_xyxy"], config.CRATER_MATCH_IOU)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.6), facecolor="black")
    for ax, title in zip(axes, ["AI PREDICTION (confidence ≥ 0.50)", "REFERENCE — Robbins (2019) boxes"], strict=True):
        ax.imshow(image, cmap="gray", interpolation="nearest")
        ax.set_title(title, color="white", fontsize=11)
        ax.axis("off")
        _north_arrow(ax, tile)
        sun = _sun_arrow(ax, tile, float(meta["SUB_SOLAR_LATITUDE"]), float(meta["SUB_SOLAR_LONGITUDE"]))
    for (x1, y1, x2, y2), s in zip(boxes[shown], scores[shown], strict=True):
        axes[0].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=CYAN, lw=0.9, alpha=0.35 + 0.65 * float(s)))
    for x1, y1, x2, y2 in arrays["reference_boxes_xyxy"]:
        axes[1].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=GREEN, lw=0.9, ls="--"))
    center_lon, center_lat = tile.center()
    fig.suptitle(
        f"CRATER · {sample} · split {run['split']} · center {_fmt(center_lat, center_lon)} · incidence "
        f"{meta['INCIDENCE_ANGLE']:.1f}° · sun bearing {sun:.1f}° toward the sub-solar point (shadows fall opposite)\n"
        f"{run['checkpoint']} · {int(shown.sum())} predictions ≥ 0.50 of {len(scores)} ≥ 0.05 · "
        f"{len(arrays['reference_boxes_xyxy'])} reference · {len(matches)} matched (IoU ≥ 0.5) · display: 643 nm band, 0.5–99.5% stretch",
        color="white", fontsize=10,
    )
    path = config.SANITY_DIR / f"crater-{sample}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor="black", bbox_inches="tight")
    plt.close(fig)

    tile_map = _tile_map(arrays)
    summary = {
        "task": "crater-detection",
        "model": "OK" if not run["missingKeys"] and not run["unexpectedKeys"] else "LOAD MISMATCH",
        "checkpoint": run["checkpoint"],
        "checkpointSha256": run["checkpointSha256"],
        "source": run["sourceFile"],
        "split": run["split"],
        "input": "×".join(map(str, run["input"]["shape"])) + f" {run['input']['dtype']} (normalized)",
        "output": f"{len(scores)} boxes ≥ {run['postprocessing']['scoreThreshold']} after NMS {run['postprocessing']['nmsIouThreshold']}",
        "predictions": f"{int(shown.sum())} with confidence ≥ 0.50",
        "confidenceRange": f"{scores[shown].min():.2f}–{scores[shown].max():.2f}" if shown.any() else "none",
        "reference": f"{len(arrays['reference_boxes_xyxy'])} boxes · {len(matches)} matched at IoU ≥ 0.5",
        "tileMetrics": tile_map,
        "geolocation": f"center {_fmt(center_lat, center_lon)} · {tile.proj4.split(' +units')[0]}",
        "device": run["device"],
        "inferenceSeconds": run["inferenceSeconds"],
        "figure": str(path),
    }
    return path, summary


def _tile_map(arrays) -> dict:
    import torch
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(iou_type="bbox", max_detection_thresholds=[100, 300, 500])
    metric.update(
        [{"boxes": torch.from_numpy(arrays["boxes_xyxy"]), "scores": torch.from_numpy(arrays["scores"]),
          "labels": torch.from_numpy(arrays["labels"])}],
        [{"boxes": torch.from_numpy(arrays["reference_boxes_xyxy"]),
          "labels": torch.ones(len(arrays["reference_boxes_xyxy"]), dtype=torch.long)}],
    )
    result = metric.compute()
    return {"mAP": round(float(result["map"]), 4), "AP50": round(float(result["map_50"]), 4), "AP75": round(float(result["map_75"]), 4),
            "note": "single held-out tile; the published figure is the test-split mean (mAP 0.2581, AP50 0.6183)"}


def ice_figure(sample: str) -> tuple[Path, dict]:
    from .ice_normalize import load_patch

    run, arrays = artifacts.read_run("ice-prospectivity", sample)
    prediction = arrays["prediction"]
    reference, layers, tile, _ = load_patch(sample)
    valid = np.isfinite(reference)
    error = prediction - reference

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="black")
    panels = [
        ("SOURCE · SLOPE (°)", layers["SLOPE"][0], "gray", None),
        ("SOURCE · PSR mask (LPSR)", layers["LPSR"][0], "gray", (0, 1)),
        ("SOURCE · max temperature (K)", layers["TMAX"][0], "inferno", None),
        ("AI PREDICTION · ice prospectivity", prediction, "viridis", (0, 1)),
        ("REFERENCE · Coyan et al. (2025) PRO", reference, "viridis", (0, 1)),
        ("PREDICTION − REFERENCE", error, "coolwarm", (-0.2, 0.2)),
    ]
    for ax, (title, data, cmap, limits) in zip(axes.ravel(), panels, strict=True):
        im = ax.imshow(data, cmap=cmap, vmin=None if limits is None else limits[0], vmax=None if limits is None else limits[1],
                       interpolation="nearest")
        ax.set_title(title, color="white", fontsize=10)
        ax.axis("off")
        _north_arrow(ax, tile)
        bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        bar.ax.tick_params(colors="white", labelsize=8)
    (ul_lon, ul_lat), _, (lr_lon, lr_lat), _ = tile.corners()
    rmse = float(np.sqrt(np.mean(error[valid] ** 2)))
    fig.suptitle(
        f"ICE · {sample} · split {run['split']} · UL {_fmt(ul_lat, ul_lon)} · LR {_fmt(lr_lat, lr_lon)} · "
        f"{tile.pixel_size_x:.2f} m/px polar stereographic\n{run['checkpoint']} · raw range "
        f"{prediction.min():.3f}–{prediction.max():.3f} · RMSE vs reference {rmse:.4f} · prospectivity is not a probability of ice",
        color="white", fontsize=10,
    )
    path = config.SANITY_DIR / f"ice-{sample}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    center_lon, center_lat = tile.center()
    summary = {
        "task": "ice-prospectivity",
        "model": "OK" if not run["missingKeys"] and not run["unexpectedKeys"] else "LOAD MISMATCH",
        "checkpoint": run["checkpoint"],
        "checkpointSha256": run["checkpointSha256"],
        "source": f"{sample} ({len(run['sourceFiles'])} input layers, 9 channels)",
        "split": run["split"],
        "input": "×".join(map(str, run["input"]["shape"])) + f" {run['input']['dtype']} (normalized)",
        "output": "×".join(map(str, prediction.shape)) + " float32 dense regression",
        "predictionRange": f"{prediction.min():.4f}–{prediction.max():.4f} (raw) · mean {prediction[valid].mean():.4f}",
        "reference": f"RMSE {rmse:.4f} · MAE {np.mean(np.abs(error[valid])):.4f} (published test RMSE 0.0293, MAE 0.0197)",
        "geolocation": f"center {_fmt(center_lat, center_lon)} · {tile.proj4.split(' +units')[0]}",
        "device": run["device"],
        "inferenceSeconds": run["inferenceSeconds"],
        "figure": str(path),
    }
    return path, summary


def imp_figure(sample: str) -> tuple[Path, dict]:
    run, arrays = artifacts.read_run("imp-segmentation", sample)
    with rasterio.open(config.IMP_DIR / "all" / f"{sample}_img.tif") as source:
        image, tile = source.read(1), raster_tile(source)
    with rasterio.open(config.IMP_DIR / "all" / f"{sample}_mask.tif") as source:
        reference = source.read(1)
    shown = build.display_stretch(image)
    mask = arrays["mask"].astype(bool)
    reference_mask = reference == 1
    union = (mask | reference_mask).sum()
    iou = float((mask & reference_mask).sum() / union) if union else float("nan")

    def overlay(binary, color):
        rgba = np.zeros((*binary.shape, 4))
        rgba[binary] = [*matplotlib.colors.to_rgb(color), 0.55]
        return rgba

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.6), facecolor="black")
    for ax in axes:
        ax.imshow(shown, cmap="gray", interpolation="nearest")
        ax.axis("off")
        _north_arrow(ax, tile)
    axes[0].set_title("SOURCE · LROC NAC", color="white", fontsize=10)
    axes[1].imshow(overlay(mask, AMBER), interpolation="nearest")
    axes[1].set_title("AI SEGMENTATION (argmax)", color="white", fontsize=10)
    axes[2].imshow(overlay(reference_mask, GREEN), interpolation="nearest")
    axes[2].set_title("REFERENCE · Hargitai et al. (2025)", color="white", fontsize=10)
    im = axes[3].imshow(arrays["softmax_imp"], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axes[3].set_title("IMP class softmax score (uncalibrated)", color="white", fontsize=10)
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.02).ax.tick_params(colors="white", labelsize=8)
    center_lon, center_lat = tile.center()
    fig.suptitle(
        f"IMP · {sample} · split {run['split']} · center {_fmt(center_lat, center_lon)} · {tile.pixel_size_x:.2f} m/px\n"
        f"{run['checkpoint']} · predicted {int(mask.sum())} px · reference {int(reference_mask.sum())} px · IoU {iou:.3f}",
        color="white", fontsize=10,
    )
    path = config.SANITY_DIR / f"imp-{sample}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    logits = arrays["logits"]
    summary = {
        "task": "imp-segmentation",
        "model": "OK" if not run["missingKeys"] and not run["unexpectedKeys"] else "LOAD MISMATCH",
        "checkpoint": run["checkpoint"],
        "checkpointSha256": run["checkpointSha256"],
        "source": run["sourceFile"],
        "split": run["split"],
        "input": "×".join(map(str, run["input"]["shape"])) + f" {run['input']['dtype']} (normalized)",
        "output": "×".join(map(str, logits.shape)) + " float32 logits → argmax mask",
        "predictions": f"{int(mask.sum())} IMP pixels ({mask.mean() * 100:.1f}% of tile)",
        "scoreRange": f"logits {logits.min():.2f}–{logits.max():.2f} · mean IMP softmax on predicted pixels "
                      f"{arrays['softmax_imp'][mask].mean():.3f}" if mask.any() else "no IMP pixels predicted",
        "reference": f"{int(reference_mask.sum())} px · IoU {iou:.4f} (published test IoU₁ 0.5709)",
        "geolocation": f"center {_fmt(center_lat, center_lon)} · {tile.proj4.split(' +units')[0]}",
        "device": run["device"],
        "inferenceSeconds": run["inferenceSeconds"],
        "figure": str(path),
    }
    return path, summary


LABELS = {
    "model": "MODEL", "checkpoint": "CHECKPOINT", "source": "SOURCE", "split": "SPLIT", "input": "INPUT",
    "output": "OUTPUT", "predictions": "PREDICTIONS", "confidenceRange": "CONF RANGE", "predictionRange": "RANGE",
    "scoreRange": "SCORES", "reference": "REFERENCE", "tileMetrics": "TILE METRICS", "geolocation": "GEOLOCATION",
    "device": "DEVICE", "inferenceSeconds": "SECONDS", "figure": "FIGURE",
}


def write_report(summaries: list[dict], build_files: list[str]) -> Path:
    lines = []
    for summary in summaries:
        lines.append(summary["task"].upper())
        for key, label in LABELS.items():
            if key in summary:
                value = summary[key]
                if isinstance(value, dict):
                    value = " · ".join(f"{k} {v}" for k, v in value.items())
                lines.append(f"{label} {'.' * max(2, 13 - len(label))} {value}")
        lines.append("")
    lines.append("NORMALIZED OUTPUT")
    lines.extend(f"  {path}" for path in build_files)
    config.SANITY_DIR.mkdir(parents=True, exist_ok=True)
    text_path = config.SANITY_DIR / "sanity-report.txt"
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (config.SANITY_DIR / "sanity-report.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return text_path
