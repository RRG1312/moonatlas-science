"""Raster encodings of the MOONATLAS data contract v2 (docs/provenance/data-contract.md › ValueRasterSchema).

Shared by the science pipeline and the fixture generator so both write exactly what the browser decodes.

* raw:     value = offset + scale · (R·256 + G), 16-bit linear over the covered range; preserves the model output
           (max quantization error scale / 2, ~8e-6 for a range of ~1.08); binary alpha = coverage.
* display: gray = round(clip(value, 0, 1) · 255); binary alpha. Only a colormap input, never a science value.
* mask:    white RGBA with binary alpha.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

RAW_LEVELS = 65535


def raw_image(values: np.ndarray, covered: np.ndarray) -> tuple[Image.Image, dict]:
    if values.shape != covered.shape:
        raise ValueError("values and coverage must have the same shape")
    if not covered.any():
        raise ValueError("raw raster has no covered pixels")
    data = values[covered].astype(np.float64)
    if not np.isfinite(data).all():
        raise ValueError("raw raster contains NaN or Inf in covered pixels")
    offset, top = float(data.min()), float(data.max())
    scale = (top - offset) / RAW_LEVELS if top > offset else 1.0
    levels = np.zeros(values.shape, dtype=np.uint32)
    levels[covered] = np.round((values[covered] - offset) / scale).astype(np.uint32)
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    rgba[..., 0] = levels >> 8
    rgba[..., 1] = levels & 0xFF
    rgba[..., 3] = covered.astype(np.uint8) * 255
    return Image.fromarray(rgba, "RGBA"), {"kind": "u16-rg-linear", "offset": offset, "scale": scale}


def decode_raw(rgba: np.ndarray, encoding: dict) -> tuple[np.ndarray, np.ndarray]:
    """(values, covered) from a decoded RGB(A) array; libwebp drops an all-opaque alpha channel."""
    levels = rgba[..., 0].astype(np.float64) * 256 + rgba[..., 1]
    covered = rgba[..., 3] == 255 if rgba.shape[-1] == 4 else np.ones(rgba.shape[:2], bool)
    return encoding["offset"] + encoding["scale"] * levels, covered


def display_image(values: np.ndarray, covered: np.ndarray) -> Image.Image:
    gray = np.where(covered, np.round(np.clip(values, 0, 1) * 255), 0).astype(np.uint8)
    return Image.fromarray(np.dstack([gray, covered.astype(np.uint8) * 255]), "LA")


def mask_image(mask: np.ndarray) -> Image.Image:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = mask.astype(bool).astype(np.uint8) * 255
    return Image.fromarray(rgba, "RGBA")


def save_lossless(image: Image.Image, path) -> None:
    image.save(path, "WEBP", lossless=True, quality=100, method=6)
