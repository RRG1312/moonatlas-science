"""Can each per-patch ICE raster be read from its pole mosaic without changing what the inspector reports?

Compares every sector's raw and display WebPs with the mosaic window at its published placement
(ice-mosaics.json › patches). Reports coverage/display mismatches, the largest raw difference and how many prediction
pixels would change at the 4 (inspector) and 3 decimals. Production data plan, docs/architecture/dataset-publishing.md.

Usage: python -m moonatlas_science.ice_patch_vs_mosaic datasets/complete-preview
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from . import encoding


def compare(root: Path) -> dict:
    """Mismatches and worst differences between every per-patch raster and its pole mosaic window."""
    mosaics = {m["pole"]: m for m in json.loads((root / "ice-mosaics.json").read_text(encoding="utf-8"))}
    sectors = json.loads((root / "ice-sectors.json").read_text(encoding="utf-8"))
    images: dict[str, np.ndarray] = {}

    def rgba(path: str) -> np.ndarray:
        if path not in images:
            images[path] = np.asarray(Image.open(root / path).convert("RGBA"))
        return images[path]

    worst = {"prediction": 0.0, "reference": 0.0}
    mask_mismatch = display_mismatch = total = 0
    changed = {4: 0, 3: 0}
    for sector in sectors:
        mosaic = mosaics[sector["pole"]]
        place = next(p for p in mosaic["patches"] if p["sectorId"] == sector["id"])
        window = np.s_[place["row"]:place["row"] + place["heightPx"], place["col"]:place["col"] + place["widthPx"]]
        for group in ("prediction", "reference"):
            patch, whole = sector[group]["heatmap"], mosaic[group]
            patch_values, patch_covered = encoding.decode_raw(rgba(patch["rawPath"]), patch["rawEncoding"])
            mosaic_values, mosaic_covered = encoding.decode_raw(rgba(whole["rawPath"]), whole["rawEncoding"])
            mosaic_values, mosaic_covered = mosaic_values[window], mosaic_covered[window]
            mask_mismatch += int((patch_covered != mosaic_covered).sum())
            difference = np.abs(patch_values - mosaic_values)[patch_covered]
            worst[group] = max(worst[group], float(difference.max()) if difference.size else 0.0)
            if group == "prediction":
                total += int(patch_covered.sum())
                for decimals in changed:
                    a, b = np.round(patch_values[patch_covered], decimals), np.round(mosaic_values[patch_covered], decimals)
                    changed[decimals] += int((a != b).sum())
            patch_display, mosaic_display = rgba(patch["displayPath"]), rgba(whole["displayPath"])[window]
            display_mismatch += int((patch_display[patch_covered] != mosaic_display[patch_covered]).any(axis=-1).sum())

    return {"patches": len(sectors), "coverageMismatches": mask_mismatch, "displayMismatches": display_mismatch,
            "maxRawDifference": worst, "predictionPixels": total,
            "changedAtDecimals": changed}


if __name__ == "__main__":
    result = compare(Path(sys.argv[1] if len(sys.argv) > 1 else "datasets/complete-preview"))
    print(f"{result['patches']} patches · coverage mismatches {result['coverageMismatches']} · "
          f"display pixel mismatches {result['displayMismatches']}")
    print(f"max |patch raw − mosaic raw|: prediction {result['maxRawDifference']['prediction']:.2e} · "
          f"reference {result['maxRawDifference']['reference']:.2e}")
    for decimals, count in result["changedAtDecimals"].items():
        share = 100 * count / result["predictionPixels"]
        print(f"prediction pixels whose {decimals}-decimal value would change: {count:,} of "
              f"{result['predictionPixels']:,} ({share:.2f}%)")
