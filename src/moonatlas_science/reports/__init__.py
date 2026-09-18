"""Diagnostic figures and batch reviews over a build.

These are development tools: they visualize values that the pipeline already computed and never produce a
published number. Figures may use an optional equirectangular basemap (MOONATLAS_BASEMAP); without one they
draw on a plain dark canvas, so every report runs on a clean clone.
"""

from __future__ import annotations

from PIL import Image

from .. import config


def globe_backdrop(width: int, dim: float = 0.5) -> Image.Image:
    """Dimmed equirectangular backdrop of `width` × `width // 2`, or a plain dark canvas without a basemap."""
    size = (width, width // 2)
    if config.GLOBE_BASEMAP and config.GLOBE_BASEMAP.exists():
        return Image.eval(Image.open(config.GLOBE_BASEMAP).convert("RGB").resize(size), lambda v: int(v * dim))
    return Image.new("RGB", size, (10, 12, 14))
