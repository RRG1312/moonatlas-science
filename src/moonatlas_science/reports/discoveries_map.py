"""Development diagnostic: where the curated discoveries of a build are (equirectangular map + polar insets).

Shows every discovery by type, marks held-out test (featured) vs in-sample, and draws the processed coverage underneath
(WAC tile centers, ice patch centers, IMP sites) so geographic diversity can be judged against what the data contains.
Output: data/artifacts/sanity/<build>/discoveries-map.png

Usage: MOONATLAS_BUILD_DIR=data/build/science-0.3.0-complete python -m moonatlas_science.reports.discoveries_map
"""

from __future__ import annotations

import json
import math
from collections import Counter

from PIL import Image, ImageDraw

from moonatlas_science import config

from ..ice_batch_report import _font
from . import globe_backdrop

TYPE_COLOR = {"CRATER_CANDIDATE": (90, 220, 255), "HIGH_PROSPECTIVITY_REGION": (200, 240, 255), "IMP_CANDIDATE": (255, 165, 61)}


def load(name: str):
    path = config.BUILD_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def main() -> None:
    discoveries = load("discoveries.json")
    regions, sectors, sites = load("crater-regions.json"), load("ice-sectors.json"), load("imp-observation-groups.json")
    width = 2400
    inset = 520
    # Polar insets sit below the map so they never hide southern discoveries.
    image = Image.new("RGB", (width, width // 2 + inset + 40), (10, 12, 14))
    image.paste(globe_backdrop(width, 0.45), (0, 0))
    draw = ImageDraw.Draw(image)
    font, small = _font(22), _font(16)
    xy = lambda lat, lon: ((lon + 180) / 360 * width, (90 - lat) / 180 * width / 2)  # noqa: E731

    for r in regions:
        x, y = xy(r["latitude"], r["longitude"])
        draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(70, 140, 160))
    for s in sites:
        x, y = xy(s["latitude"], s["longitude"])
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(150, 100, 50))
    for d in discoveries:
        x, y = xy(d["latitude"], d["longitude"])
        color = TYPE_COLOR[d["type"]]
        r = 9
        if d["featured"]:
            draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
        else:
            draw.rectangle([x - r + 2, y - r + 2, x + r - 2, y + r - 2], outline=color, width=2)

    for i, (pole, sign) in enumerate((("north", 1), ("south", -1))):
        ox, oy = 20 + i * (inset + 20), width // 2 + 20
        draw.rectangle([ox, oy, ox + inset, oy + inset], fill=(10, 12, 14), outline=(90, 90, 90))
        cx, cy, scale = ox + inset / 2, oy + inset / 2, inset / 2 / 12  # 12° of colatitude to the edge

        def pxy(lat, lon):
            c = 90 - sign * lat
            a = math.radians(lon)
            return cx + c * scale * math.sin(a), cy + (c * scale * math.cos(a) if sign < 0 else -c * scale * math.cos(a))

        for colat in (5, 10):
            draw.ellipse([cx - colat * scale, cy - colat * scale, cx + colat * scale, cy + colat * scale], outline=(60, 70, 80))
        for s in (s for s in sectors if s["pole"] == pole):
            x, y = pxy(s["latitude"], s["longitude"])
            draw.rectangle([x - 6, y - 6, x + 6, y + 6], outline=(70, 110, 130))
        for d in (d for d in discoveries if d["type"] == "HIGH_PROSPECTIVITY_REGION" and (d["latitude"] > 0) == (sign > 0)):
            x, y = pxy(d["latitude"], d["longitude"])
            color = TYPE_COLOR[d["type"]]
            if d["featured"]:
                draw.ellipse([x - 9, y - 9, x + 9, y + 9], outline=color, width=3)
            else:
                draw.rectangle([x - 7, y - 7, x + 7, y + 7], outline=color, width=2)
        draw.text((ox + 10, oy + 8), f"{pole} pole · 80° and 85° circles · squares = ice patches", fill=(220, 220, 220), font=small)

    counts = Counter(d["type"] for d in discoveries)
    featured = sum(d["featured"] for d in discoveries)
    draw.rectangle([0, 0, width, 40], fill=(10, 12, 14))
    draw.text((12, 9), f"Curated discoveries ({len(discoveries)}: craters {counts['CRATER_CANDIDATE']} · ice "
              f"{counts['HIGH_PROSPECTIVITY_REGION']} · IMP {counts['IMP_CANDIDATE']}) · circle = held-out test ({featured}) · "
              f"square = training/validation sample · dots = WAC tiles · rings = IMP sites", fill=(235, 235, 235), font=font)
    out = config.SANITY_DIR / config.BUILD_DIR.name
    out.mkdir(parents=True, exist_ok=True)
    image.save(out / "discoveries-map.png", optimize=True)
    print(out / "discoveries-map.png")


if __name__ == "__main__":
    main()
