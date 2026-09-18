"""Materialize the production dataset `datasets/real/` from a validated complete science build.

Production asset policy (docs/architecture/dataset-publishing.md), applied here and nowhere else:
  * per-patch ICE **display** rasters are not shipped: they are pixel-identical to the pole mosaic window, which this
    script re-verifies before dropping them, and the sector's `displayPath` becomes null;
  * per-tile crater detail JSON is written compact, asserting it decodes to exactly the same object;
  * everything else the explorer loads is copied unchanged (rasters, previews, masks, catalogue JSON).
Then the manifest is rebuilt with `dataMode: real`, checksums are recomputed over the published files, and the dataset
is validated. Nothing is fabricated or recomputed from the model: this is packaging, not science.

Usage: MOONATLAS_BUILD_DIR=data/build/science-0.3.1-complete python -m moonatlas_science.steps.build_dataset
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from moonatlas_science import config, manifest, validate

from ..ice_patch_vs_mosaic import compare

MODE = "real"


def write_json(path: Path, data, compact: bool) -> None:
    text = json.dumps(data, allow_nan=False, separators=(",", ":")) if compact else json.dumps(data, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    source, target = config.BUILD_DIR, config.PUBLISHED_DATASETS_DIR / MODE
    if not (source / "manifest.json").exists():
        sys.exit(f"{source} is not a science build")
    problems = validate.validate(source)
    if problems:
        sys.exit("refusing to package an invalid build:\n" + "\n".join(problems))

    equivalence = compare(source)
    if equivalence["coverageMismatches"] or equivalence["displayMismatches"]:
        sys.exit(f"ICE patch display rasters differ from the mosaic windows: {equivalence}")

    if target.exists():
        shutil.rmtree(target)
    dropped = compacted = 0
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        relative = path.relative_to(source).as_posix()
        destination = target / relative
        if relative.startswith("overlays/ice/") and "mosaic" not in relative and not relative.endswith("-raw.webp"):
            dropped += 1  # per-patch display raster: the mosaic window carries the same pixels
            continue
        if relative.startswith("crater-tiles/"):
            detail = json.loads(path.read_text(encoding="utf-8"))
            write_json(destination, detail, compact=True)
            assert json.loads(destination.read_text(encoding="utf-8")) == detail, relative
            compacted += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    sectors = json.loads((target / "ice-sectors.json").read_text(encoding="utf-8"))
    for sector in sectors:
        for group in (sector["prediction"], sector["reference"]):
            group["heatmap"]["displayPath"] = None
    write_json(target / "ice-sectors.json", sectors, compact=False)

    # The production dataset is dated by the science build it packages, never by the moment it was packaged: rebuilding
    # REAL from the same science build must reproduce the same bytes and the same pinned checksums.
    generated_at = datetime.fromisoformat(
        json.loads((source / "manifest.json").read_text(encoding="utf-8"))["generatedAt"].replace("Z", "+00:00")
    )
    config.BUILD_DIR = target  # the manifest must describe the published files, not the source build
    document = manifest.write("production", generated_at)
    problems = validate.validate(target)
    for problem in problems:
        print(f"FAIL {problem}")
    files = [p for p in target.rglob("*") if p.is_file()]
    print(f"{MODE}: {len(files)} files · {sum(p.stat().st_size for p in files) / 1048576:.1f} MB · "
          f"{dropped} per-patch display rasters dropped · {compacted} crater tile files compacted")
    print(f"manifest dataMode {json.loads((target / 'manifest.json').read_text(encoding='utf-8'))['dataMode']} · "
          f"{len(document['checksums'])} checksums · validation {'PASS' if not problems else f'FAIL ({len(problems)})'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
