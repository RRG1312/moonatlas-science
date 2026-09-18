"""Publish a validated science build as a dataset under the publish directory (MOONATLAS_PUBLISH_DIR).

Refuses to publish when validation fails or when the build's manifest dataMode differs from the target mode.
The target folder is replaced entirely, so no stale or synthetic file can survive.

Usage: python -m moonatlas_science.steps.publish_build --mode real-smoke
"""

import argparse
import json
import shutil
import sys

from moonatlas_science import config, manifest, validate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=sorted(set(manifest.DATA_MODE_BY_SCOPE.values())), required=True)
    mode = parser.parse_args().mode
    built = json.loads((config.BUILD_DIR / "manifest.json").read_text(encoding="utf-8"))
    if built["dataMode"] != mode:
        sys.exit(f"build dataMode is {built['dataMode']!r}, refusing to publish as {mode!r}")
    problems = validate.validate(config.BUILD_DIR)
    if problems:
        sys.exit("refusing to publish an invalid build:\n" + "\n".join(problems))
    target = config.PUBLISHED_DATASETS_DIR / mode
    if target.exists():
        for child in target.iterdir():  # empty in place (Windows may hold the directory handle)
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    shutil.copytree(config.BUILD_DIR, target, dirs_exist_ok=True)
    files = [p for p in target.rglob("*") if p.is_file()]
    print(f"published {len(files)} files ({sum(p.stat().st_size for p in files) / 1024:.0f} KB) to {target}")
