"""Validate a normalized MOONATLAS science build (contract v2). Exits 1 on any problem.

Usage: python -m moonatlas_science.steps.validate_data [--root data/build/science-0.2.0 | datasets/real-smoke]
"""

import argparse
import sys
from pathlib import Path

from moonatlas_science import config, validate

LABELS = [
    ("craterPredictions", "CRATER PREDICTIONS"),
    ("craterRegions", "CRATER REGIONS"),
    ("icePatches", "ICE PATCHES"),
    ("iceMosaics", "ICE MOSAICS"),
    ("impRegions", "IMP REGIONS"),
    ("discoveries", "DISCOVERIES"),
    ("syntheticObjects", "SYNTHETIC OBJECTS"),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=config.BUILD_DIR)
    parser.add_argument("--scope", choices=("full", "published"), default="full",
                        help="full (default) also cross-checks the local raw inference artifacts; "
                             "published validates a dataset on a machine that did not run the inference")
    args = parser.parse_args()
    root, scope = args.root, args.scope
    problems = validate.validate(root, check_raw=scope == "full")
    for problem in problems:
        print(f"FAIL {problem}")
    facts = validate.summary(root)
    title = {"real-smoke": "REAL SCIENCE SMOKE DATA", "batch-preview": "REAL SCIENCE BATCH PREVIEW",
             "complete-preview": "REAL SCIENCE COMPLETE PREVIEW", "real": "REAL SCIENCE DATA"}.get(facts["dataMode"], "SCIENCE DATA")
    print(f"\n{title}  ({root})\n")
    for key, label in LABELS:
        print(f"{label} {'.' * (24 - len(label))} {facts[key]}")
    print(f"SCOPE {'.' * 19} {scope.upper()}{' (raw inference artifacts not checked)' if scope == 'published' else ''}")
    print(f"VALIDATION {'.' * 14} {'PASS' if not problems else f'FAIL ({len(problems)} problems)'}")
    sys.exit(1 if problems else 0)
