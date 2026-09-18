"""SCIENCE SMOKE TEST: one real sample per task, end to end, each step as its own process.

download → inference (crater, ice, IMP) → normalization → ice mosaic → manifest → validation → sanity report.
Stops at the first failing step. Nothing is substituted when a step fails.

Usage: python -m moonatlas_science.steps.smoke_test [--skip-downloads]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    ("download models", ["download_models.py"]),
    ("download data", ["download_data.py"]),
    ("crater inference", ["run_crater_inference.py"]),
    ("ice inference", ["run_ice_inference.py"]),
    ("IMP inference", ["run_imp_inference.py"]),
    ("normalize craters", ["process_craters.py"]),
    ("normalize ice", ["process_ice.py"]),
    ("normalize IMP", ["process_imp.py"]),
    ("ice mosaic", ["build_ice_mosaic.py"]),
    ("catalog", ["build_catalog.py"]),
    ("manifest", ["build_manifest.py", "--scope", "smoke"]),
    ("validation", ["validate_data.py"]),
    ("sanity report", ["sanity_report.py"]),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-downloads", action="store_true")
    args = parser.parse_args()
    for name, (script, *extra) in STEPS:
        if args.skip_downloads and name.startswith("download"):
            continue
        print(f"\n=== {name} ===", flush=True)
        started = time.perf_counter()
        result = subprocess.run([sys.executable, str(HERE / script), *extra])
        if result.returncode != 0:
            print(f"\nSMOKE TEST FAILED at step: {name}")
            sys.exit(result.returncode)
        print(f"--- {name} ok ({time.perf_counter() - started:.1f} s)", flush=True)
    print("\nSCIENCE SMOKE TEST PASSED")
