"""Write sources.json, manifest.json and processing-manifest.json (models, datasets, counts, checksums).

Usage: python -m moonatlas_science.steps.build_manifest [--scope smoke|batch-preview|complete-preview|production]
"""

import argparse
import json

from moonatlas_science import build, manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scope", choices=list(manifest.DATA_MODE_BY_SCOPE), default="smoke")
    build.write_sources()
    document = manifest.write(parser.parse_args().scope)
    print(json.dumps(document["counts"], indent=2))
