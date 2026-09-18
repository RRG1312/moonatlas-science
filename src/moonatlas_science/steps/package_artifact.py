"""Pack `datasets/real/` into one deterministic production artifact plus the committed `production-data.json`.

The archive is the distribution format for deployments that clone the repository without the dataset (ADR-008). It is
byte-for-byte reproducible from the same dataset: entries sorted by path, fixed mtime/uid/gid/mode, USTAR format and a
gzip stream without a timestamp. Only the dataset is included: no fixtures, previews of other modes, screenshots,
diagnostics, reports, upstream downloads or checkpoints.

Usage: python -m moonatlas_science.steps.package_artifact [--out dist]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

from moonatlas_science import config

ARTIFACT_VERSION = 1  # bump when the archive layout (not its content) changes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack(dataset: Path) -> tuple[bytes, list[Path]]:
    """Deterministic .tar.gz bytes of `dataset`, with every entry under `real/`."""
    files = sorted((p for p in dataset.rglob("*") if p.is_file()), key=lambda p: p.relative_to(dataset).as_posix())
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for path in files:
            name = f"{dataset.name}/{path.relative_to(dataset).as_posix()}"
            if len(name) > 100:  # USTAR without prefix: keep names short so any minimal extractor can read them
                sys.exit(f"path too long for the archive format: {name}")
            info = tarfile.TarInfo(name)
            info.size = path.stat().st_size
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, compresslevel=9, mtime=0) as gz:
        gz.write(raw.getvalue())
    return compressed.getvalue(), files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=config.PROJECT_ROOT / "dist")
    parser.add_argument("--dataset", type=Path, default=None, help="dataset to pack (default: the published real dataset)")
    parser.add_argument("--manifest", type=Path, default=None, help="where to write the artifact manifest (default: next to the archive)")
    args = parser.parse_args()
    out, dataset = args.out, args.dataset or config.PUBLISHED_DATASETS_DIR / "real"
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dataMode"] != "real":
        sys.exit(f"{dataset} declares dataMode {manifest['dataMode']!r}: build it with `moonatlas-science publish` first")

    archive, files = pack(dataset)
    out.mkdir(parents=True, exist_ok=True)
    name = f"moonatlas-{manifest['processingVersion']}-real.tar.gz"
    (out / name).write_bytes(archive)
    document = {
        "schemaVersion": manifest["schemaVersion"],
        "scienceVersion": manifest["processingVersion"],
        "artifactVersion": ARTIFACT_VERSION,
        "dataMode": "real",
        "archive": name,
        "sha256": hashlib.sha256(archive).hexdigest(),
        "compressedBytes": len(archive),
        "uncompressedBytes": sum(p.stat().st_size for p in files),
        "fileCount": len(files),
        "rootDirectory": dataset.name,
        "datasetManifestSha256": sha256_file(dataset / "manifest.json"),
        "processingManifestSha256": sha256_file(dataset / "processing-manifest.json"),
        "generatedAt": manifest["generatedAt"],
    }
    manifest_file = args.manifest or out / f"{Path(name).stem.removesuffix('.tar')}.json"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{out / name}: {len(archive) / 1048576:.1f} MB compressed · {document['uncompressedBytes'] / 1048576:.1f} MB "
          f"in {document['fileCount']} files\nsha256 {document['sha256']}\nmanifest {manifest_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
