"""Production packaging: the compact crater detail must carry the same data, and the archive must be reproducible."""

import gzip
import io
import json
import tarfile

import pytest

from moonatlas_science.steps.build_dataset import write_json
from moonatlas_science.steps.package_artifact import pack

DETAIL = {
    "regionId": "wac-m1095814428ce-r1896-c752",
    "prediction": [[10.5, 20.25, 30.0, 40.125, 0.9312], [1.0, 2.0, 3.0, 4.0, 0.5]],
    "reference": [[11.0, 21.0, 31.0, 41.0]],
}


def test_compact_detail_decodes_to_the_same_object(tmp_path):
    indented, compact = tmp_path / "indented.json", tmp_path / "compact.json"
    write_json(indented, DETAIL, compact=False)
    write_json(compact, DETAIL, compact=True)
    text = compact.read_text(encoding="utf-8")
    assert json.loads(text) == json.loads(indented.read_text(encoding="utf-8")) == DETAIL
    assert "\n" not in text and ", " not in text and len(text) < len(indented.read_text(encoding="utf-8"))
    assert compact.read_bytes().count(b"\r") == 0


def dataset(tmp_path):
    root = tmp_path / "real"
    (root / "crater-tiles").mkdir(parents=True)
    write_json(root / "manifest.json", {"dataMode": "real"}, compact=False)
    write_json(root / "crater-tiles" / "a.json", DETAIL, compact=True)
    (root / "overlays").mkdir()
    (root / "overlays" / "x.webp").write_bytes(b"\x00\x01\x02")
    return root


def test_archive_is_byte_identical_across_runs_and_rooted_at_the_dataset(tmp_path):
    root = dataset(tmp_path)
    first, files = pack(root)
    second, _ = pack(root)
    assert first == second, "the production artifact must be reproducible"
    names = tarfile.open(fileobj=io.BytesIO(gzip.decompress(first))).getnames()
    assert names == ["real/crater-tiles/a.json", "real/manifest.json", "real/overlays/x.webp"]
    assert [p.name for p in files] == ["a.json", "manifest.json", "x.webp"]
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(first))) as archive:
        for member in archive.getmembers():
            assert member.mtime == 0 and member.uid == 0 and member.gid == 0 and member.uname == ""


def test_long_paths_are_refused_so_any_minimal_extractor_can_read_the_archive(tmp_path):
    root = dataset(tmp_path)
    (root / "crater-tiles" / f"{'x' * 110}.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        pack(root)
