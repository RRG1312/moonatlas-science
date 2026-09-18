"""Normalization helpers that decide what users see: box matching, ids, value encoding, report checks."""

import numpy as np
import pytest
from PIL import Image

from moonatlas_science import build, config, encoding
from moonatlas_science.crater_normalize import match_boxes, region_id
from moonatlas_science.ice_normalize import check_hemisphere, sector_id
from moonatlas_science.imp_normalize import region_id as imp_region_id
from moonatlas_science.validate import Report, _value_raster, _within


def test_matching_is_one_to_one_and_prefers_confident_predictions():
    pytest.importorskip("torchvision")  # upstream IoU: level 2+, skipped in the fast suite
    reference = np.array([[0, 0, 10, 10]], dtype=np.float32)
    predicted = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    scores = np.array([0.6, 0.9], dtype=np.float32)
    matches = match_boxes(predicted, scores, reference, 0.5)
    assert [(p, r) for p, r, _ in matches] == [(1, 0)]  # the higher-confidence box claims the only reference


def test_matching_respects_the_iou_threshold_and_empty_inputs():
    pytest.importorskip("torchvision")  # upstream IoU: level 2+, skipped in the fast suite
    reference = np.array([[0, 0, 10, 10]], dtype=np.float32)
    far = np.array([[20, 20, 30, 30]], dtype=np.float32)
    assert match_boxes(far, np.array([0.99], np.float32), reference, 0.5) == []
    assert match_boxes(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), reference, 0.5) == []


def test_ids_follow_the_schema_patterns():
    assert region_id("M1174144247CE_r5360_c480") == "wac-m1174144247ce-r5360-c480"
    assert sector_id("patch_0001_0002_S_80S") == "ice-s-0001-0002"
    assert sector_id("patch_0000_0003_80N") == "ice-n-0000-0003"
    assert imp_region_id("M1126915118RE.ech.cog__target_1980__idx_1980_p0") == "imp-m1126915118re-t1980-i1980"
    with pytest.raises(ValueError):
        imp_region_id("not-an-imp-tile")


def test_coordinates_are_normalized_and_never_reach_180():
    assert build.coordinate(10, 180) == {"latitude": 10, "longitude": -180.0}
    assert build.coordinate(-10, 179.9999999)["longitude"] == -180.0
    assert build.coordinate(0, -181)["longitude"] == 179.0


@pytest.fixture
def build_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILD_DIR", tmp_path)
    return tmp_path


def test_value_rasters_preserve_raw_values_and_clip_only_the_display(build_dir):
    values = np.array([[-0.0634, 0.0, 0.5], [0.8379, 1.0, 1.0157]], dtype=np.float64)
    covered = np.ones(values.shape, bool)
    covered[0, 1] = False
    raster = build.value_raster(values, covered, "sample")
    raw = np.asarray(Image.open(build_dir / raster["rawPath"]).convert("RGBA"))
    decoded, decoded_covered = encoding.decode_raw(raw, raster["rawEncoding"])
    assert (decoded_covered == covered).all()
    assert np.abs(decoded[covered] - values[covered]).max() <= raster["rawEncoding"]["scale"] / 2 + 1e-12
    assert decoded[1, 2] > 1 and decoded[0, 0] < 0  # the model output is not bounded
    display = np.asarray(Image.open(build_dir / raster["displayPath"]).convert("RGBA"))
    assert display[1, 2, 0] == 255 and display[0, 0, 0] == 0  # display only is clipped
    report = Report()
    assert _value_raster(report, build_dir, raster, values.shape, "test") is not None
    assert report.problems == []


def test_validator_flags_degenerate_rasters(build_dir):
    raster = build.value_raster(np.full((4, 4), 0.3), np.ones((4, 4), bool), "flat")
    report = Report()
    _value_raster(report, build_dir, raster, (4, 4), "test")
    assert any("constant" in problem for problem in report.problems)


def test_raw_encoding_rejects_non_finite_values():
    values = np.array([[0.1, np.nan]])
    with pytest.raises(ValueError, match="NaN"):
        encoding.raw_image(values, np.ones(values.shape, bool))


def test_projection_records_are_structured_from_proj_strings():
    record = build.projection_record("+proj=stere +lat_0=-90 +lon_0=0 +k=0.994 +x_0=500000 +y_0=500000 +R=1737400 +units=m +no_defs")
    assert record == {"kind": "polar-stereographic", "latitudeOfOriginDeg": -90, "centralMeridianDeg": 0, "scaleFactor": 0.994,
                      "falseEastingM": 500000.0, "falseNorthingM": 500000.0, "radiusM": 1737400.0}
    with pytest.raises(ValueError):
        build.projection_record("+proj=longlat +R=1737400 +no_defs")


def test_footprint_containment_handles_the_antimeridian():
    footprint = [{"latitude": 1, "longitude": 179.5}, {"latitude": 1, "longitude": -179.5},
                 {"latitude": -1, "longitude": -179.5}, {"latitude": -1, "longitude": 179.5}]
    assert _within({"latitude": 0, "longitude": 179.9}, footprint, 0)
    assert not _within({"latitude": 0, "longitude": 170}, footprint, 0)


def test_hemisphere_must_agree_between_name_tag_and_crs():
    south = build.projection_record("+proj=stere +lat_0=-90 +lon_0=0 +k=0.994 +x_0=500000 +y_0=500000 +R=1737400 +units=m +no_defs")
    north = {**south, "latitudeOfOriginDeg": 90}
    assert check_hemisphere("patch_0001_0002_S_80S", "S", south) == "south"
    assert check_hemisphere("patch_0000_0003_80N", "N", north) == "north"
    for sample, tag, projection in (("patch_0001_0002_S_80S", "N", south), ("patch_0001_0002_S_80S", "S", north),
                                    ("patch_0000_0003_80N", "N", south), ("patch_0000_0003_80N", "", north),
                                    ("patch_0000_0003", "N", north)):
        with pytest.raises(ValueError):
            check_hemisphere(sample, tag, projection)
