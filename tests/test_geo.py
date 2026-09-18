"""Geolocation tests: published SomBench metadata rows are the ground truth (copied verbatim)."""

import numpy as np
import pytest

from moonatlas_science.geo import (
    ProjectedTile,
    normalize_longitude,
    wac_tile_projection,
)

# metadata.parquet rows (Sombench-WAC-Crater-Detection @ 20f800be).
SMOKE_SOUTH_TM = {  # test split, LTM_2S
    "WAC_VIS_TILE": "images/M1174144247CE_r5360_c480.nc",
    "CENTER_LATITUDE": -53.15382890435657, "CENTER_LONGITUDE": -169.1767581231181,
    "UPPER_LEFT_LATITUDE": -52.20695843280776, "UPPER_LEFT_LONGITUDE": -170.36239141832502,
    "LOWER_RIGHT_LATITUDE": -54.09075510775035, "LOWER_RIGHT_LONGITUDE": -167.9384531824178,
    "BOUNDS_XMIN": 57899.99999999956, "BOUNDS_XMAX": 109099.99999999956,
    "BOUNDS_YMIN": 853500.0000000002, "BOUNDS_YMAX": 904700.0000000002,
}
NORTH_TM = {  # LTM_10N
    "WAC_VIS_TILE": "images/M1290203799CE_r5872_c1112.nc",
    "CENTER_LATITUDE": 13.771920141761159, "CENTER_LONGITUDE": -105.32202981444279,
    "UPPER_LEFT_LATITUDE": 14.61021216718391, "UPPER_LEFT_LONGITUDE": -106.19988111923357,
    "LOWER_RIGHT_LATITUDE": 12.929982206497655, "LOWER_RIGHT_LONGITUDE": -104.45046268680528,
    "BOUNDS_XMIN": 185499.9999999997, "BOUNDS_XMAX": 236699.9999999997,
    "BOUNDS_YMIN": 391700.0000000007, "BOUNDS_YMAX": 442900.0000000007,
}
NORTH_POLAR = {  # polar stereographic cap
    "WAC_VIS_TILE": "images/M1206744776CE_r2408_c3120.nc",
    "CENTER_LATITUDE": 63.54045013842536, "CENTER_LONGITUDE": -101.58775130396306,
    "UPPER_LEFT_LATITUDE": 62.584822826287734, "UPPER_LEFT_LONGITUDE": -102.94371490877114,
    "LOWER_RIGHT_LATITUDE": 64.48437293473515, "LOWER_RIGHT_LONGITUDE": -100.12700071926722,
    "BOUNDS_XMIN": -321060.6000000001, "BOUNDS_XMAX": -269860.6000000001,
    "BOUNDS_YMIN": 637507.4000000003, "BOUNDS_YMAX": 688707.4000000003,
}
SOUTH_EAST_TM = {  # LTM_33S, eastern hemisphere
    "WAC_VIS_TILE": "images/M1360586071CE_r2920_c1440.nc",
    "CENTER_LATITUDE": -68.081899216301, "CENTER_LONGITUDE": 79.8174455134934,
    "UPPER_LEFT_LATITUDE": -67.33205255898362, "UPPER_LEFT_LONGITUDE": 77.36757991329948,
    "LOWER_RIGHT_LATITUDE": -68.79182477210976, "LOWER_RIGHT_LONGITUDE": 82.42701751752782,
    "BOUNDS_XMIN": 312599.99999999977, "BOUNDS_XMAX": 363799.99999999977,
    "BOUNDS_YMIN": 406400.00000000023, "BOUNDS_YMAX": 457600.00000000023,
}
ROWS = [SMOKE_SOUTH_TM, NORTH_TM, NORTH_POLAR, SOUTH_EAST_TM]
TOL = 1e-9


@pytest.mark.parametrize("meta", ROWS, ids=lambda m: m["WAC_VIS_TILE"])
def test_tile_center_and_corners_reproduce_metadata(meta):
    tile = wac_tile_projection(meta)
    lon, lat = tile.pixel_to_lonlat(256, 256)
    assert lat == pytest.approx(meta["CENTER_LATITUDE"], abs=TOL)
    assert lon == pytest.approx(meta["CENTER_LONGITUDE"], abs=TOL)
    (ul_lon, ul_lat), _, (lr_lon, lr_lat), _ = tile.corners()
    assert (ul_lat, ul_lon) == pytest.approx((meta["UPPER_LEFT_LATITUDE"], meta["UPPER_LEFT_LONGITUDE"]), abs=TOL)
    assert (lr_lat, lr_lon) == pytest.approx((meta["LOWER_RIGHT_LATITUDE"], meta["LOWER_RIGHT_LONGITUDE"]), abs=TOL)


def test_projection_kinds_are_recovered():
    assert "+proj=tmerc" in wac_tile_projection(SMOKE_SOUTH_TM).proj4
    assert "+y_0=2500000" in wac_tile_projection(SMOKE_SOUTH_TM).proj4
    assert "+proj=stere +lat_0=90" in wac_tile_projection(NORTH_POLAR).proj4


def test_prediction_center_round_trips_through_the_projection():
    tile = wac_tile_projection(SMOKE_SOUTH_TM)
    col, row = 137.25, 402.75  # center of a hypothetical predicted box
    lon, lat = tile.pixel_to_lonlat(col, row)
    back_col, back_row = tile.lonlat_to_pixel(lon, lat)
    assert (float(back_col), float(back_row)) == pytest.approx((col, row), abs=1e-6)


@pytest.mark.parametrize("meta", [SMOKE_SOUTH_TM, NORTH_TM, SOUTH_EAST_TM], ids=lambda m: m["WAC_VIS_TILE"])
def test_transverse_mercator_tiles_are_north_up_and_east_right(meta):
    tile = wac_tile_projection(meta)
    lon_left, _ = tile.pixel_to_lonlat(10, 256)
    lon_right, _ = tile.pixel_to_lonlat(500, 256)
    _, lat_top = tile.pixel_to_lonlat(256, 10)
    _, lat_bottom = tile.pixel_to_lonlat(256, 500)
    assert lon_right > lon_left  # columns grow east
    assert lat_top > lat_bottom  # rows grow south


def test_south_hemisphere_latitudes_are_negative_and_north_positive():
    assert all(lat < 0 for _, lat in wac_tile_projection(SMOKE_SOUTH_TM).corners())
    assert all(lat > 0 for _, lat in wac_tile_projection(NORTH_TM).corners())


def test_rejects_metadata_that_no_projection_reproduces():
    shifted = {**NORTH_TM, "BOUNDS_XMIN": NORTH_TM["BOUNDS_XMIN"] + 1000, "BOUNDS_XMAX": NORTH_TM["BOUNDS_XMAX"] + 1000}
    with pytest.raises(ValueError, match="exactly one projection"):
        wac_tile_projection(shifted)


def test_longitudes_wrap_across_the_antimeridian():
    # Zone 1 (central meridian -176°): a point 250 km west of it lies beyond -180°.
    tile = ProjectedTile(
        proj4="+proj=tmerc +lat_0=0 +lon_0=-176 +k=0.999 +x_0=250000 +y_0=0 +R=1737400 +units=m +no_defs",
        x_min=0.0, y_max=51_200.0, pixel_size_x=100.0, pixel_size_y=100.0, width=512, height=512,
    )
    lon, _ = tile.pixel_to_lonlat(0, 0)
    assert 170 < lon < 180
    lons, _ = tile.pixel_to_lonlat([0, 512], [0, 0])
    assert all(-180 <= v < 180 for v in lons)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(180, -180), (-180, -180), (540, -180), (179.5, 179.5), (-179.5, -179.5), (360.25, 0.25), (-0.0, 0.0)],
)
def test_normalize_longitude(value, expected):
    assert normalize_longitude(value) == pytest.approx(expected)


def test_polar_stereographic_origin_is_the_pole_with_its_scale_factor():
    tile = ProjectedTile(
        proj4="+proj=stere +lat_0=-90 +lon_0=0 +k=0.994 +x_0=500000 +y_0=500000 +R=1737400 +units=m +no_defs",
        x_min=500_000.0 - 1000, y_max=500_000.0 + 1000, pixel_size_x=100.0, pixel_size_y=100.0, width=20, height=20,
    )
    _, lat = tile.pixel_to_lonlat(10, 10)
    assert lat == pytest.approx(-90, abs=1e-9)
    assert tile.scale_factor(0, -90) == pytest.approx(0.994, abs=1e-6)


def test_transverse_mercator_scale_is_k0_on_the_central_meridian():
    tile = wac_tile_projection(NORTH_TM)
    assert tile.scale_factor(-104, 0) == pytest.approx(0.999, abs=1e-6)  # numerical derivative
    ground_x, _ = tile.pixel_ground_size_m(-104, 0)
    assert ground_x == pytest.approx(100 / 0.999, rel=1e-6)


def test_grid_has_requested_shape_and_matches_corners():
    tile = wac_tile_projection(NORTH_TM)
    grid = tile.grid(9)
    assert len(grid) == 9 and all(len(row) == 9 for row in grid)
    assert grid[0][0] == pytest.approx(tile.corners()[0])
    assert grid[8][8] == pytest.approx(tile.corners()[2])
    assert np.isfinite(np.array(grid)).all()
