"""The pure-Python projections used by the fixture generator must match PROJ exactly (they mirror lib/lunar/polar.ts)."""

import numpy as np
import pytest
from pyproj import CRS, Transformer

from moonatlas_science import projection as proj

CASES = [
    (proj.polar_stereographic("south"), [(-81.450066, -35.371514), (-85.25, 120.0), (-80.02, -179.5), (-89.9, 10.0)]),
    (proj.polar_stereographic("north"), [(84.3, 45.5), (80.01, -120.0), (88.8, 179.9)]),
    (proj.transverse_mercator(32.0), [(4.342116, 33.699627), (-10.0, 30.0), (20.0, 36.0)]),
    (proj.transverse_mercator(-160.0, 2_500_000.0), [(-53.153829, -169.176758), (-52.2, -170.36), (-54.09, -167.94)]),
]


def pyproj_for(p: dict) -> CRS:
    kind = "stere" if p["kind"] == "polar-stereographic" else "tmerc"
    return CRS.from_proj4(
        f"+proj={kind} +lat_0={p['latitudeOfOriginDeg']} +lon_0={p['centralMeridianDeg']} +k={p['scaleFactor']} "
        f"+x_0={p['falseEastingM']} +y_0={p['falseNorthingM']} +R={p['radiusM']} +units=m +no_defs"
    )


@pytest.mark.parametrize(("p", "points"), CASES, ids=["south-polar", "north-polar", "tm-32E", "tm-160W-south"])
def test_forward_and_inverse_match_proj(p, points):
    crs = pyproj_for(p)
    forward = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
    for lat, lon in points:
        ex, ey = forward.transform(lon, lat)
        x, y = proj.forward(p, lat, lon)
        assert (x, y) == pytest.approx((ex, ey), abs=1e-4)  # 0.1 mm
        back_lat, back_lon = proj.inverse(p, x, y)
        assert back_lat == pytest.approx(lat, abs=1e-9)
        assert np.isclose((back_lon - lon + 180) % 360 - 180, 0, atol=1e-9)


def test_pixel_to_latlon_uses_the_grid_edge_origin():
    p = proj.polar_stereographic("south")
    g = proj.grid(320006.48, 741064.88, 238.56, 256, 256)
    lat, lon = proj.pixel_to_latlon(p, g, 0.5, 0.5)
    assert (lat, lon) == pytest.approx((-80.04938294861823, -36.742614308232426), abs=1e-8)  # pyproj on the GeoTIFF
