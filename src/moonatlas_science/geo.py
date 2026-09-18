"""Pixel ↔ lunar geographic coordinates for SomBench rasters (docs/provenance/data-contract.md › Coordinate systems).

Every conversion goes through an explicit projected CRS on the IAU 2015 lunar sphere (R = 1737.4 km):

* IMP NAC tiles and ice patches carry their CRS and geotransform in the GeoTIFF.
* WAC crater tiles carry **no** CRS in the TIFF. Their only geolocation is `metadata.parquet`
  (projected BOUNDS plus the tile center, upper-left and lower-right corners in lat/lon). The tile
  projection is recovered by finding the single candidate among the Lunar Transverse Mercator zones
  and the Lunar Polar Stereographic caps that reproduces all three published points exactly.

Pixel coordinates are continuous with the origin at the outer edge of the upper-left pixel:
pixel (col, row) spans [col, col + 1) × [row, row + 1), so its center is (col + 0.5, row + 0.5).
Longitudes are returned in [-180, 180).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from pyproj import CRS, Proj, Transformer

LUNAR_RADIUS_M = 1_737_400.0
# Published metadata points are reproduced to ~1e-13°; anything above this is not the tile projection.
MATCH_TOLERANCE_DEG = 1e-6


def normalize_longitude(longitude):
    """Wraps degrees into [-180, 180)."""
    wrapped = (np.asarray(longitude, dtype=float) + 180.0) % 360.0 - 180.0
    return float(wrapped) if np.ndim(wrapped) == 0 else wrapped


def _tm(lon0: float, false_northing: float) -> str:
    return (f"+proj=tmerc +lat_0=0 +lon_0={lon0:g} +k=0.999 +x_0=250000 +y_0={false_northing:.0f} "
            f"+R={LUNAR_RADIUS_M:.0f} +units=m +no_defs")


def _ps(lat0: float) -> str:
    return (f"+proj=stere +lat_0={lat0:g} +lon_0=0 +k=0.994 +x_0=500000 +y_0=500000 "
            f"+R={LUNAR_RADIUS_M:.0f} +units=m +no_defs")


# Lunar Transverse Mercator: 45 zones of 8° (central meridians -176…176), k0 0.999, false easting 250 km,
# false northing 0 (north) or 2500 km (south). Lunar Polar Stereographic: k0 0.994, false origin 500 km.
WAC_CANDIDATE_PROJECTIONS: tuple[str, ...] = (
    *(_tm(lon0, fn) for lon0 in range(-176, 180, 8) for fn in (0, 2_500_000)),
    _ps(90),
    _ps(-90),
)


@lru_cache(maxsize=128)
def _to_geographic(proj4: str) -> Transformer:
    crs = CRS.from_proj4(proj4)
    return Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)


@lru_cache(maxsize=128)
def _to_projected(proj4: str) -> Transformer:
    crs = CRS.from_proj4(proj4)
    return Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)


@dataclass(frozen=True)
class ProjectedTile:
    """A north-up raster grid in a projected lunar CRS."""

    proj4: str
    x_min: float
    y_max: float
    pixel_size_x: float
    pixel_size_y: float
    width: int
    height: int

    def pixel_to_projected(self, col, row):
        return self.x_min + np.asarray(col, dtype=float) * self.pixel_size_x, self.y_max - np.asarray(
            row, dtype=float
        ) * self.pixel_size_y

    def pixel_to_lonlat(self, col, row):
        """Continuous pixel coordinates → (longitude, latitude) in degrees, longitude in [-180, 180)."""
        x, y = self.pixel_to_projected(col, row)
        lon, lat = _to_geographic(self.proj4).transform(x, y)
        return normalize_longitude(lon), (float(lat) if np.ndim(lat) == 0 else np.asarray(lat))

    def lonlat_to_pixel(self, lon, lat):
        x, y = _to_projected(self.proj4).transform(lon, lat)
        return (np.asarray(x) - self.x_min) / self.pixel_size_x, (self.y_max - np.asarray(y)) / self.pixel_size_y

    def center(self) -> tuple[float, float]:
        return self.pixel_to_lonlat(self.width / 2, self.height / 2)

    def corners(self) -> list[tuple[float, float]]:
        """Outer corners as (lon, lat): upper-left, upper-right, lower-right, lower-left."""
        cols = [0, self.width, self.width, 0]
        rows = [0, 0, self.height, self.height]
        lon, lat = self.pixel_to_lonlat(cols, rows)
        return [(float(a), float(b)) for a, b in zip(lon, lat, strict=True)]

    def grid(self, n: int) -> list[list[tuple[float, float]]]:
        """n × n outer-edge grid of (lon, lat), row by row from the top."""
        steps = np.linspace(0, 1, n)
        cols, rows = np.meshgrid(steps * self.width, steps * self.height)
        lon, lat = self.pixel_to_lonlat(cols, rows)
        return [[(float(lon[r, c]), float(lat[r, c])) for c in range(n)] for r in range(n)]

    def scale_factor(self, lon: float, lat: float) -> float:
        """Point scale of the (conformal) projection: projected length / true length on the sphere."""
        factors = Proj(self.proj4).get_factors(lon, lat)
        return float((factors.meridional_scale + factors.parallel_scale) / 2)

    def pixel_ground_size_m(self, lon: float, lat: float) -> tuple[float, float]:
        k = self.scale_factor(lon, lat)
        return self.pixel_size_x / k, self.pixel_size_y / k


def _angle_error(lon_a, lat_a, lon_b, lat_b) -> float:
    return max(abs(lat_a - lat_b), abs(normalize_longitude(lon_a - lon_b)))


def wac_tile_projection(meta: dict, width: int = 512, height: int = 512) -> ProjectedTile:
    """Recovers the projection of a WAC tile from its metadata.parquet row.

    Raises ValueError unless exactly one candidate reproduces the published center, upper-left and
    lower-right coordinates — the geolocation is never approximated.
    """
    x_min, x_max = float(meta["BOUNDS_XMIN"]), float(meta["BOUNDS_XMAX"])
    y_min, y_max = float(meta["BOUNDS_YMIN"]), float(meta["BOUNDS_YMAX"])
    checks = [
        ((x_min + x_max) / 2, (y_min + y_max) / 2, meta["CENTER_LONGITUDE"], meta["CENTER_LATITUDE"]),
        (x_min, y_max, meta["UPPER_LEFT_LONGITUDE"], meta["UPPER_LEFT_LATITUDE"]),
        (x_max, y_min, meta["LOWER_RIGHT_LONGITUDE"], meta["LOWER_RIGHT_LATITUDE"]),
    ]
    matches = []
    for proj4 in WAC_CANDIDATE_PROJECTIONS:
        transform = _to_geographic(proj4)
        error = 0.0
        for x, y, lon, lat in checks:
            got_lon, got_lat = transform.transform(x, y)
            error = max(error, _angle_error(got_lon, got_lat, float(lon), float(lat)))
        if error < MATCH_TOLERANCE_DEG:
            matches.append(proj4)
    if len(matches) != 1:
        raise ValueError(
            f"{meta.get('WAC_VIS_TILE', 'tile')}: expected exactly one projection reproducing the metadata, "
            f"found {len(matches)}"
        )
    return ProjectedTile(
        proj4=matches[0],
        x_min=x_min,
        y_max=y_max,
        pixel_size_x=(x_max - x_min) / width,
        pixel_size_y=(y_max - y_min) / height,
        width=width,
        height=height,
    )


def raster_tile(dataset) -> ProjectedTile:
    """ProjectedTile from an open rasterio dataset; requires a CRS and a north-up geotransform."""
    t = dataset.transform
    if dataset.crs is None:
        raise ValueError(f"{dataset.name}: raster has no CRS")
    if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
        raise ValueError(f"{dataset.name}: expected a north-up geotransform, got {tuple(t)[:6]}")
    return ProjectedTile(
        proj4=dataset.crs.to_proj4().replace("+no_defs=True", "+no_defs"),
        x_min=t.c,
        y_max=t.f,
        pixel_size_x=t.a,
        pixel_size_y=-t.e,
        width=dataset.width,
        height=dataset.height,
    )
