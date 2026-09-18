"""Spherical lunar map projections in pure Python (no pyproj), for the fixture generator.

The science pipeline uses pyproj; tests/test_projection.py checks these formulas against it,
and lib/lunar/polar.ts mirrors the polar stereographic case. Degrees and metres; records follow
lib/validation/schemas.ts › ProjectionSchema.
"""

from __future__ import annotations

import math

LUNAR_RADIUS_M = 1_737_400.0
DEG = math.pi / 180


def polar_stereographic(pole: str, scale_factor: float = 0.994, false_origin_m: float = 500_000.0) -> dict:
    return {
        "kind": "polar-stereographic",
        "latitudeOfOriginDeg": -90 if pole == "south" else 90,
        "centralMeridianDeg": 0,
        "scaleFactor": scale_factor,
        "falseEastingM": false_origin_m,
        "falseNorthingM": false_origin_m,
        "radiusM": LUNAR_RADIUS_M,
    }


def transverse_mercator(central_meridian_deg: float, false_northing_m: float = 0.0) -> dict:
    return {
        "kind": "transverse-mercator",
        "latitudeOfOriginDeg": 0,
        "centralMeridianDeg": central_meridian_deg,
        "scaleFactor": 0.999,
        "falseEastingM": 250_000.0,
        "falseNorthingM": false_northing_m,
        "radiusM": LUNAR_RADIUS_M,
    }


def _wrap(longitude: float) -> float:
    value = (longitude + 180.0) % 360.0 - 180.0
    return -180.0 if value >= 180.0 else value


def forward(p: dict, latitude: float, longitude: float) -> tuple[float, float]:
    lam = (longitude - p["centralMeridianDeg"]) * DEG
    phi = latitude * DEG
    rk = p["radiusM"] * p["scaleFactor"]
    if p["kind"] == "polar-stereographic":
        south = p["latitudeOfOriginDeg"] < 0
        rho = 2 * rk * math.tan(math.pi / 4 + (phi if south else -phi) / 2)
        return p["falseEastingM"] + rho * math.sin(lam), p["falseNorthingM"] + (1 if south else -1) * rho * math.cos(lam)
    b = math.cos(phi) * math.sin(lam)
    x = 0.5 * rk * math.log((1 + b) / (1 - b))
    y = rk * (math.atan2(math.tan(phi), math.cos(lam)) - p["latitudeOfOriginDeg"] * DEG)
    return p["falseEastingM"] + x, p["falseNorthingM"] + y


def inverse(p: dict, x: float, y: float) -> tuple[float, float]:
    """Returns (latitude, longitude)."""
    rk = p["radiusM"] * p["scaleFactor"]
    dx, dy = x - p["falseEastingM"], y - p["falseNorthingM"]
    if p["kind"] == "polar-stereographic":
        south = p["latitudeOfOriginDeg"] < 0
        c = 2 * math.atan(math.hypot(dx, dy) / (2 * rk))
        latitude = (c - math.pi / 2) if south else (math.pi / 2 - c)
        longitude = p["centralMeridianDeg"] + math.atan2(dx, dy if south else -dy) / DEG
        return latitude / DEG, _wrap(longitude)
    d = dy / rk + p["latitudeOfOriginDeg"] * DEG
    xp = dx / rk
    latitude = math.asin(math.sin(d) / math.cosh(xp))
    longitude = p["centralMeridianDeg"] + math.atan2(math.sinh(xp), math.cos(d)) / DEG
    return latitude / DEG, _wrap(longitude)


def grid(x_min: float, y_max: float, pixel_size: float, width: int, height: int) -> dict:
    return {"xMinM": x_min, "yMaxM": y_max, "pixelSizeXM": pixel_size, "pixelSizeYM": pixel_size,
            "widthPx": width, "heightPx": height}


def pixel_to_latlon(p: dict, g: dict, col: float, row: float) -> tuple[float, float]:
    return inverse(p, g["xMinM"] + col * g["pixelSizeXM"], g["yMaxM"] - row * g["pixelSizeYM"])
