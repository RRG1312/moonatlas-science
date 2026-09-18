# SomBench

The benchmark suite released with the NASA-IBM Lunar Foundation Model. MOONATLAS uses three of its datasets:

| Dataset | Used for | Reference labels |
|---------|----------|------------------|
| `Sombench-WAC-Crater-Detection` | WAC tiles + crater boxes | Derived from Robbins (2019) |
| `Sombench-Ice-Prospectivity-Regression` | Polar input layers + target map | Coyan et al. (2025) prospectivity workflow |
| `Sombench-IMP-Segmentation` | NAC tiles + masks | Hargitai et al. (2025) |

All three are distributed under CC BY 4.0. MOONATLAS downloads them at pinned revisions and never redistributes
bulk data; see `NOTICE` for attribution and the modification notice.

## Georeferencing

SomBench WAC tiles carry no CRS in the TIFF. MOONATLAS recovers the projection from the tile metadata: the
single candidate among the 45 lunar transverse Mercator zones (k0 0.999, false easting 250 km, false northing
0 or 2500 km) and the two polar stereographic caps (k0 0.994, false origin 500 km) that reproduces the tile
centre and corners to < 1e-6°. Zero or several matches is a hard failure. In the current build all 1,000 tiles
resolve exactly (870 transverse Mercator, 130 polar stereographic; residuals ≈ 1e-13°).

`tests/test_projection.py` checks the pure-Python projection formulas against `pyproj`, and the validator
re-derives every published centre and corner from the stored `projection` + `grid`.
