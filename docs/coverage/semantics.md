# Coverage semantics

## Coverage is a footprint test

A model output covers a coordinate when the coordinate lies **inside that output's own footprint**: the
quadrilateral of a WAC or NAC tile, or a valid pixel of an ice patch in its native grid. Footprint corners are
the tile grid's own outer corners, published with every object.

**PROXIMITY IS NOT COVERAGE.** Distance to the nearest analyzed object is useful for ranking what is nearby; it
never makes a place analyzed. No radius, marker size or grouping threshold is a coverage statement.

### Known gap: the validator's containment check is a bounding box

`moonatlas_science.validate._within` tests a point against the **axis-aligned bounds** of a footprint, not the
footprint quadrilateral. For a tile whose grid is rotated relative to lat/lon — every polar stereographic tile
— those bounds are larger than the tile, so the check is conservative: it accepts a few points just outside the
true footprint. It never rejects a point that is inside.

The semantic definition above (the quadrilateral) is what a consumer should implement, and what
`examples/conformance/geometry.json` encodes. Tightening the validator is a welcome contribution, but it may
reject existing builds near tile edges, so it needs its own pull request, a processingVersion decision and
evidence of what changes.

## "No coverage" is a statement about the catalogue

*No coverage here* means this catalogue holds no model output for that location. It is **not** the absence of a
feature on the Moon, and it is not evidence that nothing is there. The Moon is globally explorable; the
catalogue is not globally covered.

## How coverage is measured

`moonatlas-science report coverage_inventory` rasterizes the union of analyzed footprints and reports covered
area per split, distinct footprints (near-duplicates collapsed) and geographic groupings. For the current build:

| Measure | Value |
|---------|-------|
| Surface covered by analyzed WAC tiles | 1,470,244 km² — 3.88 % of the Moon |
| Sum of individual tile areas | 2,621,440 km² (tiles overlap) |
| Distinct footprints (5 km threshold) | 926 |
| Test / validation / train share of the surface | 0.66 % / 0.67 % / 2.54 % |

Overlap matters twice: coverage must never be computed by adding tile areas, and detections from overlapping
tiles must never be summed into a crater count.
