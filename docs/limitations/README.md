# Limitations

What these outputs cannot tell you.

- **Coverage is partial.** The current build analyzes 3.88 % of the surface for craters, polar patches within
  ~10° of the poles for ice, and 130 NAC tiles for IMP. Absence in the catalogue is not absence on the Moon.
- **Detections are per tile.** Overlapping tiles repeat the same physical craters (53,842 of 80,454 detections
  have a likely repeat), so detection counts are not crater counts and must not be summed across tiles.
- **Crater labels have completeness limits.** The reference catalogue is near-complete only above a diameter
  threshold, so agreement metrics are descriptive, not precision and recall against truth.
- **Ice prospectivity emulates an expert map.** It inherits that map's assumptions and is not a measurement of
  ice, not a probability, and not a resource estimate. Unconstrained regression means values can leave 0–1.
- **IMP masks are segmentation on a 256 m tile,** with an uncalibrated score; they are candidates for
  geological interest, not confirmations, and not active volcanism.
- **In-sample results are not evaluation.** Train and validation predictions show behaviour, not performance.
- **Small reproduction samples.** Our held-out checks use 100 and 10 tiles; they confirm consistency, not new
  benchmark numbers.
- **Derived metrics are MOONATLAS's own.** Interest scores and agreement metrics are simple, documented and
  versioned, but they are not validated scientific measurements and are not NASA or IBM outputs.
- **Research use only.** The upstream model cards state their outputs are not validated for operational
  decisions. This pipeline does not change that.
