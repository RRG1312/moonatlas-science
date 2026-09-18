# Polar ice prospectivity

**Model:** NASA-IBM LFM ice prospectivity checkpoint, a dense regression over 8 input layers (9 channels) on
256 × 256 patches at 240 m/px, within ~10° of each pole.

## What the model predicts

It emulates a **knowledge-driven expert prospectivity map** (the reference layer distributed with SomBench).
Its output is a continuous score on a 0–1 target scale. It is **not** a probability and **not** a measurement
of ice. Because the regression is unconstrained, raw values can fall slightly outside 0–1.

## Normalization

- The raw array is kept unclipped, with `min`, `mean`, `p90`, `max` recomputed from it.
- Raw rasters are encoded losslessly; display rasters clip to 0–1 purely for colour, and the validator checks
  that display equals `clip(raw)` with the same coverage mask.
- Each patch keeps its own projection and grid; `mosaic.py` composes one seamless mosaic per pole in that
  native polar stereographic grid and rejects overlapping placements.
- The published **anchor** is an edge-safe maximum: a MOONATLAS-derived locator for pointing at a patch, not the
  model peak. Both are published so they can never be confused.

## Agreement with the reference

RMSE and MAE against the reference map are **MOONATLAS derived metrics** computed per patch. The upstream card
publishes regression errors rather than a per-split benchmark table, so MOONATLAS reports its own numbers and
does not claim a reproduction score for this task.
