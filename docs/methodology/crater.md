# WAC crater detection

**Model:** NASA-IBM LFM crater detection checkpoint (Faster R-CNN head with LoRA), run through the official
TerraTorch configuration on LROC WAC tiles of 512 × 512 px at ~100 m/px (51.2 km on a side).

## Inference

The checkpoint is loaded strictly and run per tile. The raw artifact keeps every predicted box with its score,
before any threshold.

## Postprocessing

Upstream applies a task score filter of 0.05 and NMS at IoU 0.5. MOONATLAS publishes detections at the
**deployment threshold 0.5** used by the upstream plotting path, and records the threshold with the data.

## Normalization

- The tile's projection is recovered from its metadata and verified (see `docs/datasets/sombench.md`).
- Each box becomes a lunar coordinate (centre) and a diameter in km, from the box size and the tile resolution.
- Boxes touching a tile edge are flagged as truncated; a truncated box has no reliable diameter.
- Reference boxes from the SomBench labels (derived from the Robbins catalogue) are matched greedily
  one-to-one by descending confidence at IoU ≥ 0.5, the COCO AP@50 criterion. The match flag is a **MOONATLAS
  derived metric**, not a statement about the Moon.

## What a detection is and is not

A detection is a model prediction on one tile. It is **not** a confirmed crater. Because SomBench tiles
overlap, the same physical crater appears in several tiles: in the current build 53,842 of 80,454 detections
have a likely repeat elsewhere, so a detection count is **not** a crater count. A prediction that matches no
reference box is an **unmatched prediction**, not a new crater: the reference catalogue has its own
completeness limits.
