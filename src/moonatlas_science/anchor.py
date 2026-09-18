"""Raw model peak vs MOONATLAS discovery anchor for dense ice prospectivity patches (docs/provenance/data-contract.md › IceSector).

The raw peak is the model's maximum and is never altered. The discovery anchor is a MOONATLAS-derived locator: the
maximum over covered pixels at least EDGE_MARGIN_PX from the valid edge (patch border or no-data), because every patch
is inferred independently and its outermost pixels carry edge effects (south test batch: error vs reference is 1.29×
baseline within 0–3 px and baseline from 4 px on; negative overshoot 2.2× within 0–3 px).
"""

from __future__ import annotations

import numpy as np

METHOD = "edge-safe-max-v1"
# One model token. Verified in the released checkpoint, not taken from the "ps8" file name: config
# backbone_patch_size 8, per-modality patch projections of 64 = 8×8 inputs, and a 32×32 positional grid for 256 px.
EDGE_MARGIN_PX = 8


def edge_distance(covered: np.ndarray) -> np.ndarray:
    """Chessboard distance (px) from each covered pixel to the nearest uncovered pixel or the patch border; 0 on the edge."""
    distance = np.zeros(covered.shape, dtype=np.int32)
    current = covered.astype(bool)
    step = 0
    while current.any():
        distance[current] = step
        p = np.pad(current, 1)
        current = (p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
                   & p[:-2, :-2] & p[:-2, 2:] & p[2:, :-2] & p[2:, 2:])
        step += 1
    return distance


def argmax_where(values: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    if not mask.any():
        raise ValueError("no eligible pixel")
    row, col = np.unravel_index(int(np.argmax(np.where(mask, values, -np.inf))), values.shape)
    return int(row), int(col)


def peak_and_anchor(values: np.ndarray, covered: np.ndarray, margin: int = EDGE_MARGIN_PX) -> tuple[dict, dict | None]:
    """(raw peak, discovery anchor) as pixel records {row, col, value, edgeDistancePx}.

    The anchor is None when no covered pixel is at least `margin` px from the valid edge (e.g. a thin sliver of coverage
    at the 80° boundary): such a patch has no edge-safe location and is never selected as a discovery.
    """
    if not covered.any():
        raise ValueError("patch has no covered pixel")
    distance = edge_distance(covered)
    records = []
    for mask in (covered, covered & (distance >= margin)):
        if not mask.any():
            records.append(None)
            continue
        row, col = argmax_where(values, mask)
        records.append({"row": row, "col": col, "value": float(values[row, col]), "edgeDistancePx": int(distance[row, col])})
    return records[0], records[1]


if __name__ == "__main__":
    covered = np.ones((20, 20), bool)
    covered[:, 15:] = False  # no-data band acts as an edge
    values = np.zeros((20, 20))
    values[0, 0], values[10, 13], values[9, 6] = 1.05, 0.99, 0.97
    d = edge_distance(covered)
    assert d[0, 0] == 0 and d[10, 14] == 0 and d[10, 10] == 4 and d[9, 6] == 6 and not covered[10, 15]
    peak, anchor = peak_and_anchor(values, covered, margin=4)
    assert (peak["row"], peak["col"], peak["edgeDistancePx"]) == (0, 0, 0)  # the model maximum is kept as is
    assert (anchor["row"], anchor["col"]) == (9, 6) and anchor["edgeDistancePx"] >= 4
    sliver = np.zeros((20, 20), bool)
    sliver[0, :5] = True  # one row of coverage: no pixel is 4 px from the edge
    assert peak_and_anchor(values, sliver, margin=4)[1] is None
    print("ok")
