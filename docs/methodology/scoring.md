# MOONATLAS derived metrics

Every value here is computed by MOONATLAS from model outputs. None is a NASA or IBM output, and each carries a
versioned formula id that travels with the value.

Helper: `percentileRank(x, P) = 100 × (count(v < x) + 0.5 × count(v == x)) / |P|`, rounded to one decimal; with
`|P| = 1` the result is 50.0 (`moonatlas_science/scoring.py`).

| Formula id | Applies to | Definition | Population |
|-----------|------------|------------|------------|
| `crater-diameter-rank-v1` | Crater detection | `percentileRank(diameterKm)` | Non-truncated predictions with confidence ≥ 0.5. Published as a descriptive **diameter rank** |
| `crater-region-density-rank-v1` | WAC region | `percentileRank(detectionCount)` | All processed regions |
| `ice-p90-bounded-v2` | Ice patch | `round1(100 × clamp(raw.p90, 0, 1))` | — (P90 resists single-pixel peaks; the clamp bounds the unconstrained regression and is part of the id) |
| `imp-area-rank-v1` | IMP observation | `percentileRank(areaM2)` | Observations with predicted area > 0 |

Rules:

- A formula change gets a new id. Ids are never silently redefined.
- Reference annotations never feed a score.
- Agreement metrics (IoU, RMSE, match flags) are derived metrics too, and are labelled as such.
- A score is a ranking aid, not a scientific measurement, and never a claim about the Moon.
