"""MOONATLAS DERIVED METRIC helpers — MOONATLAS scoring formulas (docs/methodology/scoring.md).

The arithmetic deliberately performs the same floating-point operations in the same
order as the TypeScript implementation, and rounds half-up like JavaScript's
Math.round (Python's round() is half-even and would drift by one step).
tests/data-integrity.test.ts checks both implementations agree exactly.
"""

from __future__ import annotations

import math


def round1(value: float) -> float:
    """Same as `Math.round(value * 10) / 10` in JavaScript."""
    return math.floor(value * 10 + 0.5) / 10


def percentile_rank(value: float, population: list[float]) -> float:
    """Mid-rank percentile in [0, 100]; ties share the average rank."""
    if not population:
        raise ValueError("percentile_rank needs a non-empty population")
    below = sum(1 for v in population if v < value)
    equal = sum(1 for v in population if v == value)
    return round1((100 * (below + 0.5 * equal)) / len(population))


def ice_p90_bounded_score(p90: float) -> float:
    """ice-p90-bounded-v2: the raw regression P90 bounded to the [0, 1] target scale (a derived display transform)."""
    return round1(100 * min(1.0, max(0.0, p90)))
