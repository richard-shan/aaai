"""Pareto-frontier analysis with the symmetric-protocol rules from the plan:

- every family contributes k dev-selected operating points, evaluated once on
  test; interpolation happens only on the upper-left convex hull, justified by
  randomized policy mixing (a coin flip between two configs attains any convex
  combination of (E[tokens], E[accuracy]));
- primary metrics: accuracy-at-matched-budget and tokens-at-matched-accuracy;
- hypervolume is secondary (min-max normalized, hull recomputed per bootstrap
  resample by the caller).
"""
from __future__ import annotations

import numpy as np


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper-left non-dominated subset of (tokens, accuracy), sorted by tokens."""
    pts = sorted(points)
    front: list[tuple[float, float]] = []
    best = -np.inf
    for t, a in pts:
        if a > best:
            front.append((t, a))
            best = a
    return front


def upper_hull(front: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper concave hull over the frontier (mixing makes chords attainable)."""
    hull: list[tuple[float, float]] = []
    for p in front:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            # drop hull[-1] if it lies below the chord hull[-2] -> p
            if (y2 - y1) * (p[0] - x1) <= (p[1] - y1) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def accuracy_at_budget(front: list[tuple[float, float]], budget: float) -> float | None:
    """Best attainable accuracy at mean-token budget via hull interpolation."""
    hull = upper_hull(pareto_frontier(front))
    if not hull or budget < hull[0][0]:
        return None
    prev = hull[0]
    for t, a in hull[1:]:
        if t > budget:
            frac = (budget - prev[0]) / (t - prev[0])
            return float(prev[1] + frac * (a - prev[1]))
        prev = (t, a)
    return float(hull[-1][1])


def tokens_at_accuracy(front: list[tuple[float, float]], target_acc: float) -> float | None:
    """Fewest mean tokens attaining target accuracy via hull interpolation."""
    hull = upper_hull(pareto_frontier(front))
    prev = None
    for t, a in hull:
        if a >= target_acc:
            if prev is None or a == prev[1]:
                return float(t)
            frac = (target_acc - prev[1]) / (a - prev[1])
            return float(prev[0] + frac * (t - prev[0]))
        prev = (t, a)
    return None


def hypervolume(front: list[tuple[float, float]], ref_tokens: float,
                ref_acc: float, t_range: tuple[float, float],
                a_range: tuple[float, float]) -> float:
    """Secondary metric; min-max normalized axes; dominated area vs reference."""
    def nt(t):
        return (t - t_range[0]) / max(t_range[1] - t_range[0], 1e-9)

    def na(a):
        return (a - a_range[0]) / max(a_range[1] - a_range[0], 1e-9)

    pts = pareto_frontier(front)
    hv, prev_t = 0.0, nt(ref_tokens)
    for t, a in sorted(pts, reverse=True):
        tn, an = nt(t), na(a)
        if an <= na(ref_acc) or tn >= prev_t:
            continue
        hv += (prev_t - tn) * (an - na(ref_acc))
        prev_t = tn
    return float(hv)
