"""Causal validation of steering directions (H2), with instruments that are
independent of the vector's construction:

1. behavioral effect over thousands of chunks: +alpha must raise / -alpha must
   lower the target-phase rate under BOTH the regex labeler and the LLM judge
   (judge breaks the regex circularity), with bootstrap CIs;
2. paired accuracy non-inferiority on >=200 paired dev rollouts: accept a
   (layer, alpha) iff the upper CI bound of the paired accuracy drop < 2%;
3. conclude-propensity test for v_conv: steering +/- v_conv must shift the
   rate of emitting </think> within a fixed horizon (extends H2 to the
   convergence direction).
"""
from __future__ import annotations

import numpy as np

from ..analysis.stats import paired_bootstrap
from ..labeling.phase_regex import label_phase


def phase_rate(texts: list[str], phase: str) -> float:
    if not texts:
        return float("nan")
    return float(np.mean([label_phase(t) == phase for t in texts]))


def phase_rate_shift(steered_chunks: list[str], unsteered_chunks: list[str],
                     phase: str, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    s = np.array([label_phase(t) == phase for t in steered_chunks], dtype=float)
    u = np.array([label_phase(t) == phase for t in unsteered_chunks], dtype=float)
    deltas = []
    for _ in range(n_boot):
        deltas.append(rng.choice(s, len(s)).mean() - rng.choice(u, len(u)).mean())
    deltas = np.array(deltas)
    return {"rate_steered": float(s.mean()), "rate_unsteered": float(u.mean()),
            "delta": float(s.mean() - u.mean()),
            "lo": float(np.quantile(deltas, 0.025)),
            "hi": float(np.quantile(deltas, 0.975))}


def accept_steering(df_steered, df_unsteered, accuracy_col: str = "correct",
                    margin: float = 0.02) -> dict:
    """Paired per-problem accuracy comparison; accept iff upper CI bound of the
    drop (unsteered - steered) is below `margin`."""
    r = paired_bootstrap(df_steered, df_unsteered, accuracy_col)
    # delta = steered - unsteered; worst plausible accuracy drop = -lower CI bound
    r["max_plausible_drop"] = float(max(0.0, -r["lo"]))
    r["accepted"] = r["max_plausible_drop"] < margin
    return r


def conclude_propensity(texts_pos: list[str], texts_neg: list[str],
                        marker: str = "</think>") -> dict:
    p = float(np.mean([marker in t for t in texts_pos])) if texts_pos else float("nan")
    n = float(np.mean([marker in t for t in texts_neg])) if texts_neg else float("nan")
    return {"rate_plus": p, "rate_minus": n, "delta": p - n}
