"""Cluster (problem-level) bootstrap statistics.

Unit of analysis = problem; all conditions share problems and per-(problem,
rollout) seeds, so deltas are paired at the problem level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _problem_means(df: pd.DataFrame, value: str) -> pd.Series:
    return df.groupby("problem_id")[value].mean()


def cluster_bootstrap_ci(df: pd.DataFrame, value: str, n_boot: int = 10000,
                         alpha: float = 0.05, seed: int = 0) -> dict:
    """CI for the mean of `value`, resampling problems with replacement."""
    means = _problem_means(df, value).to_numpy()
    rng = np.random.default_rng(seed)
    n = len(means)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = means[idx].mean(axis=1)
    return {"mean": float(means.mean()),
            "lo": float(np.quantile(boots, alpha / 2)),
            "hi": float(np.quantile(boots, 1 - alpha / 2)),
            "n_problems": n}


def paired_bootstrap(df_a: pd.DataFrame, df_b: pd.DataFrame, value: str,
                     n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> dict:
    """CI and one-sided p-values for mean(a) - mean(b), paired by problem."""
    ma, mb = _problem_means(df_a, value), _problem_means(df_b, value)
    common = ma.index.intersection(mb.index)
    d = (ma[common] - mb[common]).to_numpy()
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    return {"delta": float(d.mean()),
            "lo": float(np.quantile(boots, alpha / 2)),
            "hi": float(np.quantile(boots, 1 - alpha / 2)),
            "p_a_lt_b": float((boots >= 0).mean()),   # p-value for H1: a < b
            "p_a_gt_b": float((boots <= 0).mean()),
            "n_problems": n}


def non_inferiority(df_a: pd.DataFrame, df_b: pd.DataFrame, value: str,
                    margin: float, n_boot: int = 10000, seed: int = 0) -> dict:
    """One-sided: is mean(a) > mean(b) - margin? (a = ours, b = reference).
    Concluded when the lower CI bound of the paired delta exceeds -margin."""
    r = paired_bootstrap(df_a, df_b, value, n_boot=n_boot, seed=seed)
    r["margin"] = margin
    r["non_inferior"] = bool(r["lo"] > -margin)
    return r


def holm_correction(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out
