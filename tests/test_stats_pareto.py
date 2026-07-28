import numpy as np
import pandas as pd

from reasoncontrol.analysis.pareto import (accuracy_at_budget, hypervolume,
                                           pareto_frontier, tokens_at_accuracy,
                                           upper_hull)
from reasoncontrol.analysis.stats import (cluster_bootstrap_ci, holm_correction,
                                          non_inferiority, paired_bootstrap)


def _df(vals, seeds=4):
    rows = []
    for pid, v in enumerate(vals):
        for s in range(seeds):
            rows.append({"problem_id": f"p{pid}", "correct": v, "tokens": 100 + pid})
    return pd.DataFrame(rows)


def test_paired_bootstrap_detects_difference():
    rng = np.random.default_rng(0)
    a = _df(rng.binomial(1, 0.8, 200))
    b = _df(rng.binomial(1, 0.5, 200))
    r = paired_bootstrap(a, b, "correct")
    assert r["delta"] > 0.15 and r["lo"] > 0


def test_non_inferiority_pass_and_fail():
    vals = np.random.default_rng(1).binomial(1, 0.7, 300)
    same = non_inferiority(_df(vals), _df(vals), "correct", margin=0.02)
    assert same["non_inferior"]
    worse = non_inferiority(_df(np.zeros(300, int)), _df(np.ones(300, int)),
                            "correct", margin=0.02)
    assert not worse["non_inferior"]


def test_cluster_ci_contains_mean():
    df = _df(np.random.default_rng(2).binomial(1, 0.6, 100))
    r = cluster_bootstrap_ci(df, "correct", n_boot=2000)
    assert r["lo"] <= r["mean"] <= r["hi"]


def test_holm_monotone():
    out = holm_correction({"a": 0.01, "b": 0.04, "c": 0.3})
    assert out["a"] <= out["b"] <= out["c"] <= 1.0


def test_pareto_frontier_and_hull():
    pts = [(100, 0.5), (200, 0.7), (150, 0.6), (300, 0.65), (400, 0.9)]
    front = pareto_frontier(pts)
    assert (300, 0.65) not in front          # dominated by (200, 0.7)
    hull = upper_hull(front)
    assert hull[0] == (100, 0.5) and hull[-1] == (400, 0.9)


def test_accuracy_at_budget_interpolates():
    pts = [(100, 0.5), (300, 0.9)]
    assert abs(accuracy_at_budget(pts, 200) - 0.7) < 1e-9
    assert accuracy_at_budget(pts, 50) is None       # unattainable budget
    assert accuracy_at_budget(pts, 1000) == 0.9


def test_tokens_at_accuracy():
    pts = [(100, 0.5), (300, 0.9)]
    assert abs(tokens_at_accuracy(pts, 0.7) - 200) < 1e-9
    assert tokens_at_accuracy(pts, 0.95) is None


def test_hypervolume_orders_dominance():
    better = [(100, 0.9)]
    worse = [(200, 0.6)]
    kw = dict(ref_tokens=400, ref_acc=0.0, t_range=(0, 400), a_range=(0, 1))
    assert hypervolume(better, **kw) > hypervolume(worse, **kw)
