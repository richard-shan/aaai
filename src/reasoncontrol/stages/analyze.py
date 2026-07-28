"""Stage 10: aggregate all controller/baseline runs -> Pareto tables, paired
stats, wall-clock table, figures."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from ..analysis.pareto import (accuracy_at_budget, hypervolume, pareto_frontier,
                               tokens_at_accuracy)
from ..analysis.stats import cluster_bootstrap_ci, non_inferiority, paired_bootstrap
from ._stage import mark_done, read_jsonl_zst, setup, stage_args

BUDGETS = (2000, 4000, 8000)


def load_runs(controller_root: Path, split: str) -> pd.DataFrame:
    frames = []
    for phash_dir in sorted(controller_root.glob("*")):
        for ds_dir in sorted(p for p in phash_dir.glob("*") if p.is_dir()):
            for f in sorted(ds_dir.glob(f"results_{split}_seed*.jsonl.zst")):
                recs = read_jsonl_zst(f)
                if not recs:
                    continue
                df = pd.DataFrame([{k: r.get(k) for k in
                                    ("problem_id", "rollout_id", "n_think_tokens",
                                     "correct")} for r in recs])
                pol = recs[0].get("policy", {})
                df["kind"] = pol.get("kind", "?")
                df["phash"] = phash_dir.name
                df["dataset"] = ds_dir.name
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    args = stage_args(__doc__)
    cfg, paths, stage_dir = setup(args, "analyze")
    t0 = time.time()
    import os
    split = os.environ.get("RC_SPLIT", "dev")
    df = load_runs(paths.model_root / "controller", split)
    if df.empty:
        print("analyze: no runs found")
        return
    out_dir = paths.analysis()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows, report = [], {}
    for ds, dgrp in df.groupby("dataset"):
        points = {}
        for (kind, phash), grp in dgrp.groupby(["kind", "phash"]):
            acc = cluster_bootstrap_ci(grp, "correct", n_boot=2000)
            tokc = cluster_bootstrap_ci(grp, "n_think_tokens", n_boot=2000)
            summary_rows.append({"dataset": ds, "kind": kind, "phash": phash,
                                 "acc": acc["mean"], "acc_lo": acc["lo"],
                                 "acc_hi": acc["hi"], "tokens": tokc["mean"],
                                 "tokens_lo": tokc["lo"], "tokens_hi": tokc["hi"],
                                 "n": acc["n_problems"]})
            points.setdefault(kind, []).append((tokc["mean"], acc["mean"]))
        ds_report = {"families": {}}
        for kind, pts in points.items():
            fam = {"points": pts, "frontier": pareto_frontier(pts),
                   "acc_at_budget": {b: accuracy_at_budget(pts, b) for b in BUDGETS}}
            ds_report["families"][kind] = fam
        # paired comparisons: full controller vs each family's best point
        full = dgrp[dgrp.kind == "full"]
        if not full.empty:
            best_full_hash = full.groupby("phash")["correct"].mean().idxmax()
            best_full = full[full.phash == best_full_hash]
            comps = {}
            for kind in points:
                if kind == "full":
                    continue
                other = dgrp[dgrp.kind == kind]
                by_acc = other.groupby("phash")["correct"].mean()
                best_hash = by_acc.idxmax()
                ob = other[other.phash == best_hash]
                comps[kind] = {
                    "tokens": paired_bootstrap(best_full, ob, "n_think_tokens"),
                    "acc_non_inferiority": non_inferiority(best_full, ob, "correct",
                                                           margin=0.02)}
            ds_report["paired_vs_full"] = comps
        report[ds] = ds_report

    pd.DataFrame(summary_rows).to_csv(out_dir / f"summary_{split}.csv", index=False)
    (out_dir / f"report_{split}.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(pd.DataFrame(summary_rows).to_string(index=False))
    _plot(summary_rows, out_dir, split)
    mark_done(stage_dir, t0)


def _plot(rows, out_dir: Path, split: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    df = pd.DataFrame(rows)
    for ds, grp in df.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for kind, g in grp.groupby("kind"):
            g = g.sort_values("tokens")
            ax.errorbar(g.tokens, g.acc,
                        yerr=[g.acc - g.acc_lo, g.acc_hi - g.acc],
                        marker="o", capsize=2, label=kind)
        ax.set_xlabel("mean thinking tokens")
        ax.set_ylabel("accuracy")
        ax.set_title(f"{ds} ({split})")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"pareto_{ds}_{split}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
