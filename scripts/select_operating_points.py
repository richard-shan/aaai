#!/usr/bin/env python
"""1-SE operating-point selection on pooled dev results (math_train + gsm8k).

For each requested policy family: find the best pooled-dev accuracy config,
compute its SE over per-problem means, and among configs whose accuracy is
within 1 SE of the best, select the one with the fewest mean think tokens.
Selection uses dev ONLY (pre-registered protocol). Merges into
<model-root>/analysis/selection_dev.json so families can be selected as their
sweeps finish.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from reasoncontrol.stages._stage import read_jsonl_zst

DEV_DATASETS = ("math_train", "gsm8k")
PARAM_KEYS = ("tau_exit", "patience_k", "alpha", "budget")


def load_dev(root: Path) -> pd.DataFrame:
    rows = []
    for phash_dir in sorted((root / "controller").glob("*")):
        for ds in DEV_DATASETS:
            for f in sorted((phash_dir / ds).glob("results_dev_seed*.jsonl.zst")):
                recs = read_jsonl_zst(f)
                if not recs:
                    continue
                pol = recs[0].get("policy", {})
                hf_loop = "n_forwards" in recs[0]
                for r in recs:
                    rows.append({"kind": pol.get("kind"), "phash": phash_dir.name,
                                 "dataset": ds, "problem_id": r["problem_id"],
                                 "correct": float(r["correct"]),
                                 "tokens": float(r["n_think_tokens"]),
                                 "hf_loop": hf_loop,
                                 **{k: pol.get(k) for k in PARAM_KEYS}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-root", default="runs/r1_qwen_1p5b")
    ap.add_argument("--families", default="exit_only,static_budget,budget_prompt")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.model_root)
    out_path = Path(a.out) if a.out else root / "analysis" / "selection_dev.json"
    df = load_dev(root)
    if df.empty:
        raise SystemExit("select: no dev results found")

    sel = json.loads(out_path.read_text()) if out_path.exists() else {}
    sel.setdefault("families", {})
    for fam in a.families.split(","):
        g = df[df["kind"] == fam]
        if fam in ("full", "exit_only", "steer_only"):
            g = g[g["hf_loop"]]  # controller families must come from the HF loop
        if g.empty:
            print(f"select[{fam}]: no dev points; skipped")
            continue
        stats = []
        for ph, gg in g.groupby("phash"):
            pm = gg.groupby("problem_id")["correct"].mean().to_numpy()
            acc = float(pm.mean())
            se = float(pm.std(ddof=1) / np.sqrt(len(pm))) if len(pm) > 1 else 0.0
            tok = float(gg.groupby("problem_id")["tokens"].mean().mean())
            params = {k: gg[k].iloc[0] for k in PARAM_KEYS
                      if pd.notna(gg[k].iloc[0])}
            stats.append({"phash": ph, "acc": acc, "se": se, "tokens": tok,
                          "n_problems": int(len(pm)), "params": params})
        best = max(stats, key=lambda s: s["acc"])
        thr = best["acc"] - best["se"]
        elig = [s for s in stats if s["acc"] >= thr]
        pick = min(elig, key=lambda s: s["tokens"])
        sel["families"][fam] = {**pick, "one_se_threshold": thr,
                                "best_acc": best["acc"], "best_se": best["se"],
                                "n_configs": len(stats)}
        print(f"select[{fam}]: {len(stats)} configs; best acc={best['acc']:.3f} "
              f"(se {best['se']:.3f}); pick {pick['phash']} "
              f"params={pick['params']} acc={pick['acc']:.3f} "
              f"tokens={pick['tokens']:.0f}")
    sel["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sel, indent=2, default=str))
    print(f"select: wrote {out_path}")


if __name__ == "__main__":
    main()
