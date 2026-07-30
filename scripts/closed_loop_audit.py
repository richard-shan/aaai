#!/usr/bin/env python
"""Closed-loop calibration audit (pre-registered D6 input): does the conv
probe's confidence at the moment of intervention predict final correctness?

For every exit_only/full dev config: take p_conv at the exit boundary
(actions_log), bin it against empirical accuracy, and split accuracy/tokens by
exited_early to separate 'probe fired' rollouts from 'hit the cap' rollouts.
Writes <root>/analysis/closed_loop_audit.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from reasoncontrol.stages._stage import read_jsonl_zst

ROOT = Path("runs/r1_qwen_1p5b")
DEV_DATASETS = ("math_train", "gsm8k")
BINS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001)


def exit_p(rec) -> float | None:
    for row in rec.get("actions_log") or []:
        # rows: [chunk_idx, action, p_conv, aux, cum_tokens]
        if len(row) >= 3 and row[1] == "exit":
            return float(row[2])
    return None


def main():
    out = {"families": {}}
    for phash_dir in sorted((ROOT / "controller").glob("*")):
        rows, pol = [], None
        for ds in DEV_DATASETS:
            for f in sorted((phash_dir / ds).glob("results_dev_seed*.jsonl.zst")):
                recs = read_jsonl_zst(f)
                if not recs or "actions_log" not in recs[0]:
                    break
                k = recs[0].get("policy", {}).get("kind")
                if k not in ("exit_only", "full"):
                    break
                pol = recs[0]["policy"]
                for r in recs:
                    rows.append({"correct": float(r["correct"]),
                                 "tokens": float(r["n_think_tokens"]),
                                 "exited": bool(r.get("exited_early")),
                                 "p_exit": exit_p(r)})
        if not rows or pol is None:
            continue
        df = pd.DataFrame(rows)
        ex, nx = df[df.exited], df[~df.exited]
        rel = []
        pe = ex.dropna(subset=["p_exit"])
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            b = pe[(pe.p_exit >= lo) & (pe.p_exit < hi)]
            if len(b):
                rel.append({"bin": [lo, min(hi, 1.0)], "n": int(len(b)),
                            "mean_p": float(b.p_exit.mean()),
                            "acc": float(b.correct.mean())})
        entry = {"kind": pol["kind"], "tau_exit": pol.get("tau_exit"),
                 "patience_k": pol.get("patience_k"), "n": int(len(df)),
                 "frac_exited": float(df.exited.mean()),
                 "acc_exited": float(ex.correct.mean()) if len(ex) else None,
                 "acc_not_exited": float(nx.correct.mean()) if len(nx) else None,
                 "tokens_exited": float(ex.tokens.mean()) if len(ex) else None,
                 "tokens_not_exited": float(nx.tokens.mean()) if len(nx) else None,
                 "mean_p_at_exit": float(pe.p_exit.mean()) if len(pe) else None,
                 "reliability": rel}
        out["families"][phash_dir.name] = entry
        print(f"audit[{pol['kind']} tau={pol.get('tau_exit')} "
              f"k={pol.get('patience_k')}]: exited {entry['frac_exited']:.2f} "
              f"acc(exited)={entry['acc_exited']} acc(cap)={entry['acc_not_exited']} "
              f"mean_p_at_exit={entry['mean_p_at_exit']}")
    out["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    outp = ROOT / "analysis" / "closed_loop_audit.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, default=str))
    print("wrote", outp)


if __name__ == "__main__":
    main()
