#!/usr/bin/env python
"""1-SE operating-point selection from RE-GRADED dev rollouts.

Supersedes the selection made by scripts/select_operating_points.py, which read
the `correct` field written at generation time and was therefore computed with
the broken answer extractor (see docs/RESULTS.md, "MEASUREMENT DEFECT").

Selection is still dev-only and still the pre-registered 1-SE rule: among
configs whose pooled dev accuracy is within one SE of the best, take the one
with the fewest mean think tokens. The only change is which accuracy column
feeds it (default: correct_matched_strict = fixed extractor + the pre-registered
512-token answer budget enforced on BOTH engines).

Writes the same schema as selection_dev.json so downstream `selget` calls are
unchanged; the previous file is copied to selection_dev.buggy_grading.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

# policy-dict key that identifies each family's tuned parameter(s)
FAMILY_PARAMS = {
    "exit_only": ("tau_exit", "patience_k"),
    "full": ("tau_exit", "patience_k"),
    "steer_only": ("alpha",),
    "static_budget": ("budget",),
    "budget_prompt": ("budget",),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", default="runs/r1_qwen_1p5b")
    ap.add_argument("--records", default=None)
    ap.add_argument("--column", default="correct_matched_strict")
    ap.add_argument("--families", default="exit_only,full,static_budget,budget_prompt")
    ap.add_argument("--datasets", default="math_train,gsm8k")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    recs = args.records or os.path.join(args.model_root, "analysis",
                                        "regrade_records.parquet")
    out = args.out or os.path.join(args.model_root, "analysis",
                                   "selection_dev.json")
    df = pd.read_parquet(recs)
    dev = df[(df.split == "dev")
             & (df.dataset.isin(args.datasets.split(",")))]
    if dev.empty:
        raise SystemExit(f"no dev rollouts in {recs}")

    # policy params live in the run dirs, not the regrade parquet; recover them
    # from the original result files' policy dicts via the summary of hashes.
    from reasoncontrol.stages._stage import read_jsonl_zst
    import glob
    params_by_hash: dict[str, dict] = {}
    for p in glob.glob(os.path.join(args.model_root, "*/*/*/results_dev_*.jsonl.zst")):
        ph = p.split("/")[-3]
        if ph in params_by_hash:
            continue
        try:
            r = read_jsonl_zst(p)
        except Exception:
            continue
        if r:
            params_by_hash[ph] = r[0].get("policy") or {}

    families = {}
    for fam in args.families.split(","):
        d = dev[dev["kind"] == fam]
        if d.empty:
            print(f"select[{fam}]: no dev data, skipped")
            continue
        rows = []
        for ph, g in d.groupby("phash"):
            per = g.groupby("problem_id")[args.column].mean()
            rows.append(dict(phash=ph, acc=float(per.mean()),
                             se=float(per.std(ddof=1) / np.sqrt(len(per))),
                             tokens=float(g.n_think_tokens.mean()),
                             n_problems=int(len(per))))
        rows.sort(key=lambda r: -r["acc"])
        best = rows[0]
        thr = best["acc"] - best["se"]
        elig = [r for r in rows if r["acc"] >= thr]
        pick = min(elig, key=lambda r: r["tokens"])
        pol = params_by_hash.get(pick["phash"], {})
        keys = FAMILY_PARAMS.get(fam, ())
        families[fam] = {**pick,
                         "params": {k: pol.get(k) for k in
                                    ("tau_exit", "patience_k", "alpha", "budget")},
                         "tuned_params": {k: pol.get(k) for k in keys},
                         "one_se_threshold": thr,
                         "best_acc": best["acc"], "best_se": best["se"],
                         "n_configs": len(rows)}
        print(f"select[{fam}]: {len(rows)} configs; best acc={best['acc']:.3f} "
              f"(se {best['se']:.3f}) -> PICK {pick['phash']} "
              f"{families[fam]['tuned_params']} acc={pick['acc']:.3f} "
              f"tokens={pick['tokens']:.0f}")

    if os.path.exists(out):
        shutil.copyfile(out, out.replace(".json", ".buggy_grading.json"))
    with open(out, "w") as f:
        json.dump({"families": families,
                   "accuracy_column": args.column,
                   "source": recs,
                   "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  time.gmtime())}, f, indent=2)
    print(f"wrote {out} (column={args.column})")


if __name__ == "__main__":
    main()
