#!/usr/bin/env python
"""Pre-registered PRIMARY ENDPOINT: pooled MATH-500 + GSM8K test, hierarchical.

`stages/analyze.py` only emits the paired hierarchical test when a `full`
(exit+steer) family is present on test. Steering was rejected by the
pre-registered acceptance gate and D6 re-headlined exit-led, so no `full`
condition ever ran on test — and the primary endpoint was therefore never
computed. This script computes it directly from the re-graded records.

Hierarchy (pre-registered, in order; the second is only interpreted if the
first passes):
  1. TOKEN SUPERIORITY   exit_only uses fewer think tokens than same-engine
                         noop (one-sided, paired by problem).
  2. ACCURACY NON-INFERIORITY
                         the lower CI bound of the paired accuracy delta
                         exceeds -margin (default 0.02).

Unit of analysis is the problem; conditions share problems, so deltas are
paired and bootstrapped over problems (10k resamples).

GRADING COLUMN: `correct_strict` — the fixed extractor with budgets as-run.
NOT `correct_matched_strict`: the autopilot invoked regrade.py without
--max-answer, so that column re-truncates answers to 512 tokens, which would
re-introduce the very defect the 2048-token budget was raised to remove. All
test runs used gen.max_answer_tokens=2048 on both engines, so budgets are
already matched and no truncation correction is needed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from reasoncontrol.analysis.stats import (cluster_bootstrap_ci,
                                          non_inferiority, paired_bootstrap)

PRIMARY_DATASETS = ("math500", "gsm8k")


def load(tag: str, col: str) -> pd.DataFrame:
    p = Path("runs") / tag / "analysis" / "regrade_records.parquet"
    df = pd.read_parquet(p)
    df = df[(df.split == "test") & (df.dataset.isin(PRIMARY_DATASETS))].copy()
    df["correct"] = df[col].astype(float)
    # problem_ids are unique across datasets; pooling is a concatenation
    return df


def arm(df: pd.DataFrame, kind: str, engine: str, phash: str | None = None):
    a = df[(df.kind == kind) & (df.engine == engine)]
    if phash:
        a = a[a.phash == phash]
    return a


def describe(a: pd.DataFrame) -> dict:
    acc = cluster_bootstrap_ci(a, "correct", n_boot=10000)
    tok = cluster_bootstrap_ci(a, "n_think_tokens", n_boot=10000)
    return {"acc": acc, "tokens": tok, "n_rollouts": int(len(a)),
            "n_seeds": int(a.seed.nunique()),
            "phash": sorted(a.phash.unique())}


def compare(treat: pd.DataFrame, ref: pd.DataFrame, margin: float) -> dict:
    tok = paired_bootstrap(treat, ref, "n_think_tokens", n_boot=10000)
    ni = non_inferiority(treat, ref, "correct", margin=margin, n_boot=10000)
    # gate 1: fewer tokens, i.e. delta < 0 with the upper CI bound below zero
    token_superior = bool(tok["hi"] < 0)
    return {
        "token_delta": tok,
        "token_superiority_passed": token_superior,
        "token_reduction_pct": (100.0 * -tok["delta"]
                                / ref.groupby("problem_id")["n_think_tokens"]
                                .mean().mean()),
        "accuracy_delta": {k: ni[k] for k in ("delta", "lo", "hi",
                                              "n_problems")},
        "accuracy_non_inferior": bool(ni["non_inferior"]),
        "margin": margin,
        # the honest headline holds regardless of which way the gate lands
        "verdict": ("token superiority PASSED; accuracy "
                    + ("NON-INFERIOR" if ni["non_inferior"]
                       else f"INFERIOR (drop exceeds {margin})")
                    if token_superior else "token superiority FAILED"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="r1_qwen_1p5b")
    ap.add_argument("--column", default="correct_strict")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--exit-phash", default="079322feb6")
    args = ap.parse_args()

    df = load(args.tag, args.column)
    treat = arm(df, "exit_only", "hf", args.exit_phash)
    if treat.empty:
        raise SystemExit(f"no exit_only/hf rollouts for {args.tag}")
    hf_noop = arm(df, "noop", "hf")
    vllm_noop = arm(df, "noop", "vllm")

    out = {"tag": args.tag, "column": args.column,
           "datasets": list(PRIMARY_DATASETS),
           "arms": {"exit_only_hf": describe(treat)}, "comparisons": {}}
    if not hf_noop.empty:
        out["arms"]["noop_hf"] = describe(hf_noop)
        out["comparisons"]["vs_noop_same_engine_PRIMARY"] = compare(
            treat, hf_noop, args.margin)
    if not vllm_noop.empty:
        out["arms"]["noop_vllm"] = describe(vllm_noop)
        out["comparisons"]["vs_noop_cross_engine"] = compare(
            treat, vllm_noop, args.margin)
    sb = arm(df, "static_budget", "vllm")
    if not sb.empty:
        out["arms"]["static_budget_vllm"] = describe(sb)
        out["comparisons"]["vs_static_budget"] = compare(treat, sb, args.margin)

    out["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    p = Path("runs/analysis") / args.tag / "primary_endpoint.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=str))

    print(f"=== {args.tag} pooled {'+'.join(PRIMARY_DATASETS)} test ===")
    for name, a in out["arms"].items():
        print(f"{name:22s} acc={a['acc']['mean']:.4f} "
              f"[{a['acc']['lo']:.4f},{a['acc']['hi']:.4f}] "
              f"tok={a['tokens']['mean']:7.1f} "
              f"n={a['n_rollouts']} seeds={a['n_seeds']}")
    for name, c in out["comparisons"].items():
        d = c["accuracy_delta"]
        print(f"\n{name}: {c['verdict']}")
        print(f"  tokens delta {c['token_delta']['delta']:+.1f} "
              f"[{c['token_delta']['lo']:+.1f},{c['token_delta']['hi']:+.1f}] "
              f"({c['token_reduction_pct']:.1f}% fewer)")
        print(f"  accuracy delta {d['delta']:+.4f} "
              f"[{d['lo']:+.4f},{d['hi']:+.4f}] over {d['n_problems']} problems")
    print("\nwrote", p)


if __name__ == "__main__":
    main()
