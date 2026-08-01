#!/usr/bin/env python
"""Pre-registered steering acceptance gate + D6 framing decision.

Accept a steer_only alpha iff the upper CI bound of the paired accuracy drop
vs the SAME-ENGINE (HF-loop) noop reference is < 2% on >= 200 paired dev
rollouts. The regex phase-rate shift computed here is a screen only — a
judge-verified shift is additionally required before any steering claim ships
(flagged as judge_verification_pending when the accuracy gate passes).

D6 (pre-registered): if steering adds < 5% of exit-only's token savings at
matched accuracy — or fails acceptance — the paper re-headlines exit-led.
Writes <root>/analysis/steering_acceptance.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from reasoncontrol.analysis.stats import paired_bootstrap
from reasoncontrol.data.grading import extract_answer, grade
from reasoncontrol.stages._stage import read_jsonl_zst
from reasoncontrol.steering.validate import accept_steering, phase_rate_shift

ROOT = Path("runs/r1_qwen_1p5b")
DEV_DATASETS = ("math_train", "gsm8k")
PHASES = ("verification", "backtracking", "deduction")
MAX_ANSWER = 512   # matched answer budget; see scripts/regrade.py


def _grader():
    """Gold answers + tokenizer for matched-budget re-grading (cached).

    The `correct` field stored in the result files was computed with the
    pre-2026-07-30 extractor, which missed the "**Answer:** X" form, and under
    an answer budget that differed across engines. Both arms of this gate are
    therefore re-graded here with the fixed extractor at a common budget.
    """
    if getattr(_grader, "_cache", None) is None:
        import glob

        import pandas as pd
        from transformers import AutoTokenizer
        gold, style = {}, {}
        for p in glob.glob("runs/data/manifests/*.parquet"):
            d = pd.read_parquet(p)
            ds = Path(p).stem
            st = "mcq" if ds == "gpqa_diamond" else "math"
            for pid, ga in zip(d["problem_id"], d["gold_answer"]):
                gold[pid], style[pid] = ga, st
        tok = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        enc = tok.encode("</think>", add_special_tokens=False)
        _grader._cache = (gold, style, tok, enc[-1] if enc else None)
    return _grader._cache


def regrade(rec) -> bool:
    gold, style, tok, close_id = _grader()
    pid = rec["problem_id"]
    g, st = gold.get(pid), style.get(pid, "math")
    if g is None:
        return bool(rec.get("correct"))
    text = rec.get("text") or ""
    ids = rec.get("output_token_ids") or []
    if close_id is not None and ids:
        try:
            i = len(ids) - 1 - ids[::-1].index(close_id)
        except ValueError:
            i = None
        if i is not None and len(ids) - i - 1 > MAX_ANSWER:
            text = tok.decode(ids[i + 1:i + 1 + MAX_ANSWER],
                              skip_special_tokens=False)
    return bool(grade(extract_answer(text, style=st), g, style=st))


def load_family(kind: str, hf_only: bool = True) -> dict:
    out = {}
    for phash_dir in sorted((ROOT / "controller").glob("*")):
        frames, pol = [], None
        for ds in DEV_DATASETS:
            for f in sorted((phash_dir / ds).glob("results_dev_seed*.jsonl.zst")):
                recs = read_jsonl_zst(f)
                if not recs:
                    continue
                if recs[0].get("policy", {}).get("kind") != kind:
                    break
                if hf_only and "n_forwards" not in recs[0]:
                    break  # vLLM baseline sharing the same kind — wrong engine
                pol = recs[0]["policy"]
                frames.append(pd.DataFrame(
                    [{"problem_id": r["problem_id"],
                      "rollout_id": r["rollout_id"],
                      "correct": float(regrade(r)),
                      "n_think_tokens": float(r["n_think_tokens"]),
                      "text": r.get("text", "")} for r in recs]))
        if frames and pol is not None:
            out[phash_dir.name] = (pol, pd.concat(frames, ignore_index=True))
    return out


def paragraphs(df: pd.DataFrame, cap: int = 40000) -> list[str]:
    ps = []
    for t in df["text"]:
        ps.extend(p for p in t.split("\n\n") if len(p.strip()) > 20)
    if len(ps) > cap:
        rng = np.random.default_rng(0)
        ps = [ps[i] for i in rng.choice(len(ps), cap, replace=False)]
    return ps


def main():
    noops = load_family("noop", hf_only=True)
    if not noops:
        raise SystemExit("no HF-loop noop dev reference found "
                         "(needs the min_chunks=5-hashed controller noop run)")
    # load_family maps phash -> (policy, df); pick the reference with the most
    # dev rollouts (ties broken by hash) so the choice is deterministic.
    noop_phash, (_, noop_df) = max(sorted(noops.items()),
                                   key=lambda kv: len(kv[1][1]))
    res = {"alphas": {}, "any_accepted": False,
           "noop_reference": {"phash": noop_phash, "n_rollouts": len(noop_df)},
           "grading": f"fixed extractor, answer budget matched at {MAX_ANSWER}"}
    noop_paras = None
    for ph, (pol, df) in sorted(load_family("steer_only").items()):
        alpha = pol.get("alpha")
        acc = accept_steering(df, noop_df)
        tok = paired_bootstrap(df, noop_df, "n_think_tokens")
        entry = {"phash": ph, "alpha": alpha,
                 "n_paired_problems": int(len(set(df.problem_id)
                                              & set(noop_df.problem_id))),
                 "n_rollouts": int(len(df)),
                 "acceptance": acc,
                 "tokens_delta": {k: tok[k] for k in ("delta", "lo", "hi")
                                  if k in tok}}
        try:
            if noop_paras is None:
                noop_paras = paragraphs(noop_df)
            sp = paragraphs(df)
            entry["phase_shift_regex_paragraph_approx"] = {
                p: phase_rate_shift(sp, noop_paras, p, n_boot=300)
                for p in PHASES}
        except Exception as e:  # screen only; never blocks the gate output
            entry["phase_shift_error"] = str(e)
        if acc.get("accepted"):
            res["any_accepted"] = True
            entry["judge_verification_pending"] = True
        res["alphas"][str(alpha)] = entry
        print(f"steer alpha={alpha}: accepted={acc.get('accepted')} "
              f"max_plausible_drop={acc.get('max_plausible_drop', float('nan')):.4f} "
              f"dTokens={entry['tokens_delta'].get('delta', float('nan')):.0f}")

    d6 = {}
    sel_p = ROOT / "analysis" / "selection_dev.json"
    if sel_p.exists():
        sel = json.loads(sel_p.read_text()).get("families", {})
        noop_tok = float(noop_df.groupby("problem_id")["n_think_tokens"]
                         .mean().mean())
        if "exit_only" in sel:
            exit_tok = float(sel["exit_only"]["tokens"])
            d6 = {"noop_tokens": noop_tok, "exit_tokens": exit_tok,
                  "exit_savings": noop_tok - exit_tok}
            if "full" in sel:
                full_tok = float(sel["full"]["tokens"])
                extra = exit_tok - full_tok
                d6.update(full_tokens=full_tok,
                          full_acc=float(sel["full"]["acc"]),
                          steer_extra_savings=extra,
                          extra_over_exit_ratio=(extra / d6["exit_savings"]
                                                 if d6["exit_savings"] > 0
                                                 else None))
    ratio = d6.get("extra_over_exit_ratio")
    exit_led = (not res["any_accepted"]) or (ratio is not None and ratio < 0.05)
    res["d6"] = {**d6, "exit_led_headline": bool(exit_led)}
    res["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    outp = ROOT / "analysis" / "steering_acceptance.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2, default=str))
    print("STEERING", "ACCEPTED (judge verification still required)"
          if res["any_accepted"] else "REJECTED")
    print("D6 decision:", "EXIT-LED HEADLINE" if exit_led
          else "steer+exit headline retained")
    print("wrote", outp)


if __name__ == "__main__":
    main()
