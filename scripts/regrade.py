#!/usr/bin/env python
"""Re-grade stored rollouts offline (no GPU, no re-generation).

Fixes two measurement defects found on 2026-07-30:

1. `extract_answer` missed the "**Answer:** X" / "Answer: X" form that the
   R1-distill models emit instead of \\boxed{} in ~20-25% of completions, so
   those rollouts were silently scored incorrect.
2. The answer budget was enforced asymmetrically: the HF controller loop stops
   at `gen.max_answer_tokens` (512) while the vLLM baseline path passes
   `max_think + max_answer` as one joint budget, letting baselines write
   thousands of answer tokens. That inflated baselines relative to the
   controller and masqueraded as a cross-engine accuracy gap.

Emits, per rollout, correctness under four tiers:
  correct_strict            fixed extractor, budgets as-run
  correct_perm              + last-expression fallback (recovers truncations)
  correct_matched_strict    answer region truncated to --max-answer first
  correct_matched_perm      both corrections

`correct_matched_strict` is the apples-to-apples number: fixed extractor plus
the pre-registered 512-token answer budget enforced on BOTH engines.

Writes runs/<model>/analysis/regrade_records.parquet (per-rollout) and
regrade_summary.csv (per-run). Originals are never modified.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

from reasoncontrol.data.grading import extract_answer, grade
from reasoncontrol.stages._stage import read_jsonl_zst

THINK_CLOSE = "</think>"


def load_gold(manifest_dir: str) -> tuple[dict[str, str], dict[str, str]]:
    gold, style = {}, {}
    for p in glob.glob(os.path.join(manifest_dir, "*.parquet")):
        df = pd.read_parquet(p)
        ds = os.path.basename(p).replace(".parquet", "")
        st = "mcq" if ds == "gpqa_diamond" else "math"
        for pid, ga in zip(df["problem_id"], df["gold_answer"]):
            gold[pid] = ga
            style[pid] = st
    return gold, style


def answer_region(text: str) -> str:
    return text.rsplit(THINK_CLOSE, 1)[1] if THINK_CLOSE in text else ""


def truncated_answer_text(rec, tok, close_id, max_answer: int) -> str | None:
    """Answer region truncated to `max_answer` generated tokens, or None if the
    record already respects the budget (no truncation needed)."""
    ids = rec.get("output_token_ids") or []
    if close_id is None or not ids:
        return None
    try:
        idx = len(ids) - 1 - ids[::-1].index(close_id)
    except ValueError:
        return None
    tail = ids[idx + 1:]
    if len(tail) <= max_answer:
        return None
    return tok.decode(tail[:max_answer], skip_special_tokens=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", default="runs/r1_qwen_1p5b")
    ap.add_argument("--manifests", default="runs/data/manifests")
    ap.add_argument("--max-answer", type=int, default=512)
    ap.add_argument("--tokenizer", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    gold, style_of = load_gold(args.manifests)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    enc = tok.encode(THINK_CLOSE, add_special_tokens=False)
    close_id = enc[-1] if len(enc) == 1 else (enc[-1] if enc else None)
    if len(enc) != 1:
        print(f"warn: {THINK_CLOSE} is {len(enc)} tokens {enc}; using last id")

    rows, summary = [], []
    files = sorted(glob.glob(os.path.join(args.model_root,
                                          "*/*/*/results_*.jsonl.zst")))
    for path in files:
        try:
            recs = read_jsonl_zst(path)
        except Exception as e:  # partially written / in-flight file
            print(f"skip (unreadable): {path} ({e})")
            continue
        if not recs:
            continue
        pol = recs[0].get("policy") or {}
        kind = pol.get("kind", "?")
        phash = path.split("/")[-3]
        ds = path.split("/")[-2]
        fname = os.path.basename(path)
        split = recs[0].get("split", "?")
        seed = fname.split("seed")[-1].split(".")[0]
        engine = "hf" if "n_forwards" in recs[0] else "vllm"

        n_trunc = 0
        for r in recs:
            pid = r["problem_id"]
            g = gold.get(pid)
            st = style_of.get(pid, "math")
            text = r.get("text") or ""
            strict = extract_answer(text, style=st)
            perm = extract_answer(text, style=st, permissive=True)
            cut = truncated_answer_text(r, tok, close_id, args.max_answer)
            if cut is None:
                m_strict, m_perm = strict, perm
            else:
                n_trunc += 1
                m_strict = extract_answer(cut, style=st)
                m_perm = extract_answer(cut, style=st, permissive=True)
            rows.append(dict(
                engine=engine, kind=kind, phash=phash, dataset=ds, split=split,
                seed=int(seed), problem_id=pid,
                rollout_id=r.get("rollout_id"),
                n_think_tokens=r.get("n_think_tokens"),
                exited_early=bool(r.get("exited_early")),
                correct_orig=bool(r.get("correct")),
                parsed_orig=r.get("final_answer") not in (None, ""),
                correct_strict=bool(g is not None and grade(strict, g, style=st)),
                correct_perm=bool(g is not None and grade(perm, g, style=st)),
                correct_matched_strict=bool(g is not None and grade(m_strict, g, style=st)),
                correct_matched_perm=bool(g is not None and grade(m_perm, g, style=st)),
                parsed_strict=strict is not None,
            ))
        sub = pd.DataFrame(rows[-len(recs):])
        summary.append(dict(
            engine=engine, kind=kind, phash=phash, dataset=ds, split=split,
            seed=int(seed), n=len(recs), n_budget_truncated=n_trunc,
            acc_orig=sub.correct_orig.mean(),
            acc_strict=sub.correct_strict.mean(),
            acc_perm=sub.correct_perm.mean(),
            acc_matched_strict=sub.correct_matched_strict.mean(),
            acc_matched_perm=sub.correct_matched_perm.mean(),
            unparsed_orig=1 - sub.parsed_orig.mean(),
            unparsed_strict=1 - sub.parsed_strict.mean(),
            mean_think=sub.n_think_tokens.mean(),
        ))
        s = summary[-1]
        print(f"{engine:5s} {kind:14s} {ds:11s} {split:5s} s{seed} "
              f"orig={s['acc_orig']:.3f} strict={s['acc_strict']:.3f} "
              f"perm={s['acc_perm']:.3f} matched={s['acc_matched_strict']:.3f} "
              f"(trunc {n_trunc})")

    out = args.out_prefix or os.path.join(args.model_root, "analysis", "regrade")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_parquet(f"{out}_records.parquet", index=False)
    pd.DataFrame(summary).to_csv(f"{out}_summary.csv", index=False)
    print(f"\nwrote {out}_records.parquet and {out}_summary.csv "
          f"({len(files)} runs, {len(rows)} rollouts)")


if __name__ == "__main__":
    main()
