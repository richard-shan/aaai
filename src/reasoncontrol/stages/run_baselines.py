"""Stage 9: probe-free + trial-decode baselines on the cheap backend (vLLM on
GPU; hf for smoke). Same protocol as the controller: k dev-selected points,
identical problems and per-(problem, rollout) seeds.

Families here: noop | static_budget (sweep via --set policy.budget=...) |
concise_prompt | budget_prompt | trial_decode. HF-loop families (full,
exit_only, steer_only) run via run_controller.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict

from ..controller.baselines import (CONCISE_SUFFIX, budget_suffix, run_plain,
                                    run_static_budget, run_trial_decode)
from ..controller.policy import policy_hash
from ..data.datasets import build_prompt, load_problems, present_datasets
from ..data.grading import extract_answer, grade
from ..generation.backend import make_backend
from ..generation.hf_backend import apply_chat_template
from ._stage import is_done, mark_done, setup, stage_args, write_jsonl_zst


class _J:
    def __init__(self, problem_id, rollout_id, prompt_ids):
        self.problem_id, self.rollout_id, self.prompt_ids = problem_id, rollout_id, prompt_ids


def main():
    args = stage_args(__doc__)
    cfg, paths, stage_dir = setup(args, "run_baselines")
    t0 = time.time()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model.hf_id)
    backend = make_backend(cfg.gen.backend, cfg.model.hf_id, dtype=cfg.model.dtype,
                           batch_size=cfg.gen.batch_size,
                           max_model_len=cfg.gen.max_think_tokens
                           + cfg.gen.max_answer_tokens + 1024,
                           gpu_memory_utilization=cfg.gen.gpu_mem_util,
                           max_num_seqs=cfg.gen.max_num_seqs)
    pcfg = cfg.policy
    phash = policy_hash(pcfg)
    suffix_ids = tok.encode("\n</think>\n\n", add_special_tokens=False)
    forced_suffix_ids = tok.encode(cfg.forced.suffix_math, add_special_tokens=False)

    import os
    split = os.environ.get("RC_SPLIT", "dev")
    for ds in present_datasets(paths.manifests(), args.datasets or cfg.datasets):
        problems = load_problems(paths.manifests(), ds, split=split)
        if not problems:
            continue
        out_path = paths.controller(phash, ds) / f"results_{split}_seed{cfg.seed}.jsonl.zst"
        if out_path.exists() and not args.force:
            continue
        jobs, gold = [], {}
        for p in problems:
            style = p.meta.get("style", "math")
            prompt = build_prompt(p, style)
            if pcfg.kind == "concise_prompt":
                prompt += CONCISE_SUFFIX
            elif pcfg.kind == "budget_prompt":
                prompt += budget_suffix(pcfg.budget)
            pid = apply_chat_template(tok, prompt, think_tags=cfg.model.think_tags)
            gold[p.problem_id] = (p.gold_answer, style)
            for rid in range(cfg.gen.n_rollouts):
                jobs.append(_J(p.problem_id, rid, pid))
        kw = dict(temperature=cfg.gen.temperature, top_p=cfg.gen.top_p, seed=cfg.seed)
        if pcfg.kind in ("noop", "concise_prompt", "budget_prompt"):
            res = run_plain(backend, jobs, cfg.gen.max_think_tokens,
                            cfg.gen.max_answer_tokens, **kw)
        elif pcfg.kind == "static_budget":
            res = run_static_budget(backend, jobs, pcfg.budget, suffix_ids,
                                    cfg.gen.max_answer_tokens, **kw)
        elif pcfg.kind == "trial_decode":
            res = run_trial_decode(backend, jobs, forced_suffix_ids,
                                   agree_k=pcfg.trial_agree_k,
                                   min_chunks=pcfg.min_chunks,
                                   max_think=cfg.gen.max_think_tokens,
                                   max_answer=cfg.gen.max_answer_tokens, **kw)
        else:
            raise ValueError(f"policy kind {pcfg.kind} belongs to run_controller")
        records = []
        for r in res:
            text = r.text or tok.decode(r.output_token_ids, skip_special_tokens=False)
            g, style = gold[r.problem_id]
            final = extract_answer(text, style=style)
            records.append({"problem_id": r.problem_id, "rollout_id": r.rollout_id,
                            "output_token_ids": r.output_token_ids, "text": text,
                            "n_think_tokens": r.n_think_tokens,
                            "n_boundary_decodes": r.n_boundary_decodes,
                            "extra_decode_tokens": r.extra_decode_tokens,
                            "final_answer": final,
                            "correct": grade(final, g, style=style),
                            "policy": asdict(pcfg), "split": split})
        write_jsonl_zst(out_path, records)
        acc = sum(r["correct"] for r in records) / max(len(records), 1)
        toks = sum(r["n_think_tokens"] for r in records) / max(len(records), 1)
        print(f"run_baselines[{pcfg.kind} {phash}]: {ds}/{split} "
              f"acc={acc:.3f} mean_think={toks:.0f}")
    (stage_dir / f"{phash}.json").write_text(json.dumps(asdict(pcfg), indent=2))
    mark_done(stage_dir, t0)


if __name__ == "__main__":
    main()
