"""Stage 1: sample rollouts (vLLM on GPU; hf backend for smoke).
Sharded by 50 problems; resumable at shard granularity."""
from __future__ import annotations

import time

from ..data.datasets import build_prompt, load_problems
from ..data.grading import extract_answer, grade
from ..generation.backend import GenRequest, make_backend
from ..generation.hf_backend import apply_chat_template
from ._stage import is_done, mark_done, setup, stage_args, write_jsonl_zst

SHARD = 50


def main():
    args = stage_args(__doc__)
    cfg, paths, stage_dir = setup(args, "generate")
    if is_done(stage_dir, args.force):
        print("generate: already done")
        return
    t0 = time.time()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model.hf_id)
    backend = make_backend(cfg.gen.backend, cfg.model.hf_id, dtype=cfg.model.dtype,
                           batch_size=cfg.gen.batch_size,
                           max_model_len=cfg.gen.max_think_tokens + cfg.gen.max_answer_tokens + 1024)
    datasets = args.datasets or cfg.datasets
    for ds in datasets:
        problems = load_problems(paths.manifests(), ds)
        out_dir = paths.rollouts(ds)
        for s0 in range(0, len(problems), SHARD):
            shard_idx = s0 // SHARD
            shard_path = out_dir / f"shard_{shard_idx:03d}.jsonl.zst"
            if shard_path.exists() and not args.force:
                continue
            reqs, meta = [], []
            for p in problems[s0:s0 + SHARD]:
                style = p.meta.get("style", "math")
                prompt_ids = apply_chat_template(
                    tok, build_prompt(p, style), think_tags=cfg.model.think_tags)
                for rid in range(cfg.gen.n_rollouts):
                    reqs.append(GenRequest(
                        request_id=f"{p.problem_id}|{rid}",
                        prompt_token_ids=tuple(prompt_ids),
                        max_tokens=cfg.gen.max_think_tokens + cfg.gen.max_answer_tokens,
                        temperature=cfg.gen.temperature, top_p=cfg.gen.top_p,
                        seed=cfg.seed * 1000 + rid))
                    meta.append((p, rid, prompt_ids, style))
            records = []
            for (p, rid, prompt_ids, style), res in zip(meta, backend.generate(reqs)):
                text = res.text
                final = extract_answer(text, style=style)
                think_end = None
                if "</think>" in text:
                    think_txt = text.split("</think>")[0]
                    think_end = len(tok.encode(think_txt, add_special_tokens=False))
                records.append({
                    "problem_id": p.problem_id, "rollout_id": rid,
                    "prompt_token_ids": list(prompt_ids),
                    "output_token_ids": list(res.output_token_ids),
                    "text": text, "think_end": think_end,
                    "final_answer": final,
                    "correct": grade(final, p.gold_answer, style=style),
                    "n_think_tokens": think_end if think_end is not None
                                      else len(res.output_token_ids),
                    "style": style, "gold_answer": p.gold_answer,
                })
            write_jsonl_zst(shard_path, records)
            print(f"generate: {ds} shard {shard_idx} ({len(records)} rollouts)")
    mark_done(stage_dir, t0)


if __name__ == "__main__":
    main()
