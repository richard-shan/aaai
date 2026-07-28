"""Stage 4: forced-answer truncation labels (two-phase prefix-cached).
--audit prints 50 samples with parse rate by position."""
from __future__ import annotations

import time

import pandas as pd

from ..chunking import ChunkRecord
from ..generation.backend import make_backend
from ..labeling.convergence import (build_probes, label_chunks, run_forced_answers,
                                    select_boundaries)
from ._stage import (is_done, load_chunks_df, mark_done, read_all_rollouts,
                     setup, stage_args)


def main():
    args = stage_args(__doc__)
    cfg, paths, stage_dir = setup(args, "forced_answer")
    if is_done(stage_dir, args.force):
        print("forced_answer: already done")
        return
    t0 = time.time()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model.hf_id)
    # must cover prompt + full think trace (no </think> => whole output) + suffix
    max_model_len = cfg.gen.max_think_tokens + cfg.gen.max_answer_tokens + 1024
    backend = make_backend(cfg.gen.backend, cfg.model.hf_id, dtype=cfg.model.dtype,
                           batch_size=cfg.gen.batch_size,
                           max_model_len=max_model_len)
    hit_rates = {}
    from ..data.datasets import present_datasets
    for ds in present_datasets(paths.manifests(), args.datasets or cfg.datasets):
        chunks_df = load_chunks_df(paths.chunks(ds))
        rollouts = read_all_rollouts(paths.rollouts(ds))
        style = rollouts[0].get("style", "math") if rollouts else "math"
        suffix = cfg.forced.suffix_mcq if style == "mcq" else cfg.forced.suffix_math
        suffix_ids = tok.encode(suffix, add_special_tokens=False)
        chunks = [ChunkRecord(**{k: row[k] for k in
                                 ("problem_id", "rollout_id", "chunk_idx", "text",
                                  "tok_start", "tok_end")})
                  for row in chunks_df.to_dict("records")]
        picked = select_boundaries(chunks, cfg.forced.max_boundaries_per_rollout,
                                   cfg.forced.dense_prefix)
        prompt_ids = {(r["problem_id"], r["rollout_id"]): r["prompt_token_ids"]
                      for r in rollouts}
        output_ids = {(r["problem_id"], r["rollout_id"]): r["output_token_ids"]
                      for r in rollouts}
        final_answers = {(r["problem_id"], r["rollout_id"]): r["final_answer"]
                         for r in rollouts}
        gold = {r["problem_id"]: r["gold_answer"] for r in rollouts}
        probes = build_probes(picked, prompt_ids, output_ids, suffix_ids)
        limit = max_model_len - cfg.forced.max_new_tokens
        n_all = len(probes)
        probes = [p for p in probes if len(p.prefix_token_ids) <= limit]
        if len(probes) < n_all:   # label_chunks leaves dropped boundaries unlabeled
            print(f"forced_answer: {ds} dropped {n_all - len(probes)} "
                  f"over-length prefixes (> {limit} tokens)")
        raw = run_forced_answers(backend, probes, cfg.forced.max_new_tokens)
        picked = label_chunks(picked, raw, final_answers, gold, style=style)
        out = paths.forced(ds)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([c.__dict__ for c in picked]).to_parquet(out, index=False)
        hr = getattr(backend, "cache_hit_rate", lambda: None)()
        hit_rates[ds] = hr
        n_parsed = sum(c.forced_answer is not None for c in picked)
        print(f"forced_answer: {ds} {len(picked)} boundaries, "
              f"parse rate {n_parsed / max(len(picked), 1):.3f}, cache hit {hr}")
        if args.audit:
            _audit(picked)
    mark_done(stage_dir, t0, {"cache_hit_rates": hit_rates})


def _audit(picked, n: int = 50):
    import numpy as np
    parse_by_pos = {}
    for c in picked:
        b = c.chunk_idx // 10
        parse_by_pos.setdefault(b, []).append(c.forced_answer is not None)
    print("parse rate by chunk-idx decade:",
          {k: float(np.mean(v)) for k, v in sorted(parse_by_pos.items())})
    for c in picked[:n]:
        print(f"--- {c.problem_id} r{c.rollout_id} c{c.chunk_idx}: "
              f"forced={c.forced_answer!r} matches_final={c.conv_matches_final}")


if __name__ == "__main__":
    main()
