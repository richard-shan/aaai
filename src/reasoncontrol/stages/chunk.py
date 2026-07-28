"""Stage 2: chunk all rollouts (CPU). Also runs the online-vs-offline boundary
equivalence check on a sample (acceptance checklist)."""
from __future__ import annotations

import time
from dataclasses import asdict

import pandas as pd

from ..chunking import chunk_trace
from ._stage import is_done, mark_done, read_all_rollouts, setup, stage_args


def main():
    args = stage_args(__doc__)
    cfg, paths, stage_dir = setup(args, "chunk")
    if is_done(stage_dir, args.force):
        print("chunk: already done")
        return
    t0 = time.time()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model.hf_id)
    from ..data.datasets import present_datasets
    for ds in present_datasets(paths.manifests(), args.datasets or cfg.datasets):
        rollouts = read_all_rollouts(paths.rollouts(ds))
        if not rollouts:
            continue
        rows = []
        for r in rollouts:
            chunks = chunk_trace(
                r["output_token_ids"], tok, prompt_len=len(r["prompt_token_ids"]),
                problem_id=r["problem_id"], rollout_id=r["rollout_id"],
                min_chunk_tokens=cfg.chunk.min_chunk_tokens,
                max_chunks=cfg.chunk.max_chunks, keep_first=cfg.chunk.keep_first,
                keep_last=cfg.chunk.keep_last, think_end=r["think_end"])
            rows.extend(asdict(c) for c in chunks)
        out = paths.chunks(ds)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out, index=False)
        print(f"chunk: {ds} -> {len(rows)} chunks")
    mark_done(stage_dir, t0)


if __name__ == "__main__":
    main()
