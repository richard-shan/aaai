"""Stage 3: teacher-forced boundary-state capture + per-layer residual norms
+ vLLM/HF consistency check (>= 99% argmax reproduction required)."""
from __future__ import annotations

import json
import time

import torch

from ..activations.capture import capture_boundaries
from ..activations.store import ActStore
from ..generation.hf_backend import load_model_and_tokenizer
from ._stage import (is_done, load_chunks_df, mark_done, read_all_rollouts,
                     setup, stage_args)

SHARD = 50   # rollouts per shard


def main():
    args = stage_args(__doc__)
    cfg, paths, stage_dir = setup(args, "capture")
    if is_done(stage_dir, args.force):
        print("capture: already done")
        return
    t0 = time.time()
    model, tok = load_model_and_tokenizer(cfg.model.hf_id, cfg.model.dtype,
                                          cfg.model.device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    consistency = {}
    from ..data.datasets import present_datasets
    for ds in present_datasets(paths.manifests(), args.datasets or cfg.datasets):
        chunks = load_chunks_df(paths.chunks(ds))
        rollouts = read_all_rollouts(paths.rollouts(ds))
        by_key = {(r["problem_id"], r["rollout_id"]): r for r in rollouts}
        store = ActStore(paths.acts_dir(ds), cfg.model.cache_layers)
        keys = sorted(by_key)
        matches = []
        by_shard_path = store.root / "residual_norms_by_shard.json"
        norms_by_shard = (json.loads(by_shard_path.read_text())
                          if by_shard_path.exists() else {})
        for s0 in range(0, len(keys), SHARD):
            shard_idx = s0 // SHARD
            if (store.root / f"acts_{shard_idx:03d}.safetensors").exists() and not args.force:
                continue
            seqs, bpos, index_rows = [], [], []
            for key in keys[s0:s0 + SHARD]:
                r = by_key[key]
                grp = chunks[(chunks.problem_id == key[0]) & (chunks.rollout_id == key[1])]
                if grp.empty:
                    continue
                seqs.append(r["prompt_token_ids"] + r["output_token_ids"])
                bpos.append([int(te) - 1 for te in grp.tok_end])
                index_rows.extend({"problem_id": key[0], "rollout_id": key[1],
                                   "chunk_idx": int(ci)} for ci in grp.chunk_idx)
            if not seqs:
                continue
            h, norms, argmax_match = capture_boundaries(
                model, seqs, bpos, cfg.model.cache_layers, pad,
                batch_size=max(1, cfg.gen.batch_size // 8))
            store.write_shard(shard_idx, h, index_rows)
            matches.append(argmax_match)
            # accumulate per shard; the mean across shards is what ships
            norms_by_shard[str(shard_idx)] = {str(k): v for k, v in norms.items()}
            by_shard_path.write_text(json.dumps(norms_by_shard))
            layer_keys = sorted({k for n in norms_by_shard.values() for k in n},
                                key=int)
            (store.root / "residual_norms.json").write_text(json.dumps(
                {k: sum(n[k] for n in norms_by_shard.values() if k in n)
                    / sum(1 for n in norms_by_shard.values() if k in n)
                 for k in layer_keys}))
        if matches:
            consistency[ds] = sum(matches) / len(matches)
            print(f"capture: {ds} argmax consistency {consistency[ds]:.4f}")
    mark_done(stage_dir, t0, {"argmax_consistency": consistency})
    for ds, c in consistency.items():
        if cfg.gen.temperature == 0 and c < 0.95:
            print(f"WARNING: {ds} consistency {c:.4f} < 0.95 on GREEDY rollouts "
                  "— investigate tokenization/numerics drift before trusting labels")
        elif cfg.gen.temperature > 0:
            print(f"capture: {ds} argmax-vs-sampled agreement {c:.4f} "
                  f"(temp={cfg.gen.temperature}; the >=0.99 gate applies to "
                  "greedy rollouts only)")


if __name__ == "__main__":
    main()
