"""Stage 5: regex phase labels for all chunks + LLM-judge audit subset with
Cohen's kappa. Merges forced-answer labels into labels.parquet."""
from __future__ import annotations

import json
import time

import pandas as pd

from ..labeling.phase_judge import (cohens_kappa, judge_requests, parse_judge,
                                    stratified_sample)
from ..labeling.phase_regex import label_phase
from ._stage import is_done, load_chunks_df, mark_done, setup, stage_args

JUDGE_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"


def main():
    args = stage_args(__doc__)
    cfg, paths, stage_dir = setup(args, "label_phase")
    if is_done(stage_dir, args.force):
        print("label_phase: already done")
        return
    t0 = time.time()
    kappas = {}
    use_judge = cfg.gen.backend == "vllm"    # judge stubbed on smoke/CPU runs
    judge_backend = judge_tok = None
    from ..data.datasets import present_datasets
    for ds in present_datasets(paths.manifests(), args.datasets or cfg.datasets):
        forced_path = paths.forced(ds)
        base = load_chunks_df(forced_path if forced_path.exists()
                              else paths.chunks(ds))
        base["phase_regex"] = [label_phase(t) for t in base["text"]]
        if use_judge:
            if judge_backend is None:
                from ..generation.backend import make_backend
                from transformers import AutoTokenizer
                judge_tok = AutoTokenizer.from_pretrained(JUDGE_MODEL)
                judge_backend = make_backend("vllm", JUDGE_MODEL, dtype="auto",
                                             max_model_len=4096)
            recs = base.to_dict("records")

            class _C:                            # judge utils expect .text/.phase_regex
                def __init__(self, d):
                    self.text, self.phase_regex = d["text"], d["phase_regex"]
                    self._d = d
            sample = stratified_sample([_C(d) for d in recs], per_phase=300,
                                       seed=cfg.seed)
            res = judge_backend.generate(judge_requests(sample, judge_tok))
            judged = [parse_judge(r.text) for r in res]
            kappas[ds] = cohens_kappa([c.phase_regex for c in sample], judged)
            judge_map = {(c._d["problem_id"], c._d["rollout_id"], c._d["chunk_idx"]): j
                         for c, j in zip(sample, judged)}
            base["phase_judge"] = [judge_map.get((p, r, ci))
                                   for p, r, ci in zip(base.problem_id,
                                                       base.rollout_id, base.chunk_idx)]
        out = paths.labels(ds)
        out.parent.mkdir(parents=True, exist_ok=True)
        base.to_parquet(out, index=False)
        print(f"label_phase: {ds} {len(base)} rows"
              + (f", kappa={kappas.get(ds):.3f}" if ds in kappas else ""))
    (stage_dir / "kappa.json").write_text(json.dumps(kappas, indent=2))
    mark_done(stage_dir, t0, {"kappa": kappas})
    return judge_backend


if __name__ == "__main__":
    from ._stage import exit_stage
    exit_stage(main())
