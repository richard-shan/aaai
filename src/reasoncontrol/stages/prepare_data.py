"""Stage 0: download datasets, fix splits (seed=0), write manifests.

Split table (fixed by the plan; every test-time knob's dev set is named here):
- math_train:   500 probe_train + 200 dev sampled from MATH training split
- math500:      ALL 500 problems test (untouched)
- gsm8k:        300 probe_train / 50 dev / 250 test from the GSM8K test split
- aime:         AIME'24 (n=30) dev, AIME'25 (n=30) test (uncontaminated anchor)
- gpqa_diamond: test-only; operating points inherited from MATH dev

Also asserts the model's chat template auto-opens <think> for reasoning models.
"""
from __future__ import annotations

import time

import numpy as np

from ..data.datasets import Problem, save_manifest
from ..data.grading import extract_boxed
from ._stage import is_done, mark_done, setup, stage_args


def _mk(dataset, split, idx, q, a, meta=None) -> Problem:
    return Problem(problem_id=f"{dataset}/{split}/{idx}", dataset=dataset,
                   split=split, question=q, gold_answer=str(a), meta=meta or {})


def _math_train(out_dir, rng):
    from datasets import concatenate_datasets, load_dataset
    # EleutherAI/hendrycks_math has no "all" config — concatenate the seven
    # subject configs in sorted order so the seed-0 permutation is stable.
    subjects = ["algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory", "prealgebra",
                "precalculus"]
    ds = concatenate_datasets(
        [load_dataset("EleutherAI/hendrycks_math", s, split="train") for s in subjects])
    idx = rng.permutation(len(ds))[:800]
    probs = []
    for j, i in enumerate(idx):
        if len(probs) >= 700:
            break
        row = ds[int(i)]
        gold = extract_boxed(row["solution"])
        if not gold:
            continue
        split = "probe_train" if len(probs) < 500 else "dev"
        probs.append(_mk("math_train", split, len(probs), row["problem"], gold,
                         {"level": row.get("level"), "type": row.get("type")}))
    return save_manifest(probs, out_dir, "math_train")


def _math500(out_dir):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    probs = [_mk("math500", "test", i, r["problem"], r["answer"],
                 {"level": r.get("level"), "subject": r.get("subject")})
             for i, r in enumerate(ds)]
    return save_manifest(probs, out_dir, "math500")


def _gsm8k(out_dir, rng):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    order = rng.permutation(len(ds))[:600]
    probs = []
    for j, i in enumerate(order):
        row = ds[int(i)]
        split = "probe_train" if j < 300 else ("dev" if j < 350 else "test")
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        probs.append(_mk("gsm8k", split, j, row["question"], gold))
    return save_manifest(probs, out_dir, "gsm8k")


def _aime(out_dir):
    from datasets import load_dataset
    probs, j = [], 0
    aime24 = load_dataset("AI-MO/aimo-validation-aime", split="train")
    for r in aime24:
        if "2024" in str(r.get("url", "")) or "2024" in str(r.get("id", "")):
            probs.append(_mk("aime", "dev", j, r["problem"], r["answer"]))
            j += 1
    aime25 = load_dataset("math-ai/aime25", split="test")
    for i, r in enumerate(aime25):
        probs.append(_mk("aime", "test", j + i, r["problem"], r["answer"]))
    return save_manifest(probs, out_dir, "aime")


def _gpqa(out_dir, rng, hf_token):
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", token=hf_token)
    probs = []
    for i, r in enumerate(ds):
        choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                   r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        perm = rng.permutation(4)
        gold_letter = chr(65 + int(np.where(perm == 0)[0][0]))
        probs.append(_mk("gpqa_diamond", "test", i, r["Question"], gold_letter,
                         {"choices": [str(choices[k]) for k in perm], "style": "mcq"}))
    return save_manifest(probs, out_dir, "gpqa_diamond")


def build_manifests(out_dir, seed: int = 0, hf_token: str | None = None,
                    only: list[str] | None = None):
    rng = np.random.default_rng(seed)
    written = {}

    def want(name: str) -> bool:
        return only is None or name in only

    if want("math_train"):
        written["math_train"] = _math_train(out_dir, rng)
    if want("math500"):
        written["math500"] = _math500(out_dir)
    if want("gsm8k"):
        written["gsm8k"] = _gsm8k(out_dir, rng)
    if want("aime"):
        written["aime"] = _aime(out_dir)
    if want("gpqa_diamond"):
        try:
            written["gpqa_diamond"] = _gpqa(out_dir, rng, hf_token)
        except Exception as e:   # gated access can fail; risk-register fallback
            print(f"WARNING: GPQA unavailable ({e}); continuing without it")
    return written


def assert_think_template(model_id: str) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    txt = tok.apply_chat_template([{"role": "user", "content": "hi"}],
                                  add_generation_prompt=True, tokenize=False)
    assert txt.rstrip().endswith("<think>"), (
        f"chat template for {model_id} does not auto-open <think>; "
        "the pipeline's think-region assumptions would silently break")


def main():
    args = stage_args(__doc__)
    cfg, paths, stage_dir = setup(args, "prepare_data")
    if is_done(stage_dir, args.force):
        print("prepare_data: already done")
        return
    t0 = time.time()
    if cfg.model.think_tags:
        assert_think_template(cfg.model.hf_id)
    import os
    written = build_manifests(paths.manifests(), seed=cfg.seed,
                              hf_token=os.environ.get("HF_TOKEN"),
                              only=args.datasets or list(cfg.datasets))
    mark_done(stage_dir, t0, {"manifests": {k: str(v) for k, v in written.items()}})
    print(f"prepare_data: wrote {list(written)}")


if __name__ == "__main__":
    main()
