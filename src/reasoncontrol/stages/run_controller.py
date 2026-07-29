"""Stage 8: run the closed-loop controller (and HF-loop policies) on a split.

Grid sweeps run on dev ONLY; test runs happen at dev-selected operating points
(--set policy.* overrides pick the point). Every run logs wall-clock, tokens,
and per-boundary actions.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict

from ..config import PolicyCfg
from ..controller.loop import ControlledRunner, Job
from ..controller.policy import make_policy, policy_hash
from ..data.datasets import build_prompt, load_problems, present_datasets
from ..data.grading import extract_answer, grade
from ..generation.hf_backend import apply_chat_template, load_model_and_tokenizer
from ..labeling.phase_regex import PHASES
from ..probes.probe import LinearProbe
from ..steering.hooks import SteeringHook
from ..steering.vectors import load_vector
from ._stage import is_done, mark_done, setup, stage_args, write_jsonl_zst


def main():
    args = stage_args(__doc__)
    cfg, paths, stage_dir = setup(args, "run_controller")
    t0 = time.time()
    model, tok = load_model_and_tokenizer(cfg.model.hf_id, cfg.model.dtype,
                                          cfg.model.device)
    pcfg = cfg.policy
    policy = make_policy(pcfg)
    phash = policy_hash(pcfg)

    conv_probe = phase_probe = None
    probe_path = paths.probes("conv", cfg.probe.arch) / f"L{cfg.model.probe_layer}.pt"
    if probe_path.exists() and pcfg.kind not in ("noop", "static_budget"):
        conv_probe = LinearProbe.load(probe_path)
    phase_path = paths.probes("phase", cfg.probe.arch) / f"L{cfg.model.probe_layer}.pt"
    if phase_path.exists() and pcfg.kind in ("full", "steer_only"):
        phase_probe = LinearProbe.load(phase_path)

    suppress_hook = break_hook = None
    if pcfg.kind in ("full", "steer_only"):
        vpath = paths.steering() / f"verification_L{cfg.model.steer_layer}.pt"
        if vpath.exists():
            blob = load_vector(vpath)
            suppress_hook = SteeringHook(blob["v"], r_bar=blob["r_bar"])
            suppress_hook.attach(model, blob["layer"])
        dpath = paths.steering() / f"deduction_L{cfg.model.steer_layer}.pt"
        if dpath.exists() and pcfg.break_loops:
            blob = load_vector(dpath)
            break_hook = SteeringHook(blob["v"], r_bar=blob["r_bar"])
            break_hook.attach(model, blob["layer"])

    runner = ControlledRunner(
        model, tok, policy, gen_cfg=cfg.gen, policy_cfg=pcfg,
        conv_probe=conv_probe, phase_probe=phase_probe, phase_names=PHASES,
        probe_layer=cfg.model.probe_layer if conv_probe else None,
        suppress_hook=suppress_hook, break_hook=break_hook,
        steer_layer=cfg.model.steer_layer, global_seed=cfg.seed,
        think_tags=cfg.model.think_tags)

    import os
    split = os.environ.get("RC_SPLIT", "dev")
    for ds in present_datasets(paths.manifests(), args.datasets or cfg.datasets):
        problems = load_problems(paths.manifests(), ds, split=split)
        if not problems:
            continue
        out_dir = paths.controller(phash, ds)
        out_path = out_dir / f"results_{split}_seed{cfg.seed}.jsonl.zst"
        if out_path.exists() and not args.force:
            print(f"run_controller: {ds} exists, skipping")
            continue
        jobs = []
        for p in problems:
            style = p.meta.get("style", "math")
            pid = apply_chat_template(tok, build_prompt(p, style),
                                      think_tags=cfg.model.think_tags)
            for rid in range(cfg.gen.n_rollouts):
                jobs.append(Job(problem_id=p.problem_id, rollout_id=rid,
                                prompt_ids=pid, gold_answer=p.gold_answer,
                                style=style))
        results = runner.run(
            jobs, batch_size=cfg.gen.batch_size,
            progress_cb=lambda done, pending: print(
                f"run_controller[{pcfg.kind}]: {ds} {done}/{len(jobs)} rollouts "
                f"({pending} queued) t+{time.time() - t0:.0f}s", flush=True))
        gold = {p.problem_id: (p.gold_answer, p.meta.get("style", "math"))
                for p in problems}
        records = []
        for r in results:
            g, style = gold[r.problem_id]
            final = extract_answer(r.text, style=style)
            records.append({**asdict(r), "final_answer": final,
                            "correct": grade(final, g, style=style),
                            "policy": asdict(pcfg), "split": split})
        write_jsonl_zst(out_path, records)
        n_tok = sum(r.n_think_tokens for r in results) / max(len(results), 1)
        acc = sum(rec["correct"] for rec in records) / max(len(records), 1)
        print(f"run_controller[{pcfg.kind} {phash}]: {ds}/{split} "
              f"acc={acc:.3f} mean_think={n_tok:.0f}")
    (stage_dir / f"{phash}.json").write_text(json.dumps(asdict(pcfg), indent=2))
    mark_done(stage_dir, t0)


if __name__ == "__main__":
    main()
