"""Stage 7: diff-of-means steering vectors + orthogonalization against probe
weights + causal validation artifacts.

Writes steering/{phase}_L{layer}.pt with r_bar baked in. The full causal
validation (steered generations, judge phase-rate shift, paired accuracy
gate) runs via run_controller with steer_only policies on dev — this stage
builds the vectors and the injection/ablation quick check.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from ..activations.store import ActStore
from ..probes.probe import LinearProbe
from ..steering.vectors import diff_of_means, orthogonalize, save_vector
from ._stage import is_done, mark_done, setup, stage_args

TARGET_PHASES = ("verification", "backtracking", "deduction")


def main():
    args = stage_args(__doc__)
    cfg, paths, stage_dir = setup(args, "build_steering")
    if is_done(stage_dir, args.force):
        print("build_steering: already done")
        return
    t0 = time.time()
    ds = (args.datasets or ["math_train"])[0]
    labels = pd.read_parquet(paths.labels(ds))
    store = ActStore(paths.acts_dir(ds), cfg.model.cache_layers)
    norms = json.loads((store.root / "residual_norms.json").read_text())

    # "next chunk is <phase>": shift phase labels back one boundary
    labels = labels.sort_values(["problem_id", "rollout_id", "chunk_idx"])
    labels["next_phase"] = labels.groupby(["problem_id", "rollout_id"])["phase_regex"].shift(-1)
    labels = labels[labels.next_phase.notna()]

    # probe directions to orthogonalize against (sensor/actuator decoupling)
    probe_dirs = []
    probe_dir_path = paths.probes("conv", cfg.probe.arch) / f"L{cfg.model.steer_layer}.pt"
    if probe_dir_path.exists():
        probe_dirs.append(LinearProbe.load(probe_dir_path).direction)

    layer = cfg.model.steer_layer
    h = store.gather(labels[["problem_id", "rollout_id", "chunk_idx"]], layer)
    reports = {}
    for phase in TARGET_PHASES:
        is_target = (labels.next_phase == phase).to_numpy()
        if is_target.sum() < 20:
            print(f"build_steering: skipping {phase} (only {int(is_target.sum())} boundaries)")
            continue
        v = diff_of_means(h, is_target)
        v_orth, rep = orthogonalize(v, probe_dirs)
        save_vector(paths.steering() / f"{phase}_L{layer}.pt", v_orth, layer,
                    r_bar=float(norms[str(layer)]),
                    meta={"phase": phase, "n_target": int(is_target.sum()),
                          "orthogonalization": rep})
        reports[phase] = rep
        print(f"build_steering: {phase} L{layer} n_target={int(is_target.sum())} "
              f"cos_vs_probe before/after: "
              f"{rep.get('cos_before_0', 'n/a')}/{rep.get('cos_after_0', 'n/a')}")
    (paths.steering() / "validation.json").write_text(json.dumps(reports, indent=2))
    mark_done(stage_dir, t0)


if __name__ == "__main__":
    main()
