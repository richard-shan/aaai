"""Stage 6: probes with layer sweep, position-controlled evaluation, shallow
baselines, and calibration on dev.

Go/no-go (D4): conv probe beats the shallow/position baseline by >= 0.05 AUC
within position strata AND pooled AUC >= 0.75 on MATH dev.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from ..activations.store import ActStore
from ..labeling.phase_regex import PHASES
from ..probes.probe import LinearProbe
from ..probes.shallow import fit_shallow, position_stratified_auc, shallow_features
from ._stage import is_done, mark_done, setup, stage_args

TRAIN_DATASETS = ("math_train", "gsm8k")     # probe_train splits only


def _labels(paths, ds) -> pd.DataFrame:
    df = pd.read_parquet(paths.labels(ds))
    df["dataset"] = ds
    return df


def main():
    args = stage_args(__doc__)
    cfg, paths, stage_dir = setup(args, "train_probes")
    if is_done(stage_dir, args.force):
        print("train_probes: already done")
        return
    t0 = time.time()
    datasets = args.datasets or [d for d in TRAIN_DATASETS if d in cfg.datasets] \
        or list(cfg.datasets)[:1]
    frames = [_labels(paths, ds) for ds in datasets if paths.labels(ds).exists()]
    df = pd.concat(frames, ignore_index=True)
    conv = df[df.conv_matches_final.notna()].copy()
    conv["y"] = conv.conv_matches_final.astype(int)
    groups = conv.problem_id.to_numpy()
    max_len = int(conv.tok_end.max())

    report: dict = {"conv": {}, "phase": {}, "shallow": {}}
    stores = {ds: ActStore(paths.acts_dir(ds), cfg.model.cache_layers)
              for ds in datasets}

    def gather(sub: pd.DataFrame, layer: int) -> torch.Tensor:
        hs = []
        for ds, grp in sub.groupby("dataset"):
            hs.append((grp.index.to_numpy(),
                       stores[ds].gather(grp[["problem_id", "rollout_id", "chunk_idx"]],
                                         layer)))
        out = torch.empty(len(sub), hs[0][1].shape[-1])
        pos_map = {ix: k for k, ix in enumerate(sub.index.to_numpy())}
        for idxs, h in hs:
            for row_i, ix in enumerate(idxs):
                out[pos_map[ix]] = h[row_i]
        return out

    # shallow baseline (position + cue features) — H1's control
    shallow_probe, shallow_m = fit_shallow(conv, conv.y.to_numpy(), groups, max_len,
                                           seed=cfg.probe.seeds[0])
    report["shallow"]["conv"] = shallow_m.__dict__
    Xs = torch.tensor(shallow_features(conv, max_len).to_numpy(dtype=np.float32))
    shallow_scores = shallow_probe.predict_proba(Xs)[:, 1].numpy()

    best = {"layer": None, "auc": -1}
    for layer in cfg.model.cache_layers:
        X = gather(conv, layer)
        probe = LinearProbe(d_model=X.shape[1],
                            hidden=128 if cfg.probe.arch == "mlp" else 0)
        m = probe.fit(X, torch.tensor(conv.y.to_numpy()), groups,
                      l2=cfg.probe.l2, epochs=cfg.probe.epochs, lr=cfg.probe.lr,
                      n_folds=cfg.probe.n_folds, seed=cfg.probe.seeds[0])
        scores = probe.predict_proba(X)[:, 1].numpy()
        strat = position_stratified_auc(scores, conv.y.to_numpy(),
                                        conv.tok_end.to_numpy())
        strat_shallow = position_stratified_auc(shallow_scores, conv.y.to_numpy(),
                                                conv.tok_end.to_numpy())
        report["conv"][layer] = {"metrics": m.__dict__, "stratified": strat,
                                 "stratified_shallow": strat_shallow}
        out_dir = paths.probes("conv", cfg.probe.arch)
        probe.save(out_dir / f"L{layer}.pt", m)
        if m.auc > best["auc"]:
            best = {"layer": layer, "auc": m.auc}
    report["conv"]["best"] = best

    # phase probe at the best conv layer (and the configured probe layer)
    ph = df[df.phase_regex.notna()].copy()
    ph = ph[ph.phase_regex.isin(PHASES)]
    phase_to_i = {p: i for i, p in enumerate(PHASES)}
    for layer in {best["layer"], cfg.model.probe_layer} - {None}:
        Xp = gather(ph, layer)
        probe = LinearProbe(d_model=Xp.shape[1], n_classes=len(PHASES))
        y = torch.tensor([phase_to_i[p] for p in ph.phase_regex])
        m = probe.fit(Xp, y, ph.problem_id.to_numpy(), l2=cfg.probe.l2,
                      epochs=cfg.probe.epochs, lr=cfg.probe.lr,
                      n_folds=cfg.probe.n_folds, seed=cfg.probe.seeds[0])
        probe.save(paths.probes("phase", cfg.probe.arch) / f"L{layer}.pt", m)
        report["phase"][layer] = m.__dict__

    (stage_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    # go/no-go summary
    bl = best["layer"]
    if bl is not None:
        strat = report["conv"][bl]["stratified"]
        strat_sh = report["conv"][bl]["stratified_shallow"]
        margins = [strat[k]["auc"] - strat_sh[k]["auc"] for k in strat
                   if not (np.isnan(strat[k]["auc"]) or np.isnan(strat_sh[k]["auc"]))]
        go = best["auc"] >= 0.75 and (min(margins) if margins else 0) >= 0.05
        print(f"GO/NO-GO: conv AUC={best['auc']:.3f} at L{bl}; "
              f"min within-stratum margin over shallow={min(margins) if margins else float('nan'):.3f} "
              f"=> {'GO' if go else 'NO-GO (see plan risk register)'}")
    mark_done(stage_dir, t0, {"best": best})


if __name__ == "__main__":
    main()
