"""Shallow-feature baselines for H1: if a logistic regression on position and
surface cues matches the hidden-state probe, the probe decodes nothing latent.

Features per boundary: chunk_idx, tokens_so_far, frac_of_max_len, last-chunk
length, cue-pattern counts. Position-stratified AUC utilities live here too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..labeling.phase_regex import cue_counts
from .probe import LinearProbe, ProbeMetrics, _auc

FEATURES = ["chunk_idx", "tokens_so_far", "frac_len", "chunk_len",
            "cues_exploration", "cues_deduction", "cues_verification", "cues_backtracking"]


def shallow_features(chunks: pd.DataFrame, max_len: int) -> pd.DataFrame:
    rows = []
    for r in chunks.itertuples(index=False):
        f = {"chunk_idx": r.chunk_idx,
             "tokens_so_far": r.tok_end,
             "frac_len": r.tok_end / max_len,
             "chunk_len": r.tok_end - r.tok_start}
        f.update(cue_counts(r.text))
        rows.append(f)
    return pd.DataFrame(rows)[FEATURES]


def fit_shallow(chunks: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                max_len: int, seed: int = 0) -> tuple[LinearProbe, ProbeMetrics]:
    X = torch.tensor(shallow_features(chunks, max_len).to_numpy(dtype=np.float32))
    probe = LinearProbe(d_model=X.shape[1], n_classes=2)
    metrics = probe.fit(X, torch.tensor(y), groups, seed=seed)
    return probe, metrics


def position_stratified_auc(scores: np.ndarray, labels: np.ndarray,
                            tokens_so_far: np.ndarray, n_strata: int = 3) -> dict:
    """AUC within position strata (by tokens_so_far tercile). The go/no-go
    compares probe-vs-shallow within strata, not on the pooled set."""
    qs = np.quantile(tokens_so_far, np.linspace(0, 1, n_strata + 1))
    out = {}
    for i in range(n_strata):
        lo, hi = qs[i], qs[i + 1]
        mask = (tokens_so_far >= lo) & (tokens_so_far <= hi if i == n_strata - 1
                                        else tokens_so_far < hi)
        out[f"stratum_{i}"] = {"n": int(mask.sum()),
                               "auc": _auc(scores[mask], labels[mask])}
    return out
