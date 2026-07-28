"""Linear / MLP probes on boundary residual-stream states.

fit() does grouped-CV training (groups = problem_id so no problem leaks across
folds), returns AUC/acc/ECE; calibrate() applies temperature scaling on dev.
predict_proba at inference is one matvec + softmax (the controller path).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProbeMetrics:
    auc: float
    acc: float
    ece: float
    n: int
    per_fold_auc: list[float]


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    # rank-sum (Mann-Whitney) AUC with tie-averaged ranks
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg = (cum - (counts - 1) / 2.0)
    ranks = avg[inv]
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())
    return float(ece)


def group_kfold(groups: np.ndarray, n_folds: int, seed: int = 0):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_folds)
    for f in folds:
        test_mask = np.isin(groups, f)
        yield ~test_mask, test_mask


class LinearProbe(nn.Module):
    def __init__(self, d_model: int, n_classes: int = 2, hidden: int = 0):
        super().__init__()
        self.d_model, self.n_classes, self.hidden = d_model, n_classes, hidden
        if hidden > 0:
            self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(),
                                     nn.Linear(hidden, n_classes))
        else:
            self.net = nn.Linear(d_model, n_classes)
        self.register_buffer("temperature", torch.ones(1))
        self.register_buffer("mu", torch.zeros(d_model))
        self.register_buffer("sigma", torch.ones(d_model))

    # ---- training ------------------------------------------------------
    def _train_once(self, X, y, l2, epochs, lr, seed):
        torch.manual_seed(seed)
        if self.hidden > 0:
            model = nn.Sequential(nn.Linear(self.d_model, self.hidden), nn.ReLU(),
                                  nn.Linear(self.hidden, self.n_classes))
        else:
            model = nn.Linear(self.d_model, self.n_classes)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.cross_entropy(model(X), y)
            loss.backward()
            opt.step()
        return model

    def fit(self, X: torch.Tensor, y: torch.Tensor, groups: np.ndarray,
            l2: float = 1e-3, epochs: int = 200, lr: float = 1e-2,
            n_folds: int = 5, seed: int = 0) -> ProbeMetrics:
        X = X.float()
        y = y.long()
        self.mu = X.mean(0)
        self.sigma = X.std(0).clamp(min=1e-6)
        Xn = (X - self.mu) / self.sigma
        oof_probs = np.zeros(len(y))
        fold_aucs = []
        for tr, te in group_kfold(groups, n_folds, seed):
            m = self._train_once(Xn[tr], y[tr], l2, epochs, lr, seed)
            with torch.no_grad():
                p = F.softmax(m(Xn[te]), dim=-1)[:, 1].numpy()
            oof_probs[te] = p
            fold_aucs.append(_auc(p, y[te].numpy()))
        # final model on all data
        final = self._train_once(Xn, y, l2, epochs, lr, seed)
        self.net = final
        labels = y.numpy()
        preds = (oof_probs > 0.5).astype(int)
        return ProbeMetrics(auc=_auc(oof_probs, labels), acc=float((preds == labels).mean()),
                            ece=expected_calibration_error(oof_probs, labels),
                            n=len(labels), per_fold_auc=[float(a) for a in fold_aucs])

    # ---- inference -----------------------------------------------------
    @torch.no_grad()
    def logits(self, h: torch.Tensor) -> torch.Tensor:
        hn = (h.float() - self.mu) / self.sigma
        return self.net(hn) / self.temperature

    @torch.no_grad()
    def predict_proba(self, h: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.logits(h), dim=-1)

    def calibrate(self, X_val: torch.Tensor, y_val: torch.Tensor) -> float:
        """Temperature scaling; returns fitted temperature."""
        with torch.no_grad():
            hn = (X_val.float() - self.mu) / self.sigma
            raw = self.net(hn)
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(raw / log_t.exp(), y_val.long())
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = log_t.detach().exp()
        return float(self.temperature)

    @property
    def direction(self) -> torch.Tensor:
        """Class-1 weight direction in RAW activation space (for steering
        orthogonalization and logit-lens). Linear probes only."""
        if self.hidden > 0:
            raise ValueError("direction undefined for MLP probe")
        w = self.net.weight.detach()          # [n_classes, d]
        v = (w[1] - w[0]) / self.sigma        # undo standardization
        return v / v.norm()

    # ---- persistence ---------------------------------------------------
    def save(self, path: str | Path, metrics: ProbeMetrics | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.state_dict(),
                    "d_model": self.d_model, "n_classes": self.n_classes,
                    "hidden": self.hidden}, path)
        if metrics is not None:
            path.with_suffix(".metrics.json").write_text(json.dumps(asdict(metrics), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LinearProbe":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        probe = cls(blob["d_model"], blob["n_classes"], blob["hidden"])
        probe.load_state_dict(blob["state"])
        return probe
