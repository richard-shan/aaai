"""Difference-of-means steering directions with sensor/actuator decoupling.

v = mean(h at boundaries whose NEXT chunk is <phase>) - mean(h at other
boundaries), unit-normalized, then orthogonalized against the probe weight
directions so steering cannot mechanically move the probes it is gated on
(the probe layer is also strictly below the steer layer; this is defense in
depth for the same failure mode).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def diff_of_means(h: torch.Tensor, is_target: np.ndarray) -> torch.Tensor:
    """h: [n, d] boundary states; is_target: bool mask (next chunk is phase)."""
    t = torch.as_tensor(is_target, dtype=torch.bool)
    if t.sum() == 0 or (~t).sum() == 0:
        raise ValueError("need both target and non-target boundaries")
    v = h[t].float().mean(0) - h[~t].float().mean(0)
    return v / v.norm()


def orthogonalize(v: torch.Tensor, against: list[torch.Tensor]) -> tuple[torch.Tensor, dict]:
    """Project v off each direction in `against`; report cosines before/after."""
    report = {}
    out = v.clone().float()
    for i, u in enumerate(against):
        u = u.float()
        u = u / u.norm()
        report[f"cos_before_{i}"] = float(torch.dot(out / out.norm(), u))
        out = out - torch.dot(out, u) * u
    out = out / out.norm()
    for i, u in enumerate(against):
        u = u.float()
        u = u / u.norm()
        report[f"cos_after_{i}"] = float(torch.dot(out, u))
    return out, report


def save_vector(path: str | Path, v: torch.Tensor, layer: int, r_bar: float,
                meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"v": v.detach().float(), "layer": layer, "r_bar": r_bar,
                "meta": meta or {}}, path)


def load_vector(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
