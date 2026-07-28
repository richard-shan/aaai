"""Boundary-activation storage: safetensors shards + a parquet index.

Shard tensor "h": [n_boundaries, n_layers, d] (fp16 by default, int8 behind a
flag). Index rows map (problem_id, rollout_id, chunk_idx) -> (shard, row).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file


class ActStore:
    def __init__(self, root: str | Path, layers: tuple[int, ...], quantize_int8: bool = False):
        self.root = Path(root)
        self.layers = tuple(layers)
        self.quantize_int8 = quantize_int8
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- writing -------------------------------------------------------
    def write_shard(self, shard_idx: int, h: torch.Tensor,
                    index_rows: list[dict]) -> None:
        """h: [n, n_layers, d]; index_rows: one dict per row with keys
        problem_id, rollout_id, chunk_idx (+ anything else useful)."""
        assert h.shape[0] == len(index_rows)
        assert h.shape[1] == len(self.layers)
        path = self.root / f"acts_{shard_idx:03d}.safetensors"
        if self.quantize_int8:
            scale = h.abs().amax(dim=(0, 2), keepdim=True).clamp(min=1e-6) / 127.0
            q = (h / scale).round().clamp(-127, 127).to(torch.int8)
            save_file({"h_int8": q, "scale": scale.to(torch.float16)}, str(path))
        else:
            save_file({"h": h.to(torch.float16)}, str(path))
        df = pd.DataFrame(index_rows)
        df["shard"] = shard_idx
        df["row"] = np.arange(len(index_rows))
        idx_path = self.root / "index.parquet"
        if idx_path.exists():
            df = pd.concat([pd.read_parquet(idx_path), df], ignore_index=True)
            df = df.drop_duplicates(subset=["problem_id", "rollout_id", "chunk_idx"], keep="last")
        df.to_parquet(idx_path, index=False)

    # ---- reading -------------------------------------------------------
    def index(self) -> pd.DataFrame:
        return pd.read_parquet(self.root / "index.parquet")

    def load_shard(self, shard_idx: int) -> torch.Tensor:
        path = self.root / f"acts_{shard_idx:03d}.safetensors"
        d = load_file(str(path))
        if "h_int8" in d:
            return d["h_int8"].float() * d["scale"].float()
        return d["h"].float()

    def gather(self, keys: pd.DataFrame, layer: int) -> torch.Tensor:
        """keys: DataFrame with problem_id/rollout_id/chunk_idx (order preserved).
        Returns [len(keys), d] fp32 states at `layer`."""
        li = self.layers.index(layer)
        idx = self.index().merge(keys.reset_index(names="_ord"),
                                 on=["problem_id", "rollout_id", "chunk_idx"], how="inner")
        if len(idx) != len(keys):
            missing = len(keys) - len(idx)
            raise KeyError(f"{missing} requested boundaries missing from ActStore")
        outs = []
        for shard, grp in idx.groupby("shard"):
            h = self.load_shard(int(shard))[grp["row"].to_numpy(), li]
            grp = grp.assign(_pos=np.arange(len(grp)))
            outs.append((grp["_ord"].to_numpy(), h))
        result = torch.empty(len(keys), outs[0][1].shape[-1])
        for order, h in outs:
            result[torch.as_tensor(order)] = h
        return result
