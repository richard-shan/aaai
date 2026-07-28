"""Canonical artifact layout. One source of truth for where every stage reads/writes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path
    model_tag: str

    @classmethod
    def create(cls, runs_dir: str | Path, model_tag: str) -> "RunPaths":
        return cls(root=Path(runs_dir), model_tag=model_tag)

    @property
    def model_root(self) -> Path:
        return self.root / self.model_tag

    def manifests(self) -> Path:
        return self.root / "data" / "manifests"

    def rollouts(self, dataset: str) -> Path:
        return self.model_root / "rollouts" / dataset

    def chunks(self, dataset: str) -> Path:
        return self.model_root / "chunks" / dataset / "chunks.parquet"

    def acts_dir(self, dataset: str) -> Path:
        return self.model_root / "acts" / dataset

    def forced(self, dataset: str) -> Path:
        return self.model_root / "forced" / dataset / "forced.parquet"

    def labels(self, dataset: str) -> Path:
        return self.model_root / "labels" / dataset / "labels.parquet"

    def probes(self, task: str, arch: str) -> Path:
        return self.model_root / "probes" / f"{task}_{arch}"

    def steering(self) -> Path:
        return self.model_root / "steering"

    def controller(self, policy_hash: str, dataset: str) -> Path:
        return self.model_root / "controller" / policy_hash / dataset

    def analysis(self) -> Path:
        return self.root / "analysis" / self.model_tag
