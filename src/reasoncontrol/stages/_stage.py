"""Shared stage scaffolding: config loading, resolved-config dump, done
markers, idempotency, rollout/chunk IO helpers."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd
import zstandard

from ..config import RunCfg, dump_config, load_config
from ..paths import RunPaths


def stage_args(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--audit", action="store_true")
    return ap.parse_args()


def setup(args, stage_name: str) -> tuple[RunCfg, RunPaths, Path]:
    cfg = load_config(args.config, args.overrides)
    paths = RunPaths.create(cfg.runs_dir, cfg.model.tag)
    stage_dir = paths.model_root / "_stages" / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, stage_dir / "resolved_config.yaml")
    return cfg, paths, stage_dir


def done_marker(stage_dir: Path) -> Path:
    return stage_dir / "stage.done"


def is_done(stage_dir: Path, force: bool) -> bool:
    return done_marker(stage_dir).exists() and not force


def mark_done(stage_dir: Path, t_start: float, extra: dict | None = None) -> None:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
    except Exception:
        sha = "unknown"
    done_marker(stage_dir).write_text(json.dumps(
        {"git_sha": sha, "wall_s": time.time() - t_start, **(extra or {})}, indent=2))


# ---- rollout jsonl.zst IO ---------------------------------------------------

def write_jsonl_zst(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstandard.ZstdCompressor()
    with open(path, "wb") as f, cctx.stream_writer(f) as w:
        for r in records:
            w.write((json.dumps(r) + "\n").encode())


def read_jsonl_zst(path: Path) -> list[dict]:
    dctx = zstandard.ZstdDecompressor()
    out = []
    with open(path, "rb") as f, dctx.stream_reader(f) as r:
        buf = r.read().decode()
    for line in buf.splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def read_all_rollouts(rollout_dir: Path) -> list[dict]:
    out = []
    for shard in sorted(rollout_dir.glob("shard_*.jsonl.zst")):
        out.extend(read_jsonl_zst(shard))
    return out


def load_chunks_df(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
