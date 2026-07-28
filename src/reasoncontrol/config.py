"""YAML + dataclass config layer.

Later files override earlier ones; dotted --set overrides apply last.
Every stage dumps its resolved config next to its outputs.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelCfg:
    hf_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tag: str = "r1_qwen_1p5b"
    dtype: str = "float16"
    # residual-stream layers cached by stage capture (post-block hidden states)
    cache_layers: tuple[int, ...] = (9, 12, 15, 18, 21, 24, 26, 27)
    probe_layer: int = 18          # sensor: strictly below steer_layer
    steer_layer: int = 21          # actuator
    think_tags: bool = True        # False for non-reasoning smoke models
    device: str = "auto"           # auto -> cuda if available else cpu


@dataclass(frozen=True)
class GenCfg:
    temperature: float = 0.6
    top_p: float = 0.95
    max_think_tokens: int = 16384
    max_answer_tokens: int = 512
    n_rollouts: int = 4
    backend: str = "vllm"          # vllm | hf
    batch_size: int = 32
    limit_problems: int = 0        # 0 = all; >0 = pilot on the first N per dataset


@dataclass(frozen=True)
class ChunkCfg:
    min_chunk_tokens: int = 12
    max_chunks: int = 160
    keep_first: int = 10
    keep_last: int = 5


@dataclass(frozen=True)
class ForcedAnswerCfg:
    suffix_math: str = "\n</think>\n\nThe final answer is \\boxed{"
    suffix_mcq: str = "\n</think>\n\nThe final answer is ("
    max_new_tokens: int = 32
    max_boundaries_per_rollout: int = 60
    dense_prefix: int = 20         # label all of the first N boundaries, then every 2nd


@dataclass(frozen=True)
class ProbeCfg:
    arch: str = "linear"           # linear | mlp
    l2: float = 1e-3
    epochs: int = 200
    lr: float = 1e-2
    n_folds: int = 5
    seeds: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class PolicyCfg:
    kind: str = "full"             # full | noop | static_budget | exit_only | steer_only |
                                   # trial_decode | concise_prompt | budget_prompt
    tau_exit: float = 0.9
    tau_steer: float = 0.7
    alpha: float = 6.0             # units of mean residual norm at steer layer
    patience_k: int = 2
    min_chunks: int = 4
    steer_phases: tuple[str, ...] = ("verification", "backtracking")
    break_loops: bool = True       # not-converged + sustained verify/backtrack -> steer to deduction
    loop_patience: int = 3
    hysteresis: float = 0.1
    budget: int = 4096             # static_budget / budget_prompt
    trial_agree_k: int = 2         # trial_decode: consecutive agreeing forced answers


@dataclass(frozen=True)
class RunCfg:
    model: ModelCfg = field(default_factory=ModelCfg)
    gen: GenCfg = field(default_factory=GenCfg)
    chunk: ChunkCfg = field(default_factory=ChunkCfg)
    forced: ForcedAnswerCfg = field(default_factory=ForcedAnswerCfg)
    probe: ProbeCfg = field(default_factory=ProbeCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    runs_dir: str = "runs"
    datasets: tuple[str, ...] = ("math500", "gsm8k", "aime", "gpqa_diamond")
    seed: int = 0


def _coerce(tp: Any, val: Any) -> Any:
    origin = getattr(tp, "__origin__", None)
    if is_dataclass(tp) and isinstance(val, dict):
        return _from_dict(tp, val)
    if origin is tuple and isinstance(val, (list, tuple)):
        args = tp.__args__
        inner = args[0] if args and args[-1] is Ellipsis else None
        return tuple(_coerce(inner, v) if inner else v for v in val)
    if tp is float and isinstance(val, int):
        return float(val)
    return val


def _from_dict(cls: type, d: dict) -> Any:
    kwargs = {}
    valid = {f.name: f for f in fields(cls)}
    for k, v in d.items():
        if k not in valid:
            raise KeyError(f"unknown config key {cls.__name__}.{k}")
        kwargs[k] = _coerce(_resolve(cls, k), v)
    return cls(**kwargs)


def _resolve(cls: type, name: str) -> Any:
    import typing
    hints = typing.get_type_hints(cls)
    return hints[name]


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_set(expr: str) -> tuple[list[str], Any]:
    key, _, raw = expr.partition("=")
    if not _:
        raise ValueError(f"--set expects key=value, got {expr!r}")
    val = yaml.safe_load(raw)
    return key.split("."), val


def load_config(paths: list[str | Path], overrides: list[str] | None = None) -> RunCfg:
    merged: dict = {}
    for p in paths:
        with open(p) as f:
            doc = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, doc)
    for expr in overrides or []:
        keys, val = _parse_set(expr)
        node = merged
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = val
    return _from_dict(RunCfg, merged)


def dump_config(cfg: Any, path: str | Path) -> None:
    def to_plain(obj: Any) -> Any:
        if is_dataclass(obj):
            return {f.name: to_plain(getattr(obj, f.name)) for f in fields(obj)}
        if isinstance(obj, (list, tuple)):
            return [to_plain(v) for v in obj]
        return obj

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(to_plain(cfg), f, sort_keys=False)
