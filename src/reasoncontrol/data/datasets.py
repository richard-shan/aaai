"""Problem loading. HF downloads happen only in stage prepare_data (GPU box);
everything downstream reads the pinned parquet manifests, so unit tests and
offline work never touch the hub.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# split name -> role; every test-time knob's dev set is fixed here (see plan)
SPLITS = ("probe_train", "dev", "test")


@dataclass(frozen=True)
class Problem:
    problem_id: str          # "{dataset}/{split}/{idx}"
    dataset: str             # math500 | math_train | gsm8k | aime | gpqa_diamond
    split: str
    question: str
    gold_answer: str         # canonical string; GPQA: letter "A".."D"
    meta: dict = field(default_factory=dict)


def load_problems(manifest_dir: str | Path, dataset: str,
                  split: str | None = None) -> list[Problem]:
    path = Path(manifest_dir) / f"{dataset}.parquet"
    df = pd.read_parquet(path)
    if split is not None:
        df = df[df["split"] == split]
    out = []
    for row in df.itertuples(index=False):
        # meta is stored as a JSON string (arrow cannot write empty structs)
        if isinstance(row.meta, str):
            meta = json.loads(row.meta) if row.meta else {}
        else:
            meta = row.meta if isinstance(row.meta, dict) else {}
        out.append(Problem(problem_id=row.problem_id, dataset=row.dataset,
                           split=row.split, question=row.question,
                           gold_answer=str(row.gold_answer), meta=meta))
    return out


def save_manifest(problems: list[Problem], manifest_dir: str | Path, dataset: str) -> Path:
    Path(manifest_dir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{
        "problem_id": p.problem_id, "dataset": p.dataset, "split": p.split,
        "question": p.question, "gold_answer": p.gold_answer,
        "meta": json.dumps(p.meta or {}),
    } for p in problems])
    path = Path(manifest_dir) / f"{dataset}.parquet"
    df.to_parquet(path, index=False)
    return path


def build_prompt(problem: Problem, style: str = "math") -> str:
    if style == "mcq":
        choices = problem.meta.get("choices", [])
        lettered = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
        return (f"{problem.question}\n\n{lettered}\n\n"
                "Answer with the letter of the correct choice.")
    return (f"{problem.question}\n\n"
            "Please reason step by step, and put your final answer within \\boxed{}.")
