"""LLM-judge phase annotation for a stratified audit subset.

Runs a local judge model (default Qwen2.5-32B-Instruct-AWQ on vLLM) over
sampled chunks; reports Cohen's kappa vs the regex labels. kappa < 0.5
triggers the judge-only probe-training fallback (risk register).
"""
from __future__ import annotations

import re

import numpy as np

from ..generation.backend import GenBackend, GenRequest
from .phase_regex import PHASES

JUDGE_PROMPT = """You are annotating one step of a model's math reasoning trace.

Classify the step into exactly one category:
- exploration: trying a new approach, hypothesis, or example
- deduction: carrying out a derivation or computation step
- verification: checking, confirming, or re-deriving an earlier result
- backtracking: noticing a mistake or abandoning the current line ("Wait...")
- other: none of the above

Step:
---
{chunk}
---
Answer with only the category name."""


def judge_requests(chunks: list, tokenizer, max_chars: int = 1200) -> list[GenRequest]:
    reqs = []
    for i, c in enumerate(chunks):
        prompt = JUDGE_PROMPT.format(chunk=c.text[:max_chars])
        msgs = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        reqs.append(GenRequest(request_id=str(i), prompt_token_ids=tuple(ids),
                               max_tokens=8, greedy=True))
    return reqs


def parse_judge(text: str) -> str:
    t = text.strip().lower()
    for ph in PHASES:
        if re.search(rf"\b{ph}\b", t):
            return ph
    return "other"


def cohens_kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(a)
    conf = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        conf[idx[x], idx[y]] += 1
    po = np.trace(conf) / n
    pe = float((conf.sum(0) * conf.sum(1)).sum()) / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def stratified_sample(chunks: list, per_phase: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    by_phase: dict[str, list] = {}
    for c in chunks:
        by_phase.setdefault(c.phase_regex or "other", []).append(c)
    out = []
    for ph, cs in sorted(by_phase.items()):
        take = min(per_phase, len(cs))
        out.extend(list(rng.choice(np.array(cs, dtype=object), take, replace=False)))
    return out
