"""Baseline runners that do NOT need the HF closed-loop (they run on any
GenBackend, i.e. vLLM on the GPU box — ~10x cheaper than the HF loop):

- run_plain:        NoOp reference / ConcisePrompt / BudgetPrompt (prompt-level)
- run_static_budget: s1-style budget forcing (truncate at B, splice suffix,
                     greedy answer) — never truncation-without-answer
- run_trial_decode: DEER/Dynasor-style exit — segment-wise generation
                    (stop="\n\n"), forced answer at each boundary, exit on k
                    consecutive math-verify-agreeing forced answers. Its
                    per-boundary cost (a <=32-token decode) vs our probe's one
                    matvec is a headline comparison.

HF-loop policies (full/exit_only/steer_only/noop) live in policy.py.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..data.grading import answers_equal, extract_forced_answer
from ..generation.backend import GenBackend, GenRequest

CONCISE_SUFFIX = "\n\nBe concise: keep your reasoning as short as possible."


def budget_suffix(budget: int) -> str:
    return f"\n\nSolve this using at most {budget} tokens of reasoning."


@dataclass
class PlainResult:
    problem_id: str
    rollout_id: int
    output_token_ids: list[int]
    text: str
    n_think_tokens: int
    n_boundary_decodes: int = 0
    extra_decode_tokens: int = 0


def run_plain(backend: GenBackend, jobs: list, max_think: int, max_answer: int,
              temperature: float = 0.6, top_p: float = 0.95,
              seed: int = 0) -> list[PlainResult]:
    reqs = [GenRequest(request_id=f"{j.problem_id}|{j.rollout_id}",
                       prompt_token_ids=tuple(j.prompt_ids),
                       max_tokens=max_think + max_answer,
                       temperature=temperature, top_p=top_p,
                       seed=seed + j.rollout_id) for j in jobs]
    out = []
    for j, r in zip(jobs, backend.generate(reqs)):
        text = r.text
        n_think = len(r.output_token_ids)
        if "</think>" in text:
            think_part = text.split("</think>")[0]
            # approximation is fine here: think length re-measured in analysis
            n_think = int(len(r.output_token_ids) * max(len(think_part), 1) / max(len(text), 1))
        out.append(PlainResult(j.problem_id, j.rollout_id, list(r.output_token_ids),
                               text, n_think))
    return out


def run_static_budget(backend: GenBackend, jobs: list, budget: int,
                      exit_suffix_ids: list[int], max_answer: int,
                      temperature: float = 0.6, top_p: float = 0.95,
                      seed: int = 0) -> list[PlainResult]:
    think_reqs = [GenRequest(request_id=f"{j.problem_id}|{j.rollout_id}",
                             prompt_token_ids=tuple(j.prompt_ids),
                             max_tokens=budget, temperature=temperature,
                             top_p=top_p, seed=seed + j.rollout_id,
                             stop=("</think>",)) for j in jobs]
    thinks = backend.generate(think_reqs)
    ans_reqs, metas = [], []
    for j, t in zip(jobs, thinks):
        prefix = tuple(j.prompt_ids) + tuple(t.output_token_ids) + tuple(exit_suffix_ids)
        ans_reqs.append(GenRequest(request_id=t.request_id, prompt_token_ids=prefix,
                                   max_tokens=max_answer, greedy=True))
        metas.append((j, t))
    out = []
    for (j, t), a in zip(metas, backend.generate(ans_reqs)):
        out.append(PlainResult(j.problem_id, j.rollout_id,
                               list(t.output_token_ids) + list(exit_suffix_ids)
                               + list(a.output_token_ids),
                               t.text + "\n</think>\n\n" + a.text,
                               n_think_tokens=len(t.output_token_ids)))
    return out


def run_trial_decode(backend: GenBackend, jobs: list, suffix_ids: list[int],
                     agree_k: int = 2, min_chunks: int = 4, max_think: int = 16384,
                     max_answer: int = 512, forced_max_tokens: int = 32,
                     temperature: float = 0.6, top_p: float = 0.95,
                     seed: int = 0, style: str = "math",
                     segment_stop: str = "\n\n") -> list[PlainResult]:
    """Segment-wise: generate to the next boundary, force an answer, exit when
    the last `agree_k` forced answers agree (math-verify equivalence)."""
    out = []
    for j in jobs:
        prefix = list(j.prompt_ids)
        n_think = 0
        history: list[str | None] = []
        n_probes, extra_tokens = 0, 0
        chunk_i = 0
        done_thinking = False
        while n_think < max_think:
            seg = backend.generate([GenRequest(
                request_id="seg", prompt_token_ids=tuple(prefix),
                max_tokens=min(1024, max_think - n_think),
                temperature=temperature, top_p=top_p,
                seed=seed + j.rollout_id + 31 * chunk_i,
                stop=(segment_stop, "</think>"))])[0]
            prefix += list(seg.output_token_ids)
            n_think += len(seg.output_token_ids)
            chunk_i += 1
            if "</think>" in seg.text or not seg.output_token_ids:
                done_thinking = "</think>" in seg.text
                break
            forced = backend.generate([GenRequest(
                request_id="probe", prompt_token_ids=tuple(prefix) + tuple(suffix_ids),
                max_tokens=forced_max_tokens, greedy=True, stop=("\n\n",))])[0]
            n_probes += 1
            extra_tokens += len(forced.output_token_ids)
            ans = extract_forced_answer(forced.text, style=style)
            history.append(ans)
            if (chunk_i >= min_chunks and len(history) >= agree_k
                    and all(a is not None for a in history[-agree_k:])
                    and all(answers_equal(history[-1], a) for a in history[-agree_k:])):
                break
        answer_prefix = tuple(prefix) + (() if done_thinking else tuple(suffix_ids))
        ans = backend.generate([GenRequest(request_id="ans",
                                           prompt_token_ids=answer_prefix,
                                           max_tokens=max_answer, greedy=True)])[0]
        # token ids are authoritative; callers decode text with the tokenizer
        out.append(PlainResult(j.problem_id, j.rollout_id,
                               list(answer_prefix[len(j.prompt_ids):]) + list(ans.output_token_ids),
                               "", n_think_tokens=n_think,
                               n_boundary_decodes=n_probes,
                               extra_decode_tokens=extra_tokens))
    return out
