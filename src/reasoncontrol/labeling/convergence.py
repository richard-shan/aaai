"""Forced-answer truncation: the central label engine.

For each chunk boundary b of a rollout, build prefix = prompt + output[:tok_end]
and force the suffix "\n</think>\n\nThe final answer is \\boxed{" (math). The
continuation is decoded greedily for a FIXED <=32 tokens (never a '}' stop
string — nested braces), the answer extracted by balanced-brace matching, and
compared with math-verify equivalence to (a) the rollout's own final answer
(conv_matches_final — primary self-supervised target) and (b) gold
(conv_correct).

Two-phase submission preserves vLLM prefix-cache hits under chunked-prefill
co-scheduling: phase A sends only each rollout's LONGEST prefix (disjoint
across rollouts), phase B sends the remaining boundaries, which then reuse
block-aligned cached prefixes.

The same machinery doubles as the DEER/Dynasor-style trial-decode baseline
(controller/baselines.py) — there the forced answers are used online.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..chunking import ChunkRecord
from ..data.grading import answers_equal, extract_forced_answer, grade
from .phase_regex import label_phase
from ..generation.backend import GenBackend, GenRequest


@dataclass(frozen=True)
class BoundaryProbe:
    problem_id: str
    rollout_id: int
    chunk_idx: int
    prefix_token_ids: tuple[int, ...]


def select_boundaries(chunks: list[ChunkRecord], max_per_rollout: int = 60,
                      dense_prefix: int = 20) -> list[ChunkRecord]:
    """All of the first `dense_prefix` boundaries, then every 2nd, capped."""
    by_rollout: dict[tuple, list[ChunkRecord]] = {}
    for c in chunks:
        by_rollout.setdefault((c.problem_id, c.rollout_id), []).append(c)
    out = []
    for _, cs in sorted(by_rollout.items()):
        cs = sorted(cs, key=lambda c: c.chunk_idx)
        picked = cs[:dense_prefix] + cs[dense_prefix::2]
        out.extend(picked[:max_per_rollout])
    return out


def build_probes(chunks: list[ChunkRecord], prompt_ids: dict[tuple, list[int]],
                 output_ids: dict[tuple, list[int]], suffix_ids: list[int]) -> list[BoundaryProbe]:
    probes = []
    for c in chunks:
        key = (c.problem_id, c.rollout_id)
        prompt = prompt_ids[key]
        out = output_ids[key]
        prefix = tuple(prompt) + tuple(out[:c.tok_end - len(prompt)]) + tuple(suffix_ids)
        probes.append(BoundaryProbe(c.problem_id, c.rollout_id, c.chunk_idx, prefix))
    return probes


def run_forced_answers(backend: GenBackend, probes: list[BoundaryProbe],
                       max_new_tokens: int = 32,
                       two_phase: bool = True,
                       shard_rollouts: int = 100) -> dict[tuple, str]:
    """Returns (problem_id, rollout_id, chunk_idx) -> raw continuation text.

    Rollouts are processed in shards of `shard_rollouts` so that phase-A
    prefixes are still resident in the vLLM prefix cache when phase B of the
    same shard runs (whole-corpus submission evicts them: ~10k tokens/rollout
    vs ~2M cached tokens on an 80 GB card)."""
    def req(p: BoundaryProbe) -> GenRequest:
        return GenRequest(request_id=f"{p.problem_id}|{p.rollout_id}|{p.chunk_idx}",
                          prompt_token_ids=p.prefix_token_ids,
                          max_tokens=max_new_tokens, greedy=True,
                          stop=("\n\n",))

    by_rollout: dict[tuple, list[BoundaryProbe]] = {}
    for p in probes:
        by_rollout.setdefault((p.problem_id, p.rollout_id), []).append(p)
    keys = sorted(by_rollout)
    results: dict[tuple, str] = {}
    for s0 in range(0, len(keys), max(shard_rollouts, 1)):
        shard = [p for k in keys[s0:s0 + max(shard_rollouts, 1)] for p in by_rollout[k]]
        if two_phase:
            longest: dict[tuple, BoundaryProbe] = {}
            for p in shard:
                key = (p.problem_id, p.rollout_id)
                if key not in longest or len(p.prefix_token_ids) > len(longest[key].prefix_token_ids):
                    longest[key] = p
            phase_a = list(longest.values())
            phase_a_ids = {(p.problem_id, p.rollout_id, p.chunk_idx) for p in phase_a}
            phase_b = [p for p in shard
                       if (p.problem_id, p.rollout_id, p.chunk_idx) not in phase_a_ids]
            batches = [phase_a, phase_b]
        else:
            batches = [shard]
        for batch in batches:
            if not batch:
                continue
            for res in backend.generate([req(p) for p in batch]):
                pid, rid, cid = res.request_id.split("|")
                results[(pid, int(rid), int(cid))] = res.text
    return results


def label_chunks(chunks: list[ChunkRecord], forced_raw: dict[tuple, str],
                 final_answers: dict[tuple, str | None],
                 gold_answers: dict[str, str], style: str = "math") -> list[ChunkRecord]:
    """Fill phase_regex, forced_answer, conv_matches_final, conv_correct in place."""
    for c in chunks:
        c.phase_regex = label_phase(c.text)
        raw = forced_raw.get((c.problem_id, c.rollout_id, c.chunk_idx))
        if raw is None:
            continue
        ans = extract_forced_answer(raw, style=style)
        c.forced_answer = ans
        final = final_answers.get((c.problem_id, c.rollout_id))
        if style == "mcq":
            c.conv_matches_final = (ans is not None and final is not None
                                    and ans.strip().upper() == final.strip().upper())
        else:
            c.conv_matches_final = answers_equal(ans, final)
        c.conv_correct = grade(ans, gold_answers[c.problem_id], style=style)
    return chunks
