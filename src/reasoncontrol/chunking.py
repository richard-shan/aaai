"""Reasoning-trace chunking with ONE boundary detector shared by the offline
chunker and the online controller loop.

A boundary falls after token t iff the decoded text of the rolling 2-token tail
(t-1, t) ends with "\n\n". Using a 2-token window catches "\n"+"\n" split across
tokens as well as merged tokens like ".\n\n"; consecutive-blank-line runs
("\n\n\n") produce one boundary per token whose tail ends in "\n\n", which the
offline chunker collapses via the min-chunk merge.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChunkRecord:
    problem_id: str
    rollout_id: int
    chunk_idx: int
    text: str
    tok_start: int               # index into the FULL sequence (prompt + output)
    tok_end: int                 # exclusive; boundary state = hidden state at tok_end - 1
    phase_regex: str | None = None
    phase_judge: str | None = None
    forced_answer: str | None = None
    conv_matches_final: bool | None = None
    conv_correct: bool | None = None


class BoundaryDetector:
    """Streaming detector. Feed one token id at a time; returns True if a chunk
    boundary falls after that token. Deterministic function of the token id
    stream, so offline replay reproduces online decisions exactly."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._tail: list[int] = []

    def reset(self) -> None:
        self._tail = []

    def push(self, token_id: int) -> bool:
        self._tail.append(token_id)
        if len(self._tail) > 2:
            self._tail.pop(0)
        text = self.tokenizer.decode(self._tail, skip_special_tokens=False)
        return text.endswith("\n\n")


def find_boundaries(token_ids: list[int], tokenizer) -> list[int]:
    """Offline replay of the streaming detector.

    Returns positions p (0-based, relative to token_ids) such that a boundary
    falls AFTER token p.
    """
    det = BoundaryDetector(tokenizer)
    return [i for i, t in enumerate(token_ids) if det.push(t)]


def _subsample(n: int, max_chunks: int, keep_first: int, keep_last: int) -> list[int]:
    if n <= max_chunks:
        return list(range(n))
    head = list(range(keep_first))
    tail = list(range(n - keep_last, n))
    middle = [i for i in range(keep_first, n - keep_last)]
    budget = max_chunks - keep_first - keep_last
    stride = max(1, len(middle) // budget)
    mid = middle[::stride][:budget]
    return sorted(set(head + mid + tail))


def chunk_trace(output_token_ids: list[int], tokenizer, prompt_len: int,
                problem_id: str = "", rollout_id: int = 0,
                min_chunk_tokens: int = 12, max_chunks: int = 160,
                keep_first: int = 10, keep_last: int = 5,
                think_end: int | None = None) -> list[ChunkRecord]:
    """Chunk the thinking region of a rollout.

    output_token_ids: tokens generated after the prompt. think_end: index into
    output_token_ids of the first token of "</think>" (None = whole output is
    the trace). tok_start/tok_end in the returned records are FULL-sequence
    indices (prompt_len offset applied).
    """
    trace = output_token_ids[:think_end] if think_end is not None else output_token_ids
    bounds = find_boundaries(trace, tokenizer)
    # segment [start, b] for each boundary b, plus the trailing remainder
    segments: list[tuple[int, int]] = []  # [start, end) in trace coords
    start = 0
    for b in bounds:
        segments.append((start, b + 1))
        start = b + 1
    if start < len(trace):
        segments.append((start, len(trace)))
    # merge short chunks into their successor
    merged: list[tuple[int, int]] = []
    for seg in segments:
        if merged and (merged[-1][1] - merged[-1][0]) < min_chunk_tokens:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)
    # a short trailing chunk merges backwards
    if len(merged) >= 2 and (merged[-1][1] - merged[-1][0]) < min_chunk_tokens:
        merged[-2] = (merged[-2][0], merged[-1][1])
        merged.pop()
    keep = _subsample(len(merged), max_chunks, keep_first, keep_last)
    records = []
    for new_idx, i in enumerate(keep):
        s, e = merged[i]
        records.append(ChunkRecord(
            problem_id=problem_id, rollout_id=rollout_id, chunk_idx=new_idx,
            text=tokenizer.decode(trace[s:e], skip_special_tokens=False),
            tok_start=prompt_len + s, tok_end=prompt_len + e,
        ))
    return records
