"""Batched manual decode loop with per-row closed-loop interventions.

Design (see plan):
- StaticCache, left-padded batch so every row shares cache slots and a single
  1-D cache_position; per-row position_ids and an explicit 2-D padding mask.
- Compaction instead of in-place refill: when enough rows finish (or the cache
  fills), alive rows + new jobs are re-prefilled into a fresh cache. Steered
  spans are tracked per row and replayed via the hook's prefill alpha map, so
  recomputed KV matches the originally-steered trajectory.
- Boundary semantics match offline labeling exactly: a boundary is detected
  when the sampled token completes "\n\n" (shared BoundaryDetector); the
  probe reads that token's hidden state on the NEXT forward (its state at
  input time = the state at tok_end - 1 offline), and EXIT splices the suffix
  immediately after it — identical to the forced-answer prefix construction.
- Per-row torch.Generator for sampling; finished rows keep attending to their
  own tokens (never fully masked) and their outputs are discarded.
- Probe cost: one fp32 matvec per boundary on a state the forward already
  computed. No extra model passes, ever.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn.functional as F
from transformers import StaticCache

from ..chunking import BoundaryDetector
from ..config import GenCfg, PolicyCfg
from ..controller.policy import Action, ControllerPolicy, RowState
from ..steering.hooks import ProbeTap, SteeringHook


class Mode(Enum):
    THINK = "think"
    ANSWER = "answer"           # natural </think>: keep sampling
    ANSWER_GREEDY = "answer_greedy"   # after forced EXIT: greedy (matches labels)


@dataclass
class Job:
    problem_id: str
    rollout_id: int
    prompt_ids: list[int]
    gold_answer: str = ""
    style: str = "math"


@dataclass
class ControllerResult:
    problem_id: str
    rollout_id: int
    output_token_ids: list[int]
    text: str
    n_think_tokens: int
    exited_early: bool
    actions_log: list[tuple]     # (chunk_idx, action, p_conv, phase, tokens_used)
    wall_s: float
    n_forwards: int


@dataclass
class _Row:
    job: Job
    out_ids: list[int] = field(default_factory=list)
    mode: Mode = Mode.THINK
    state: RowState = field(default_factory=RowState)
    detector: BoundaryDetector = None
    end_detector: BoundaryDetector = None   # reused class for </think> detection
    gen: torch.Generator = None
    alpha: float = 0.0                       # current steering alpha (suppress<0)
    steer_kind: str | None = None
    steer_spans: list[list] = field(default_factory=list)  # [start, end|None, alpha]
    pending_boundary: bool = False
    force_queue: list[int] = field(default_factory=list)
    n_think: int = 0
    n_answer: int = 0
    finished: bool = False
    actions_log: list[tuple] = field(default_factory=list)
    t_start: float = 0.0
    n_forwards: int = 0

    @property
    def real_len(self) -> int:
        return len(self.job.prompt_ids) + len(self.out_ids)


class ThinkEndDetector:
    """Streaming detector for the literal '</think>' substring over decoded tail."""

    def __init__(self, tokenizer, window: int = 6):
        self.tokenizer = tokenizer
        self.window = window
        self._tail: list[int] = []

    def push(self, token_id: int) -> bool:
        self._tail.append(token_id)
        if len(self._tail) > self.window:
            self._tail.pop(0)
        return "</think>" in self.tokenizer.decode(self._tail, skip_special_tokens=False)


def _row_seed(problem_id: str, rollout_id: int, global_seed: int) -> int:
    h = hashlib.sha256(f"{problem_id}|{rollout_id}|{global_seed}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**63)


def sample_token(logits: torch.Tensor, temperature: float, top_p: float,
                 gen: torch.Generator, greedy: bool = False) -> int:
    if greedy or temperature <= 0:
        return int(logits.argmax().item())
    probs = F.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(descending=True)
        cum = sorted_probs.cumsum(-1)
        keep = cum - sorted_probs < top_p     # keep tokens whose cumsum-before < p
        sorted_probs = sorted_probs * keep
        sorted_probs = sorted_probs / sorted_probs.sum()
        pick = torch.multinomial(sorted_probs, 1, generator=gen)
        return int(sorted_idx[pick].item())
    return int(torch.multinomial(probs, 1, generator=gen).item())


class ControlledRunner:
    def __init__(self, model, tokenizer, policy: ControllerPolicy,
                 gen_cfg: GenCfg, policy_cfg: PolicyCfg,
                 conv_probe=None, phase_probe=None, phase_names: tuple[str, ...] = (),
                 probe_layer: int | None = None,
                 suppress_hook: SteeringHook | None = None,
                 break_hook: SteeringHook | None = None,
                 steer_layer: int | None = None,
                 exit_suffix: str = "\n</think>\n\n",
                 global_seed: int = 0,
                 cache_headroom: int = 4096,
                 refill_frac: float = 0.25,
                 think_tags: bool = True):
        self.model = model
        self.tok = tokenizer
        self.policy = policy
        self.gen_cfg = gen_cfg
        self.policy_cfg = policy_cfg
        self.conv_probe = conv_probe
        self.phase_probe = phase_probe
        self.phase_names = phase_names
        self.probe_layer = probe_layer
        self.suppress_hook = suppress_hook
        self.break_hook = break_hook
        self.steer_layer = steer_layer
        self.exit_suffix_ids = tokenizer.encode(exit_suffix, add_special_tokens=False)
        self.global_seed = global_seed
        self.cache_headroom = cache_headroom
        self.refill_frac = refill_frac
        self.think_tags = think_tags
        self.device = next(model.parameters()).device
        self.tap = ProbeTap()
        self.eos_ids = set()
        if tokenizer.eos_token_id is not None:
            eos = tokenizer.eos_token_id
            self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])

    # ------------------------------------------------------------------
    def run(self, jobs: list[Job], batch_size: int = 8,
            progress_cb=None) -> list[ControllerResult]:
        pending = list(jobs)[::-1]           # pop() from the end
        results: list[ControllerResult] = []
        rows: list[_Row] = []
        taps_attached = []
        if self.probe_layer is not None:
            taps_attached.append(self.tap.attach(self.model, self.probe_layer))
        try:
            while pending or rows:
                # (re)build the batch: keep alive rows, add new ones
                rows = [r for r in rows if not r.finished]
                while pending and len(rows) < batch_size:
                    rows.append(self._new_row(pending.pop()))
                if not rows:
                    break
                results.extend(self._run_batch_until_compaction(rows))
                if progress_cb:
                    progress_cb(len(results), len(pending))
            return results
        finally:
            self.tap.detach()

    # ------------------------------------------------------------------
    def _new_row(self, job: Job) -> _Row:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(_row_seed(job.problem_id, job.rollout_id, self.global_seed))
        row = _Row(job=job, detector=BoundaryDetector(self.tok),
                   end_detector=ThinkEndDetector(self.tok), gen=gen)
        row.t_start = time.time()
        if not self.think_tags:
            # smoke models without <think>: whole output treated as trace,
            # natural end only via EOS
            row.end_detector = None
        return row

    # ------------------------------------------------------------------
    def _prefill(self, rows: list[_Row]):
        seqs = [r.job.prompt_ids + r.out_ids for r in rows]
        maxlen = max(len(s) for s in seqs)
        B = len(rows)
        pad_id = self.tok.pad_token_id or (next(iter(self.eos_ids)) if self.eos_ids else 0)
        ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
        mask = torch.zeros((B, maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, maxlen - len(s):] = torch.tensor(s, dtype=torch.long)
            mask[i, maxlen - len(s):] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        max_new = self.gen_cfg.max_think_tokens + self.gen_cfg.max_answer_tokens \
            + len(self.exit_suffix_ids)
        remaining = max(max_new - min(len(r.out_ids) for r in rows), 256)
        cache_len = maxlen + min(remaining, self.cache_headroom)
        cache = StaticCache(config=self.model.config, max_cache_len=cache_len)
        # replay steered spans into the prefill alpha maps
        for hook, kind in ((self.suppress_hook, "suppress"), (self.break_hook, "break")):
            if hook is None:
                continue
            amap = torch.zeros((B, maxlen))
            any_span = False
            for i, r in enumerate(rows):
                off = maxlen - len(seqs[i])
                for start, end, alpha, k in [(sp[0], sp[1], sp[2], sp[3]) for sp in r.steer_spans]:
                    if k != kind:
                        continue
                    e = end if end is not None else r.real_len
                    amap[i, off + start:off + e] = alpha
                    any_span = True
            hook.set_prefill(amap if any_span else None)
        ids, mask, pos = ids.to(self.device), mask.to(self.device), pos.to(self.device)
        with torch.no_grad():
            # logits_to_keep=1: only the last position is sampled from; full
            # B x maxlen x vocab logits OOM on compaction re-prefills
            out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                             past_key_values=cache, use_cache=True,
                             logits_to_keep=1,
                             cache_position=torch.arange(maxlen, device=self.device))
        for r in rows:
            r.n_forwards += 1
        return cache, out.logits[:, -1], mask, maxlen, cache_len

    # ------------------------------------------------------------------
    def _run_batch_until_compaction(self, rows: list[_Row]) -> list[ControllerResult]:
        cache, last_logits, mask, written, cache_len = self._prefill(rows)
        B = len(rows)
        finished_results: list[ControllerResult] = []
        # boundary decisions pending from prefill (only when resuming mid-trace
        # after compaction we do NOT re-decide old boundaries; state carries over)
        step = 0
        while True:
            alive = [r for r in rows if not r.finished]
            n_finished = B - len(alive)
            if not alive:
                break
            if n_finished / B >= self.refill_frac and n_finished > 0:
                break                      # compaction: caller rebuilds batch
            if cache_len is not None and written >= cache_len:
                break                      # cache full: rebuild with more room
            # ---- decide next token per row (probe decisions first) -----
            next_ids = torch.zeros((B, 1), dtype=torch.long)
            for i, r in enumerate(rows):
                if r.finished:
                    next_ids[i, 0] = self.tok.pad_token_id or 0
                    continue
                if r.pending_boundary:
                    # the boundary token's hidden state was stashed by the
                    # forward that just ran (it was the input token)
                    if r.mode is Mode.THINK:
                        self._decide_at_boundary(r, i)
                    r.pending_boundary = False
                next_ids[i, 0] = self._next_token(r, last_logits[i])
            # ---- bookkeeping on the sampled tokens ---------------------
            for i, r in enumerate(rows):
                if r.finished:
                    continue
                t = int(next_ids[i, 0])
                r.out_ids.append(t)
                self._track_token(r, t)
            # ---- forward ----------------------------------------------
            alive_any = any(not r.finished for r in rows)
            if not alive_any:
                break
            if self.suppress_hook is not None:
                self.suppress_hook.set_rows(torch.tensor(
                    [r.alpha if r.steer_kind == "suppress" and not r.finished else 0.0
                     for r in rows]))
            if self.break_hook is not None:
                self.break_hook.set_rows(torch.tensor(
                    [r.alpha if r.steer_kind == "break" and not r.finished else 0.0
                     for r in rows]))
            mask = torch.cat([mask, torch.ones((B, 1), dtype=mask.dtype, device=self.device)],
                             dim=-1)
            pos = torch.tensor([[r.real_len - 1] for r in rows], device=self.device)
            with torch.no_grad():
                out = self.model(input_ids=next_ids.to(self.device), attention_mask=mask,
                                 position_ids=pos, past_key_values=cache, use_cache=True,
                                 cache_position=torch.tensor([written], device=self.device))
            for r in rows:
                if not r.finished:
                    r.n_forwards += 1
            last_logits = out.logits[:, -1]
            written += 1
            step += 1
            # collect newly finished
            for r in rows:
                if r.finished and not getattr(r, "_collected", False):
                    finished_results.append(self._result(r))
                    r._collected = True
        for r in rows:
            if r.finished and not getattr(r, "_collected", False):
                finished_results.append(self._result(r))
                r._collected = True
        return finished_results

    # ------------------------------------------------------------------
    def _decide_at_boundary(self, r: _Row, row_idx: int) -> None:
        h = self.tap.last[row_idx] if self.tap.last is not None else None
        p_conv = None
        phase = None
        if self.conv_probe is not None and h is not None:
            p_conv = float(self.conv_probe.predict_proba(h.unsqueeze(0).cpu())[0, 1])
        if self.phase_probe is not None and h is not None and self.phase_names:
            pi = int(self.phase_probe.predict_proba(h.unsqueeze(0).cpu())[0].argmax())
            phase = self.phase_names[pi]
        action = self.policy.decide(r.state, p_conv, phase)
        r.actions_log.append((r.state.chunk_count, action.value, p_conv, phase,
                              len(r.out_ids)))
        if action is Action.EXIT:
            self._begin_exit(r)
        elif action is Action.STEER_SUPPRESS:
            self._set_steer(r, "suppress", -abs(self.policy_cfg.alpha))
        elif action is Action.STEER_BREAK:
            self._set_steer(r, "break", abs(self.policy_cfg.alpha))
        elif action is Action.STOP_STEER:
            self._set_steer(r, None, 0.0)

    def _set_steer(self, r: _Row, kind: str | None, alpha: float) -> None:
        if r.steer_kind == kind and r.alpha == alpha:
            return
        # close open span
        for sp in r.steer_spans:
            if sp[1] is None:
                sp[1] = r.real_len
        r.steer_kind, r.alpha = kind, alpha
        if kind is not None:
            r.steer_spans.append([r.real_len, None, alpha, kind])

    def _begin_exit(self, r: _Row) -> None:
        self._set_steer(r, None, 0.0)
        r.force_queue = list(self.exit_suffix_ids)
        r.mode = Mode.ANSWER_GREEDY

    def _next_token(self, r: _Row, logits: torch.Tensor) -> int:
        # hard budget (s1-style budget forcing) applies in THINK mode
        hb = self.policy.hard_budget
        if (r.mode is Mode.THINK and not r.force_queue
                and (r.n_think >= self.gen_cfg.max_think_tokens
                     or (hb is not None and r.n_think >= hb))):
            self._begin_exit(r)
        if r.force_queue:
            r._just_forced = True
            return r.force_queue.pop(0)
        r._just_forced = False
        greedy = r.mode is Mode.ANSWER_GREEDY
        return sample_token(logits.cpu(), self.gen_cfg.temperature,
                            self.gen_cfg.top_p, r.gen, greedy=greedy)

    def _track_token(self, r: _Row, t: int) -> None:
        if r.mode is Mode.THINK:
            r.n_think += 1
            if r.end_detector is not None and r.end_detector.push(t):
                r.mode = Mode.ANSWER
                return
            if r.detector.push(t):
                r.pending_boundary = True
            if t in self.eos_ids:
                r.finished = True   # degenerate: EOS mid-think (smoke models)
            return
        # ANSWER / ANSWER_GREEDY; forced suffix tokens count as neither
        if getattr(r, "_just_forced", False):
            return
        r.n_answer += 1
        if t in self.eos_ids or r.n_answer >= self.gen_cfg.max_answer_tokens:
            r.finished = True

    def _result(self, r: _Row) -> ControllerResult:
        return ControllerResult(
            problem_id=r.job.problem_id, rollout_id=r.job.rollout_id,
            output_token_ids=list(r.out_ids),
            text=self.tok.decode(r.out_ids, skip_special_tokens=False),
            n_think_tokens=r.n_think,
            exited_early=r.state.exited,
            actions_log=r.actions_log,
            wall_s=time.time() - r.t_start,
            n_forwards=r.n_forwards)
