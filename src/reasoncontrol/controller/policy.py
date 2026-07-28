"""Controller policy: a per-row state machine evaluated at chunk boundaries.

Actions:
- CONTINUE       no-op
- STEER_SUPPRESS anti-verification steering (converged-but-still-verifying)
- STEER_BREAK    break unproductive verify/backtrack loops when NOT converged
                 (the phase channel's accuracy-relevant action)
- STOP_STEER     lapse any active steering (hysteresis crossed)
- EXIT           splice </think> and answer greedily

Baselines share this interface so every condition runs the same loop code path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..config import PolicyCfg


class Action(Enum):
    CONTINUE = "continue"
    STEER_SUPPRESS = "steer_suppress"
    STEER_BREAK = "steer_break"
    STOP_STEER = "stop_steer"
    EXIT = "exit"


@dataclass
class RowState:
    chunk_count: int = 0
    consecutive_converged: int = 0
    consec_loop_phase: int = 0
    tokens_used: int = 0
    steering: str | None = None      # None | "suppress" | "break"
    exited: bool = False


class ControllerPolicy:
    """Full closed-loop policy. Subclasses toggle channels off."""

    use_exit = True
    use_steer = True

    def __init__(self, cfg: PolicyCfg):
        self.cfg = cfg

    # hard token budget enforced by the loop every step (None = no budget)
    @property
    def hard_budget(self) -> int | None:
        return None

    def decide(self, s: RowState, p_conv: float | None,
               phase: str | None) -> Action:
        c = self.cfg
        s.chunk_count += 1
        if p_conv is not None and p_conv >= c.tau_exit:
            s.consecutive_converged += 1
        else:
            s.consecutive_converged = 0
        in_loop_phase = phase in c.steer_phases
        converged_soft = p_conv is not None and p_conv >= c.tau_steer
        if in_loop_phase and not converged_soft:
            s.consec_loop_phase += 1
        else:
            s.consec_loop_phase = 0

        if s.chunk_count < c.min_chunks:
            return Action.CONTINUE
        if self.use_exit and s.consecutive_converged >= c.patience_k:
            s.exited = True
            return Action.EXIT
        if self.use_steer and p_conv is not None:
            if s.steering == "suppress":
                # hysteresis: keep steering until p_conv drops below tau - h
                if p_conv < c.tau_steer - c.hysteresis or not in_loop_phase:
                    s.steering = None
                    return Action.STOP_STEER
                return Action.STEER_SUPPRESS
            if s.steering == "break":
                if not in_loop_phase or converged_soft:
                    s.steering = None
                    return Action.STOP_STEER
                return Action.STEER_BREAK
            if converged_soft and in_loop_phase:
                s.steering = "suppress"
                return Action.STEER_SUPPRESS
            if c.break_loops and not converged_soft and s.consec_loop_phase >= c.loop_patience:
                s.steering = "break"
                return Action.STEER_BREAK
        return Action.CONTINUE


class NoOpPolicy(ControllerPolicy):
    use_exit = False
    use_steer = False

    def decide(self, s: RowState, p_conv, phase) -> Action:
        s.chunk_count += 1
        return Action.CONTINUE


class ExitOnlyPolicy(ControllerPolicy):
    use_steer = False


class SteerOnlyPolicy(ControllerPolicy):
    use_exit = False


class StaticBudgetPolicy(NoOpPolicy):
    """s1-style budget forcing: the loop truncates at exactly `budget` think
    tokens, splices the exit suffix, and decodes the answer greedily."""

    @property
    def hard_budget(self) -> int | None:
        return self.cfg.budget


def make_policy(cfg: PolicyCfg) -> ControllerPolicy:
    kinds = {
        "full": ControllerPolicy,
        "noop": NoOpPolicy,
        "exit_only": ExitOnlyPolicy,
        "steer_only": SteerOnlyPolicy,
        "static_budget": StaticBudgetPolicy,
    }
    if cfg.kind not in kinds:
        raise ValueError(f"policy kind {cfg.kind!r} not runnable in the HF loop "
                         f"(vLLM-side baselines live in controller/baselines.py)")
    return kinds[cfg.kind](cfg)


def policy_hash(cfg: PolicyCfg) -> str:
    import hashlib
    import json
    from dataclasses import asdict
    blob = json.dumps(asdict(cfg), sort_keys=True, default=list)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]
