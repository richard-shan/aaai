from reasoncontrol.config import PolicyCfg
from reasoncontrol.controller.policy import (Action, ControllerPolicy,
                                             ExitOnlyPolicy, NoOpPolicy,
                                             RowState, StaticBudgetPolicy,
                                             SteerOnlyPolicy, make_policy,
                                             policy_hash)

CFG = PolicyCfg(tau_exit=0.9, tau_steer=0.7, patience_k=2, min_chunks=2,
                loop_patience=3, hysteresis=0.1)


def test_min_chunks_guard():
    p = ControllerPolicy(CFG)
    s = RowState()
    assert p.decide(s, 0.99, "deduction") is Action.CONTINUE   # chunk 1 < min_chunks


def test_patience_exit():
    p = ControllerPolicy(CFG)
    s = RowState()
    p.decide(s, 0.95, "deduction")                 # chunk 1: guard
    assert p.decide(s, 0.95, "deduction") is Action.EXIT       # 2 consecutive >= tau
    assert s.exited


def test_patience_resets():
    p = ControllerPolicy(CFG)
    s = RowState()
    p.decide(s, 0.95, "deduction")
    p.decide(s, 0.5, "deduction")                  # reset
    assert s.consecutive_converged == 0
    a = p.decide(s, 0.95, "deduction")
    assert a is not Action.EXIT                    # only 1 consecutive again


def test_steer_suppress_and_hysteresis():
    p = ControllerPolicy(CFG)
    s = RowState()
    p.decide(s, 0.1, "deduction")                  # warmup chunk
    assert p.decide(s, 0.75, "verification") is Action.STEER_SUPPRESS
    # within hysteresis band: keep steering
    assert p.decide(s, 0.65, "verification") is Action.STEER_SUPPRESS
    # below tau - h: stop
    assert p.decide(s, 0.55, "verification") is Action.STOP_STEER


def test_break_loop_when_stuck():
    p = ControllerPolicy(CFG)
    s = RowState()
    p.decide(s, 0.1, "deduction")
    for _ in range(2):
        assert p.decide(s, 0.2, "verification") is Action.CONTINUE
    assert p.decide(s, 0.2, "verification") is Action.STEER_BREAK


def test_exit_only_never_steers():
    p = ExitOnlyPolicy(CFG)
    s = RowState()
    p.decide(s, 0.1, "verification")
    for _ in range(5):
        a = p.decide(s, 0.75, "verification")
        assert a in (Action.CONTINUE,)


def test_steer_only_never_exits():
    p = SteerOnlyPolicy(CFG)
    s = RowState()
    for _ in range(10):
        assert p.decide(s, 0.99, "deduction") is not Action.EXIT


def test_noop_and_budget():
    assert NoOpPolicy(CFG).decide(RowState(), 0.99, "verification") is Action.CONTINUE
    assert StaticBudgetPolicy(PolicyCfg(kind="static_budget", budget=123)).hard_budget == 123
    assert ControllerPolicy(CFG).hard_budget is None


def test_make_policy_and_hash():
    assert isinstance(make_policy(PolicyCfg(kind="full")), ControllerPolicy)
    h1 = policy_hash(PolicyCfg(kind="full", tau_exit=0.9))
    h2 = policy_hash(PolicyCfg(kind="full", tau_exit=0.8))
    assert h1 != h2 and len(h1) == 10
