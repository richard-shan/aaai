import torch

from reasoncontrol.config import GenCfg, PolicyCfg
from reasoncontrol.controller.loop import ControlledRunner, Job, Mode
from reasoncontrol.controller.policy import (Action, ControllerPolicy,
                                             NoOpPolicy, StaticBudgetPolicy)
from reasoncontrol.steering.hooks import SteeringHook

from util import ToyTokenizer, bias_model_toward, tiny_qwen2

GEN = GenCfg(temperature=1.0, top_p=0.95, max_think_tokens=40,
             max_answer_tokens=8, backend="hf")
PCFG = PolicyCfg(min_chunks=1, patience_k=1)


def _jobs(tok, n=4):
    return [Job(problem_id=f"p{i}", rollout_id=0,
                prompt_ids=tok.encode(f"Suppose x = {i}. So solve"))
            for i in range(n)]


def _runner(model, tok, policy, **kw):
    defaults = dict(gen_cfg=GEN, policy_cfg=PCFG, think_tags=False,
                    global_seed=7, cache_headroom=64)
    defaults.update(kw)
    return ControlledRunner(model, tok, policy, **defaults)


def test_noop_token_identical_to_hook_detached():
    tok = ToyTokenizer()
    m1, m2 = tiny_qwen2(seed=3), tiny_qwen2(seed=3)
    jobs = _jobs(tok)
    r_plain = _runner(m1, tok, NoOpPolicy(PCFG)).run(list(jobs), batch_size=4)
    hook = SteeringHook(torch.randn(64))
    runner = _runner(m2, tok, NoOpPolicy(PCFG), suppress_hook=hook, steer_layer=2)
    hook.attach(m2, 2)
    r_hooked = runner.run(list(jobs), batch_size=4)
    hook.detach()
    a = {r.problem_id: r.output_token_ids for r in r_plain}
    b = {r.problem_id: r.output_token_ids for r in r_hooked}
    assert a == b


def test_exit_splices_suffix_and_greedy_answer():
    tok = ToyTokenizer()
    nn = tok.vocab.index("\n\n")
    model = bias_model_toward(tiny_qwen2(seed=1), nn)   # emits "\n\n" every step

    class AlwaysExit(ControllerPolicy):
        def decide(self, s, p_conv, phase):
            s.chunk_count += 1
            if s.chunk_count >= 2:
                s.exited = True
                return Action.EXIT
            return Action.CONTINUE

    runner = _runner(model, tok, AlwaysExit(PCFG))
    res = runner.run(_jobs(tok, 2), batch_size=2)
    suffix = tok.encode("\n</think>\n\n")
    for r in res:
        assert r.exited_early
        # the forced suffix appears verbatim in the output ids
        ids = r.output_token_ids
        assert any(ids[i:i + len(suffix)] == suffix for i in range(len(ids)))
        assert r.n_think_tokens <= GEN.max_think_tokens
        assert any(a[1] == "exit" for a in r.actions_log)


def test_static_budget_forces_exit_at_budget():
    tok = ToyTokenizer()
    nn = tok.vocab.index("\n\n")
    model = bias_model_toward(tiny_qwen2(seed=2), nn)
    policy = StaticBudgetPolicy(PolicyCfg(kind="static_budget", budget=10))
    res = _runner(model, tok, policy).run(_jobs(tok, 2), batch_size=2)
    for r in res:
        assert r.n_think_tokens <= 11
        assert len(r.output_token_ids) > r.n_think_tokens   # answer follows


def test_compaction_returns_all_results():
    tok = ToyTokenizer()
    model = tiny_qwen2(seed=4)
    # tiny headroom forces repeated cache rebuilds; batch < jobs forces refills
    runner = _runner(model, tok, NoOpPolicy(PCFG), cache_headroom=16)
    res = runner.run(_jobs(tok, 5), batch_size=2)
    assert {r.problem_id for r in res} == {f"p{i}" for i in range(5)}
    assert all(r.n_forwards > 0 for r in res)


def test_steering_changes_trajectory_and_records_spans():
    tok = ToyTokenizer()
    nn = tok.vocab.index("\n\n")
    m1, m2 = (bias_model_toward(tiny_qwen2(seed=5), nn, strength=3.0) for _ in range(2))

    class AlwaysSteer(ControllerPolicy):
        def decide(self, s, p_conv, phase):
            s.chunk_count += 1
            return Action.STEER_SUPPRESS

    jobs = _jobs(tok, 2)
    base = _runner(m1, tok, NoOpPolicy(PCFG)).run(list(jobs), batch_size=2)
    hook = SteeringHook(torch.randn(64), r_bar=4.0)
    runner = _runner(m2, tok, AlwaysSteer(PolicyCfg(alpha=8.0, min_chunks=1)),
                     suppress_hook=hook, steer_layer=1)
    hook.attach(m2, 1)
    steered = runner.run(list(jobs), batch_size=2)
    hook.detach()
    assert any(a != b for a, b in zip(
        [r.output_token_ids for r in sorted(base, key=lambda r: r.problem_id)],
        [r.output_token_ids for r in sorted(steered, key=lambda r: r.problem_id)]))


def test_probe_reads_at_boundaries():
    tok = ToyTokenizer()
    nn = tok.vocab.index("\n\n")
    model = bias_model_toward(tiny_qwen2(seed=6), nn)

    class FakeProbe:
        def predict_proba(self, h):
            assert h.shape[-1] == 64          # real hidden state reached the probe
            return torch.tensor([[0.25, 0.75]])   # 0.75 is exact in float32

    policy = ControllerPolicy(PolicyCfg(min_chunks=1, patience_k=2, tau_exit=0.7))
    runner = _runner(model, tok, policy, conv_probe=FakeProbe(), probe_layer=2)
    res = runner.run(_jobs(tok, 1), batch_size=1)
    log = res[0].actions_log
    assert len(log) >= 2
    assert all(entry[2] == 0.75 for entry in log)     # p_conv recorded
    assert res[0].exited_early                        # patience 2 at p=0.75 >= 0.7
