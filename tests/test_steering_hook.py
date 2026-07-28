import torch

from reasoncontrol.steering.hooks import ProbeTap, SteeringHook

from util import ToyTokenizer, tiny_qwen2


def _states(model, ids, layer):
    tap = ProbeTap()
    tap.attach(model, layer)
    with torch.no_grad():
        model(ids)
    tap.detach()
    return tap.last.clone()


def test_masked_rows_get_exact_addition():
    model = tiny_qwen2()
    ids = torch.randint(2, 500, (3, 1))            # decode-shaped input [B, 1]
    v = torch.randn(64)
    hook = SteeringHook(v, r_bar=2.0)
    hook.attach(model, 1)

    tap = ProbeTap()
    tap.attach(model, 1)                           # read the SAME layer's output
    hook.set_rows(torch.tensor([0.0, 0.0, 0.0]))
    with torch.no_grad():
        model(ids)
    base = tap.last.clone()

    hook.set_rows(torch.tensor([0.0, 3.0, 0.0]))
    with torch.no_grad():
        model(ids)
    steered = tap.last.clone()
    tap.detach()
    hook.detach()

    delta = steered - base
    expected = 3.0 * 2.0 * (v / v.norm())
    assert torch.allclose(delta[1], expected, atol=1e-4)
    assert torch.allclose(delta[0], torch.zeros(64), atol=1e-6)
    assert torch.allclose(delta[2], torch.zeros(64), atol=1e-6)


def test_zero_alpha_is_bit_exact_noop():
    model = tiny_qwen2()
    ids = torch.randint(2, 500, (2, 5))
    logits_plain = model(ids).logits
    hook = SteeringHook(torch.randn(64))
    hook.attach(model, 2)
    hook.set_rows(torch.zeros(2))
    logits_hooked = model(ids).logits
    hook.detach()
    assert torch.equal(logits_plain, logits_hooked)


def test_prefill_alpha_map():
    model = tiny_qwen2()
    ids = torch.randint(2, 500, (2, 6))
    v = torch.randn(64)
    hook = SteeringHook(v, r_bar=1.0)
    hook.attach(model, 1)
    tap = ProbeTap()
    # tap on a LATER layer so the addition propagates
    tap.attach(model, 3)
    with torch.no_grad():
        model(ids)
    base = tap.last.clone()
    amap = torch.zeros(2, 6)
    amap[0, 2:4] = 5.0
    hook.set_prefill(amap)
    with torch.no_grad():
        model(ids)
    steered = tap.last.clone()
    tap.detach()
    hook.detach()
    assert not torch.allclose(base[0], steered[0])   # row 0 perturbed upstream
    assert torch.allclose(base[1], steered[1])       # row 1 untouched
    assert hook._prefill_alphas is None              # cleared after one use
