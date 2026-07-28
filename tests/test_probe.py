import numpy as np
import torch

from reasoncontrol.probes.probe import (LinearProbe, expected_calibration_error,
                                        group_kfold)


def _planted(n=600, d=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=d)
    w /= np.linalg.norm(w)
    logits = X @ w * 3.0
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(np.int64)
    groups = np.repeat(np.arange(n // 4), 4)[:n]
    return torch.tensor(X), torch.tensor(y), groups, w


def test_recovers_planted_direction():
    X, y, groups, w = _planted()
    probe = LinearProbe(d_model=X.shape[1])
    m = probe.fit(X, y, groups, epochs=150)
    assert m.auc > 0.9
    cos = float(np.dot(probe.direction.numpy(), w))
    assert abs(cos) > 0.8


def test_group_kfold_no_leak():
    groups = np.repeat(np.arange(10), 5)
    for tr, te in group_kfold(groups, 5):
        assert set(groups[tr]).isdisjoint(set(groups[te]))


def test_save_load_roundtrip(tmp_path):
    X, y, groups, _ = _planted(n=200)
    probe = LinearProbe(d_model=X.shape[1])
    probe.fit(X, y, groups, epochs=50)
    p1 = probe.predict_proba(X[:10])
    probe.save(tmp_path / "p.pt")
    loaded = LinearProbe.load(tmp_path / "p.pt")
    assert torch.allclose(loaded.predict_proba(X[:10]), p1)


def test_calibration_reduces_ece():
    X, y, groups, _ = _planted(n=1000, seed=1)
    probe = LinearProbe(d_model=X.shape[1])
    probe.fit(X[:600], y[:600], groups[:600], epochs=300, l2=0.0)
    # overconfident probe: sharpen logits artificially
    with torch.no_grad():
        probe.net.weight.mul_(4.0)
        probe.net.bias.mul_(4.0)
    before = expected_calibration_error(
        probe.predict_proba(X[600:])[:, 1].numpy(), y[600:].numpy())
    probe.calibrate(X[600:800], y[600:800])
    after = expected_calibration_error(
        probe.predict_proba(X[800:])[:, 1].numpy(), y[800:].numpy())
    assert after < before


def test_probe_matvec_is_fast():
    import time
    probe = LinearProbe(d_model=1536)
    h = torch.randn(1, 1536)
    probe.predict_proba(h)              # warmup
    t0 = time.perf_counter()
    for _ in range(100):
        probe.predict_proba(h)
    per_call = (time.perf_counter() - t0) / 100
    assert per_call < 1e-3              # < 1 ms per boundary read
