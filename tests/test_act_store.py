import pandas as pd
import torch

from reasoncontrol.activations.store import ActStore


def _rows(pid, n):
    return [{"problem_id": pid, "rollout_id": 0, "chunk_idx": i} for i in range(n)]


def test_write_read_gather(tmp_path):
    store = ActStore(tmp_path, layers=(3, 7))
    h0 = torch.randn(5, 2, 16)
    h1 = torch.randn(4, 2, 16)
    store.write_shard(0, h0, _rows("a", 5))
    store.write_shard(1, h1, _rows("b", 4))

    keys = pd.DataFrame([
        {"problem_id": "b", "rollout_id": 0, "chunk_idx": 2},
        {"problem_id": "a", "rollout_id": 0, "chunk_idx": 4},
    ])
    out = store.gather(keys, layer=7)
    assert out.shape == (2, 16)
    assert torch.allclose(out[0], h1[2, 1], atol=1e-2)   # fp16 round-trip
    assert torch.allclose(out[1], h0[4, 1], atol=1e-2)


def test_missing_key_raises(tmp_path):
    store = ActStore(tmp_path, layers=(0,))
    store.write_shard(0, torch.randn(2, 1, 8), _rows("a", 2))
    keys = pd.DataFrame([{"problem_id": "zz", "rollout_id": 0, "chunk_idx": 0}])
    try:
        store.gather(keys, layer=0)
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_int8_mode(tmp_path):
    store = ActStore(tmp_path, layers=(0,), quantize_int8=True)
    h = torch.randn(6, 1, 32)
    store.write_shard(0, h, _rows("a", 6))
    back = store.load_shard(0)
    rel = (back - h).abs().max() / h.abs().max()
    assert rel < 0.02
