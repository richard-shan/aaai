"""Test utilities: a deterministic toy tokenizer with realistic newline merges
(".\n\n", "\n\n", "\n\n\n" tokens) and a tiny-random Qwen2 model — no downloads.
"""
from __future__ import annotations

import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

# id 0 = pad, id 1 = eos. Multi-char merges FIRST so greedy longest-match
# reproduces the adversarial merge cases the chunker must handle.
_BASE_VOCAB = [
    "<pad>", "<eos>",
    ".\n\n", "\n\n\n", "\n\n", "\n</think>", "</think>", "<think>",
    "\n", ".", ",", " ", "=", "{", "}", "(", ")",
    "Wait", "Let", " me", " check", " verify", "So", " the", " answer",
    " is", "The", " final", "Therefore", "Alternatively", "Suppose",
    " x", " y", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", " 0",
    "boxed", "\\", "frac", "sqrt", " a", " b", " c", " step", " solve",
]


class ToyTokenizer:
    def __init__(self, vocab_size: int = 512):
        fillers = [f"<w{i}>" for i in range(vocab_size - len(_BASE_VOCAB))]
        self.vocab = _BASE_VOCAB + fillers
        assert len(self.vocab) == vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        # longest-match-first order
        self._by_len = sorted(range(vocab_size), key=lambda i: -len(self.vocab[i]))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out = []
        i = 0
        while i < len(text):
            for tid in self._by_len:
                tok = self.vocab[tid]
                if tok and not tok.startswith("<w") and not tok.startswith("<pad") \
                        and not tok.startswith("<eos") and text.startswith(tok, i):
                    out.append(tid)
                    i += len(tok)
                    break
            else:
                i += 1        # unknown char: skip
        return out

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        toks = []
        for i in ids:
            t = self.vocab[int(i)]
            if skip_special_tokens and t in ("<pad>", "<eos>"):
                continue
            if t.startswith("<w"):
                t = " w"      # filler tokens decode to a plain word
            toks.append(t)
        return "".join(toks)


def tiny_qwen2(vocab_size: int = 512, seed: int = 0) -> Qwen2ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2,
                      vocab_size=vocab_size, max_position_embeddings=4096,
                      pad_token_id=0, eos_token_id=1)
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model


def bias_model_toward(model: Qwen2ForCausalLM, token_id: int, strength: float = 10.0):
    """Make the model overwhelmingly prefer one token (deterministic-ish traces)
    by swapping in a biased lm_head."""
    import torch.nn as nn
    old = model.lm_head
    new = nn.Linear(old.in_features, old.out_features, bias=True)
    with torch.no_grad():
        new.weight.copy_(old.weight * 0.01)
        new.bias.zero_()
        new.bias[token_id] = strength
    model.lm_head = new
    return model
