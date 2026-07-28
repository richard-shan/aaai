"""HF transformers backend: shared model/tokenizer loading + a simple batched
sampler used for CPU smoke runs and the teacher-forced capture pass.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .backend import GenRequest, GenResult


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(model_id: str, dtype: str = "float16", device: str = "auto"):
    device = resolve_device(device)
    torch_dtype = getattr(torch, dtype) if device != "cpu" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch_dtype, attn_implementation="sdpa")
    model.to(device).eval()
    return model, tok


def apply_chat_template(tokenizer, user_prompt: str, think_tags: bool = True,
                        system: str | None = None) -> list[int]:
    """One place where prompts become token ids. R1-distill templates
    auto-open <think>; prepare_data asserts this for reasoning models."""
    messages = ([{"role": "system", "content": system}] if system else []) \
        + [{"role": "user", "content": user_prompt}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                        tokenize=True)
    # transformers 5.x returns a BatchEncoding (UserDict, NOT a dict subclass)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):   # batched form
        ids = ids[0]
    return [int(i) for i in ids]


class HFBackend:
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "auto",
                 batch_size: int = 4, **_):
        self.model, self.tokenizer = load_model_and_tokenizer(model_id, dtype, device)
        self.device = next(self.model.parameters()).device
        self.batch_size = batch_size

    @torch.no_grad()
    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        out: list[GenResult] = []
        for i in range(0, len(requests), self.batch_size):
            out.extend(self._batch(requests[i:i + self.batch_size]))
        return out

    def _batch(self, reqs: list[GenRequest]) -> list[GenResult]:
        tok = self.tokenizer
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        maxlen = max(len(r.prompt_token_ids) for r in reqs)
        ids = torch.full((len(reqs), maxlen), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, r in enumerate(reqs):
            ids[i, maxlen - len(r.prompt_token_ids):] = torch.tensor(r.prompt_token_ids)
            mask[i, maxlen - len(r.prompt_token_ids):] = 1
        r0 = reqs[0]
        do_sample = not r0.greedy and r0.temperature > 0
        # transformers sampling uses the global RNG; seed it per batch
        torch.manual_seed(r0.seed)
        res = self.model.generate(
            ids.to(self.device), attention_mask=mask.to(self.device),
            max_new_tokens=r0.max_tokens, do_sample=do_sample,
            temperature=r0.temperature if do_sample else None,
            top_p=r0.top_p if do_sample else None,
            pad_token_id=pad)
        outs = []
        for i, r in enumerate(reqs):
            new_ids = res[i, maxlen:].tolist()
            if tok.eos_token_id is not None and tok.eos_token_id in new_ids:
                new_ids = new_ids[:new_ids.index(tok.eos_token_id) + 1]
            text = tok.decode(new_ids, skip_special_tokens=False)
            for s in r.stop:
                if s in text:
                    text = text.split(s)[0]
                    new_ids = tok.encode(text, add_special_tokens=False)
                    break
            outs.append(GenResult(request_id=r.request_id, output_token_ids=new_ids,
                                  text=text))
        return outs
