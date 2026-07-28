"""Generation backend protocol. Token-ids-only interface: ALL tokenization goes
through one HF AutoTokenizer (loaded in hf_backend.load_model_and_tokenizer);
vLLM only ever receives prompt_token_ids, never raw strings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class GenRequest:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    temperature: float = 0.6
    top_p: float = 0.95
    seed: int = 0
    stop: tuple[str, ...] = ()
    greedy: bool = False


@dataclass
class GenResult:
    request_id: str
    output_token_ids: list[int]
    text: str
    finish_reason: str = ""


class GenBackend(Protocol):
    def generate(self, requests: list[GenRequest]) -> list[GenResult]: ...


def make_backend(name: str, model_id: str, **kw) -> "GenBackend":
    if name == "vllm":
        from .vllm_backend import VLLMBackend   # lazy: vllm absent on CPU boxes
        return VLLMBackend(model_id, **kw)
    if name == "hf":
        from .hf_backend import HFBackend
        return HFBackend(model_id, **kw)
    raise ValueError(f"unknown backend {name!r}")
