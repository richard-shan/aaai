"""vLLM backend (GPU box only; import is lazy via make_backend).

enable_prefix_caching is on by default: stage forced_answer relies on it with
two-phase submission (each rollout's longest prefix first, then the rest) —
see labeling/convergence.py. The pilot gate requires >=90% prefix-cache hits.
"""
from __future__ import annotations

from .backend import GenRequest, GenResult


class VLLMBackend:
    def __init__(self, model_id: str, dtype: str = "float16",
                 max_model_len: int = 20480, gpu_memory_utilization: float = 0.92,
                 enable_prefix_caching: bool = True, seed: int = 0,
                 max_num_seqs: int = 0, **_):
        from vllm import LLM
        kw = {"max_num_seqs": max_num_seqs} if max_num_seqs else {}
        self.llm = LLM(model=model_id, dtype=dtype, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization,
                       enable_prefix_caching=enable_prefix_caching, seed=seed,
                       disable_log_stats=False,   # cache_hit_rate needs stats
                       **kw)

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        from vllm import SamplingParams, TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=list(r.prompt_token_ids))
                   for r in requests]
        params = [SamplingParams(
            temperature=0.0 if r.greedy else r.temperature,
            top_p=1.0 if r.greedy else r.top_p,
            max_tokens=r.max_tokens, seed=r.seed,
            stop=list(r.stop) or None) for r in requests]
        outs = self.llm.generate(prompts, params)
        results = []
        for r, o in zip(requests, outs):
            comp = o.outputs[0]
            results.append(GenResult(request_id=r.request_id,
                                     output_token_ids=list(comp.token_ids),
                                     text=comp.text,
                                     finish_reason=str(comp.finish_reason)))
        return results

    def shutdown(self) -> None:
        """Terminate EngineCore procs; atexit joins on them can hang forever."""
        try:
            self.llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass

    def cache_hit_rate(self) -> float | None:
        try:
            metrics = self.llm.llm_engine.get_metrics()   # vLLM v1 API
            hits = queries = None
            for m in metrics:
                # exact names: vllm:external_prefix_cache_* would shadow these
                # under a suffix match
                if m.name == "vllm:prefix_cache_hits":
                    hits = m.value
                elif m.name == "vllm:prefix_cache_queries":
                    queries = m.value
            if hits is not None and queries:
                return hits / queries
        except Exception:
            pass
        return None
