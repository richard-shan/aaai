"""SAE decomposition of probe/steering directions (Llama-8B fork).

Uses the open layer-19 SAE for DeepSeek-R1-Distill-Llama-8B
(`qresearch/DeepSeek-R1-Distill-Llama-8B-SAE-l19`, Apache). Loading is lazy
and GPU-box-only; this module only needs the decoder matrix.
"""
from __future__ import annotations

import torch


def load_sae_decoder(repo_id: str = "qresearch/DeepSeek-R1-Distill-Llama-8B-SAE-l19"):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import json, os
    # repo layout: try common filenames
    for fname in ("sae.safetensors", "model.safetensors", "sae_weights.safetensors"):
        try:
            path = hf_hub_download(repo_id, fname)
            weights = load_file(path)
            break
        except Exception:
            continue
    else:
        raise FileNotFoundError(f"no known SAE weight file in {repo_id}; inspect the repo")
    for key in ("W_dec", "decoder.weight", "w_dec"):
        if key in weights:
            W = weights[key]
            return W if W.shape[0] > W.shape[1] else W.T   # [n_features, d]
    raise KeyError(f"no decoder matrix in {list(weights)[:10]}...")


@torch.no_grad()
def decompose_direction(v: torch.Tensor, W_dec: torch.Tensor, k: int = 30) -> list[tuple[int, float]]:
    """Top-k SAE features by cosine similarity between the direction and each
    feature's decoder vector. Feature interpretation happens via the SAE's
    companion tooling / manual inspection of max-activating examples."""
    v = v.float() / v.float().norm()
    W = W_dec.float()
    W = W / W.norm(dim=-1, keepdim=True)
    sims = W @ v
    top = sims.topk(k)
    return list(zip(top.indices.tolist(), top.values.tolist()))
