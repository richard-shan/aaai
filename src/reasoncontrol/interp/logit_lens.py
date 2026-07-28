"""Logit-lens on probe/steering directions: project a residual-stream
direction through the final norm + unembedding and report top-k promoted /
suppressed vocabulary items. Part of the star faithfulness section.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def direction_vocab(model, tokenizer, v: torch.Tensor, k: int = 30) -> dict:
    v = v.float().to(next(model.parameters()).device)
    norm = model.model.norm
    # RMSNorm is scale-sensitive: apply the norm's weight to the direction
    scaled = v * norm.weight.float()
    logits = model.lm_head.weight.float() @ scaled
    top = logits.topk(k)
    bot = (-logits).topk(k)
    return {
        "promoted": [(tokenizer.decode([i]), float(s))
                     for i, s in zip(top.indices.tolist(), top.values.tolist())],
        "suppressed": [(tokenizer.decode([i]), float(s))
                       for i, s in zip(bot.indices.tolist(), (-bot.values).tolist())],
    }


@torch.no_grad()
def cross_model_vocab_similarity(model_a, tok_a, v_a: torch.Tensor,
                                 model_b, tok_b, v_b: torch.Tensor,
                                 k: int = 200) -> float:
    """Jaccard overlap of top-k promoted token STRINGS (vocab-space transfer
    comparison across models with different tokenizers)."""
    ta = {t for t, _ in direction_vocab(model_a, tok_a, v_a, k)["promoted"]}
    tb = {t for t, _ in direction_vocab(model_b, tok_b, v_b, k)["promoted"]}
    return len(ta & tb) / max(len(ta | tb), 1)
