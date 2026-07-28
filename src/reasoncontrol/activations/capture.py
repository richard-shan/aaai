"""Teacher-forced boundary-state capture.

One forward per (length-bucketed) batch of rollouts with hooks on the cached
layer set; gather the residual-stream state at tok_end - 1 for every chunk
boundary. Also computes per-layer mean residual norms (steering alpha units)
and the vLLM/HF consistency check (teacher-forced argmax vs sampled tokens).
"""
from __future__ import annotations

import numpy as np
import torch


class MultiTap:
    def __init__(self, model, layers: tuple[int, ...]):
        self.layers = layers
        self.states: dict[int, torch.Tensor] = {}
        self._handles = []
        for li in layers:
            self._handles.append(model.model.layers[li].register_forward_hook(
                self._make_hook(li)))

    def _make_hook(self, li: int):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            self.states[li] = h.detach()
            return output
        return hook

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []


@torch.no_grad()
def capture_boundaries(model, sequences: list[list[int]],
                       boundary_positions: list[list[int]],
                       layers: tuple[int, ...], pad_id: int,
                       batch_size: int = 4, device: str | None = None):
    """sequences: full token id lists (prompt+output). boundary_positions[i]:
    full-sequence indices (tok_end - 1) to capture for sequence i.

    Returns (h, norms, argmax_match):
      h: [total_boundaries, n_layers, d] fp16, ordered by (sequence, position)
      norms: dict layer -> mean residual norm over real tokens
      argmax_match: fraction of next-token argmax predictions that reproduce
                    the actual next token (consistency check)
    """
    device = device or next(model.parameters()).device
    tap = MultiTap(model, layers)
    outs, matches, totals = [], 0, 0
    norm_sums = {li: 0.0 for li in layers}
    norm_counts = {li: 0 for li in layers}
    try:
        for i in range(0, len(sequences), batch_size):
            seqs = sequences[i:i + batch_size]
            bpos = boundary_positions[i:i + batch_size]
            maxlen = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for b, s in enumerate(seqs):
                ids[b, :len(s)] = torch.tensor(s)      # RIGHT padding: positions align
                mask[b, :len(s)] = 1
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device))
            # consistency: argmax at position p predicts token p+1
            logits = out.logits
            for b, s in enumerate(seqs):
                pred = logits[b, :len(s) - 1].argmax(-1).cpu()
                tgt = torch.tensor(s[1:])
                matches += int((pred == tgt).sum())
                totals += len(s) - 1
            for li in layers:
                h = tap.states[li]                     # [B, L, d]
                for b, s in enumerate(seqs):
                    norm_sums[li] += float(h[b, :len(s)].float().norm(dim=-1).sum())
                    norm_counts[li] += len(s)
            stacked = torch.stack([tap.states[li] for li in layers], dim=2)  # [B, L, nl, d]
            for b, s in enumerate(seqs):
                pos = torch.tensor(bpos[b], dtype=torch.long)
                outs.append(stacked[b, pos].to(torch.float16).cpu())          # [nb, nl, d]
    finally:
        tap.detach()
    h = torch.cat(outs, dim=0) if outs else torch.zeros(0, len(layers), model.config.hidden_size)
    norms = {li: norm_sums[li] / max(norm_counts[li], 1) for li in layers}
    argmax_match = matches / max(totals, 1)
    return h, norms, argmax_match
