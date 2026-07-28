"""Per-batch-row masked steering hook, transformers 5.x-safe.

5.x decoder layers may return a bare Tensor (verified for Qwen2 in 5.14) or a
tuple in older/other archs — handle both, mutate in place.

Two modes:
- decode: per-row scalar alphas [B], applied to the (single) current position;
- prefill: per-row-per-position alphas [B, S] so compaction re-prefills
  reproduce the KV of previously-steered spans exactly.

alphas are in units of the layer's mean residual norm (r_bar), so alpha=6.0
means 6.0 * r_bar * v_hat added to the residual stream.
"""
from __future__ import annotations

import torch


class SteeringHook:
    def __init__(self, vector: torch.Tensor, r_bar: float = 1.0):
        v = vector.detach().float()
        self.v_hat = (v / v.norm())
        self.r_bar = float(r_bar)
        self._decode_alphas: torch.Tensor | None = None   # [B]
        self._prefill_alphas: torch.Tensor | None = None  # [B, S]
        self._handle = None

    # ---- control -------------------------------------------------------
    def set_rows(self, alphas: torch.Tensor | None) -> None:
        """Per-row alpha for decode steps; None or all-zeros = bit-exact no-op."""
        if alphas is not None and torch.all(alphas == 0):
            alphas = None
        self._decode_alphas = alphas

    def set_prefill(self, alphas: torch.Tensor | None) -> None:
        """[B, S] alpha map for the next prefill forward; cleared after use."""
        self._prefill_alphas = alphas

    # ---- hook ----------------------------------------------------------
    def __call__(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if self._prefill_alphas is not None:
            a = self._prefill_alphas.to(h.device, h.dtype)          # [B, S]
            if a.shape[1] == h.shape[1]:
                v = self.v_hat.to(h.device, h.dtype)
                h += (a * self.r_bar).unsqueeze(-1) * v
                self._prefill_alphas = None
                return output
        if self._decode_alphas is not None and h.shape[1] == 1:
            a = self._decode_alphas.to(h.device, h.dtype)           # [B]
            v = self.v_hat.to(h.device, h.dtype)
            h += (a * self.r_bar).view(-1, 1, 1) * v
        return output

    # ---- lifecycle -----------------------------------------------------
    def attach(self, model, layer: int):
        self._handle = model.model.layers[layer].register_forward_hook(self)
        return self._handle

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class ProbeTap:
    """Forward hook on the probe layer that stashes the last-position hidden
    state each forward (no output_hidden_states; zero extra compute)."""

    def __init__(self):
        self.last: torch.Tensor | None = None   # [B, d]
        self._handle = None

    def __call__(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        self.last = h[:, -1].detach()
        return output

    def attach(self, model, layer: int):
        self._handle = model.model.layers[layer].register_forward_hook(self)
        return self._handle

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
