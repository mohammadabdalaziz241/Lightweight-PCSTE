"""Hz-gated cross-band exchange — the central PC-STE mechanism.

Explicit information exchange across physical frequency bands,
conditioned on ABSOLUTE frequency:

  Step A  a_j    = w2^T tanh(W1 [h_j ; phi(f_j)])       (score per band)
          alpha  = softmax over VALID bands only
  Step B  c      = sum_j alpha_j V h_j                  (shared context)
  Step C  g_i    = sigmoid(G [h_i ; phi(f_i) ; c])
          h'_i   = h_i + g_i * W_c c                    (gated residual)

Properties (unit-tested): h'_i depends on every other VALID band through
c; phi(f) = deterministic Fourier features of absolute f (kHz)
participates in both the attention scores and the gates; masked/padded
bands receive zero attention weight and cannot contribute; datasets with
fewer bands (HIT: 17) simply present a shorter valid set — no fake
high-frequency tokens exist anywhere.
"""
from __future__ import annotations

import torch
import torch.nn as nn

PHI_DIM = 16   # freq-only Fourier features from FourierCoordEncoder


class HzGatedCrossBandMixer(nn.Module):
    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.score = nn.Sequential(
            nn.Linear(d_model + PHI_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, 1))
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(2 * d_model + PHI_DIM, d_model)
        self.context = nn.Linear(d_model, d_model)

    def forward(self, h: torch.Tensor, phi_f: torch.Tensor,
                band_mask: torch.Tensor) -> torch.Tensor:
        """h (B, F, d) band summaries; phi_f (B, F, PHI_DIM);
        band_mask (B, F) bool. Returns h' (B, F, d)."""
        hn = self.norm(h)
        a = self.score(torch.cat([hn, phi_f], dim=-1)).squeeze(-1)
        a = a.masked_fill(~band_mask, float("-inf"))
        alpha = torch.softmax(a, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)      # all-masked safety
        alpha = alpha * band_mask.to(alpha.dtype)
        c = torch.einsum("bf,bfd->bd", alpha, self.value(hn))  # (B, d)
        c_b = c.unsqueeze(1).expand_as(h)
        g = torch.sigmoid(self.gate(
            torch.cat([hn, phi_f, c_b], dim=-1)))
        out = h + g * self.context(c_b)
        return out * band_mask.unsqueeze(-1).to(out.dtype)
