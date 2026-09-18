"""Absolute physical-coordinate Fourier features (frozen C2 strategy).

Deterministic multi-scale sin/cos features of the ABSOLUTE patch centre
coordinates — frequency in kHz and time in seconds — followed by one
learned linear projection. No normalized f/Nyquist, no dataset-specific
index embeddings: the same mechanical frequency (e.g. 3.2 kHz) receives
the same deterministic features in every dataset.

Scales (fixed buffers, log-spaced):
  frequency wavelengths: 8 values in [0.1, 51.2] kHz
  time wavelengths:      8 values in [0.02, 2.56] s
features(x; lambdas) = [sin(2*pi*x/l), cos(2*pi*x/l)]  -> 16 per
coordinate, 32 total -> Linear(32, d).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

N_SCALES = 8
F_RANGE_KHZ = (0.1, 51.2)
T_RANGE_S = (0.02, 2.56)


def _wavelengths(lo: float, hi: float) -> torch.Tensor:
    return torch.logspace(math.log10(lo), math.log10(hi), N_SCALES)


def fourier_1d(x: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
    """x (...,) -> (..., 2*N_SCALES) deterministic features."""
    ang = 2.0 * math.pi * x.unsqueeze(-1) / lambdas.to(x.device)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class FourierCoordEncoder(nn.Module):
    """(f_centre_khz, t_centre_s) -> d-dim coordinate embedding."""

    COORD_DIM = 4 * N_SCALES  # 32

    def __init__(self, d_model: int):
        super().__init__()
        self.register_buffer("lam_f", _wavelengths(*F_RANGE_KHZ))
        self.register_buffer("lam_t", _wavelengths(*T_RANGE_S))
        self.proj = nn.Linear(self.COORD_DIM, d_model)

    def features(self, f_khz: torch.Tensor,
                 t_s: torch.Tensor) -> torch.Tensor:
        return torch.cat([fourier_1d(f_khz, self.lam_f),
                          fourier_1d(t_s, self.lam_t)], dim=-1)

    def forward(self, f_khz: torch.Tensor,
                t_s: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(f_khz, t_s))

    def freq_features(self, f_khz: torch.Tensor) -> torch.Tensor:
        """16-dim deterministic phi(f) used by the cross-band mixer."""
        return fourier_1d(f_khz, self.lam_f)
