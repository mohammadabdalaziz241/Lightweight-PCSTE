"""PC-STE v2 envelope / order-domain branch (spec: analysis/pcste_v2_diagnostics/PROPOSED_PCSTE_V2_ARCHITECTURE.md §2).

Deterministic numpy preprocessing (no learnable parameters):
  raw 1 s window -> mean removal -> 4th-order zero-phase Butterworth band-pass (fixed per dataset) -> Hilbert analytic
  signal -> squared envelope -> mean removal -> Hann -> rfft zero-padded to 4 s -> magnitude -> resample on a shaft-order
  grid (0.05 ... 12.0 orders, step 0.05, 240 cells) using the manifest rpm -> divide by the window's own median -> log1p
  (level removal: exactly gain-invariant; no cross-window level information survives in this branch).

Torch branch: 240 cells -> 15 tokens of 16 cells -> linear stem + Fourier order coordinates (plain order and
BPFO-normalised order) + branch-type / factor-known embeddings. Tokens are appended to the STFT band summaries and take
part in the Hz-gated mixer and the validity-masked global mean (fusion at the mixer, spec §2 "Fusion").
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from src.methodology_v2.encoder.coords import fourier_1d

ORDER_MIN, ORDER_MAX, ORDER_STEP = 0.05, 12.0, 0.05
N_ORDER_CELLS = int(round((ORDER_MAX - ORDER_MIN) / ORDER_STEP)) + 1     # 240
ORDER_PATCH = 16                                                        # -> 15 tokens
N_ORDER_TOKENS = N_ORDER_CELLS // ORDER_PATCH
ENVELOPE_VERSION = "pcste_v2.envelope.v2"
NFFT_SECONDS = 4.0

# fixed resonance bands (Hz) per dataset — spec §2 table
ENVELOPE_BANDS_HZ = {"CWRU": (2000.0, 8000.0), "JNU": (2000.0, 12000.0),
                     "HIT": (2000.0, 8000.0), "MAFAULDA": (2000.0, 10000.0)}
# pre-registered 3-band fallback (spec §2 "Alternative for the ablation": 0.5-2 kHz, 2-8 kHz, 8 kHz-0.8*Nyquist, stacked as
# three envelope channels; spec §3 names it the fallback when the fixed band misses the resonance). A band whose lower edge is not
# below 0.8*Nyquist (e.g. the 8 kHz band at 12 kHz sampling) is physically absent: its channel is zero and flagged invalid.
ENVELOPE3_VERSION = "pcste_v2.envelope3.v1"
N_ENVELOPE3_BANDS = 3
ENVELOPE3_NYQUIST_FRACTION = 0.8


def envelope3_bands_hz(fs: float) -> list[tuple[float, float] | None]:
    """The three fixed bands for a sampling rate; None for a band that does not fit below 0.8*Nyquist."""
    top = ENVELOPE3_NYQUIST_FRACTION * fs / 2.0
    bands = [(500.0, 2000.0), (2000.0, 8000.0), (8000.0, top)]
    return [(lo, min(hi, top)) if lo < min(hi, top) else None for lo, hi in bands]


def envelope3_order_matrix(x: np.ndarray, fs: float, rpm: float) -> tuple[np.ndarray, np.ndarray]:
    """(3, N_ORDER_CELLS) float32 level-removed log envelope order spectra (one row per band; zeros for an absent band) and the
    (3,) band-valid flags. Each row is computed by envelope_order_vector, so it is exactly gain-invariant like the single band."""
    out = np.zeros((N_ENVELOPE3_BANDS, N_ORDER_CELLS), dtype=np.float32); valid = np.zeros(N_ENVELOPE3_BANDS, dtype=bool)
    for i, band in enumerate(envelope3_bands_hz(fs)):
        if band is None:
            continue
        out[i] = envelope_order_vector(x, fs, rpm, band); valid[i] = True
    return out, valid


# bearing characteristic multipliers (x shaft frequency); 'BSF' is the ball-spin frequency (site value / 2 for CWRU)
BEARING_FACTORS = {
    "CWRU_DE_6205": {"BPFO": 3.5848, "BPFI": 5.4152, "BSF": 2.35675, "FTF": 0.39828},
    "CWRU_FE_6203": {"BPFO": 3.0530, "BPFI": 4.9469, "BSF": 1.9937, "FTF": 0.3817},
    "MAFAULDA": {"BPFO": 2.9980, "BPFI": 5.0020, "BSF": 1.8710, "FTF": 0.3750},
}
UNKNOWN_FACTORS = "UNKNOWN"


def order_grid() -> np.ndarray:
    return ORDER_MIN + ORDER_STEP * np.arange(N_ORDER_CELLS)


def order_token_centres() -> np.ndarray:
    return order_grid().reshape(N_ORDER_TOKENS, ORDER_PATCH).mean(axis=1)


def envelope_order_vector(x: np.ndarray, fs: float, rpm: float, band_hz: tuple[float, float]) -> np.ndarray:
    """Level-removed log envelope order spectrum, float32 (N_ORDER_CELLS,). Deterministic."""
    import scipy.signal as sg
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x - x.mean()
    lo, hi = band_hz
    hi = min(hi, 0.95 * fs / 2)
    sos = sg.butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    xb = sg.sosfiltfilt(sos, x)
    env = np.abs(sg.hilbert(xb)) ** 2
    env = env - env.mean()
    n = int(round(NFFT_SECONDS * fs))
    A = np.abs(np.fft.rfft(env * np.hanning(env.size), n=n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    fr = float(rpm) / 60.0
    Ai = np.interp(order_grid() * fr, f, A)
    med = float(np.median(Ai))
    v = np.log1p(Ai / med) if med > 0 else np.zeros_like(Ai)        # ratio to the window's own median: exactly gain-invariant
    return v.astype(np.float32)


class OrderCoordEncoder(nn.Module):
    """Fourier features of (order centre, BPFO-normalised order centre) -> d_model; 8 log-spaced wavelengths in orders."""

    N_SCALES = 8

    def __init__(self, d_model: int):
        super().__init__()
        self.register_buffer("lam_o", torch.logspace(math.log10(0.1), math.log10(12.8), self.N_SCALES))
        self.proj = nn.Linear(4 * self.N_SCALES, d_model)

    def forward(self, order_c: torch.Tensor, order_bpfo_norm: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([fourier_1d(order_c, self.lam_o), fourier_1d(order_bpfo_norm, self.lam_o)], dim=-1))

    def phi(self, order_c: torch.Tensor) -> torch.Tensor:
        """16-dim deterministic features used in the mixer score/gate (same layout as the Hz features)."""
        return fourier_1d(order_c, self.lam_o)


class EnvelopeBranch(nn.Module):
    """240-cell level-removed order spectrum -> 15 tokens (B, 15, d) with mixer features (B, 15, 16) and mask (B, 15).
    n_bands=1: the Stage B branch (unchanged). n_bands=3: the pre-registered 3-band fallback; the three band channels of each
    16-cell patch are stacked into the stem input (48 -> d) and a band-validity embedding marks physically absent bands."""

    def __init__(self, d_model: int, n_bands: int = 1):
        super().__init__()
        self.n_bands = int(n_bands)
        self.stem = nn.Linear(self.n_bands * ORDER_PATCH, d_model)
        self.coords = OrderCoordEncoder(d_model)
        self.branch_type = nn.Parameter(torch.zeros(d_model))
        self.factor_known = nn.Embedding(2, d_model)
        if self.n_bands > 1:
            self.band_valid_embed = nn.Linear(self.n_bands, d_model, bias=False)
        self.register_buffer("order_c", torch.tensor(order_token_centres(), dtype=torch.float32))

    def forward(self, env: torch.Tensor, env_valid: torch.Tensor, bpfo_factor: torch.Tensor,
                factor_known: torch.Tensor, env_band_valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = env.shape[0]
        if self.n_bands == 1:
            tok = self.stem(env.reshape(b, N_ORDER_TOKENS, ORDER_PATCH))
        else:                                                    # (B, n_bands, 240) -> (B, 15, n_bands*16), band-major within a patch
            tok = self.stem(env.reshape(b, self.n_bands, N_ORDER_TOKENS, ORDER_PATCH).permute(0, 2, 1, 3).reshape(b, N_ORDER_TOKENS, self.n_bands * ORDER_PATCH))
            tok = tok + self.band_valid_embed(env_band_valid.to(tok.dtype)).unsqueeze(1)
        oc = self.order_c.unsqueeze(0).expand(b, -1)
        norm = torch.where(factor_known.bool().unsqueeze(1), oc / bpfo_factor.clamp(min=1e-3).unsqueeze(1), oc)
        tok = tok + self.coords(oc, norm) + self.branch_type + self.factor_known(factor_known.long()).unsqueeze(1)
        mask = env_valid.bool().unsqueeze(1).expand(-1, N_ORDER_TOKENS)
        tok = tok * mask.unsqueeze(-1).to(tok.dtype)
        return tok, self.coords.phi(oc), mask
