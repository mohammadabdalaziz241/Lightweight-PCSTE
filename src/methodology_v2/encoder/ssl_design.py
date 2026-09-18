"""Part 5C — SSL masking/decoder DESIGN utilities (no training).

Contains: the deterministic mask generator, redundancy/difficulty
diagnostics, non-learned reconstruction baselines (P0/P1/P2), and a
bounded prototype of the masked-reconstruction path used ONLY to prove
the gradient route through the Hz-gated mixer. There is no optimizer,
no epoch loop, no trainer, and no dataset-scale training anywhere here.

Masked-encoding design (candidate under study):
  patchify -> for masked VALID patches, the patch content NEVER reaches
  the stem: the token embedding is replaced by a learned mask token;
  coordinate embeddings are still added (the model may know a patch
  exists at 7.5 kHz / 0.4 s but not its values) -> shared temporal
  encoder -> Hz-gated mixer -> per-band post-mixer context h'_i is
  projected and added to each temporal token of band i (X1) -> small
  per-token MLP decoder predicts the 16x8 patch -> masked valid-cell
  MSE, averaged per window then over the batch.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn as nn

from .patchify import PATCH_F, PATCH_T, patch_centres, patchify

MASK_GEOMETRIES = {          # block shape (bands, time patches)
    "M1_random": (1, 1),
    "M2_block": (2, 3),      # ~1.5-1.56 kHz x ~123-128 ms
    "M3_time_span": (1, 4),
    "M4_band_span": (3, 1),
}


def window_rng(global_seed: int, epoch: int, window_id: str
               ) -> np.random.Generator:
    """Deterministic per-(seed, epoch, window) generator: same seed
    reproduces exact masks; different epochs expose different masks; no
    mask storage required."""
    digest = hashlib.sha256(
        f"{global_seed}|{epoch}|{window_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def generate_mask(token_valid: np.ndarray, ratio: float, geometry: str,
                  rng: np.random.Generator) -> np.ndarray:
    """Boolean (F, T) mask over VALID patches only, hitting exactly
    round(ratio * n_valid) masked patches. Padding/invalid patches are
    never masked and never count toward the ratio."""
    fb, tp = token_valid.shape
    n_valid = int(token_valid.sum())
    target = int(round(ratio * n_valid))
    mask = np.zeros_like(token_valid, dtype=bool)
    if geometry == "M5_mixed":
        shapes = [MASK_GEOMETRIES[k] for k in
                  ("M2_block", "M3_time_span", "M4_band_span")]
    else:
        shapes = [MASK_GEOMETRIES[geometry]]
    guard = 0
    while mask.sum() < target and guard < 10_000:
        bf, bt = shapes[rng.integers(len(shapes))]
        f0 = int(rng.integers(0, fb))
        t0 = int(rng.integers(0, tp))
        blk = np.zeros_like(mask)
        blk[f0:f0 + bf, t0:t0 + bt] = True
        mask |= blk & token_valid
        guard += 1
    # exact-count trim (deterministic via rng order)
    excess = int(mask.sum()) - target
    if excess > 0:
        idx = np.argwhere(mask)
        drop = idx[rng.permutation(len(idx))[:excess]]
        mask[drop[:, 0], drop[:, 1]] = False
    return mask


# ---------------------------------------------------------------------------
# redundancy / difficulty diagnostics (train-only inputs)
# ---------------------------------------------------------------------------

def _patch_grid(x: np.ndarray) -> np.ndarray:
    """(bins, frames) -> (F, T, 128) zero-completed patch grid + valid."""
    bins, frames = x.shape
    fb = -(-bins // PATCH_F)
    tp = -(-frames // PATCH_T)
    g = np.zeros((fb * PATCH_F, tp * PATCH_T), dtype=np.float64)
    g[:bins, :frames] = x
    return (g.reshape(fb, PATCH_F, tp, PATCH_T)
            .transpose(0, 2, 1, 3).reshape(fb, tp, -1))


def redundancy_metrics(x: np.ndarray) -> dict:
    """Descriptive patch-level redundancy for one window (no labels)."""
    g = _patch_grid(x)
    fb, tp, _ = g.shape

    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else 0.0

    out = {}
    for lag in (1, 2, 4):
        cs = [corr(g[f, t], g[f, t + lag])
              for f in range(fb) for t in range(tp - lag)]
        out[f"temporal_corr_lag{lag}"] = round(float(np.mean(cs)), 4)
    for dist in (1, 2, 4):
        cs = [corr(g[f, t], g[f + dist, t])
              for f in range(fb - dist) for t in range(tp)]
        out[f"freq_corr_dist{dist}"] = round(float(np.mean(cs)), 4)
    # predictability from immediate temporal neighbours
    num = den = 0.0
    for f in range(fb):
        for t in range(1, tp - 1):
            pred = 0.5 * (g[f, t - 1] + g[f, t + 1])
            num += ((g[f, t] - pred) ** 2).sum()
            den += ((g[f, t] - g[f, t].mean()) ** 2).sum()
    out["neighbour_interp_residual_ratio"] = round(num / max(den, 1e-12), 4)
    out["patch_energy_p95_over_median"] = round(float(
        np.percentile((g ** 2).mean(-1), 95)
        / max(np.median((g ** 2).mean(-1)), 1e-12)), 2)
    return out


# ---------------------------------------------------------------------------
# non-learned reconstruction baselines (P0/P1/P2)
# ---------------------------------------------------------------------------

def _cell_valid(bins: int, frames: int) -> np.ndarray:
    fb = -(-bins // PATCH_F)
    tp = -(-frames // PATCH_T)
    v = np.zeros((fb * PATCH_F, tp * PATCH_T), dtype=bool)
    v[:bins, :frames] = True
    return (v.reshape(fb, PATCH_F, tp, PATCH_T)
            .transpose(0, 2, 1, 3).reshape(fb, tp, -1))


def baseline_mses(x: np.ndarray, mask: np.ndarray) -> dict:
    """Masked valid-cell MSE of P0 (zero), P1 (temporal-neighbour mean),
    P2 (nearest visible frequency neighbour) for one window."""
    g = _patch_grid(x)
    cv = _cell_valid(*x.shape)
    fb, tp, _ = g.shape
    vis = ~mask

    def nearest_visible_t(f, t):
        for d in range(1, tp):
            cands = []
            if t - d >= 0 and vis[f, t - d]:
                cands.append(g[f, t - d])
            if t + d < tp and vis[f, t + d]:
                cands.append(g[f, t + d])
            if cands:
                return np.mean(cands, axis=0)
        return None

    def nearest_visible_f(f, t):
        for d in range(1, fb):
            cands = []
            if f - d >= 0 and vis[f - d, t]:
                cands.append(g[f - d, t])
            if f + d < fb and vis[f + d, t]:
                cands.append(g[f + d, t])
            if cands:
                return np.mean(cands, axis=0)
        return None

    errs = {"P0_zero": [], "P1_temporal_neighbour": [],
            "P2_frequency_neighbour": []}
    for f in range(fb):
        for t in range(tp):
            if not mask[f, t] or not cv[f, t].any():
                continue
            tgt, valid = g[f, t], cv[f, t]
            errs["P0_zero"].append(((tgt - 0.0) ** 2)[valid].mean())
            p1 = nearest_visible_t(f, t)
            if p1 is None:
                p1 = nearest_visible_f(f, t)
            errs["P1_temporal_neighbour"].append(
                ((tgt - (p1 if p1 is not None else 0)) ** 2)[valid].mean())
            p2 = nearest_visible_f(f, t)
            if p2 is None:
                p2 = nearest_visible_t(f, t)
            errs["P2_frequency_neighbour"].append(
                ((tgt - (p2 if p2 is not None else 0)) ** 2)[valid].mean())
    return {k: float(np.mean(v)) for k, v in errs.items()}


# ---------------------------------------------------------------------------
# masked-reconstruction PROTOTYPE (gradient-path proof only)
# ---------------------------------------------------------------------------

class ReconstructionProbe(nn.Module):
    """Design-candidate decoder D1 + X1 context injection, built ON a
    frozen-architecture PCSTE instance. Prototype for the gradient-path
    proof and parameter accounting — NOT a trainer.

    q_{i,t} = z_{i,t} + P(h'_i)        (X1 additive post-mixer context)
    pred    = MLP(q): d -> 256 -> 128  (one 16x8 patch per token)
    loss    = per-window mean over masked VALID cells of (pred-target)^2,
              then mean over windows.
    Gradient path: loss -> pred -> q -> [z (temporal encoder -> stem/
    coords)] AND [h' (mixer -> all bands -> temporal encoder ...)] —
    the mixer is on the path for every masked prediction.
    """

    def __init__(self, encoder, d_dec: int = 256):
        super().__init__()
        self.encoder = encoder
        d = encoder.cfg.d_model
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.ctx_proj = nn.Linear(d, d)
        self.decoder = nn.Sequential(
            nn.Linear(d, d_dec), nn.GELU(),
            nn.Linear(d_dec, PATCH_F * PATCH_T))

    def decoder_parameter_count(self) -> int:
        return (sum(p.numel() for p in self.ctx_proj.parameters())
                + sum(p.numel() for p in self.decoder.parameters())
                + self.mask_token.numel())

    def forward(self, spec, frequency_hz, time_seconds, cell_mask,
                patch_mask: torch.Tensor) -> dict:
        enc = self.encoder
        patches, pcell, token_mask, band_mask = patchify(spec, cell_mask)
        f_khz, t_s = patch_centres(frequency_hz, time_seconds, cell_mask)

        tok = enc.stem(patches)
        # LEAKAGE CONTROL: masked valid patches never contribute content —
        # their stem embedding is fully replaced by the learned mask token
        m = (patch_mask & token_mask).unsqueeze(-1)
        tok = torch.where(m, self.mask_token.expand_as(tok), tok)
        fb, tp = tok.shape[1], tok.shape[2]
        tok = tok + enc.coords(f_khz.unsqueeze(2).expand(-1, fb, tp),
                               t_s.unsqueeze(1).expand(-1, fb, tp))
        tok = tok * token_mask.unsqueeze(-1).to(tok.dtype)

        b, _, _, d = tok.shape
        z = enc.temporal(tok.reshape(b * fb, tp, d)).reshape(b, fb, tp, d)
        z = z * token_mask.unsqueeze(-1).to(z.dtype)
        tm = token_mask.to(z.dtype).unsqueeze(-1)
        band = z.sum(2) / tm.sum(2).clamp(min=1.0)
        band = band * band_mask.unsqueeze(-1).to(band.dtype)
        mixed = enc.mixer(band, enc.coords.freq_features(f_khz), band_mask)

        q = z + self.ctx_proj(mixed).unsqueeze(2)      # X1 injection
        pred = self.decoder(q)                          # (B,F,T,128)
        target = patches.squeeze(3).reshape(*pred.shape)
        cellv = pcell.squeeze(3).reshape(*pred.shape)

        loss_mask = (patch_mask & token_mask).unsqueeze(-1) & cellv
        se = (pred - target) ** 2
        per_win = ((se * loss_mask).sum(dim=(1, 2, 3))
                   / loss_mask.sum(dim=(1, 2, 3)).clamp(min=1))
        return {"loss": per_win.mean(), "per_window_mse": per_win,
                "pred": pred, "loss_mask": loss_mask}
