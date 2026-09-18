"""16x8 time-frequency patchification with minimal completion padding.

Axis semantics are explicit throughout: dim -2 = frequency bins,
dim -1 = time frames. Padding is mechanical tensor completion only
(zeros, masked); no interpolation, no fabricated frequency content.
"""
from __future__ import annotations

import torch

PATCH_F = 16   # frequency bins per patch (frozen)
PATCH_T = 8    # time frames per patch (frozen)


def pad_to_multiple(x: torch.Tensor, cell_mask: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad (B, bins, frames) so both axes divide the patch size."""
    b, n_bins, n_frames = x.shape
    pf = (-n_bins) % PATCH_F
    pt = (-n_frames) % PATCH_T
    x = torch.nn.functional.pad(x, (0, pt, 0, pf))
    cell_mask = torch.nn.functional.pad(cell_mask, (0, pt, 0, pf))
    return x, cell_mask


def patchify(x: torch.Tensor, cell_mask: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                        torch.Tensor]:
    """Returns (patches, patch_cell_mask, token_mask, band_mask).

    patches:         (B, F_bands, T_patches, 1, PATCH_F, PATCH_T)
    patch_cell_mask: same spatial layout, bool
    token_mask:      (B, F_bands, T_patches) — patch has >=1 valid cell
    band_mask:       (B, F_bands) — band has >=1 valid token
    Invalid cells are zeroed BEFORE any learnable layer sees them.
    """
    x = x * cell_mask.to(x.dtype)
    x, cell_mask = pad_to_multiple(x, cell_mask)
    b, n_bins, n_frames = x.shape
    fb, tp = n_bins // PATCH_F, n_frames // PATCH_T
    patches = (x.reshape(b, fb, PATCH_F, tp, PATCH_T)
               .permute(0, 1, 3, 2, 4).unsqueeze(3))
    pmask = (cell_mask.reshape(b, fb, PATCH_F, tp, PATCH_T)
             .permute(0, 1, 3, 2, 4).unsqueeze(3))
    token_mask = pmask.any(dim=(-1, -2)).squeeze(-1)
    band_mask = token_mask.any(dim=-1)
    return patches, pmask, token_mask, band_mask


def patch_centres(freq_hz: torch.Tensor, time_s: torch.Tensor,
                  cell_mask: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Physical centres from REAL cells only (padded entries excluded).

    freq_hz: (B, bins) with zeros in padding; time_s: (B, frames).
    Returns f_centre_khz (B, F_bands), t_centre_s (B, T_patches).
    """
    b, n_bins, n_frames = cell_mask.shape
    fvalid = cell_mask.any(dim=2)          # (B, bins)
    tvalid = cell_mask.any(dim=1)          # (B, frames)
    pf = (-n_bins) % PATCH_F
    pt = (-n_frames) % PATCH_T
    freq_hz = torch.nn.functional.pad(freq_hz, (0, pf))
    fvalid = torch.nn.functional.pad(fvalid, (0, pf))
    time_s = torch.nn.functional.pad(time_s, (0, pt))
    tvalid = torch.nn.functional.pad(tvalid, (0, pt))

    fb = freq_hz.shape[1] // PATCH_F
    tp = time_s.shape[1] // PATCH_T
    fg = freq_hz.reshape(b, fb, PATCH_F)
    fm_ = fvalid.reshape(b, fb, PATCH_F).to(freq_hz.dtype)
    f_centre = (fg * fm_).sum(-1) / fm_.sum(-1).clamp(min=1.0)
    tg = time_s.reshape(b, tp, PATCH_T)
    tm_ = tvalid.reshape(b, tp, PATCH_T).to(time_s.dtype)
    t_centre = (tg * tm_).sum(-1) / tm_.sum(-1).clamp(min=1.0)
    return f_centre / 1000.0, t_centre     # kHz, seconds
