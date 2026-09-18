"""PC-STE — Physically Calibrated Spectro-Temporal Encoder (Part 5B).

Frozen N2 spectrogram -> 16x8 patches -> local patch embedding ->
absolute-coordinate features -> shared per-band temporal backbone ->
Hz-gated cross-band exchange -> validity-masked pooling -> global
embedding (d=192).

The identical class serves future S0 (random init + supervised) and S1
(SSL init + the same supervised training); the encoder consumes only
tensor values, physical coordinates and masks — never labels or
dataset-conditional branches.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn

from .coords import FourierCoordEncoder
from .mixer import HzGatedCrossBandMixer
from .patchify import PATCH_F, PATCH_T, patch_centres, patchify
from .ssm import build_backbone


@dataclass(frozen=True)
class PCSTEConfig:
    name: str = "PC-STE"
    version: str = "part5b.v1"
    d_model: int = 192
    n_temporal_blocks: int = 4
    backbone: str = "bimamba"        # "bimamba" | "transformer" (A3)
    patch_freq_bins: int = PATCH_F
    patch_time_frames: int = PATCH_T
    stem_conv_channels: int = 8
    coordinate_encoder: str = "fourier_absolute_khz_seconds"
    use_coordinates: bool = True     # False = ablation A1 (index PE none)
    use_mixer: bool = True           # False = ablation A2 (mean agg)
    bidirectional_fusion: str = "mean_of_directions"
    pooling: str = "validity_masked_mean"
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class PatchStem(nn.Module):
    """Compact local patch encoder: 3x3 depth conv over the 16x8 patch
    (receptive field 3x3 inside the patch) then full-patch linear
    projection to d. Shared across datasets and bands."""

    def __init__(self, cfg: PCSTEConfig):
        super().__init__()
        c = cfg.stem_conv_channels
        self.conv = nn.Conv2d(1, c, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.proj = nn.Linear(c * PATCH_F * PATCH_T, cfg.d_model)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        b, fb, tp = patches.shape[:3]
        x = patches.reshape(b * fb * tp, 1, PATCH_F, PATCH_T)
        x = self.act(self.conv(x)).reshape(b * fb * tp, -1)
        return self.proj(x).reshape(b, fb, tp, -1)


class PCSTE(nn.Module):
    def __init__(self, cfg: PCSTEConfig | None = None):
        super().__init__()
        self.cfg = cfg or PCSTEConfig()
        self.stem = PatchStem(self.cfg)
        self.coords = FourierCoordEncoder(self.cfg.d_model)
        self.temporal = build_backbone(self.cfg.backbone,
                                       self.cfg.d_model,
                                       self.cfg.n_temporal_blocks)
        self.mixer = HzGatedCrossBandMixer(self.cfg.d_model)

    # -- core forward -----------------------------------------------------
    def forward(self, spec: torch.Tensor, frequency_hz: torch.Tensor,
                time_seconds: torch.Tensor,
                cell_mask: torch.Tensor) -> dict:
        patches, _, token_mask, band_mask = patchify(spec, cell_mask)
        f_khz, t_s = patch_centres(frequency_hz, time_seconds, cell_mask)

        tok = self.stem(patches)
        if self.cfg.use_coordinates:
            fb, tp = tok.shape[1], tok.shape[2]
            ce = self.coords(
                f_khz.unsqueeze(2).expand(-1, fb, tp),
                t_s.unsqueeze(1).expand(-1, fb, tp))
            tok = tok + ce
        tok = tok * token_mask.unsqueeze(-1).to(tok.dtype)

        b, fb, tp, d = tok.shape
        z = self.temporal(tok.reshape(b * fb, tp, d)).reshape(b, fb, tp, d)
        z = z * token_mask.unsqueeze(-1).to(z.dtype)

        tm = token_mask.to(z.dtype).unsqueeze(-1)
        band = z.sum(dim=2) / tm.sum(dim=2).clamp(min=1.0)
        band = band * band_mask.unsqueeze(-1).to(band.dtype)

        if self.cfg.use_mixer:
            phi_f = self.coords.freq_features(f_khz)
            mixed = self.mixer(band, phi_f, band_mask)
        else:                                   # ablation A2
            mixed = band

        bm = band_mask.to(mixed.dtype).unsqueeze(-1)
        z_global = (mixed * bm).sum(dim=1) / bm.sum(dim=1).clamp(min=1.0)
        return {"global_embedding": z_global,
                "band_summaries": band, "mixed_band_summaries": mixed,
                "tokens": z, "token_mask": token_mask,
                "band_mask": band_mask,
                "band_freq_khz": f_khz, "time_patch_centres_s": t_s}

    # -- public API -------------------------------------------------------
    def encode_global(self, spec, frequency_hz, time_seconds,
                      cell_mask) -> torch.Tensor:
        return self(spec, frequency_hz, time_seconds,
                    cell_mask)["global_embedding"]

    def encode_tokens(self, spec, frequency_hz, time_seconds,
                      cell_mask) -> dict:
        return self(spec, frequency_hz, time_seconds, cell_mask)

    # -- registry helpers -------------------------------------------------
    def parameter_breakdown(self) -> dict:
        def cnt(m):
            return sum(p.numel() for p in m.parameters()
                       if p.requires_grad)
        return {"patch_stem": cnt(self.stem),
                "coordinate_encoder": cnt(self.coords),
                "temporal_backbone": cnt(self.temporal),
                "cross_band_mixer": cnt(self.mixer),
                "total": cnt(self)}
