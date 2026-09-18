"""PC-STE v2 model wrapper (spec: analysis/pcste_v2_diagnostics/PROPOSED_PCSTE_V2_ARCHITECTURE.md).

Configuration flags select optional streams; with every flag off the forward pass delegates to the frozen PC-STE core
(bit-identical outputs, test-enforced), so the control arm IS the dissertation architecture.

Streams are named g{grid}_c{channel}. grid 0 = the frozen STFT; grid 1 = the second multi-resolution grid; channel 0 =
the primary channel (dataset's fixed channel); channel 1 = the optional second channel (MaFaulDa underhang axial).
Per-channel encoding: each channel's bands pass through the shared core (channel-type embedding added to its tokens) and
its own mixer pass; channel global embeddings are fused by a validity-masked mean (missing channels masked). Envelope
tokens (from the primary channel) join the mixer pass of channel 0 as extra "bands".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn

from src.methodology_v2.encoder.pcste import PCSTE, PCSTEConfig
from src.methodology_v2.encoder.patchify import patch_centres, patchify
from .envelope import EnvelopeBranch, N_ORDER_CELLS, N_ENVELOPE3_BANDS

MODEL_VERSION = "pcste_v2.model.v1"


@dataclass(frozen=True)
class PCSTEv2Config:
    envelope_branch: bool = False
    multires: bool = False
    n_channels: int = 1
    envelope_bands: int = 1          # 1 = Stage B branch; 3 = pre-registered 3-band fallback
    core: PCSTEConfig = field(default_factory=PCSTEConfig)

    def to_dict(self) -> dict:
        d = asdict(self); d["version"] = MODEL_VERSION
        if self.envelope_bands == 1:
            d.pop("envelope_bands")   # keeps config_sha256 of every Stage B variant unchanged
        return d

    @property
    def n_grids(self) -> int:
        return 2 if self.multires else 1

    @property
    def is_control(self) -> bool:
        return not self.envelope_branch and not self.multires and self.n_channels == 1


class PCSTEv2(nn.Module):
    def __init__(self, cfg: PCSTEv2Config | None = None):
        super().__init__()
        self.cfg = cfg or PCSTEv2Config()
        self.core = PCSTE(self.cfg.core)
        d = self.cfg.core.d_model
        if self.cfg.envelope_branch:
            self.env = EnvelopeBranch(d, n_bands=self.cfg.envelope_bands)
        if self.cfg.multires:
            self.grid_embed = nn.Embedding(2, d)
        if self.cfg.n_channels > 1:
            self.chan_embed = nn.Embedding(self.cfg.n_channels, d)

    # -- band summaries for one stream (replicates PCSTE.forward up to the band summaries) -----------------------
    def _bands(self, spec, frequency_hz, time_seconds, cell_mask, extra: torch.Tensor | None):
        c = self.core
        patches, _, token_mask, band_mask = patchify(spec, cell_mask)
        f_khz, t_s = patch_centres(frequency_hz, time_seconds, cell_mask)
        tok = c.stem(patches)
        if c.cfg.use_coordinates:
            fb, tp = tok.shape[1], tok.shape[2]
            tok = tok + c.coords(f_khz.unsqueeze(2).expand(-1, fb, tp), t_s.unsqueeze(1).expand(-1, fb, tp))
        if extra is not None:
            tok = tok + extra.view(1, 1, 1, -1)
        tok = tok * token_mask.unsqueeze(-1).to(tok.dtype)
        b, fb, tp, d = tok.shape
        z = c.temporal(tok.reshape(b * fb, tp, d)).reshape(b, fb, tp, d)
        z = z * token_mask.unsqueeze(-1).to(z.dtype)
        tm = token_mask.to(z.dtype).unsqueeze(-1)
        band = z.sum(dim=2) / tm.sum(dim=2).clamp(min=1.0)
        band = band * band_mask.unsqueeze(-1).to(band.dtype)
        return band, band_mask, c.coords.freq_features(f_khz)

    def _mix_pool(self, band, band_mask, phi):
        c = self.core
        mixed = c.mixer(band, phi, band_mask) if c.cfg.use_mixer else band
        bm = band_mask.to(mixed.dtype).unsqueeze(-1)
        return (mixed * bm).sum(dim=1) / bm.sum(dim=1).clamp(min=1.0), mixed

    def forward(self, **batch) -> dict:
        if self.cfg.is_control:
            out = self.core(spec=batch["spec_g0_c0"], frequency_hz=batch["frequency_hz_g0_c0"],
                            time_seconds=batch["time_seconds_g0_c0"], cell_mask=batch["cell_mask_g0_c0"])
            return {"global_embedding": out["global_embedding"], "mixed_band_summaries": out["mixed_band_summaries"],
                    "band_mask": out["band_mask"]}
        b = batch["spec_g0_c0"].shape[0]
        stream_valid = batch["stream_valid"]                        # (B, n_grids, n_channels) bool
        env_tok = env_phi = env_mask = None
        if self.cfg.envelope_branch:
            env_tok, env_phi, env_mask = self.env(batch["env"], batch["env_valid"], batch["env_bpfo"], batch["env_known"], batch.get("env_band_valid"))
        chan_z, chan_ok, mixed_all = [], [], []
        for ci in range(self.cfg.n_channels):
            bands, masks, phis = [], [], []
            for gi in range(self.cfg.n_grids):
                key = f"g{gi}_c{ci}"
                extra = None
                if self.cfg.multires:
                    extra = self.grid_embed.weight[gi]
                if self.cfg.n_channels > 1:
                    extra = self.chan_embed.weight[ci] if extra is None else extra + self.chan_embed.weight[ci]
                band, bmask, phi = self._bands(batch[f"spec_{key}"], batch[f"frequency_hz_{key}"], batch[f"time_seconds_{key}"], batch[f"cell_mask_{key}"], extra)
                sv = stream_valid[:, gi, ci].unsqueeze(1)
                bands.append(band); masks.append(bmask & sv); phis.append(phi)
            if ci == 0 and env_tok is not None:
                bands.append(env_tok); masks.append(env_mask); phis.append(env_phi)
            band = torch.cat(bands, dim=1); bmask = torch.cat(masks, dim=1); phi = torch.cat(phis, dim=1)
            z, mixed = self._mix_pool(band, bmask, phi)
            chan_z.append(z); chan_ok.append(stream_valid[:, 0, ci]); mixed_all.append(mixed)
        Z = torch.stack(chan_z, dim=1)                              # (B, C, d)
        ok = torch.stack(chan_ok, dim=1).to(Z.dtype).unsqueeze(-1)  # (B, C, 1)
        z_global = (Z * ok).sum(dim=1) / ok.sum(dim=1).clamp(min=1.0)
        return {"global_embedding": z_global, "mixed_band_summaries": mixed_all[0], "band_mask": None,
                "channel_embeddings": Z}

    def parameter_breakdown(self) -> dict:
        d = self.core.parameter_breakdown()
        d["envelope_branch"] = sum(p.numel() for p in self.env.parameters()) if self.cfg.envelope_branch else 0
        d["grid_embed"] = self.grid_embed.weight.numel() if self.cfg.multires else 0
        d["chan_embed"] = self.chan_embed.weight.numel() if self.cfg.n_channels > 1 else 0
        d["total_v2"] = sum(p.numel() for p in self.parameters())
        return d


# ---------------------------------------------------------------------------------------------------------------
# collation of v2 items
# ---------------------------------------------------------------------------------------------------------------

def collate_v2(items: list[dict], n_grids: int, n_channels: int, envelope: bool, envelope_bands: int = 1) -> dict:
    """items: dicts with 'streams': {'g0_c0': (x (bins, frames) float32, freq_hz (bins,), time_s (frames,)), ...},
    optional 'env': (vector (240,) float32 or None, bpfo_factor float, known 0/1) or, for envelope_bands=3,
    (matrix (3, 240) float32, bpfo_factor, known, band_valid (3,) bool)."""
    n = len(items); out = {}
    stream_valid = torch.zeros(n, n_grids, n_channels, dtype=torch.bool)
    for gi in range(n_grids):
        for ci in range(n_channels):
            key = f"g{gi}_c{ci}"
            present = [it["streams"].get(key) for it in items]
            shapes = [p[0].shape for p in present if p is not None]
            if not shapes:                       # stream absent from the whole batch: masked placeholder
                shapes = [it["streams"]["g0_c0"][0].shape for it in items]
                present = [None] * n
            max_b = max(s[0] for s in shapes); max_t = max(s[1] for s in shapes)
            spec = torch.zeros(n, max_b, max_t, dtype=torch.float32); mask = torch.zeros(n, max_b, max_t, dtype=torch.bool)
            freq = torch.zeros(n, max_b, dtype=torch.float32); time = torch.zeros(n, max_t, dtype=torch.float32)
            for i, p in enumerate(present):
                if p is None:
                    continue
                x, f, t = p; bb, tt = x.shape
                spec[i, :bb, :tt] = torch.as_tensor(np.asarray(x, dtype=np.float32)); mask[i, :bb, :tt] = True
                freq[i, :bb] = torch.as_tensor(np.asarray(f, dtype=np.float32)); time[i, :tt] = torch.as_tensor(np.asarray(t, dtype=np.float32))
                stream_valid[i, gi, ci] = True
            out[f"spec_{key}"] = spec; out[f"cell_mask_{key}"] = mask; out[f"frequency_hz_{key}"] = freq; out[f"time_seconds_{key}"] = time
    out["stream_valid"] = stream_valid
    if envelope and envelope_bands == 1:
        env = torch.zeros(n, N_ORDER_CELLS, dtype=torch.float32); valid = torch.zeros(n, dtype=torch.bool)
        bpfo = torch.ones(n, dtype=torch.float32); known = torch.zeros(n, dtype=torch.long)
        for i, it in enumerate(items):
            e = it.get("env")
            if e is None or e[0] is None:
                continue
            env[i] = torch.as_tensor(np.asarray(e[0], dtype=np.float32)); valid[i] = True; bpfo[i] = float(e[1]); known[i] = int(e[2])
        out["env"] = env; out["env_valid"] = valid; out["env_bpfo"] = bpfo; out["env_known"] = known
    elif envelope:
        env = torch.zeros(n, N_ENVELOPE3_BANDS, N_ORDER_CELLS, dtype=torch.float32); valid = torch.zeros(n, dtype=torch.bool)
        bpfo = torch.ones(n, dtype=torch.float32); known = torch.zeros(n, dtype=torch.long); band_valid = torch.zeros(n, N_ENVELOPE3_BANDS, dtype=torch.bool)
        for i, it in enumerate(items):
            e = it.get("env")
            if e is None or e[0] is None:
                continue
            env[i] = torch.as_tensor(np.asarray(e[0], dtype=np.float32)); valid[i] = True; bpfo[i] = float(e[1]); known[i] = int(e[2])
            band_valid[i] = torch.as_tensor(np.asarray(e[3], dtype=bool))
        out["env"] = env; out["env_valid"] = valid; out["env_bpfo"] = bpfo; out["env_known"] = known; out["env_band_valid"] = band_valid
    return out
