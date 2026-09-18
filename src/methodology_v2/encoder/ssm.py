"""Temporal backbones for PC-STE: reference-faithful bidirectional Mamba
and the pre-registered equal-capacity Transformer drop-in.

DEPENDENCY DISCLOSURE (Part-5B Mamba rule): the official `mamba_ssm`
CUDA package cannot be built in this environment (torch 2.12.0+cu130 has
no compatible causal-conv1d/mamba-ssm wheel; source build fails — see
PCSTE_ENCODER_REPORT.md). This module therefore implements the OFFICIAL
Mamba-1 selective-SSM formulation (Gu & Dao 2023; parameterization and
recurrence exactly as mamba_ssm.modules.mamba_simple / selective_scan
reference) in pure PyTorch:

  in_proj: x,z = W_in u                (d -> 2*d_inner, d_inner = 2d)
  x <- SiLU(causal depthwise Conv1d_k4(x))
  dt, B, C = W_x x    (dt_rank=ceil(d/16), d_state=16)
  Delta = softplus(W_dt dt + b_dt)
  h_t = exp(Delta_t A) h_{t-1} + (Delta_t B_t) x_t   (selective scan)
  y_t = C_t h_t + D x_t
  out = W_out (y * SiLU(z))

This is NOT an RNN substitute: it is the same architecture and the same
recurrence, executed without the fused kernel — computationally
irrelevant at the per-band sequence length of 23-24 used here. Swapping
in the official kernel later changes speed, not the architecture.

Bidirectional design (documented, compact): per layer,
  y = x + 0.5 * ( MambaFwd(LN(x)) + flip( MambaBwd( flip(LN(x)) ) ) )
with two independently parameterized directional Mamba blocks.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan(x: torch.Tensor, delta: torch.Tensor,
                   a_mat: torch.Tensor, b: torch.Tensor,
                   c: torch.Tensor, d_skip: torch.Tensor) -> torch.Tensor:
    """Reference selective scan (official Mamba-1 recurrence):
    h_t = exp(delta_t A) h_{t-1} + (delta_t B_t) x_t;  y_t = C_t h_t + D x_t.
    x, delta: (N, T, d_inner); a_mat: (d_inner, d_state);
    b, c: (N, T, d_state); d_skip: (d_inner,). Parity against the vendored
    official selective_scan_ref is enforced by the Part-5B parity gate."""
    n, t, _ = x.shape
    h = x.new_zeros(n, x.shape[2], a_mat.shape[1])
    ys = []
    for k in range(t):
        da = torch.exp(delta[:, k].unsqueeze(-1) * a_mat)
        db = delta[:, k].unsqueeze(-1) * b[:, k].unsqueeze(1)
        h = da * h + db * x[:, k].unsqueeze(-1)
        ys.append(torch.einsum("nds,ns->nd", h, c[:, k]))
    return torch.stack(ys, dim=1) + d_skip * x


class MambaRefBlock(nn.Module):
    """Official Mamba-1 block, reference (loop) selective scan."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner,
                                kernel_size=d_conv, groups=self.d_inner,
                                padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner,
                                self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (N, T, d_model)
        n, t, _ = u.shape
        xz = self.in_proj(u)                       # (N, T, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)
        x = self.conv1d(x.transpose(1, 2))[:, :, :t].transpose(1, 2)
        x = F.silu(x)                              # (N, T, d_inner)

        dbc = self.x_proj(x)
        dt, b, c = torch.split(
            dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))       # (N, T, d_inner)
        a_mat = -torch.exp(self.A_log)             # (d_inner, d_state)
        y = selective_scan(x, delta, a_mat, b, c, self.D)
        return self.out_proj(y * F.silu(z))


class BiMambaLayer(nn.Module):
    """Pre-norm residual bidirectional Mamba (mean fusion)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = MambaRefBlock(d_model)
        self.bwd = MambaRefBlock(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + 0.5 * (self.fwd(h)
                          + self.bwd(h.flip(1)).flip(1))


class BiMambaBackbone(nn.Module):
    kind = "bimamba_reference"

    def __init__(self, d_model: int = 192, n_blocks: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(BiMambaLayer(d_model)
                                    for _ in range(n_blocks))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TransformerBackbone(nn.Module):
    """Equal-capacity pre-norm Transformer drop-in (ablation A3 only —
    instantiated for parameter/forward verification, never trained
    here). Replaces ONLY the temporal backbone; everything else in
    PC-STE stays identical."""
    kind = "transformer"

    def __init__(self, d_model: int = 192, n_blocks: int = 4,
                 n_heads: int = 4, mlp_ratio: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads,
                                              batch_first=True),
                "ln2": nn.LayerNorm(d_model),
                "mlp": nn.Sequential(
                    nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(),
                    nn.Linear(mlp_ratio * d_model, d_model)),
            }))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            h = blk["ln1"](x)
            a, _ = blk["attn"](h, h, h, need_weights=False)
            x = x + a
            x = x + blk["mlp"](blk["ln2"](x))
        return self.norm(x)


def build_backbone(kind: str, d_model: int, n_blocks: int) -> nn.Module:
    if kind == "bimamba":
        return BiMambaBackbone(d_model, n_blocks)
    if kind == "transformer":
        return TransformerBackbone(d_model, n_blocks)
    raise ValueError(f"unknown backbone kind {kind}")
