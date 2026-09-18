"""Student architectures and deterministic surgery from primary checkpoints.

Student-D  = the FROZEN PC-STE class with n_temporal_blocks=2 (same
             stem, coordinates, d_model 192, d_inner 384, d_state 16,
             Hz mixer, dataset heads) -> 1,375,185 encoder params.
Student-DW = PC-STE with d_model 128 (d_inner 256 by the frozen
             expand=2), 2 blocks, optional low-rank stem, 128-d heads.
Half-student alternative for the Stage-2 rule: 4 layers x 1 direction
             (UniMambaLayer) — equal-cost comparator, never the primary.

Surgery (K1/K0/P1 init): copy stem, coords, retained layers (explicit
deterministic mapping student layer i <- teacher layer retained[i]),
final backbone LayerNorm, mixer and heads from the same-cell primary
best.pt. Strict: every student tensor must be filled from an explicitly
named source tensor with identical shape; extra teacher tensors are
exactly the dropped layers; anything else fails closed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoder import PCSTE, PCSTEConfig
from ..encoder.pcste import PatchStem
from ..encoder.patchify import PATCH_F, PATCH_T
from ..encoder.ssm import BiMambaLayer, MambaRefBlock
from ..experiment.heads import CLASS_ORDERS, DatasetHeads
from .guards import Part6GuardError
from .protocol import (D_MODEL, FULL_BLOCKS, STUDENT_D_BLOCKS, STUDENT_DW)


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StudentSpec:
    name: str                       # "full" | "student_d" | "student_dw" | "half_4x1"
    n_blocks: int
    d_model: int = D_MODEL
    directions: int = 2             # 2 = BiMamba, 1 = forward-only
    stem_rank: int | None = None    # low-rank stem (DW only), None = full
    uni_residual: str = "mean_of_remaining"   # half_4x1 only

    def pcste_config(self) -> PCSTEConfig:
        return PCSTEConfig(d_model=self.d_model, n_temporal_blocks=self.n_blocks)

    def to_dict(self) -> dict:
        return {"name": self.name, "n_blocks": self.n_blocks,
                "d_model": self.d_model, "directions": self.directions,
                "stem_rank": self.stem_rank, "uni_residual": self.uni_residual,
                "d_inner": 2 * self.d_model, "d_state": 16,
                "dt_rank": math.ceil(self.d_model / 16)}


FULL_SPEC = StudentSpec("full", FULL_BLOCKS)
STUDENT_D_SPEC = StudentSpec("student_d", STUDENT_D_BLOCKS)


def student_dw_spec(stem_rank: int | None = None) -> StudentSpec:
    return StudentSpec("student_dw", STUDENT_DW["n_blocks"],
                       d_model=STUDENT_DW["d_model"], stem_rank=stem_rank)


def half_4x1_spec(uni_residual: str = "mean_of_remaining") -> StudentSpec:
    return StudentSpec("half_4x1", FULL_BLOCKS, directions=1,
                       uni_residual=uni_residual)


# ---------------------------------------------------------------------------
# modules used only by non-primary variants
# ---------------------------------------------------------------------------
class LowRankPatchStem(nn.Module):
    """1024 -> r -> d factorised patch projection (DW option). Same 3x3
    conv front-end as the frozen stem; only the dense projection is
    factorised. Not an SVD 'folding' of anything trained."""

    def __init__(self, cfg: PCSTEConfig, rank: int):
        super().__init__()
        c = cfg.stem_conv_channels
        self.rank = rank
        self.conv = nn.Conv2d(1, c, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.proj_a = nn.Linear(c * PATCH_F * PATCH_T, rank, bias=False)
        self.proj_b = nn.Linear(rank, cfg.d_model)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        b, fb, tp = patches.shape[:3]
        x = patches.reshape(b * fb * tp, 1, PATCH_F, PATCH_T)
        x = self.act(self.conv(x)).reshape(b * fb * tp, -1)
        return self.proj_b(self.proj_a(x)).reshape(b, fb, tp, -1)


class UniMambaLayer(nn.Module):
    """Pre-norm residual single-direction Mamba layer (equal-cost
    comparator '4 layers x 1 direction'). residual='mean_of_remaining':
    y = x + fwd(LN x); 'keep_half_scale': y = x + 0.5*fwd(LN x)."""

    def __init__(self, d_model: int, residual: str = "mean_of_remaining"):
        super().__init__()
        if residual not in ("mean_of_remaining", "keep_half_scale"):
            raise Part6GuardError(f"unknown residual mode {residual}")
        self.norm = nn.LayerNorm(d_model)
        self.fwd = MambaRefBlock(d_model)
        self.scale = 1.0 if residual == "mean_of_remaining" else 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.fwd(self.norm(x))


class DatasetHeadsDW(nn.Module):
    """128-d single-linear heads for Student-DW (same class orders)."""

    def __init__(self, d_model: int, init_seed: int | None = None):
        super().__init__()
        if init_seed is not None:
            torch.manual_seed(init_seed)
        self.heads = nn.ModuleDict({
            ds: nn.Linear(d_model, len(classes))
            for ds, classes in CLASS_ORDERS.items()})

    def forward(self, z: torch.Tensor, dataset: str) -> torch.Tensor:
        return self.heads[dataset](z)


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def build_encoder(spec: StudentSpec, seed: int | None = None) -> PCSTE:
    """Instantiate an encoder for `spec`. Student-D is literally the frozen
    PCSTE class with 2 blocks; DW/half variants patch only the modules
    their spec names."""
    if seed is not None:
        torch.manual_seed(seed)
    enc = PCSTE(spec.pcste_config())
    if spec.stem_rank is not None:
        enc.stem = LowRankPatchStem(enc.cfg, spec.stem_rank)
    if spec.directions == 1:
        enc.temporal.layers = nn.ModuleList(
            UniMambaLayer(spec.d_model, spec.uni_residual)
            for _ in range(spec.n_blocks))
    return enc


def build_heads(spec: StudentSpec, init_seed: int) -> nn.Module:
    if spec.d_model == D_MODEL:
        return DatasetHeads(init_seed=init_seed)       # frozen class
    return DatasetHeadsDW(spec.d_model, init_seed=init_seed)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def n_layers(enc: PCSTE) -> int:
    return len(enc.temporal.layers)


def scan_steps_per_forward(enc: PCSTE, band_seq_len: int) -> int:
    """Sequential recurrent steps for one window: layers x directions x T."""
    dirs = 0
    for layer in enc.temporal.layers:
        dirs += 2 if isinstance(layer, BiMambaLayer) else 1
    return dirs * band_seq_len


def recurrent_work_R(enc: PCSTE, band_seq_len: int) -> int:
    """R = layers x dirs x T x d_inner x d_state (per band)."""
    blk = _first_block(enc)
    return scan_steps_per_forward(enc, band_seq_len) * blk.d_inner * blk.d_state


def _first_block(enc: PCSTE) -> MambaRefBlock:
    layer = enc.temporal.layers[0]
    return layer.fwd


def architecture_summary(enc: PCSTE, heads: nn.Module | None = None) -> dict:
    blk = _first_block(enc)
    return {"encoder_params": count_params(enc),
            "head_params": count_params(heads) if heads is not None else None,
            "total_params": count_params(enc) + (count_params(heads)
                                                 if heads is not None else 0),
            "n_blocks": n_layers(enc),
            "directions": 2 if isinstance(enc.temporal.layers[0],
                                          BiMambaLayer) else 1,
            "d_model": enc.cfg.d_model, "d_inner": blk.d_inner,
            "d_state": blk.d_state, "dt_rank": blk.dt_rank,
            "scan_steps_T23": scan_steps_per_forward(enc, 23),
            "scan_steps_T24": scan_steps_per_forward(enc, 24),
            "R_per_band_T24": recurrent_work_R(enc, 24),
            "breakdown": enc.parameter_breakdown()}


# ---------------------------------------------------------------------------
# surgery: full checkpoint -> student state
# ---------------------------------------------------------------------------
LAYER_PREFIX = "temporal.layers."


def layer_index_of_key(key: str) -> int | None:
    if not key.startswith(LAYER_PREFIX):
        return None
    return int(key[len(LAYER_PREFIX):].split(".")[0])


def student_state_from_full(full_encoder_state: dict,
                            retained_layers: list[int],
                            n_full_layers: int = FULL_BLOCKS) -> tuple[dict, dict]:
    """Map a full-PC-STE encoder state_dict onto a Student-D state_dict.

    student layer i <- teacher layer retained_layers[i]; every non-layer
    tensor (stem, coords, temporal.norm, mixer) copied verbatim.
    Returns (student_state, mapping_report). Fails closed on duplicate or
    out-of-range retained indices, or on unexpected keys."""
    if len(retained_layers) != STUDENT_D_BLOCKS:
        raise Part6GuardError(f"retained_layers must have {STUDENT_D_BLOCKS} "
                              f"entries, got {retained_layers}")
    if len(set(retained_layers)) != len(retained_layers):
        raise Part6GuardError(f"duplicate retained layers {retained_layers}")
    if any(not (0 <= r < n_full_layers) for r in retained_layers):
        raise Part6GuardError(f"retained layers out of range {retained_layers}")
    seen_layers = {layer_index_of_key(k) for k in full_encoder_state
                   if layer_index_of_key(k) is not None}
    if seen_layers != set(range(n_full_layers)):
        raise Part6GuardError(f"teacher state has layers {sorted(seen_layers)}, "
                              f"expected {list(range(n_full_layers))}")
    student, mapping, dropped = {}, {}, set()
    for k, v in full_encoder_state.items():
        li = layer_index_of_key(k)
        if li is None:
            student[k] = v.clone()
            mapping[k] = k
            continue
        if li in retained_layers:
            new_i = retained_layers.index(li)
            nk = f"{LAYER_PREFIX}{new_i}." + k[len(LAYER_PREFIX):].split(".", 1)[1]
            student[nk] = v.clone()
            mapping[nk] = k
        else:
            dropped.add(li)
    report = {"retained_layers": list(retained_layers),
              "dropped_layers": sorted(dropped),
              "n_student_tensors": len(student),
              "n_teacher_tensors": len(full_encoder_state),
              "copied_non_layer_tensors": sum(1 for k in mapping
                                             if layer_index_of_key(k) is None),
              "mapping": mapping}
    return student, report


def load_student_from_full(student: PCSTE, full_encoder_state: dict,
                           retained_layers: list[int]) -> dict:
    """Strict load: shapes must match, no missing/unexpected keys."""
    sd, report = student_state_from_full(full_encoder_state, retained_layers)
    target = student.state_dict()
    if set(sd) != set(target):
        missing = sorted(set(target) - set(sd))
        unexpected = sorted(set(sd) - set(target))
        raise Part6GuardError(f"surgery key mismatch: missing={missing[:5]} "
                              f"unexpected={unexpected[:5]}")
    for k in target:
        if tuple(target[k].shape) != tuple(sd[k].shape):
            raise Part6GuardError(f"shape mismatch at {k}: {tuple(target[k].shape)} "
                                  f"vs {tuple(sd[k].shape)}")
    res = student.load_state_dict(sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    return report


def load_heads_from_full(heads: nn.Module, full_heads_state: dict) -> None:
    res = heads.load_state_dict(full_heads_state, strict=True)
    assert not res.missing_keys and not res.unexpected_keys


def half_4x1_state_from_full(full_encoder_state: dict, keep: str = "fwd"
                             ) -> tuple[dict, dict]:
    """Training-free '4 layers x 1 direction' state: drop the other
    direction's tensors from every layer, keep everything else."""
    if keep not in ("fwd", "bwd"):
        raise Part6GuardError("keep must be fwd|bwd")
    drop = "bwd" if keep == "fwd" else "fwd"
    out, mapping = {}, {}
    for k, v in full_encoder_state.items():
        li = layer_index_of_key(k)
        if li is not None:
            sub = k[len(LAYER_PREFIX):].split(".", 1)[1]
            if sub.startswith(drop + "."):
                continue
            if sub.startswith(keep + "."):
                nk = f"{LAYER_PREFIX}{li}.fwd." + sub.split(".", 1)[1]
                out[nk] = v.clone()
                mapping[nk] = k
                continue
        out[k] = v.clone()
        mapping[k] = k
    return out, {"kept_direction": keep, "mapping": mapping}


def load_half4x1_from_full(student: PCSTE, full_encoder_state: dict,
                           keep: str = "fwd") -> dict:
    """Strict 4x1 surgery load: drop the other direction of every layer,
    keep everything else verbatim; shapes checked, no missing/unexpected
    keys. The loaded model is bit-identical to the training-free
    drop_direction evaluation model (test-proven)."""
    sd, rep = half_4x1_state_from_full(full_encoder_state, keep)
    target = student.state_dict()
    if set(sd) != set(target):
        missing = sorted(set(target) - set(sd))
        unexpected = sorted(set(sd) - set(target))
        raise Part6GuardError(f"4x1 surgery key mismatch: missing="
                              f"{missing[:5]} unexpected={unexpected[:5]}")
    for k in target:
        if tuple(target[k].shape) != tuple(sd[k].shape):
            raise Part6GuardError(f"shape mismatch at {k}")
    res = student.load_state_dict(sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    return {"kept_direction": rep["kept_direction"],
            "n_student_tensors": len(sd),
            "n_teacher_tensors": len(full_encoder_state),
            "dropped": "one direction per layer"}


# ---------------------------------------------------------------------------
# a tiny wrapper so a single optimizer / state_dict covers encoder+heads
# ---------------------------------------------------------------------------
class EncoderWithHeads(nn.Module):
    def __init__(self, encoder: nn.Module, heads: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.heads = heads

    def forward(self, batch: dict) -> dict:
        return self.encoder(**batch)
