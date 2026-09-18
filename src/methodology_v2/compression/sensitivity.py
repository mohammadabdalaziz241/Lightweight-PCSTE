"""Stage-2 training-free structural sensitivity tools (VALIDATION-only).

Every tool takes a loaded full PC-STE (+heads), returns a MODIFIED COPY,
and is scored by `evaluate_split` which structurally refuses TEST window
ids (guards.assert_no_test_windows) and never writes anything named
test_*. Importance statistics (Taylor, |D|, mean softplus(Delta),
activation variance, d_state decay activity) are accumulated on TRAIN
windows only (asserted).

Tools: drop-one-layer, drop-one-direction (two residual conventions),
d_inner channel pruning 25/50 % (+ fallback 384->320) by 4 importance
scores, d_state 16->8 (+ fallback 16->12) by decay activity, 2:1
time-token merging, per-band occlusion, low-rank stem (SVD truncation of
stem.proj — an ANALYSIS of the trained stem, not an initialisation),
half-student comparison (2x2 vs 4x1) framework, fallback triggers.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoder import PCSTE, collate_representations
from ..encoder.ssm import BiMambaLayer, MambaRefBlock
from ..experiment.heads import CLASS_ORDERS, LABEL_FIELD, DatasetHeads
from ..experiment.metrics import classification_report, macro_domain_f1
from .guards import Part6GuardError, assert_no_test_windows, split_of_window_id
from .student import (UniMambaLayer, half_4x1_spec, student_state_from_full,
                      STUDENT_D_SPEC, build_encoder)

DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")


# ---------------------------------------------------------------------------
# validation evaluator (TEST-refusing)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_split(encoder: nn.Module, heads: nn.Module, rep_fn,
                   ids_by_ds: dict, manifest, device: str = "cpu",
                   chunk: int = 64, allowed_split: str = "validation") -> dict:
    """MacroDomainF1 + per-dataset reports on the given ids. Refuses any
    id whose split is not `allowed_split` (default validation) — TEST is
    impossible here regardless of the caller."""
    if allowed_split == "test":
        raise Part6GuardError("sensitivity evaluator never scores TEST")
    all_ids = [w for ds in DATASETS for w in ids_by_ds.get(ds, [])]
    assert_no_test_windows(all_ids, "sensitivity evaluate_split")
    for w in all_ids:
        if split_of_window_id(w) != allowed_split:
            raise Part6GuardError(f"{w}: split != {allowed_split}")
    encoder.eval()
    heads.eval()
    reports = {}
    for ds in DATASETS:
        wids = ids_by_ds.get(ds, [])
        y_true = [str(manifest.loc[w, LABEL_FIELD[ds]]) for w in wids]
        y_pred = []
        for lo in range(0, len(wids), chunk):
            sub = wids[lo:lo + chunk]
            batch = collate_representations([rep_fn(w) for w in sub])
            batch = {k: v.to(device) for k, v in batch.items()}
            z = encoder(**batch)["global_embedding"]
            logits = heads(z, ds)
            y_pred.extend(CLASS_ORDERS[ds][int(i)] for i in logits.argmax(-1))
        reports[ds] = classification_report(y_true, y_pred, ds)
    return {"macro_domain_f1": macro_domain_f1(reports),
            "per_dataset_macro_f1": {ds: reports[ds]["macro_f1"]
                                     for ds in DATASETS},
            "reports": reports, "split": allowed_split}


# ---------------------------------------------------------------------------
# structural edits (all return deep copies)
# ---------------------------------------------------------------------------
def drop_layer(encoder: PCSTE, layer_idx: int) -> PCSTE:
    m = copy.deepcopy(encoder)
    layers = list(m.temporal.layers)
    if not (0 <= layer_idx < len(layers)):
        raise Part6GuardError(f"layer {layer_idx} out of range")
    del layers[layer_idx]
    m.temporal.layers = nn.ModuleList(layers)
    return m


class _DirectionDropped(nn.Module):
    """BiMamba layer with one direction removed, training-free.
    residual='keep_half_scale': y = x + 0.5*kept ; 'mean_of_remaining':
    y = x + kept."""

    def __init__(self, layer: BiMambaLayer, drop: str, residual: str):
        super().__init__()
        self.norm = layer.norm
        self.kept = layer.bwd if drop == "fwd" else layer.fwd
        self.flip = drop == "fwd"          # kept = bwd -> needs flips
        self.scale = 0.5 if residual == "keep_half_scale" else 1.0

    def forward(self, x):
        h = self.norm(x)
        y = self.kept(h.flip(1)).flip(1) if self.flip else self.kept(h)
        return x + self.scale * y


def drop_direction(encoder: PCSTE, layer_idx: int, drop: str,
                   residual: str = "mean_of_remaining") -> PCSTE:
    if drop not in ("fwd", "bwd"):
        raise Part6GuardError("drop must be fwd|bwd")
    m = copy.deepcopy(encoder)
    layer = m.temporal.layers[layer_idx]
    if not isinstance(layer, BiMambaLayer):
        raise Part6GuardError("drop_direction needs a BiMambaLayer")
    m.temporal.layers[layer_idx] = _DirectionDropped(layer, drop, residual)
    return m


def merge_time_tokens_2to1(encoder: PCSTE) -> PCSTE:
    """2:1 adjacent time-token averaging BEFORE the temporal backbone
    (R-MeeTo-style training-free reduction). Implemented by wrapping the
    backbone: tokens (N, T, d) -> pairs averaged -> backbone -> repeat
    back to T (nearest) so downstream masking/pooling is unchanged."""
    m = copy.deepcopy(encoder)
    inner = m.temporal

    class Merged(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, x):
            n, t, d = x.shape
            tp = t + (t % 2)
            xp = F.pad(x, (0, 0, 0, tp - t)) if tp != t else x
            merged = xp.reshape(n, tp // 2, 2, d).mean(2)
            if tp != t:                    # last odd token stands alone
                merged = torch.cat([merged[:, :-1], xp[:, t - 1:t]], dim=1)
            y = self.backbone(merged)
            y = y.repeat_interleave(2, dim=1)[:, :t]
            return y

    m.temporal = Merged(inner)
    return m


def occlude_band(encoder: PCSTE, band_idx: int) -> PCSTE:
    """Zero (and mask out) one frequency band's tokens for every window —
    a training-free per-band occlusion probe."""
    m = copy.deepcopy(encoder)
    stem = m.stem

    class OccludedStem(nn.Module):
        def __init__(self, s):
            super().__init__()
            self.s = s

        def forward(self, patches):
            p = patches.clone()
            if band_idx < p.shape[1]:
                p[:, band_idx] = 0.0
            return self.s(p)

    m.stem = OccludedStem(stem)
    return m


def low_rank_stem(encoder: PCSTE, rank: int) -> tuple[PCSTE, dict]:
    """SVD-truncate the TRAINED stem projection W (d x 1024) to rank r —
    analysis of the stem's spectrum (energy captured), evaluated
    training-free. NOT an initialisation of anything."""
    m = copy.deepcopy(encoder)
    w = m.stem.proj.weight.data
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    energy = float((s[:rank] ** 2).sum() / (s ** 2).sum())
    w_r = (u[:, :rank] * s[:rank]) @ vh[:rank]
    m.stem.proj.weight.data.copy_(w_r)
    return m, {"rank": rank, "energy_captured": energy,
               "singular_values": [float(v) for v in s[:min(32, len(s))]]}


# ---------------------------------------------------------------------------
# d_inner channel importance + pruning
# ---------------------------------------------------------------------------
class ChannelStats:
    """Accumulates per-block d_inner channel statistics on TRAIN windows:
    |D|, mean softplus(Delta), activation variance of x (post conv+SiLU),
    and first-order Taylor |w * dw| on out_proj columns."""

    def __init__(self, encoder: PCSTE):
        self.enc = encoder
        self.blocks = {f"temporal.layers.{i}.{d}": getattr(l, d)
                       for i, l in enumerate(encoder.temporal.layers)
                       for d in ("fwd", "bwd") if isinstance(l, BiMambaLayer)}
        self.delta_sum = {k: torch.zeros(b.d_inner) for k, b in self.blocks.items()}
        self.act_sum = {k: torch.zeros(b.d_inner) for k, b in self.blocks.items()}
        self.act_sq = {k: torch.zeros(b.d_inner) for k, b in self.blocks.items()}
        self.taylor = {k: torch.zeros(b.d_inner) for k, b in self.blocks.items()}
        self.count = {k: 0 for k in self.blocks}
        self._hooks = []
        for k, b in self.blocks.items():
            self._hooks.append(b.dt_proj.register_forward_hook(
                lambda mod, a, out, k=k: self._dt(k, out)))
            self._hooks.append(b.x_proj.register_forward_pre_hook(
                lambda mod, a, k=k: self._act(k, a[0])))

    def _dt(self, k, out):
        d = F.softplus(out.detach())                # (N,T,d_inner)
        self.delta_sum[k] += d.sum(dim=(0, 1)).cpu()

    def _act(self, k, x):
        x = x.detach()
        self.act_sum[k] += x.sum(dim=(0, 1)).cpu()
        self.act_sq[k] += (x ** 2).sum(dim=(0, 1)).cpu()
        self.count[k] += x.shape[0] * x.shape[1]

    def accumulate_taylor(self) -> None:
        """Call AFTER a backward pass: |w * grad| on out_proj columns."""
        for k, b in self.blocks.items():
            w = b.out_proj.weight
            if w.grad is not None:
                self.taylor[k] += (w.detach() * w.grad.detach()).abs().sum(0).cpu()

    def remove(self):
        for h in self._hooks:
            h.remove()

    def scores(self) -> dict:
        out = {}
        for k, b in self.blocks.items():
            n = max(self.count[k], 1)
            mean = self.act_sum[k] / n
            var = self.act_sq[k] / n - mean ** 2
            out[k] = {"abs_D": b.D.detach().abs().cpu(),
                      "mean_softplus_delta": self.delta_sum[k] / n,
                      "activation_variance": var.clamp_min(0),
                      "taylor": self.taylor[k]}
        return out


def accumulate_train_stats(encoder: PCSTE, heads: DatasetHeads, rep_fn,
                           train_ids_by_ds: dict, manifest,
                           device: str = "cpu", chunk: int = 32,
                           max_per_dataset: int | None = None) -> dict:
    """Run TRAIN windows through the model with a hard-label CE backward
    (Taylor) — statistics from TRAIN only (asserted)."""
    ids = [w for ds in DATASETS for w in train_ids_by_ds.get(ds, [])]
    assert_no_test_windows(ids, "channel stats")
    for w in ids:
        if split_of_window_id(w) != "train":
            raise Part6GuardError(f"importance statistics need TRAIN windows: {w}")
    stats = ChannelStats(encoder)
    encoder.eval()
    heads.eval()
    try:
        for ds in DATASETS:
            wids = train_ids_by_ds.get(ds, [])
            if max_per_dataset:
                wids = wids[:max_per_dataset]
            for lo in range(0, len(wids), chunk):
                sub = wids[lo:lo + chunk]
                batch = collate_representations([rep_fn(w) for w in sub])
                batch = {k: v.to(device) for k, v in batch.items()}
                encoder.zero_grad(set_to_none=True)
                z = encoder(**batch)["global_embedding"]
                logits = heads(z, ds)
                y = torch.tensor([CLASS_ORDERS[ds].index(
                    str(manifest.loc[w, LABEL_FIELD[ds]])) for w in sub],
                    device=device)
                F.cross_entropy(logits, y).backward()
                stats.accumulate_taylor()
        return stats.scores()
    finally:
        stats.remove()
        encoder.zero_grad(set_to_none=True)


class PrunedMambaRefBlock(MambaRefBlock):
    """MambaRefBlock with an explicit (smaller) d_inner and/or d_state,
    built by copying the retained channels/states of a trained block."""

    def __init__(self, src: MambaRefBlock, keep_channels: torch.Tensor,
                 keep_states: torch.Tensor | None = None):
        nn.Module.__init__(self)
        keep_channels = torch.as_tensor(keep_channels).long().sort().values
        keep_states = (torch.arange(src.d_state) if keep_states is None
                       else torch.as_tensor(keep_states).long().sort().values)
        di, ds = len(keep_channels), len(keep_states)
        self.d_model, self.d_inner, self.d_state = src.d_model, di, ds
        self.d_conv, self.dt_rank = src.d_conv, src.dt_rank
        both = torch.cat([keep_channels, keep_channels + src.d_inner])
        self.in_proj = nn.Linear(src.d_model, 2 * di, bias=False)
        self.in_proj.weight.data = src.in_proj.weight.data[both].clone()
        self.conv1d = nn.Conv1d(di, di, kernel_size=src.d_conv, groups=di,
                                padding=src.d_conv - 1)
        self.conv1d.weight.data = src.conv1d.weight.data[keep_channels].clone()
        self.conv1d.bias.data = src.conv1d.bias.data[keep_channels].clone()
        # x_proj rows: [dt_rank | B (d_state) | C (d_state)]
        rows = torch.cat([torch.arange(src.dt_rank),
                          src.dt_rank + keep_states,
                          src.dt_rank + src.d_state + keep_states])
        self.x_proj = nn.Linear(di, src.dt_rank + 2 * ds, bias=False)
        self.x_proj.weight.data = src.x_proj.weight.data[rows][:, keep_channels].clone()
        self.dt_proj = nn.Linear(src.dt_rank, di, bias=True)
        self.dt_proj.weight.data = src.dt_proj.weight.data[keep_channels].clone()
        self.dt_proj.bias.data = src.dt_proj.bias.data[keep_channels].clone()
        self.A_log = nn.Parameter(src.A_log.data[keep_channels][:, keep_states].clone())
        self.D = nn.Parameter(src.D.data[keep_channels].clone())
        self.out_proj = nn.Linear(di, src.d_model, bias=False)
        self.out_proj.weight.data = src.out_proj.weight.data[:, keep_channels].clone()


def prune_channels(encoder: PCSTE, scores: dict, score_name: str,
                   keep_fraction: float | None = None,
                   keep_n: int | None = None) -> PCSTE:
    """Structural d_inner pruning of EVERY direction block by the named
    importance score (keep the top-k channels). keep_fraction 0.75/0.5
    for the 25/50 % maps; keep_n=320 for the fallback 384->320."""
    m = copy.deepcopy(encoder)
    for i, layer in enumerate(m.temporal.layers):
        for d in ("fwd", "bwd"):
            if not hasattr(layer, d):
                continue
            blk = getattr(layer, d)
            sc = scores[f"temporal.layers.{i}.{d}"][score_name]
            k = keep_n if keep_n is not None else int(round(
                blk.d_inner * keep_fraction))
            keep = torch.topk(sc, k).indices
            setattr(layer, d, PrunedMambaRefBlock(blk, keep))
    return m


@torch.no_grad()
def state_decay_activity(encoder: PCSTE, rep_fn, train_ids_by_ds: dict,
                         device: str = "cpu", chunk: int = 32,
                         max_per_dataset: int | None = None) -> dict:
    """Per block, per state s: mean over TRAIN windows/time/channels of
    |exp(Delta A[:, s]) - 1| ... i.e. how much the state actually decays/
    integrates (PerfMamba-style activity); states with ~zero activity
    (exp(Delta A) ~ 1 everywhere) carry no selective dynamics."""
    ids = [w for ds in DATASETS for w in train_ids_by_ds.get(ds, [])]
    assert_no_test_windows(ids, "state activity")
    for w in ids:
        if split_of_window_id(w) != "train":
            raise Part6GuardError(f"state activity needs TRAIN windows: {w}")
    blocks = {f"temporal.layers.{i}.{d}": getattr(l, d)
              for i, l in enumerate(encoder.temporal.layers)
              for d in ("fwd", "bwd") if hasattr(l, d)}
    acc = {k: torch.zeros(b.d_state) for k, b in blocks.items()}
    cnt = {k: 0 for k in blocks}
    hooks = []

    def mk(k, b):
        def hook(mod, a, out):
            delta = F.softplus(out.detach())                 # (N,T,d_inner)
            a_mat = -torch.exp(b.A_log.detach())             # (d_inner, s)
            decay = torch.exp(delta.unsqueeze(-1) * a_mat)   # (N,T,d,s)
            acc[k] += (1.0 - decay).mean(dim=(0, 1, 2)).cpu()
            cnt[k] += 1
        return hook
    for k, b in blocks.items():
        hooks.append(b.dt_proj.register_forward_hook(mk(k, b)))
    encoder.eval()
    try:
        for ds in DATASETS:
            wids = train_ids_by_ds.get(ds, [])
            if max_per_dataset:
                wids = wids[:max_per_dataset]
            for lo in range(0, len(wids), chunk):
                batch = collate_representations([rep_fn(w) for w in wids[lo:lo + chunk]])
                batch = {kk: v.to(device) for kk, v in batch.items()}
                encoder(**batch)
    finally:
        for h in hooks:
            h.remove()
    return {k: acc[k] / max(cnt[k], 1) for k in blocks}


def prune_states(encoder: PCSTE, activity: dict, keep_n: int = 8) -> PCSTE:
    """d_state 16 -> keep_n (8 map, 12 fallback) by decay activity."""
    m = copy.deepcopy(encoder)
    for i, layer in enumerate(m.temporal.layers):
        for d in ("fwd", "bwd"):
            if not hasattr(layer, d):
                continue
            blk = getattr(layer, d)
            act = activity[f"temporal.layers.{i}.{d}"]
            keep_s = torch.topk(act, keep_n).indices
            setattr(layer, d, PrunedMambaRefBlock(
                blk, torch.arange(blk.d_inner), keep_s))
    return m


# ---------------------------------------------------------------------------
# half-student comparison framework (pre-registered rule)
# ---------------------------------------------------------------------------
def half_student_2x2(encoder: PCSTE, retained_layers: list[int]) -> PCSTE:
    """Training-free '2 layers x 2 directions' from a full model."""
    sd, _ = student_state_from_full(encoder.state_dict(), retained_layers)
    m = build_encoder(STUDENT_D_SPEC)
    m.load_state_dict(sd, strict=True)
    return m


def half_student_4x1(encoder: PCSTE, keep: str = "fwd",
                     residual: str = "mean_of_remaining") -> PCSTE:
    m = copy.deepcopy(encoder)
    drop = "bwd" if keep == "fwd" else "fwd"
    for i in range(len(m.temporal.layers)):
        m = drop_direction(m, i, drop, residual)
    return m


def choose_half_student(val_f1_2x2: list[float], val_f1_4x1: list[float]
                        ) -> dict:
    """Pre-registered rule: choose by MEDIAN validation MacroDomainF1
    after training-free removal (higher wins; exact tie -> 2x2, the
    a-priori primary). Inputs: one value per (fold, seed) S1 checkpoint."""
    m22 = float(np.median(val_f1_2x2))
    m41 = float(np.median(val_f1_4x1))
    return {"median_2x2": m22, "median_4x1": m41,
            "chosen": "2x2" if m22 >= m41 else "4x1",
            "rule": "median validation MacroDomainF1 after training-free "
                    "removal; tie -> 2x2 (a-priori primary)"}


def fallback_trigger(median_drop: float, threshold: float) -> dict:
    """Fallback pruning ratios (384->320, 16->12) trigger only if the
    median validation drop of the chosen half student exceeds the sealed
    threshold (a PENDING numeric decision passed explicitly)."""
    return {"median_drop": median_drop, "threshold": threshold,
            "fallback_active": bool(median_drop > threshold),
            "fallbacks": {"d_inner": 320, "d_state": 12}}
