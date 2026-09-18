"""Frozen trainer infrastructure for S0/S1 — NO real training here.

Both trainers expose step-level methods so Part-5D smoke tests can run
1-2 bounded steps; the real 60/50-epoch matrix is launched only after
explicit human authorization, driven by the sealed run registry.

FROZEN OPTIMIZER RECIPES (identical for SSL and downstream, and
identical between S0 and S1 downstream — single LR for encoder+heads,
no discriminative LRs, no early stopping, retrospective validation
checkpoint selection):
  AdamW(lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.05)
  grad clip global-norm 1.0
  linear warm-up 5 epochs -> cosine decay to min lr 1e-6
  SSL: 60 epochs; downstream: 50 epochs
  steps_per_epoch = ceil(full TRAIN windows of the fold / 64) for BOTH
  SSL and every label fraction (fixed compute across fractions; low
  fractions revisit labelled windows via replacement sampling).
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoder import PCSTE, collate_representations
from ..encoder.ssl_design import (ReconstructionProbe, generate_mask,
                                  window_rng)
from .heads import CLASS_ORDERS, DatasetHeads
from .metrics import (classification_report, macro_domain_f1,
                      macro_domain_recon_mse)

OPTIMIZER_SPEC = {"optimizer": "AdamW", "lr": 3e-4, "betas": (0.9, 0.95),
                  "eps": 1e-8, "weight_decay": 0.05,
                  "grad_clip_global_norm": 1.0, "min_lr": 1e-6,
                  "warmup_epochs": 5}
SSL_EPOCHS = 60
DOWNSTREAM_EPOCHS = 50
EFFECTIVE_BATCH = 64
MASK_RATIO = 0.60
MASK_GEOMETRY = "M1_random"


def make_optimizer(module: nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(module.parameters(),
                             lr=OPTIMIZER_SPEC["lr"],
                             betas=OPTIMIZER_SPEC["betas"],
                             eps=OPTIMIZER_SPEC["eps"],
                             weight_decay=OPTIMIZER_SPEC["weight_decay"])


def lr_lambda(max_epochs: int, steps_per_epoch: int):
    warm = OPTIMIZER_SPEC["warmup_epochs"] * steps_per_epoch
    total = max_epochs * steps_per_epoch
    floor = OPTIMIZER_SPEC["min_lr"] / OPTIMIZER_SPEC["lr"]

    def fn(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(total - warm, 1)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))
    return fn


def steps_per_epoch(n_train_windows: int) -> int:
    return math.ceil(n_train_windows / EFFECTIVE_BATCH)


def validation_mask_seed(seed: int) -> int:
    """Fixed validation-mask seed derived from the experimental seed —
    the SAME validation window hides the SAME patches at every epoch."""
    return int.from_bytes(
        hashlib.sha256(f"valmask|{seed}".encode()).digest()[:4], "little")


def build_patch_mask(items_meta: list[tuple[str, str]], grids: dict,
                     seed: int, epoch: int,
                     fixed_validation: bool) -> torch.Tensor:
    """(B, F_max, T_max) SSL mask. TRAIN masks vary by (seed, epoch,
    window); validation masks use (validation_mask_seed(seed), epoch=0)
    so they are constant across epochs."""
    fmax = max(g[0] for g in grids.values())
    tmax = max(g[1] for g in grids.values())
    pm = torch.zeros(len(items_meta), fmax, tmax, dtype=torch.bool)
    for i, (ds, wid) in enumerate(items_meta):
        fb, tp = grids[ds]
        tv = np.ones((fb, tp), dtype=bool)
        rng = (window_rng(validation_mask_seed(seed), 0, wid)
               if fixed_validation else window_rng(seed, epoch, wid))
        m = generate_mask(tv, MASK_RATIO, MASK_GEOMETRY, rng)
        pm[i, :fb, :tp] = torch.from_numpy(m)
    return pm


GRIDS = {"CWRU": (33, 23), "JNU": (33, 24), "HIT": (17, 24),
         "MAFAULDA": (33, 24)}


class SSLTrainer:
    """Masked-reconstruction pretraining machinery (frozen Part-5C
    objective). Provides train_step/validation_metrics; NO epoch loop
    is executed in Part 5D."""

    def __init__(self, seed: int, device: str = "cpu"):
        torch.manual_seed(seed)
        self.encoder = PCSTE()
        self.model = ReconstructionProbe(self.encoder).to(device)
        self.optimizer = make_optimizer(self.model)
        self.seed = seed
        self.device = device

    def train_step(self, reps: list, metas: list[tuple[str, str]],
                   epoch: int, scheduler=None,
                   micro_batch: int = EFFECTIVE_BATCH) -> float:
        """One optimizer step over the EFFECTIVE batch. If micro_batch <
        len(reps), gradients are accumulated over equal-size chunks —
        a mechanical memory adjustment that preserves the exact
        batch-mean loss (mean of equal-size chunk means)."""
        n = len(reps)
        self.optimizer.zero_grad()
        total = 0.0
        for lo in range(0, n, micro_batch):
            hi = min(lo + micro_batch, n)
            batch = collate_representations(reps[lo:hi])
            batch = {k: v.to(self.device) for k, v in batch.items()}
            pm = build_patch_mask(metas[lo:hi], GRIDS, self.seed, epoch,
                                  fixed_validation=False).to(self.device)
            out = self.model(**batch, patch_mask=pm)
            w = (hi - lo) / n
            (out["loss"] * w).backward()
            total += float(out["loss"].detach()) * w
            del out, batch, pm
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            OPTIMIZER_SPEC["grad_clip_global_norm"])
        self.optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return total

    @torch.no_grad()
    def validation_mses(self, reps: list,
                        metas: list[tuple[str, str]]) -> dict:
        """Per-dataset window-mean masked-cell MSE with FIXED masks."""
        batch = collate_representations(reps)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        pm = build_patch_mask(metas, GRIDS, self.seed, 0,
                              fixed_validation=True).to(self.device)
        out = self.model(**batch, patch_mask=pm)
        per_ds: dict[str, list[float]] = {}
        for i, (ds, _) in enumerate(metas):
            per_ds.setdefault(ds, []).append(
                float(out["per_window_mse"][i]))
        return {ds: float(np.mean(v)) for ds, v in per_ds.items()}

    def checkpoint_metric(self, per_dataset_mse: dict) -> float:
        return macro_domain_recon_mse(per_dataset_mse)   # minimize

    def encoder_state(self) -> dict:
        return {k: v.detach().cpu().clone()
                for k, v in self.encoder.state_dict().items()}


class SupervisedTrainer:
    """S0/S1 downstream machinery. IDENTICAL for both arms except the
    encoder initialization source; heads initialized from the shared
    deterministic head seed of the paired run."""

    def __init__(self, seed: int, head_init_seed: int,
                 encoder_state: dict | None = None,
                 device: str = "cpu"):
        torch.manual_seed(seed)
        self.encoder = PCSTE()
        if encoder_state is not None:                  # S1 arm
            self.encoder.load_state_dict(encoder_state)
        self.heads = DatasetHeads(init_seed=head_init_seed)
        self.encoder.to(device)
        self.heads.to(device)
        self.params = list(self.encoder.parameters()) \
            + list(self.heads.parameters())
        self.optimizer = torch.optim.AdamW(
            self.params, lr=OPTIMIZER_SPEC["lr"],
            betas=OPTIMIZER_SPEC["betas"], eps=OPTIMIZER_SPEC["eps"],
            weight_decay=OPTIMIZER_SPEC["weight_decay"])
        self.device = device

    def compute_loss(self, reps: list,
                     triples: list[tuple[str, str, str]]) -> torch.Tensor:
        """L_sup = mean over datasets of that dataset's mean CE —
        equal dataset influence regardless of batch composition."""
        batch = collate_representations(reps)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        z = self.encoder(**batch)["global_embedding"]
        losses = []
        for ds in sorted(CLASS_ORDERS):
            idx = [i for i, (d, _, _) in enumerate(triples) if d == ds]
            if not idx:
                continue
            logits = self.heads(z[idx], ds)
            classes = CLASS_ORDERS[ds]
            y = torch.tensor([classes.index(triples[i][1])
                              for i in idx], device=self.device)
            losses.append(F.cross_entropy(logits, y))
        return torch.stack(losses).mean()

    def train_step(self, reps, triples, scheduler=None,
                   micro_batch: int = EFFECTIVE_BATCH) -> float:
        """One optimizer step. Micro-batching chunks the DATASET-ORDERED
        batch (16 per dataset) so every chunk holds whole datasets; the
        accumulated loss mean(chunk losses weighted by datasets/chunk)
        equals the full-batch mean over the four dataset means exactly."""
        n = len(reps)
        self.optimizer.zero_grad()
        total = 0.0
        n_ds_total = len({d for d, _, _ in triples})
        for lo in range(0, n, micro_batch):
            hi = min(lo + micro_batch, n)
            loss = self.compute_loss(reps[lo:hi], triples[lo:hi])
            k = len({d for d, _, _ in triples[lo:hi]})
            w = k / n_ds_total
            (loss * w).backward()
            total += float(loss.detach()) * w
            del loss
        torch.nn.utils.clip_grad_norm_(
            self.params, OPTIMIZER_SPEC["grad_clip_global_norm"])
        self.optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return total

    @torch.no_grad()
    def predict(self, reps, ds_of: list[str]) -> list[str]:
        batch = collate_representations(reps)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        z = self.encoder(**batch)["global_embedding"]
        preds = []
        for i, ds in enumerate(ds_of):
            logits = self.heads(z[i:i + 1], ds)
            preds.append(CLASS_ORDERS[ds][int(logits.argmax())])
        return preds

    @staticmethod
    def checkpoint_metric(reports: dict[str, dict]) -> float:
        return macro_domain_f1(reports)                 # maximize;
        # exact ties resolved by choosing the EARLIER epoch (registry)
