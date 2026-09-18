"""Part-6 trainer machinery (NO epoch loop here — see the CLI script).

Reuses the frozen recipe verbatim from experiment.trainers:
  make_optimizer -> AdamW(3e-4, (0.9,0.95), 1e-8, wd 0.05) over
  encoder+heads (single LR), lr_lambda (warm-up 5 -> cosine 1e-6),
  DOWNSTREAM_EPOCHS 50, EFFECTIVE_BATCH 64, grad clip 1.0, checkpoint =
  max MacroDomainF1_val with strict '>' (ties keep the earlier epoch).
No Part-6 code path accepts an override of any of these values.

Dataset-bucketed micro-batching: the 64-window step (16 per dataset in
the frozen sampler order) is split into ONE micro-batch per dataset so
HIT collates to its own 17-band grid instead of being padded to 33 in a
mixed chunk. Each micro-batch's loss (that dataset's window mean) is
back-propagated with weight 1/n_datasets_present, so the accumulated
gradient equals that of the full-batch 'mean of per-dataset means'
exactly — the same identity the primary executor relies on with its
2-dataset chunks (test: gradient equivalence on toy data).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..encoder import collate_representations
from ..experiment.heads import CLASS_ORDERS, DatasetHeads
from ..experiment.metrics import macro_domain_f1
from ..experiment.trainers import (DOWNSTREAM_EPOCHS, EFFECTIVE_BATCH,
                                   OPTIMIZER_SPEC, lr_lambda, make_optimizer)
from .guards import Part6GuardError, assert_no_test_windows
from .losses import LossConfig, part6_loss
from .student import (EncoderWithHeads, StudentSpec, build_encoder,
                      build_heads, load_half4x1_from_full,
                      load_heads_from_full, load_student_from_full)
from .teachers import AlphaRecorder, TeacherCache, ensemble_soft_targets

DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")


@dataclass
class ArmConfig:
    arm: str
    fold: int
    seed: int
    spec: StudentSpec
    loss: LossConfig
    init_source: str | None            # "s1" | "s0" | "ssl" | None
    teacher_set: str | None            # "s1" | "s0" | "s1_l010" | None
    retained_layers: list | None = None   # Student-D (2x2) surgery mapping
    kept_direction: str | None = None     # half_4x1 surgery mapping
    head_init_seed: int | None = None

    def to_dict(self) -> dict:
        return {"arm": self.arm, "fold": self.fold, "seed": self.seed,
                "architecture": self.spec.to_dict(),
                "loss": self.loss.to_dict(), "init_source": self.init_source,
                "teacher_set": self.teacher_set,
                "retained_layers": self.retained_layers,
                "kept_direction": self.kept_direction,
                "head_init_seed": self.head_init_seed}


class Part6Trainer:
    """Student/full model + frozen optimizer + bucketed step + predict."""

    def __init__(self, cfg: ArmConfig, device: str = "cpu",
                 init_encoder_state: dict | None = None,
                 init_heads_state: dict | None = None,
                 teacher_cache: TeacherCache | None = None):
        cfg.loss.validate()
        self.cfg = cfg
        self.device = device
        torch.manual_seed(cfg.seed)                    # same as primary
        self.encoder = build_encoder(cfg.spec)
        if cfg.head_init_seed is None:
            raise Part6GuardError("head_init_seed required (paired rule)")
        self.heads = build_heads(cfg.spec, cfg.head_init_seed)
        self.surgery_report = None
        if init_encoder_state is not None:
            if cfg.spec.name == "student_d":
                if cfg.retained_layers is None:
                    raise Part6GuardError("Student-D init needs retained_layers")
                self.surgery_report = load_student_from_full(
                    self.encoder, init_encoder_state, cfg.retained_layers)
            elif cfg.spec.name == "half_4x1":
                if cfg.kept_direction not in ("fwd", "bwd"):
                    raise Part6GuardError("half_4x1 init needs kept_direction")
                self.surgery_report = load_half4x1_from_full(
                    self.encoder, init_encoder_state, cfg.kept_direction)
            elif cfg.spec.name == "full":
                res = self.encoder.load_state_dict(init_encoder_state, strict=True)
                assert not res.missing_keys and not res.unexpected_keys
            else:
                raise Part6GuardError(f"no init path for {cfg.spec.name}")
        if init_heads_state is not None:
            load_heads_from_full(self.heads, init_heads_state)
        self.model = EncoderWithHeads(self.encoder, self.heads).to(device)
        self.optimizer = make_optimizer(self.model)     # frozen recipe
        self.params = list(self.model.parameters())
        self.teacher_cache = teacher_cache
        needs_teacher = cfg.loss.kind in ("kd_ensemble",
                                          "kd_ensemble+relational",
                                          "fewshot_kd")
        if needs_teacher and teacher_cache is None:
            raise Part6GuardError(f"{cfg.arm}: loss needs a teacher cache")
        if not needs_teacher and teacher_cache is not None:
            raise Part6GuardError(f"{cfg.arm}: hard-label arm must not "
                                  "receive a teacher cache")
        self._alpha_rec = None
        if cfg.loss.kind == "kd_ensemble+relational":
            self._alpha_rec = AlphaRecorder(self.encoder.mixer)

    # -- frozen recipe helpers -------------------------------------------
    @staticmethod
    def scheduler_for(optimizer, steps_per_epoch: int):
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda(DOWNSTREAM_EPOCHS, steps_per_epoch))

    @staticmethod
    def frozen_settings() -> dict:
        return {"optimizer": dict(OPTIMIZER_SPEC), "epochs": DOWNSTREAM_EPOCHS,
                "effective_batch": EFFECTIVE_BATCH,
                "checkpoint_rule": "max MacroDomainF1_val, strict >, "
                                   "tie -> earlier epoch, no early stopping"}

    @staticmethod
    def is_better(candidate: float, best: float) -> bool:
        return candidate > best                     # strict: ties keep earlier

    # -- one micro-batch (single dataset) --------------------------------
    def _loss_single_dataset(self, ds: str, reps: list, classes: list[str],
                             window_ids: list[str],
                             labelled_mask: list[bool] | None = None
                             ) -> tuple[torch.Tensor, dict]:
        batch = collate_representations(reps)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.encoder(**batch)
        z = out["global_embedding"]
        logits = self.heads(z, ds)
        y = torch.tensor([CLASS_ORDERS[ds].index(c) for c in classes],
                         device=self.device)
        kw = {}
        kind = self.cfg.loss.kind
        if kind in ("kd_ensemble", "kd_ensemble+relational", "fewshot_kd"):
            per_seed = self.teacher_cache.per_seed_logits(ds, window_ids)
            pT = ensemble_soft_targets(per_seed.to(self.device),
                                       self.cfg.loss.ensemble_rule,
                                       self.cfg.loss.temperature)
            kw["teacher_probs_by_ds"] = {ds: pT}
        if kind == "kd_ensemble+relational":
            a_t = self.teacher_cache.single_alpha(self.cfg.seed, window_ids)
            bm = out["band_mask"]
            kw["alpha_student_by_ds"] = {ds: self._alpha_rec.alpha}
            kw["alpha_teacher_by_ds"] = {ds: a_t.to(self.device)}
            kw["band_mask_by_ds"] = {ds: bm}
        if kind == "fewshot_kd":
            if labelled_mask is None:
                raise Part6GuardError("fewshot_kd needs labelled_mask")
            kw["labelled_mask_by_ds"] = {ds: torch.tensor(labelled_mask,
                                                          device=self.device)}
        return part6_loss(self.cfg.loss, {ds: logits}, {ds: y}, **kw)

    def compute_loss(self, reps: list, triples: list[tuple[str, str, str]],
                     labelled_mask: list[bool] | None = None
                     ) -> tuple[torch.Tensor, dict]:
        """Full-batch loss (all datasets in one graph) — used by tests
        for the equivalence proof; the training step uses bucketing."""
        assert_no_test_windows([w for _, _, w in triples], "train batch")
        losses, terms = [], {}
        for ds in sorted(CLASS_ORDERS):
            idx = [i for i, (d, _, _) in enumerate(triples) if d == ds]
            if not idx:
                continue
            l, t = self._loss_single_dataset(
                ds, [reps[i] for i in idx], [triples[i][1] for i in idx],
                [triples[i][2] for i in idx],
                None if labelled_mask is None else [labelled_mask[i] for i in idx])
            losses.append(l)
            terms[ds] = t
        return torch.stack(losses).mean(), terms

    def train_step_bucketed(self, reps: list,
                            triples: list[tuple[str, str, str]],
                            scheduler=None,
                            labelled_mask: list[bool] | None = None,
                            per_dataset_micro: int | None = None) -> float:
        """One optimizer step over the effective batch, one micro-batch
        per dataset (optionally further split within a dataset — every
        split keeps the exact per-dataset mean by weighting with
        n_split/n_ds)."""
        assert_no_test_windows([w for _, _, w in triples], "train batch")
        self.optimizer.zero_grad()
        present = sorted({d for d, _, _ in triples})
        total = 0.0
        for ds in present:
            idx = [i for i, (d, _, _) in enumerate(triples) if d == ds]
            n_ds = len(idx)
            step = per_dataset_micro or n_ds
            for lo in range(0, n_ds, step):
                sub = idx[lo:lo + step]
                l, _ = self._loss_single_dataset(
                    ds, [reps[i] for i in sub], [triples[i][1] for i in sub],
                    [triples[i][2] for i in sub],
                    None if labelled_mask is None
                    else [labelled_mask[i] for i in sub])
                w = (len(sub) / n_ds) / len(present)
                (l * w).backward()
                total += float(l.detach()) * w
        torch.nn.utils.clip_grad_norm_(
            self.params, OPTIMIZER_SPEC["grad_clip_global_norm"])
        self.optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return total

    @torch.no_grad()
    def predict(self, reps: list, ds_of: list[str]) -> list[str]:
        was_training = self.model.training
        self.model.eval()
        try:
            batch = collate_representations(reps)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            z = self.encoder(**batch)["global_embedding"]
            preds = []
            for i, ds in enumerate(ds_of):
                logits = self.heads(z[i:i + 1], ds)
                preds.append(CLASS_ORDERS[ds][int(logits.argmax())])
            return preds
        finally:
            self.model.train(was_training)

    @staticmethod
    def checkpoint_metric(reports: dict) -> float:
        return macro_domain_f1(reports)

    def encoder_state(self) -> dict:
        return {k: v.detach().cpu().clone()
                for k, v in self.encoder.state_dict().items()}

    def heads_state(self) -> dict:
        return {k: v.detach().cpu().clone()
                for k, v in self.heads.state_dict().items()}
