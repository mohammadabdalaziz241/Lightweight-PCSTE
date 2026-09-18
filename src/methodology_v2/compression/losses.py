"""Part-6 losses — exactly the plan's forms, nothing tuned.

Per window (own dataset head only, class logits z_s student, teacher soft
targets p_T from the same-fold ensemble at temperature T):

  L_KD(window) = (1 - alpha) * CE(z_s, y)
               + alpha * T^2 * KL( p_T || softmax(z_s / T) )
  KL(p || q)   = sum_c p_c (log p_c - log q_c)   (teacher || student,
                 forward KL, summed over classes)
  T = 4, alpha = 0.5 (fixed a priori)

Relational term (K1/K0; weight lambda_rel is a PENDING decision, passed
explicitly — no default value exists in code):
  L_rel(window) = KL( alpha_teacher || alpha_student ) over VALID bands
  (both distributions renormalised over the valid bands; invalid/padded
  bands are excluded from the sum, never treated as zero-probability
  events; teacher = same-cell single S1/S0 model, cached).

Reduction (identical to the primary L_sup): per-dataset MEAN over the
windows of that dataset present in the batch, then MEAN over the
datasets present — every dataset weighs 25 % regardless of composition.
CE and KD share the same reduction; L_rel is reduced the same way and
added with weight lambda_rel.

C_small / P1 : hard-label CE only (same reduction as the primary).
B0           : CE with label_smoothing = 0.1 (torch semantics).
F1           : (1 - alpha) * CE on labelled windows + alpha * T^2 * KL on
               all windows (teacher = registered 10 % S1 cell).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..experiment.heads import CLASS_ORDERS
from .guards import Part6GuardError
from .protocol import (KD_ALPHA, KD_TEMPERATURE, LABEL_SMOOTHING_B0)


@dataclass(frozen=True)
class LossConfig:
    kind: str                       # ce_hard | kd_ensemble | kd_ensemble+relational | ce_label_smoothing | fewshot_kd
    temperature: float = KD_TEMPERATURE
    alpha: float = KD_ALPHA
    relational_weight: float | None = None   # REQUIRED for +relational
    label_smoothing: float = 0.0
    ensemble_rule: str | None = None         # REQUIRED for kd kinds

    def validate(self) -> "LossConfig":
        if self.kind not in ("ce_hard", "kd_ensemble",
                             "kd_ensemble+relational", "ce_label_smoothing",
                             "fewshot_kd"):
            raise Part6GuardError(f"unknown loss kind {self.kind}")
        if self.kind.startswith("kd") or self.kind == "fewshot_kd":
            if self.temperature != KD_TEMPERATURE or self.alpha != KD_ALPHA:
                raise Part6GuardError(
                    "T and alpha are fixed a priori (4.0, 0.5) — refusing "
                    f"T={self.temperature}, alpha={self.alpha}")
            if self.ensemble_rule is None:
                raise Part6GuardError("kd loss requires an explicit "
                                      "ensemble_rule (pending decision)")
        if self.kind == "kd_ensemble+relational" and self.relational_weight is None:
            raise Part6GuardError(
                "relational_weight is a PENDING pre-registration decision: "
                "pass it explicitly (no default exists)")
        if self.kind == "ce_label_smoothing" and self.label_smoothing != LABEL_SMOOTHING_B0:
            raise Part6GuardError("B0 label smoothing is fixed at 0.1")
        if self.kind == "ce_hard" and self.label_smoothing != 0.0:
            raise Part6GuardError("ce_hard must not smooth labels")
        return self

    def to_dict(self) -> dict:
        return {"kind": self.kind, "temperature": self.temperature,
                "alpha": self.alpha, "relational_weight": self.relational_weight,
                "label_smoothing": self.label_smoothing,
                "ensemble_rule": self.ensemble_rule,
                "kl_direction": "KL(teacher_T || student_T), sum over classes",
                "reduction": "mean over datasets of per-dataset window means"}


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def kd_kl_term(student_logits: torch.Tensor, teacher_probs_T: torch.Tensor,
               temperature: float = KD_TEMPERATURE) -> torch.Tensor:
    """T^2 * KL(p_T || softmax(z_s/T)) per window -> (B,)."""
    log_q = F.log_softmax(student_logits / temperature, dim=-1)
    p = teacher_probs_T
    kl = (p * (torch.log(p.clamp_min(1e-12)) - log_q)).sum(-1)
    return (temperature ** 2) * kl


def alpha_relational_kl(alpha_teacher: torch.Tensor,
                        alpha_student: torch.Tensor,
                        band_mask: torch.Tensor,
                        eps: float = 1e-12) -> torch.Tensor:
    """KL(alpha_t || alpha_s) over valid bands per window -> (B,).
    Both renormalised over the valid bands (they already sum to 1 over
    valid bands by construction; renormalising makes the term robust to
    padded/zero entries and to different padded widths)."""
    m = band_mask.to(alpha_teacher.dtype)
    if alpha_student.shape[1] != alpha_teacher.shape[1]:
        # pad the narrower to the wider (padded bands are masked anyway)
        w = max(alpha_student.shape[1], alpha_teacher.shape[1])
        alpha_student = F.pad(alpha_student, (0, w - alpha_student.shape[1]))
        alpha_teacher = F.pad(alpha_teacher, (0, w - alpha_teacher.shape[1]))
        m = F.pad(m, (0, w - m.shape[1]))
    at = alpha_teacher * m
    as_ = alpha_student * m
    at = at / at.sum(-1, keepdim=True).clamp_min(eps)
    as_ = as_ / as_.sum(-1, keepdim=True).clamp_min(eps)
    kl = at * (torch.log(at.clamp_min(eps)) - torch.log(as_.clamp_min(eps)))
    return (kl * m).sum(-1)


def per_dataset_mean_of_means(per_window: dict[str, torch.Tensor]
                              ) -> torch.Tensor:
    """{dataset: (B_ds,)} -> mean over datasets of per-dataset means."""
    if not per_window:
        raise Part6GuardError("empty batch")
    return torch.stack([v.mean() for _, v in sorted(per_window.items())]).mean()


def class_targets(ds: str, classes: list[str], device) -> torch.Tensor:
    order = CLASS_ORDERS[ds]
    return torch.tensor([order.index(c) for c in classes], device=device)


# ---------------------------------------------------------------------------
# full loss for one micro-batch of ONE OR MORE datasets
# ---------------------------------------------------------------------------
def part6_loss(cfg: LossConfig, logits_by_ds: dict[str, torch.Tensor],
               targets_by_ds: dict[str, torch.Tensor],
               teacher_probs_by_ds: dict[str, torch.Tensor] | None = None,
               alpha_student_by_ds: dict[str, torch.Tensor] | None = None,
               alpha_teacher_by_ds: dict[str, torch.Tensor] | None = None,
               band_mask_by_ds: dict[str, torch.Tensor] | None = None,
               labelled_mask_by_ds: dict[str, torch.Tensor] | None = None,
               ) -> tuple[torch.Tensor, dict]:
    """Returns (loss, per-term dict). Every dict is keyed by dataset and
    holds only that dataset's windows (own-head logits)."""
    cfg.validate()
    ce_terms, kd_terms, rel_terms = {}, {}, {}
    for ds, z in logits_by_ds.items():
        y = targets_by_ds[ds]
        if z.shape[-1] != len(CLASS_ORDERS[ds]):
            raise Part6GuardError(f"{ds}: logits are not from the {ds} head")
        if cfg.kind == "ce_label_smoothing":
            ce = F.cross_entropy(z, y, reduction="none",
                                 label_smoothing=cfg.label_smoothing)
        else:
            ce = F.cross_entropy(z, y, reduction="none")
        if cfg.kind == "fewshot_kd":
            lm = labelled_mask_by_ds[ds].to(ce.dtype)
            # CE only over the labelled windows of this dataset (if any)
            ce_terms[ds] = (ce * lm).sum() / lm.sum().clamp_min(1.0) \
                if lm.sum() > 0 else None
        else:
            ce_terms[ds] = ce
        if cfg.kind in ("kd_ensemble", "kd_ensemble+relational", "fewshot_kd"):
            if teacher_probs_by_ds is None or ds not in teacher_probs_by_ds:
                raise Part6GuardError(f"{ds}: teacher soft targets missing")
            kd_terms[ds] = kd_kl_term(z, teacher_probs_by_ds[ds],
                                      cfg.temperature)
        if cfg.kind == "kd_ensemble+relational":
            if (alpha_student_by_ds is None or alpha_teacher_by_ds is None
                    or band_mask_by_ds is None):
                raise Part6GuardError("relational term needs student/teacher "
                                      "alpha and band masks")
            rel_terms[ds] = alpha_relational_kl(
                alpha_teacher_by_ds[ds], alpha_student_by_ds[ds],
                band_mask_by_ds[ds])
    terms = {}
    if cfg.kind in ("ce_hard", "ce_label_smoothing"):
        loss = per_dataset_mean_of_means(ce_terms)
        terms["ce"] = float(loss.detach())
        return loss, terms
    if cfg.kind == "fewshot_kd":
        ce_present = {ds: v.unsqueeze(0) for ds, v in ce_terms.items()
                      if v is not None}
        kd = per_dataset_mean_of_means(kd_terms)
        if ce_present:
            ce = per_dataset_mean_of_means(ce_present)
            loss = (1 - cfg.alpha) * ce + cfg.alpha * kd
            terms["ce"] = float(ce.detach())
        else:
            loss = cfg.alpha * kd
            terms["ce"] = None
        terms["kd"] = float(kd.detach())
        return loss, terms
    ce = per_dataset_mean_of_means(ce_terms)
    kd = per_dataset_mean_of_means(kd_terms)
    loss = (1 - cfg.alpha) * ce + cfg.alpha * kd
    terms.update({"ce": float(ce.detach()), "kd": float(kd.detach())})
    if cfg.kind == "kd_ensemble+relational":
        rel = per_dataset_mean_of_means(rel_terms)
        loss = loss + cfg.relational_weight * rel
        terms["relational"] = float(rel.detach())
    return loss, terms
