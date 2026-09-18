"""Part-6 protocol constants, paths, spec-file I/O and pending decisions.

Everything numeric that the lightweight plan
(docs/methodology_v2_lightweight_plan.md) FIXES a priori lives here as a
module constant. Everything the plan leaves genuinely unresolved lives in
PENDING_DECISIONS with status "PENDING_PREREG" and a recommendation; the
seal command refuses to seal while any pending decision is unresolved,
and no code path invents a value silently (tests pass values
explicitly).

Spec files under methodology_v2/part6_compression/ follow the repository
convention (JSON content, .yaml extension, sort_keys, indent=1) so the
existing sha256 seal helpers apply unchanged.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..integrity import sha256_file
from ..registry import REPO_ROOT
from ..experiment.label_subsets import FOLDS, SEEDS  # (1,2,3), (42,1337,2026)
from ..experiment.trainers import (DOWNSTREAM_EPOCHS, EFFECTIVE_BATCH,
                                   OPTIMIZER_SPEC)

PART6_VERSION = "part6.v1"
PART6_DIR = REPO_ROOT / "methodology_v2" / "part6_compression"
PART6_RESULTS = REPO_ROOT / "results" / "methodology_v2" / "part6_compression"
PRIMARY_RESULTS = REPO_ROOT / "results" / "methodology_v2"
PRIMARY_DOWNSTREAM = PRIMARY_RESULTS / "downstream"
PRIMARY_SSL = PRIMARY_RESULTS / "ssl"

# ---------------------------------------------------------------------------
# FIXED a priori by the plan (never tuned, never moved after results)
# ---------------------------------------------------------------------------
KD_TEMPERATURE = 4.0          # T
KD_ALPHA = 0.5                # alpha in (1-alpha)*CE + alpha*T^2*KL
LABEL_SMOOTHING_B0 = 0.1      # B0 control
NI_MARGIN_ARCH = 0.02         # non-inferiority margin, architecture contrasts
NI_MARGIN_PTQ = 0.01          # non-inferiority margin, PTQ contrasts
PUSH_MIN_DELTA = 0.02         # a "push" needs mean delta >= 0.02 AND p < 0.05
PUSH_ALPHA = 0.05
N_SIGN_FLIPS = 2 ** 9         # 512 exact patterns for 9 paired cells
CONFIRMATORY_FAMILY_M = 3     # H1 K1 vs S1; H2 K1 vs C_small; H3 Q8(K1) vs K1
SECONDARY_FAMILY_M = 4        # K0 vs S0 NI; K1 vs K0; Q8(S1) vs S1 NI; Q8(S0) vs S0 NI

STUDENT_D_BLOCKS = 2          # Student-D: 2 BiMamba blocks (full model: 4)
FULL_BLOCKS = 4
D_MODEL = 192
D_INNER = 384
D_STATE = 16
STUDENT_DW = {"d_model": 128, "d_inner": 256, "n_blocks": 2}
FEW_SHOT_TEACHER_FRACTION = 0.10   # F1 teacher = registered 10%-label S1 cell

CORE_ARMS = ("k1", "c_small", "k0")            # Stage 3, 27 runs
OPTIONAL_ARMS = ("p1", "dw_k1")                # Stage 3 budget-gated
PUSH_ARMS = ("b1", "b0", "f1")                 # Stage 4, disabled by default
ALL_ARMS = CORE_ARMS + OPTIONAL_ARMS + PUSH_ARMS

TEACHER_SET_OF_ARM = {"k1": "s1", "k0": "s0", "b1": "s1", "dw_k1": "s1",
                      "f1": "s1_l010", "c_small": None, "p1": None,
                      "b0": None}
INIT_OF_ARM = {"k1": "s1", "k0": "s0", "p1": "s1", "c_small": None,
               "dw_k1": None, "b1": "ssl", "b0": "ssl", "f1": "ssl"}
ARCH_OF_ARM = {"k1": "student_d", "k0": "student_d", "c_small": "student_d",
               "p1": "student_d", "dw_k1": "student_dw", "b1": "full",
               "b0": "full", "f1": "full"}
LOSS_OF_ARM = {"k1": "kd_ensemble+relational", "k0": "kd_ensemble+relational",
               "c_small": "ce_hard", "p1": "ce_hard", "dw_k1": "kd_ensemble",
               "b1": "kd_ensemble", "b0": "ce_label_smoothing_0.1",
               "f1": "fewshot_kd"}

# frozen recipe re-exported (single source of truth stays in trainers.py)
FROZEN_RECIPE = {"optimizer": dict(OPTIMIZER_SPEC),
                 "epochs": DOWNSTREAM_EPOCHS,
                 "effective_batch": EFFECTIVE_BATCH,
                 "checkpoint_rule": "max MacroDomainF1_val; exact tie -> "
                                    "earlier epoch; no early stopping",
                 "steps_per_epoch_rule": "ceil(full fold TRAIN windows / 64)"}


# ---------------------------------------------------------------------------
# PENDING pre-registration decisions (plan leaves these unresolved)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PendingDecision:
    key: str
    question: str
    recommended: object
    alternatives: tuple
    rationale: str
    status: str = "PENDING_PREREG"
    value: object = None            # set only via resolve_pending()


PENDING_DECISIONS: dict[str, PendingDecision] = {
    "relational_alpha_kl_weight": PendingDecision(
        key="relational_alpha_kl_weight",
        question="Fixed weight lambda_rel of the relational term "
                 "KL(alpha_teacher || alpha_student) over valid mixer bands "
                 "(same-cell single teacher) added to the K1/K0 loss.",
        recommended=1.0,
        alternatives=(0.0, 0.1, 1.0),
        rationale="Plan §4 Stage 3 says 'fixed-weight' without a number. "
                  "The band-attention KL is bounded by log(33)=3.5 and is "
                  "typically O(1e-2..1e-1) for trained models, so weight 1.0 "
                  "keeps it a mild regulariser next to the CE/KD terms; 0.0 "
                  "recovers the pure Hinton loss. Must be fixed before "
                  "sealing; never tuned on validation."),
    "student_d_retained_layers": PendingDecision(
        key="student_d_retained_layers",
        question="Which two of the four teacher BiMamba layers initialise "
                 "Student-D layers (0, 1) for K1/K0/P1 (plan says 'kept "
                 "layers' without naming them).",
        recommended=[0, 2],
        alternatives=([0, 2], [0, 3], [0, 1], [2, 3]),
        rationale="[0, 2] = every-other-layer inheritance (DistilBERT-style "
                  "depth truncation); [0, 3] keeps the layer whose output "
                  "trained the final LayerNorm. Stage-2 drop-one-layer "
                  "sensitivity (validation-only, descriptive) may be cited "
                  "as explanation but the plan fixes the mapping a priori."),
    "ensemble_rule": PendingDecision(
        key="ensemble_rule",
        question="How the three same-fold teacher logit vectors are combined "
                 "into the soft target at temperature T.",
        recommended="mean_prob_at_T",
        alternatives=("mean_prob_at_T", "mean_logits"),
        rationale="mean_prob_at_T = mean_k softmax(z_k / T) (the ensemble "
                  "predictive distribution, Hinton 2015 / Lakshminarayanan "
                  "2017); mean_logits = softmax(mean_k z_k / T). Both are "
                  "implemented and deterministic; one must be sealed."),
    "student_dw_stem_rank": PendingDecision(
        key="student_dw_stem_rank",
        question="Rank r of the optional low-rank Student-DW patch stem "
                 "(Linear 1024->r->128 replacing Linear 1024->128).",
        recommended=None,
        alternatives=(None, 32, 64),
        rationale="Plan says '(+ low-rank stem)' without a rank. None = keep "
                  "the full stem (679,633 encoder params); r=64 -> ~622k; "
                  "r=32 -> ~588k. Optional arm; decide only if DW is run."),
    "half_student_direction_variant": PendingDecision(
        key="half_student_direction_variant",
        question="For the Stage-2 equal-cost alternative '4 layers x 1 "
                 "direction': which direction is kept and how the residual "
                 "is scaled after removing one direction.",
        recommended={"keep": "fwd", "residual": "mean_of_remaining"},
        alternatives=({"keep": "fwd", "residual": "mean_of_remaining"},
                      {"keep": "fwd", "residual": "keep_half_scale"},
                      {"keep": "bwd", "residual": "mean_of_remaining"}),
        rationale="mean_of_remaining: y = x + fwd(LN x) (mean over the one "
                  "remaining direction); keep_half_scale: y = x + 0.5*fwd(LN "
                  "x) (training-free removal without rescaling). Both are "
                  "reported descriptively by the sensitivity tool."),
    "stage2_fallback_threshold": PendingDecision(
        key="stage2_fallback_threshold",
        question="Fixed validation threshold that triggers the fallback "
                 "pruning ratios (d_inner 384->320, d_state 16->12).",
        recommended={"metric": "median validation MacroDomainF1 drop after "
                               "training-free removal (fold-wise, S1 "
                               "checkpoints)", "max_drop": 0.05},
        alternatives=(0.02, 0.05, 0.10),
        rationale="Plan §4 Stage 2 names the fallbacks but not the numeric "
                  "threshold. Recommendation: fallbacks activate only if the "
                  "chosen half student loses > 0.05 median validation "
                  "MacroDomainF1 training-free; otherwise unchanged."),
    "compact_student_variant": PendingDecision(
        key="compact_student_variant",
        question="Which equal-cost half student the K1/C_small/K0 core arms "
                 "train: '2x2' (2 BiMamba layers x 2 directions, retained "
                 "layers per student_d_retained_layers) or '4x1' (4 layers x "
                 "forward direction only).",
        recommended="outcome of the pre-registered Stage-2 rule (median "
                    "validation MacroDomainF1 after training-free removal; "
                    "tie -> 2x2)",
        alternatives=("2x2", "4x1"),
        rationale="Plan §4 Stage 2: the rule may only choose between these "
                  "two equal-cost variants. The rule was executed BEFORE "
                  "sealing (half_student_decision.json); the sealed value "
                  "records its outcome."),
    "f1_batch_composition": PendingDecision(
        key="f1_batch_composition",
        question="F1 few-shot KD step composition: hard labels on the "
                 "registered 10% subset + KD on 100% of TRAIN.",
        recommended={"kd_stream": "label-free SSL sampler over full TRAIN, "
                                  "64 windows/step",
                     "ce_stream": "registered 10% supervised sampler, 64 "
                                  "windows/step",
                     "note": "128 windows per optimizer step (2x compute of "
                             "a standard step) — or 32+32 to keep 64"},
        alternatives=("64+64", "32+32", "masked-CE inside the KD batch"),
        rationale="Plan fixes the mechanism (SimCLRv2-style) but not the "
                  "batch composition; F1 depends on registered 10% S1 runs "
                  "that do not exist yet."),
}


def pending_summary() -> list[dict]:
    return [asdict(d) for d in PENDING_DECISIONS.values()]


def unresolved_pending(resolved: dict | None) -> list[str]:
    """Keys still lacking an explicit resolved value."""
    resolved = resolved or {}
    return [k for k in PENDING_DECISIONS if k not in resolved]


# ---------------------------------------------------------------------------
# spec I/O (JSON content, .yaml extension: repository convention)
# ---------------------------------------------------------------------------
def dump_spec(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True,
                               default=_jsonable) + "\n")


def load_spec(path: Path):
    return json.loads(Path(path).read_text())


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:            # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def config_hash(obj) -> str:
    """Canonical sha256 of any JSON-able configuration object."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=_jsonable).encode()
    ).hexdigest()


def master_hash(files: list[Path]) -> str:
    entries = sorted((p.name, sha256_file(p)) for p in files)
    src = "".join(f"{n}:{h}\n" for n, h in entries)
    return hashlib.sha256(src.encode()).hexdigest()


# ---------------------------------------------------------------------------
# protocol document (written to methodology_v2/part6_compression/)
# ---------------------------------------------------------------------------
def protocol_document(resolved: dict | None = None) -> dict:
    """The Part-6 protocol as a JSON-able dict. Pending decisions carry
    their recommendation and status; `resolved` supplies sealed values."""
    resolved = resolved or {}
    pend = {}
    for k, d in PENDING_DECISIONS.items():
        entry = asdict(d)
        if k in resolved:
            entry["value"] = resolved[k]
            entry["status"] = "RESOLVED"
        pend[k] = entry
    return {
        "part": "methodology_v2 Part 6 — PC-STE lightweight study",
        "version": PART6_VERSION,
        "plan_document": "docs/methodology_v2_lightweight_plan.md",
        "fixed_a_priori": {
            "kd_temperature": KD_TEMPERATURE, "kd_alpha": KD_ALPHA,
            "label_smoothing_b0": LABEL_SMOOTHING_B0,
            "ni_margin_architecture": NI_MARGIN_ARCH,
            "ni_margin_ptq": NI_MARGIN_PTQ,
            "push_min_mean_delta": PUSH_MIN_DELTA, "push_alpha": PUSH_ALPHA,
            "n_sign_flip_patterns": N_SIGN_FLIPS,
            "confirmatory_family_m": CONFIRMATORY_FAMILY_M,
            "secondary_family_m": SECONDARY_FAMILY_M,
            "student_d_blocks": STUDENT_D_BLOCKS,
            "student_dw": STUDENT_DW,
            "folds": list(FOLDS), "seeds": list(SEEDS),
            "core_arms": list(CORE_ARMS), "optional_arms": list(OPTIONAL_ARMS),
            "push_arms": list(PUSH_ARMS),
            "teacher_rule": "same-fold 3-seed ensemble only; cross-fold "
                            "teachers forbidden; missing seed = hard fail",
            "kd_head_rule": "KD on the window's OWN dataset head only",
            "loss_reduction": "mean over datasets of per-dataset window "
                              "means (identical to the primary L_sup)",
        },
        "frozen_recipe": FROZEN_RECIPE,
        "pending_decisions": pend,
        "test_policy": "one sealed TEST session after ALL Part-6 training; "
                       "pre_test_ledger.csv committed first; test_seal.json "
                       "written before TEST is loaded; every touch appended "
                       "to test_touch_ledger.csv; one evaluation per "
                       "registered final model",
        "primary_untouched": "primary S0/S1/SSL registries, checkpoints, "
                             "splits, STFT, N2, architecture, optimizer, "
                             "checkpoint rule and TEST policy are read-only "
                             "inputs",
    }
