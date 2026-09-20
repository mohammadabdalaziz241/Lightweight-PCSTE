"""Frozen Part-2 split protocol constants — methodology_v2.

Everything in this module is FROZEN protocol, declared before any split was
generated. Changing any value after the manifests are sealed invalidates
the seal (verify_frozen_hashes fails closed).

Sources of authority:
- Part-1 audit (methodology_v2/part1_audit/) and its recording manifest;
- CWRU_GROUPING_RECHECK.md (approved): physical specimen across all loads,
  OR clock positions merged;
- the approved Part-2 instruction (2026-08-11): rotation tables, seed
  governance, priority orders reproduced here verbatim in structure.
"""
from __future__ import annotations

METHODOLOGY_VERSION = "methodology_v2.part2.v1"

FOLD_IDS = (1, 2, 3)
SPLITS = ("train", "validation", "test")

# ---------------------------------------------------------------------------
# CWRU — deterministic Latin-style rotation over 9 physical fault specimens
# (48 kHz DE family; 0.007"/0.014"/0.021"; Healthy and 0.028" and the 12 kHz
# family excluded). Specimen = fault spec across ALL loads, OR positions
# merged. Table copied exactly from the approved instruction.
# ---------------------------------------------------------------------------
CWRU_SPECIMENS = ("IR007", "IR014", "IR021",
                  "B007", "B014", "B021",
                  "OR007", "OR014", "OR021")

CWRU_ROTATION: dict[int, dict[str, tuple[str, ...]]] = {
    1: {"train": ("IR007", "B014", "OR021"),
        "validation": ("IR014", "B021", "OR007"),
        "test": ("IR021", "B007", "OR014")},
    2: {"train": ("IR014", "B021", "OR007"),
        "validation": ("IR021", "B007", "OR014"),
        "test": ("IR007", "B014", "OR021")},
    3: {"train": ("IR021", "B007", "OR014"),
        "validation": ("IR007", "B014", "OR021"),
        "test": ("IR014", "B021", "OR007")},
}

# ---------------------------------------------------------------------------
# JNU — within-recording temporal holdout. Five contiguous macro-block
# slots A-E per recording at nominal fractions i/5; guards are SYMBOLIC
# until Part 3 freezes the effective window span.
# ---------------------------------------------------------------------------
JNU_BLOCKS = ("A", "B", "C", "D", "E")

JNU_ROTATION: dict[int, dict[str, tuple[str, ...]]] = {
    1: {"train": ("A", "B", "C"), "validation": ("D",), "test": ("E",)},
    2: {"train": ("B", "C", "D"), "validation": ("E",), "test": ("A",)},
    3: {"train": ("C", "D", "E"), "validation": ("A",), "test": ("B",)},
}

JNU_EVALUATION_LABEL = "within-recording temporal holdout"

# The guard requirement is deliberately symbolic: Part 3 must instantiate
# guard_samples G >= effective window span (window length plus any
# augmentation/receptive-field extension), then each internal nominal
# boundary b expands to a discarded interval [b - ceil(G/2), b + ceil(G/2))
# carved from the two adjacent macro-blocks. Future windows must lie
# entirely inside one usable (post-carve) macro-block.
JNU_GUARD_RULE = {
    "status": "symbolic-uninstantiated",
    "constraint": "guard_samples >= future_effective_window_span_samples",
    "instantiation": ("internal boundary b -> discard "
                      "[b - ceil(G/2), b + ceil(G/2))"),
    "windows_must_not_cross": ["macro-block boundaries", "split boundaries",
                               "guard regions"],
}

# ---------------------------------------------------------------------------
# Seed governance (HIT and MaFaulDa) — predeclared BEFORE generation.
# A candidate seed may be rejected ONLY by the structural acceptance
# criteria below (never by downstream model performance). On rejection the
# replacement is seed + SEED_REPLACEMENT_INCREMENT, recorded in
# rejected_split_seeds.json, up to MAX_SEED_ATTEMPTS attempts.
# ---------------------------------------------------------------------------
HIT_SEEDS: dict[int, int] = {1: 101, 2: 102, 3: 103}
MAFAULDA_SEEDS: dict[int, int] = {1: 201, 2: 202, 3: 203}
SEED_REPLACEMENT_INCREMENT = 1000
MAX_SEED_ATTEMPTS = 5

# ---------------------------------------------------------------------------
# HIT — grouped split over the 134 audited (session x speed-group)
# recordings, stratified by session (class is session-determined), target
# 70/15/15 subordinated to structure.
# ---------------------------------------------------------------------------
HIT_TARGET = {"train": 0.70, "validation": 0.15, "test": 0.15}

# per-session allocation: n_val = max(1, round(0.15*n)),
# n_test = max(1, round(0.15*n)), n_train = n - n_val - n_test
HIT_ALLOCATION_RULE = "per-session round(0.15*n) to val and test, floor>=1"

# structural acceptance criteria (frozen before generation):
#  H1 no atomic group crosses partitions (enforced by construction, asserted)
#  H2 every class has >= HIT_MIN_GROUPS_PER_CLASS groups in every partition
#  H3 every session contributes >= 1 group to every partition
#  H4 LP-speed coverage: train spans all 3 global LP tertiles;
#     validation and test each span >= 2
HIT_MIN_GROUPS_PER_CLASS = 2
HIT_TRAIN_MIN_LP_TERTILES = 3
HIT_VALTEST_MIN_LP_TERTILES = 2

# ---------------------------------------------------------------------------
# MaFaulDa — fault classes grouped by fault configuration (41 configs);
# Normal (single configuration) grouped by original recording (49 units) —
# an explicitly weaker unit, documented as a limitation. Operational folder
# taxonomy used as-is (known documentation inconsistency NOT remapped).
# ---------------------------------------------------------------------------
MAFAULDA_TARGET = {"train": 0.70, "validation": 0.15, "test": 0.15}

# per-stratum allocation over group units (stratum = original_label):
# n_val = max(1, round(0.15*c)), n_test = max(1, round(0.15*c)),
# n_train = c - n_val - n_test.
# With only 4 configurations for several classes this deliberately deviates
# from 70/15/15 (frozen priority: class coverage > proportions).
MAFAULDA_ALLOCATION_RULE = ("per-class-stratum round(0.15*c) to val and "
                            "test, floor>=1, over group units")

# structural acceptance criteria (frozen before generation):
#  M1 no fault configuration crosses partitions (construction, asserted)
#  M2 no Normal recording crosses partitions (construction, asserted)
#  M3 every one of the 10 operational classes has >= 1 group unit in every
#     partition
#  M4 Normal RPM coverage: validation and test each span >= 2 of the 3
#     global normal-speed tertiles
MAFAULDA_VALTEST_MIN_NORMAL_RPM_TERTILES = 2

# ---------------------------------------------------------------------------
# S0/S1 fairness — encoded for downstream stages
# ---------------------------------------------------------------------------
USAGE_RULES = {
    "S0_supervised": {
        "optimize_on": "train (labelled)",
        "model_selection_on": "validation",
        "final_evaluation_on": "test (once, sealed until then)",
    },
    "S1_ssl_pretraining": {
        "signals": "train ONLY (labels withheld)",
        "forbidden": "validation and test signals, even unlabelled",
    },
    "S1_finetuning": {
        "labelled_examples": "exactly the S0 train set of the same fold",
        "model_selection_on": "validation",
        "final_evaluation_on": "test (once, sealed until then)",
    },
    "shared_rule": ("S0 and S1 comparisons must consume identical frozen "
                    "fold assignments; fold_id is a data partition, "
                    "model_seed is future training stochasticity — "
                    "never conflate them"),
}
