#!/usr/bin/env python
"""
NON-EXPERIMENTAL compatibility smoke test.

Purpose:
Use the ORIGINAL dissertation Part-6 implementation unchanged with:
  - new pcste_v2 final-v1 S1 checkpoints,
  - new final-v1 N2b representations,
  - new exact final-v1 supervised stream.

No TEST windows are used.
No methodology_v2 Part-6 source file is modified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

# Make repository imports behave exactly like scripts/pcste_v2/run.py.
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import torch

from src.methodology_v2.encoder import PCSTE
from src.methodology_v2.experiment.heads import DatasetHeads, head_seed
from src.methodology_v2.experiment.samplers import SupervisedSampler

# ORIGINAL dissertation Part-6 implementation
from src.methodology_v2.compression.guards import CheckpointRef
from src.methodology_v2.compression.student import StudentSpec
from src.methodology_v2.compression.losses import LossConfig
from src.methodology_v2.compression.teachers import (
    TeacherSet,
    TeacherCache,
    build_teacher_cache,
)
from src.methodology_v2.compression.trainer import (
    ArmConfig,
    Part6Trainer,
)

# New final-v1 data/protocol only
from src.pcste_v2.protocol import label_subset_frame
from src.pcste_v2.representation import ItemBuilder
import src.pcste_v2.representation as REPR
from src.pcste_v2.protocol_cwru_native12 import load_allowlist


PROTOCOL = "global_v2_final_s0s1_v1"
FOLD = 1
SEED = 42
TEACHER_SEEDS = (42, 1337, 2026)

PDIR = REPO / "pcste_v2/protocol" / PROTOCOL
OUT = REPO / "analysis/pcste_v2_lightweight_v1/smoke_original_part6"

OUT.mkdir(parents=True, exist_ok=True)

MAN = pd.read_csv(
    PDIR / f"global_v2_fold_{FOLD}.csv"
)

# Part6 cache code expects window_id index.
MAN_I = MAN.set_index("window_id", drop=False)

# ------------------------------------------------------------------
# Runtime-only compatibility for the ORIGINAL Part-6 TEST guard.
#
# The dissertation guard parses old Part-3B window-id syntax.
# New final-v1/native12 IDs use a different syntax, so we resolve
# their split from the authoritative frozen final-v1 manifest.
#
# We DO NOT replace assert_no_test_windows itself.
# We DO NOT modify guards.py on disk.
# Unknown/non-v2 IDs fall back to the original parser.
# ------------------------------------------------------------------
import src.methodology_v2.compression.guards as P6_GUARDS

_ORIGINAL_SPLIT_OF_WINDOW_ID = P6_GUARDS.split_of_window_id


def _final_v1_split_of_window_id(window_id: str) -> str:
    if window_id in MAN_I.index:
        split = str(MAN_I.loc[window_id, "split"])

        if split not in ("train", "validation", "test"):
            raise RuntimeError(
                f"unexpected final-v1 split for {window_id}: {split}"
            )

        return split

    return _ORIGINAL_SPLIT_OF_WINDOW_ID(window_id)


P6_GUARDS.split_of_window_id = _final_v1_split_of_window_id

# The smoke must never contain TEST.
assert set(
    MAN_I.loc[
        MAN_I["split"].isin(["train", "validation"]),
        "split"
    ].unique()
) <= {"train", "validation"}


# ------------------------------------------------------------------
# Native-12 CWRU read gate.
# ------------------------------------------------------------------
REPR.NATIVE12_ALLOWLIST = load_allowlist(
    REPO / "pcste_v2/protocol/cwru_native12_load_v1"
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def s1_path(fold: int, seed: int) -> Path:
    return REPO / (
        f"results/pcste_v2/final_v1/"
        f"pcstev2_final_v1_s1_gf{fold}_s{seed}/best.pt"
    )


def old_encoder_state(ck: dict) -> dict:
    """
    Compatibility mapping only:
      PCSTEv2 control state: core.X
      frozen PCSTE state:   X

    Tensor values are unchanged.
    """
    out = {}

    for k, v in ck["model"].items():
        if not k.startswith("core."):
            raise RuntimeError(
                f"unexpected non-core tensor in control checkpoint: {k}"
            )

        out[k[len("core."):]] = v

    return out


# ------------------------------------------------------------------
# Exact final-v1 supervised stream.
# ------------------------------------------------------------------
subset = label_subset_frame(MAN)

run_manifest = json.loads(
    (
        REPO /
        "results/pcste_v2/final_v1/"
        "pcstev2_final_v1_s1_gf1_s42/"
        "RUN_MANIFEST.yaml"
    ).read_text()
)

SPE = int(run_manifest["steps_per_epoch"])
EPOCHS = int(run_manifest["epochs"])

torch.manual_seed(SEED)
np.random.seed(SEED)

sampler = SupervisedSampler(
    subset,
    1.0,
    SEED,
)

stream = [
    sampler.next_batch()
    for _ in range(EPOCHS * SPE)
]

stream_sha = hashlib.sha256(
    "".join(
        "|".join(map(str, item)) + "\n"
        for batch in stream
        for item in batch
    ).encode()
).hexdigest()

assert stream_sha == run_manifest["batch_stream_sha256"]

# We only train on the FIRST exact final-v1 optimizer batch.
batch = stream[0]

assert len(batch) == 64
assert set(d for d, _, _ in batch) == {
    "CWRU", "JNU", "HIT", "MAFAULDA"
}

smoke_ids = [w for _, _, w in batch]

# Explicit split verification through the NEW authoritative manifest.
for wid in smoke_ids:
    assert MAN_I.loc[wid, "split"] == "train"


# ------------------------------------------------------------------
# New representation -> original Part6 representation tuple.
# ------------------------------------------------------------------
builder = ItemBuilder(
    PDIR,
    FOLD,
    grids=("v1",),
    channels={},
    envelope=False,
)

REP = {}


def rep(wid: str):
    if wid not in REP:
        row = MAN_I.loc[wid]
        item = builder.build(row)

        # This tuple is exactly what original collate_representations expects.
        REP[wid] = item["streams"]["g0_c0"]

    return REP[wid]


# Materialise exactly the smoke batch now.
for wid in smoke_ids:
    rep(wid)


# ------------------------------------------------------------------
# Construct SAME-FOLD 3-seed S1 teachers.
#
# We use original TeacherSet/build_teacher_cache.
# Only checkpoint namespace conversion is new.
# ------------------------------------------------------------------
refs = []
teacher_models = {}

for tseed in TEACHER_SEEDS:

    p = s1_path(FOLD, tseed)

    assert p.exists(), p

    ck = torch.load(
        p,
        map_location="cpu",
        weights_only=False,
    )

    expected_run_id = (
        f"pcstev2_final_v1_s1_gf{FOLD}_s{tseed}"
    )

    assert ck["run_id"] == expected_run_id

    state = old_encoder_state(ck)

    enc = PCSTE()
    r = enc.load_state_dict(
        state,
        strict=True,
    )
    assert not r.missing_keys
    assert not r.unexpected_keys

    heads = DatasetHeads()
    r = heads.load_state_dict(
        ck["heads"],
        strict=True,
    )
    assert not r.missing_keys
    assert not r.unexpected_keys

    enc = enc.cuda()
    heads = heads.cuda()
    enc.eval()
    heads.eval()

    ref = CheckpointRef(
        run_id=expected_run_id,
        kind="downstream",
        arm="s1",
        fold=FOLD,
        seed=tseed,
        label_pct=100,
        path=p,
        sha256=sha256_file(p),
        best_epoch=int(ck["epoch"]),
        best_val_macro_f1=float(ck["metric"]),
    )

    refs.append(ref)

    # ORIGINAL cache builder explicitly supports injected,
    # already-resolved teacher models for compatibility/tests.
    teacher_models[ref.run_id] = (
        enc,
        heads,
    )


teacher_set = TeacherSet(
    name="s1",
    fold=FOLD,
    refs=tuple(refs),
    ensemble_rule="mean_prob_at_T",
)


# ------------------------------------------------------------------
# ORIGINAL teacher cache generation.
#
# TRAIN only for this smoke.
# Real run will cache complete TRAIN+VAL.
# ------------------------------------------------------------------
print("===== BUILD ORIGINAL PART6 TEACHER CACHE =====")

build_teacher_cache(
    teacher_set,
    MAN_I,
    rep,
    out_root=OUT,
    device="cuda",
    chunk=16,
    with_band_summaries=True,
    window_ids=smoke_ids,
    encoder_state_override=teacher_models,
)

cache = TeacherCache(
    "s1",
    FOLD,
    root=OUT,
    expected_hashes=teacher_set.hashes,
)

assert cache.meta["teacher_set"]["ensemble_rule"] == "mean_prob_at_T"
assert cache.meta["fold"] == FOLD
assert cache.meta["n_windows"] == len(smoke_ids)

print(
    "teacher cache:",
    cache.meta["n_windows"],
    "windows, teachers=",
    cache.members,
)


# ------------------------------------------------------------------
# Same-cell S1 initialisation for K1.
# ------------------------------------------------------------------
student_p = s1_path(FOLD, SEED)

student_ck = torch.load(
    student_p,
    map_location="cpu",
    weights_only=False,
)

init_encoder = old_encoder_state(
    student_ck
)

init_heads = student_ck["heads"]


# ------------------------------------------------------------------
# EXACT dissertation K1 architecture and loss.
# ------------------------------------------------------------------
spec = StudentSpec(
    name="half_4x1",
    n_blocks=4,
    d_model=192,
    directions=1,
    stem_rank=None,
    uni_residual="mean_of_remaining",
)

loss = LossConfig(
    kind="kd_ensemble+relational",
    temperature=4.0,
    alpha=0.5,
    relational_weight=1.0,
    label_smoothing=0.0,
    ensemble_rule="mean_prob_at_T",
)

cfg = ArmConfig(
    arm="k1",
    fold=FOLD,
    seed=SEED,
    spec=spec,
    loss=loss,
    init_source="s1",
    teacher_set="s1",
    retained_layers=None,
    kept_direction="fwd",
    head_init_seed=head_seed(FOLD, SEED),
)


# ------------------------------------------------------------------
# ORIGINAL dissertation Part6Trainer.
# ------------------------------------------------------------------
print()
print("===== CONSTRUCT ORIGINAL PART6 TRAINER =====")

trainer = Part6Trainer(
    cfg,
    device="cuda",
    init_encoder_state=init_encoder,
    init_heads_state=init_heads,
    teacher_cache=cache,
)

n_encoder = sum(
    p.numel()
    for p in trainer.encoder.parameters()
)

assert n_encoder == 1_375_953, n_encoder

print("encoder params:", f"{n_encoder:,}")
print("architecture  :", trainer.cfg.spec.to_dict())
print("loss          :", trainer.cfg.loss.to_dict())
print("surgery       :", {
    k: v
    for k, v in trainer.surgery_report.items()
    if k != "mapping"
})


# ------------------------------------------------------------------
# ORIGINAL scheduler, using REAL fold-1 steps/epoch.
# ------------------------------------------------------------------
sched = Part6Trainer.scheduler_for(
    trainer.optimizer,
    SPE,
)


# ------------------------------------------------------------------
# ONE original Part6 training step on the FIRST exact final-v1 batch.
# ------------------------------------------------------------------
print()
print("===== ONE ORIGINAL PART6 TRAIN STEP =====")

reps = [
    rep(w)
    for _, _, w in batch
]

loss_value = trainer.train_step_bucketed(
    reps,
    batch,
    sched,
)

assert np.isfinite(loss_value)

print("loss:", loss_value)
print("lr  :", sched.get_last_lr()[0])


# ------------------------------------------------------------------
# Provenance report.
# ------------------------------------------------------------------
report = {
    "status": "PASS",
    "kind": "NOT_AN_EXPERIMENT",
    "fold": FOLD,
    "seed": SEED,
    "full_final_v1_stream_sha256": stream_sha,
    "first_batch_size": len(batch),
    "teacher_seeds": list(TEACHER_SEEDS),
    "teacher_checkpoint_sha256": teacher_set.hashes,
    "architecture": spec.to_dict(),
    "encoder_params": n_encoder,
    "loss": loss.to_dict(),
    "one_step_loss": float(loss_value),
    "scheduler_lr_after_one_step": float(
        sched.get_last_lr()[0]
    ),
    "original_part6_trainer": (
        "src/methodology_v2/compression/trainer.py"
    ),
    "original_part6_losses": (
        "src/methodology_v2/compression/losses.py"
    ),
    "original_part6_teacher_cache": (
        "src/methodology_v2/compression/teachers.py"
    ),
    "adapter_changes": [
        "remove core. namespace prefix from pcste_v2 control checkpoint",
        "extract streams[g0_c0] tuple from pcste_v2 ItemBuilder",
        "resolve new final-v1 S1 checkpoint paths"
    ],
    "test_windows_used": False,
}

(
    OUT / "SMOKE_REPORT.json"
).write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
)

print()
print("==============================================")
print("PASS: ORIGINAL dissertation Part6Trainer")
print("      completed one K1 KD training step")
print("      on the exact new final-v1 data stream.")
print("      No TEST data were used.")
print("==============================================")
