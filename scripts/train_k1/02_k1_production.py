#!/usr/bin/env python
"""
Final PCSTE-v2 K1 production executor.

Scientific implementation:
  ORIGINAL src/methodology_v2/compression Part-6 code.

Adapter responsibilities only:
  1. final-v1 checkpoint model["core.*"] -> original PCSTE namespace;
  2. final-v1 ItemBuilder g0_c0 -> original representation tuple;
  3. final-v1 manifest split lookup for original Part-6 TEST guard;
  4. frozen final-v1 S1 checkpoint resolution.

PRE-FLIGHT performs no training and no teacher-cache inference.
RUN performs TRAIN+VAL teacher caching/training/validation only.
TEST waveform inference is never implemented in this executor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# ORIGINAL dissertation implementation
# ---------------------------------------------------------------------------
from src.methodology_v2.encoder import PCSTE
from src.methodology_v2.experiment.heads import (
    CLASS_ORDERS,
    DatasetHeads,
    head_seed,
)
from src.methodology_v2.experiment.metrics import (
    classification_report,
    macro_domain_f1,
)
from src.methodology_v2.experiment.samplers import SupervisedSampler

from src.methodology_v2.compression import protocol as P6P
import src.methodology_v2.compression.guards as P6_GUARDS

from src.methodology_v2.compression.guards import (
    CheckpointRef,
    Part6GuardError,
)
from src.methodology_v2.compression.student import (
    half_4x1_spec,
    build_encoder,
)
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


# ---------------------------------------------------------------------------
# New final-v1 data/protocol only
# ---------------------------------------------------------------------------
from src.pcste_v2.protocol import label_subset_frame
from src.pcste_v2.representation import ItemBuilder
import src.pcste_v2.representation as REPR
from src.pcste_v2 import native12_freeze as N12


STUDY = "pcste_v2_lightweight_v1"
PROTOCOL = "global_v2_final_s0s1_v1"

A = REPO / "analysis/pcste_v2_lightweight_v1"
PDIR = REPO / "pcste_v2/protocol" / PROTOCOL
RESULTS = REPO / "results/pcste_v2/lightweight_v1"

CELLS = A / "K1_REGISTERED_CELLS.csv"
CONTRACT = A / "K1_PRODUCTION_PROTOCOL.json"
CODE_HASHES = A / "ORIGINAL_PART6_CODE_HASHES.sha256"

REGISTERED_CELLS_SHA256 = (
    "6d7e91b0bbd4ade3befbb97fccf18394526be33e8b7e54ef9c9fbcd0a5cee753"
)

REGISTERED_PROTOCOL_SHA256 = (
    "f2417ad057709b10b9d928ffc84114b5bfc135d818be58e664fb89e2d9134cb3"
)

TEACHER_SEEDS = (42, 1337, 2026)
DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")

NATIVE12_FREEZE_DIGEST_SHA256 = (
    "ddf4d32573c016b08da65b4f878ee0511a9bae4b15444626e1d932d24ebf39e4"
)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        text=True,
    ).strip()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str, out: Path | None = None) -> None:
    s = f"[{now()}] {msg}"
    print(s, flush=True)

    if out is not None:
        with open(out / "train_log.txt", "a") as f:
            f.write(s + "\n")


def verify_hash_manifest() -> None:
    for raw in CODE_HASHES.read_text().splitlines():
        raw = raw.strip()

        if not raw:
            continue

        expected, rel = raw.split(None, 1)
        rel = rel.strip().lstrip("*")

        p = REPO / rel

        got = sha256_file(p)

        if got != expected:
            raise RuntimeError(
                f"ORIGINAL Part6 source changed: {rel}\n"
                f"expected {expected}\n"
                f"got      {got}"
            )


def verify_frozen_contract() -> None:
    verify_hash_manifest()

    assert sha256_file(CELLS) == REGISTERED_CELLS_SHA256
    assert sha256_file(CONTRACT) == REGISTERED_PROTOCOL_SHA256


def install_native12_gate(
    fold: int,
    check_bytes: bool = True,
) -> dict:
    """Apply the SAME frozen native-12 launch gate as final-v1.

    This verifies the frozen bundle digest, protocol status,
    CWRU 12-kHz-only constraint, allowlisted source identity,
    raw-file bytes when requested, and normalizer registry.
    """
    allow = N12.preflight(
        REPO,
        PDIR,
        fold,
        NATIVE12_FREEZE_DIGEST_SHA256,
        check_bytes=check_bytes,
    )

    REPR.NATIVE12_ALLOWLIST = allow

    return allow


def registered_row(fold: int, seed: int) -> pd.Series:
    t = pd.read_csv(CELLS)

    q = t[
        (t["global_fold"].astype(int) == fold)
        & (t["seed"].astype(int) == seed)
    ]

    if len(q) != 1:
        raise RuntimeError(
            f"registered K1 cell missing/duplicated: fold={fold}, seed={seed}"
        )

    return q.iloc[0]


def manifest(fold: int) -> pd.DataFrame:
    p = PDIR / f"global_v2_fold_{fold}.csv"

    m = pd.read_csv(p)

    if "class" not in m.columns:
        raise RuntimeError("final-v1 manifest missing canonical 'class' field")

    if m["window_id"].duplicated().any():
        raise RuntimeError("duplicate final-v1 window_id")

    return m


def install_final_v1_guard(man_i: pd.DataFrame) -> None:
    """
    Keep ORIGINAL assert_no_test_windows unchanged.

    Only adapt its window-id -> split parser for final-v1 IDs.
    """
    original = P6_GUARDS.split_of_window_id

    def split_of_window_id(window_id: str) -> str:
        if window_id in man_i.index:
            split = str(man_i.loc[window_id, "split"])

            if split not in ("train", "validation", "test"):
                raise Part6GuardError(
                    f"{window_id}: invalid final-v1 split {split}"
                )

            return split

        return original(window_id)

    P6_GUARDS.split_of_window_id = split_of_window_id


def old_encoder_state(ck: dict) -> dict:
    out = {}

    for k, v in ck["model"].items():
        if not k.startswith("core."):
            raise RuntimeError(
                f"unexpected final-v1 model tensor outside core: {k}"
            )

        out[k[len("core."):]] = v

    return out


def source_checkpoint(row: pd.Series) -> tuple[Path, dict]:
    p = REPO / str(row["source_s1_checkpoint"])

    if not p.exists():
        raise RuntimeError(f"source S1 checkpoint missing: {p}")

    got = sha256_file(p)
    expected = str(row["source_s1_sha256"])

    if got != expected:
        raise RuntimeError(
            f"source S1 checkpoint hash mismatch:\n{p}\n"
            f"expected {expected}\n"
            f"got      {got}"
        )

    ck = torch.load(
        p,
        map_location="cpu",
        weights_only=False,
    )

    if int(ck["epoch"]) != int(row["source_s1_epoch"]):
        raise RuntimeError("source S1 selected epoch mismatch")

    return p, ck


def make_ref(fold: int, seed: int) -> CheckpointRef:
    row = registered_row(fold, seed)

    p, ck = source_checkpoint(row)

    return CheckpointRef(
        run_id=str(ck["run_id"]),
        kind="downstream",
        arm="s1",
        fold=fold,
        seed=seed,
        label_pct=100,
        path=p,
        sha256=str(row["source_s1_sha256"]),
        best_epoch=int(row["source_s1_epoch"]),
        best_val_macro_f1=float(ck["metric"]),
    )


def teacher_set(fold: int) -> TeacherSet:
    refs = tuple(
        make_ref(fold, s)
        for s in TEACHER_SEEDS
    )

    return TeacherSet(
        name="s1",
        fold=fold,
        refs=refs,
        ensemble_rule="mean_prob_at_T",
    )


def build_stream(
    man: pd.DataFrame,
    fold: int,
    seed: int,
) -> tuple[list, str, int, int]:

    row = registered_row(fold, seed)

    epochs = int(row["epochs"])
    spe = int(row["steps_per_epoch"])

    assert epochs == 50

    subset = label_subset_frame(man)

    torch.manual_seed(seed)
    np.random.seed(seed)

    sampler = SupervisedSampler(
        subset,
        1.0,
        seed,
    )

    stream = [
        sampler.next_batch()
        for _ in range(epochs * spe)
    ]

    h = hashlib.sha256(
        "".join(
            "|".join(map(str, item)) + "\n"
            for batch in stream
            for item in batch
        ).encode()
    ).hexdigest()

    expected = str(row["batch_stream_sha256"])

    if h != expected:
        raise RuntimeError(
            f"batch stream mismatch fold={fold} seed={seed}\n"
            f"expected {expected}\n"
            f"got      {h}"
        )

    return stream, h, epochs, spe


def frozen_spec_and_loss():
    spec = half_4x1_spec(
        "mean_of_remaining"
    )

    enc = build_encoder(
        spec,
        seed=0,
    )

    n = sum(
        p.numel()
        for p in enc.parameters()
    )

    if n != 1_375_953:
        raise RuntimeError(
            f"K1 encoder parameter count changed: {n}"
        )

    loss = LossConfig(
        kind="kd_ensemble+relational",
        temperature=4.0,
        alpha=0.5,
        relational_weight=1.0,
        label_smoothing=0.0,
        ensemble_rule="mean_prob_at_T",
    )

    loss.validate()

    return spec, loss, n


def preflight(fold: int, seed: int) -> None:
    verify_frozen_contract()

    install_native12_gate(
        fold,
        check_bytes=True,
    )

    m = manifest(fold)
    mi = m.set_index("window_id", drop=False)

    install_final_v1_guard(mi)

    train_ids = list(
        m.loc[m["split"] == "train", "window_id"]
    )

    val_ids = list(
        m.loc[m["split"] == "validation", "window_id"]
    )

    test_ids = list(
        m.loc[m["split"] == "test", "window_id"]
    )

    if not train_ids or not val_ids or not test_ids:
        raise RuntimeError("missing final-v1 split")

    # ORIGINAL guard, adapted only at window-id parsing.
    P6_GUARDS.assert_no_test_windows(
        train_ids,
        "K1 preflight TRAIN",
    )

    P6_GUARDS.assert_no_test_windows(
        val_ids,
        "K1 preflight VAL",
    )

    # Prove TEST is rejected, without reading any TEST waveform.
    rejected = False

    try:
        P6_GUARDS.assert_no_test_windows(
            [test_ids[0]],
            "K1 preflight TEST rejection probe",
        )
    except Part6GuardError:
        rejected = True

    if not rejected:
        raise RuntimeError(
            "ORIGINAL Part6 TEST guard failed to reject final-v1 TEST ID"
        )

    stream, sh, epochs, spe = build_stream(
        m,
        fold,
        seed,
    )

    row = registered_row(
        fold,
        seed,
    )

    p, ck = source_checkpoint(row)

    # Prove source checkpoint converts strictly into ORIGINAL PCSTE.
    enc = PCSTE()

    r = enc.load_state_dict(
        old_encoder_state(ck),
        strict=True,
    )

    if r.missing_keys or r.unexpected_keys:
        raise RuntimeError("strict original-PCSTE source load failed")

    heads = DatasetHeads()

    r = heads.load_state_dict(
        ck["heads"],
        strict=True,
    )

    if r.missing_keys or r.unexpected_keys:
        raise RuntimeError("strict original head load failed")

    spec, loss, n_params = frozen_spec_and_loss()

    ts = teacher_set(fold)

    print("===== K1 PRODUCTION PREFLIGHT =====")
    print("git head              :", git_head())
    print("fold / seed           :", fold, seed)
    print("source S1             :", ck["run_id"])
    print("source checkpoint     :", p.relative_to(REPO))
    print("source SHA256         :", sha256_file(p))
    print("TRAIN windows         :", len(train_ids))
    print("VAL windows           :", len(val_ids))
    print("TEST windows metadata :", len(test_ids))
    print("epochs                :", epochs)
    print("steps/epoch           :", spe)
    print("optimizer steps       :", len(stream))
    print("stream SHA256         :", sh)
    print("K1 encoder params     :", f"{n_params:,}")
    print("architecture          :", spec.to_dict())
    print("loss                  :", loss.to_dict())
    print(
        "teachers              :",
        [r.run_id for r in ts.refs],
    )
    print("TEST rejection probe  : PASS")
    print("native-12 byte gate   : PASS")
    print()
    print("PASS: production preflight")
    print("NO teacher-cache inference performed.")
    print("NO training performed.")
    print("NO TEST representation/model inference performed.")


def make_rep_store(
    man: pd.DataFrame,
    fold: int,
    out: Path,
):
    mi = man.set_index(
        "window_id",
        drop=False,
    )

    builder = ItemBuilder(
        PDIR,
        fold,
        grids=("v1",),
        channels={},
        envelope=False,
    )

    ids = list(
        man.loc[
            man["split"].isin(
                ["train", "validation"]
            ),
            "window_id",
        ]
    )

    P6_GUARDS.assert_no_test_windows(
        ids,
        "K1 representation preload",
    )

    store = {}

    t0 = time.time()

    for i, wid in enumerate(ids, 1):
        item = builder.build(
            mi.loc[wid]
        )

        store[wid] = item[
            "streams"
        ]["g0_c0"]

        if i % 500 == 0:
            log(
                f"preloaded {i}/{len(ids)} TRAIN+VAL representations",
                out,
            )

    log(
        f"preloaded {len(store)} TRAIN+VAL representations "
        f"in {time.time() - t0:.1f}s",
        out,
    )

    return store


def make_teacher_models(
    tset: TeacherSet,
    device: str,
):
    models = {}

    for ref in tset.refs:
        ck = torch.load(
            ref.path,
            map_location="cpu",
            weights_only=False,
        )

        if sha256_file(ref.path) != ref.sha256:
            raise RuntimeError(
                f"teacher source hash changed: {ref.run_id}"
            )

        enc = PCSTE()

        r = enc.load_state_dict(
            old_encoder_state(ck),
            strict=True,
        )

        if r.missing_keys or r.unexpected_keys:
            raise RuntimeError(
                f"teacher encoder strict-load failed: {ref.run_id}"
            )

        heads = DatasetHeads()

        r = heads.load_state_dict(
            ck["heads"],
            strict=True,
        )

        if r.missing_keys or r.unexpected_keys:
            raise RuntimeError(
                f"teacher heads strict-load failed: {ref.run_id}"
            )

        enc.to(device).eval()
        heads.to(device).eval()

        for p in list(enc.parameters()) + list(heads.parameters()):
            p.requires_grad_(False)

        models[ref.run_id] = (
            enc,
            heads,
        )

    return models


@torch.no_grad()
def validation_reports(
    trainer: Part6Trainer,
    store: dict,
    man: pd.DataFrame,
) -> dict:

    reports = {}

    for ds in DATASETS:

        wids = list(
            man.loc[
                (man["split"] == "validation")
                & (man["dataset"] == ds),
                "window_id",
            ]
        )

        P6_GUARDS.assert_no_test_windows(
            wids,
            "K1 validation",
        )

        mi = man.set_index(
            "window_id",
            drop=False,
        )

        y_true = [
            str(mi.loc[w, "class"])
            for w in wids
        ]

        y_pred = []

        for lo in range(
            0,
            len(wids),
            64,
        ):
            chunk = wids[
                lo:lo + 64
            ]

            y_pred.extend(
                trainer.predict(
                    [
                        store[w]
                        for w in chunk
                    ],
                    [ds] * len(chunk),
                )
            )

        reports[ds] = classification_report(
            y_true,
            y_pred,
            ds,
        )

    return reports


def run_cell(
    fold: int,
    seed: int,
    device: str,
    resume: bool,
    authorization: str,
) -> None:

    if authorization != REGISTERED_PROTOCOL_SHA256:
        raise SystemExit(
            "RUN REFUSED: --authorize must equal the frozen "
            "K1_PRODUCTION_PROTOCOL.json SHA256"
        )

    verify_frozen_contract()

    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")

    # Same fail-closed native-12 gate as the baseline executor.
    # This runs BEFORE the K1 result directory is created.
    install_native12_gate(
        fold,
        check_bytes=True,
    )

    run_id = (
        f"pcstev2_lightweight_v1_k1_gf{fold}_s{seed}"
    )

    out = RESULTS / run_id

    if out.exists() and not resume:
        raise SystemExit(
            f"{out} already exists; use --resume only for the same frozen run"
        )

    if resume and not (out / "last.pt").exists():
        raise SystemExit(
            "--resume requested but last.pt is missing"
        )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    m = manifest(fold)
    mi = m.set_index(
        "window_id",
        drop=False,
    )

    install_final_v1_guard(mi)

    train_ids = list(
        m.loc[
            m["split"] == "train",
            "window_id",
        ]
    )

    val_ids = list(
        m.loc[
            m["split"] == "validation",
            "window_id",
        ]
    )

    tv_ids = train_ids + val_ids

    P6_GUARDS.assert_no_test_windows(
        tv_ids,
        "K1 production TRAIN+VAL",
    )

    stream, stream_sha, epochs, spe = build_stream(
        m,
        fold,
        seed,
    )

    row = registered_row(
        fold,
        seed,
    )

    source_path, source_ck = source_checkpoint(
        row
    )

    spec, loss, n_params = frozen_spec_and_loss()

    tset = teacher_set(
        fold
    )

    # ------------------------------------------------------------
    # Preload EXACT final-v1 TRAIN+VAL representation once.
    # ------------------------------------------------------------
    store = make_rep_store(
        m,
        fold,
        out,
    )

    def rep_fn(wid: str):
        return store[wid]

    # ------------------------------------------------------------
    # ORIGINAL teacher cache.
    # One cache per fold; reusable across the 3 K1 seeds.
    # ------------------------------------------------------------
    cache_npz = (
        RESULTS
        / "teacher_cache"
        / f"teacher_s1_f{fold}.npz"
    )

    cache_json = (
        RESULTS
        / "teacher_cache"
        / f"teacher_s1_f{fold}.json"
    )

    if not (
        cache_npz.exists()
        and cache_json.exists()
    ):
        log(
            f"{run_id}: building ORIGINAL Part6 full TRAIN+VAL teacher cache",
            out,
        )

        models = make_teacher_models(
            tset,
            device,
        )

        try:
            build_teacher_cache(
                tset,
                mi,
                rep_fn,
                out_root=RESULTS,
                device=device,
                chunk=64,
                with_band_summaries=True,
                window_ids=tv_ids,
                encoder_state_override=models,
            )

        finally:
            models.clear()

            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    cache = TeacherCache(
        "s1",
        fold,
        root=RESULTS,
        expected_hashes=tset.hashes,
    )

    if set(cache.meta["splits"]) != {
        "train",
        "validation",
    }:
        raise RuntimeError(
            "teacher cache contains unexpected split"
        )

    if int(cache.meta["n_windows"]) != len(tv_ids):
        raise RuntimeError(
            "teacher cache window-count mismatch"
        )

    expected_cache_identity = [
        (
            str(w),
            str(mi.loc[w, "dataset"]),
            str(mi.loc[w, "split"]),
        )
        for w in tv_ids
    ]

    actual_cache_identity = list(
        zip(
            map(str, cache.arrays["window_id"]),
            map(str, cache.arrays["dataset"]),
            map(str, cache.arrays["split"]),
        )
    )

    if actual_cache_identity != expected_cache_identity:
        raise RuntimeError(
            "teacher cache TRAIN+VAL identity/order mismatch"
        )

    # ------------------------------------------------------------
    # ORIGINAL K1 construction.
    # ------------------------------------------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    acfg = ArmConfig(
        arm="k1",
        fold=fold,
        seed=seed,
        spec=spec,
        loss=loss,
        init_source="s1",
        teacher_set="s1",
        retained_layers=None,
        kept_direction="fwd",
        head_init_seed=head_seed(
            fold,
            seed,
        ),
    )

    trainer = Part6Trainer(
        acfg,
        device=device,
        init_encoder_state=old_encoder_state(
            source_ck
        ),
        init_heads_state=source_ck[
            "heads"
        ],
        teacher_cache=cache,
    )

    got_n = sum(
        p.numel()
        for p in trainer.encoder.parameters()
    )

    if got_n != n_params:
        raise RuntimeError(
            f"runtime K1 param mismatch: {got_n} != {n_params}"
        )

    sched = Part6Trainer.scheduler_for(
        trainer.optimizer,
        spe,
    )

    config_obj = {
        "study": STUDY,
        "run_id": run_id,
        "fold": fold,
        "seed": seed,
        "protocol": PROTOCOL,
        "production_protocol_sha256":
            REGISTERED_PROTOCOL_SHA256,
        "registered_cells_sha256":
            REGISTERED_CELLS_SHA256,
        "source_s1_checkpoint":
            str(source_path.relative_to(REPO)),
        "source_s1_sha256":
            sha256_file(source_path),
        "stream_sha256":
            stream_sha,
        "spec":
            spec.to_dict(),
        "loss":
            loss.to_dict(),
        "teacher_set":
            tset.to_dict(),
        "frozen_settings":
            Part6Trainer.frozen_settings(),
    }

    config_hash = P6P.config_hash(
        config_obj
    )

    start_epoch = 0

    best = {
        "metric": float("-inf"),
        "epoch": None,
        "reports": None,
    }

    if resume:
        ck = torch.load(
            out / "last.pt",
            map_location=device,
            weights_only=False,
        )

        if ck["config_hash"] != config_hash:
            raise RuntimeError(
                "resume config hash mismatch"
            )

        if ck["stream_hash"] != stream_sha:
            raise RuntimeError(
                "resume stream hash mismatch"
            )

        trainer.encoder.load_state_dict(
            ck["encoder"]
        )

        trainer.heads.load_state_dict(
            ck["heads"]
        )

        trainer.optimizer.load_state_dict(
            ck["optimizer"]
        )

        sched.load_state_dict(
            ck["scheduler"]
        )

        torch.set_rng_state(
            ck["torch_rng"].cpu()
        )

        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(
                [
                    s.cpu()
                    for s in ck["cuda_rng"]
                ]
            )

        start_epoch = int(
            ck["epoch"]
        ) + 1

        best = ck["best"]

        log(
            f"{run_id}: RESUMED at epoch {start_epoch}",
            out,
        )

    state = {
        "run_id": run_id,
        "kind": "pcste_v2_lightweight_k1",
        "status": "RUNNING",
        "fold": fold,
        "seed": seed,
        "config_hash": config_hash,
        "stream_sha256": stream_sha,
        "source_s1_checkpoint":
            str(source_path.relative_to(REPO)),
        "source_s1_sha256":
            sha256_file(source_path),
        "production_protocol_sha256":
            REGISTERED_PROTOCOL_SHA256,
        "registered_cells_sha256":
            REGISTERED_CELLS_SHA256,
        "arm_config":
            acfg.to_dict(),
        "teacher_cache":
            cache.meta["teacher_set"],
        "surgery_report": {
            k: v
            for k, v
            in trainer.surgery_report.items()
            if k != "mapping"
        },
        "frozen_settings":
            Part6Trainer.frozen_settings(),
        "encoder_params":
            got_n,
        "preload_windows":
            len(store),
        "started_at":
            now(),
        "pid":
            os.getpid(),
        "host":
            socket.gethostname(),
        "device":
            device,
        "gpu":
            (
                torch.cuda.get_device_name(0)
                if device.startswith("cuda")
                else None
            ),
        "git_head":
            git_head(),
        "test_used":
            False,
    }

    (
        out / "state.json"
    ).write_text(
        json.dumps(
            state,
            indent=1,
            sort_keys=True,
            default=P6P._jsonable,
        )
    )

    hist = open(
        out / "epoch_metrics.jsonl",
        "a",
    )

    try:
        for epoch in range(
            start_epoch,
            epochs,
        ):

            t0 = time.time()

            losses = []

            # ----------------------------------------------------
            # EXACT ORIGINAL Part6 production training semantics.
            # ----------------------------------------------------
            for k in range(spe):

                batch = stream[
                    epoch * spe + k
                ]

                reps = [
                    store[w]
                    for _, _, w
                    in batch
                ]

                losses.append(
                    trainer.train_step_bucketed(
                        reps,
                        batch,
                        sched,
                    )
                )

            t_train = (
                time.time() - t0
            )

            # ----------------------------------------------------
            # ORIGINAL validation semantics.
            # ----------------------------------------------------
            t0 = time.time()

            reports = validation_reports(
                trainer,
                store,
                m,
            )

            macro = macro_domain_f1(
                reports
            )

            t_val = (
                time.time() - t0
            )

            if Part6Trainer.is_better(
                macro,
                best["metric"],
            ):
                best = {
                    "metric": macro,
                    "epoch": epoch,
                    "reports": reports,
                }

                torch.save(
                    {
                        "run_id":
                            run_id,
                        "config_hash":
                            config_hash,
                        "epoch":
                            epoch,
                        "macro_f1_val":
                            macro,
                        "val_reports":
                            reports,
                        "encoder":
                            trainer.encoder_state(),
                        "heads":
                            trainer.heads_state(),
                    },
                    out / "best.pt",
                )

            torch.save(
                {
                    "run_id":
                        run_id,
                    "config_hash":
                        config_hash,
                    "stream_hash":
                        stream_sha,
                    "epoch":
                        epoch,
                    "best":
                        best,
                    "encoder":
                        trainer.encoder_state(),
                    "heads":
                        trainer.heads_state(),
                    "optimizer":
                        trainer.optimizer.state_dict(),
                    "scheduler":
                        sched.state_dict(),
                    "torch_rng":
                        torch.get_rng_state(),
                    "cuda_rng":
                        (
                            torch.cuda.get_rng_state_all()
                            if device.startswith("cuda")
                            else []
                        ),
                },
                out / "last.pt",
            )

            rec = {
                "epoch":
                    epoch,
                "train_loss_mean":
                    float(np.mean(losses)),
                "val_per_dataset_macro_f1":
                    {
                        ds:
                            reports[ds]["macro_f1"]
                        for ds in DATASETS
                    },
                "val_macro_domain_f1":
                    macro,
                "best_epoch":
                    best["epoch"],
                "lr":
                    sched.get_last_lr()[0],
                "seconds_train":
                    round(t_train, 1),
                "seconds_val":
                    round(t_val, 1),
                "at":
                    now(),
            }

            hist.write(
                json.dumps(
                    rec,
                    default=P6P._jsonable,
                )
                + "\n"
            )

            hist.flush()

            log(
                f"{run_id}: epoch {epoch + 1}/{epochs} "
                f"loss={rec['train_loss_mean']:.4f} "
                f"valF1={macro:.4f} "
                f"best={best['metric']:.4f}@{best['epoch']} "
                f"({t_train:.0f}s+{t_val:.0f}s)",
                out,
            )

        ckpt_hash = sha256_file(
            out / "best.pt"
        )

        completion = {
            "run_id":
                run_id,
            "best_epoch":
                best["epoch"],
            "best_val_macro_f1":
                best["metric"],
            "best_checkpoint_sha256":
                ckpt_hash,
            "config_hash":
                config_hash,
            "stream_sha256":
                stream_sha,
            "epochs_completed":
                epochs,
            "test_used":
                False,
            "note":
                (
                    "NO TEST evaluation here. "
                    "K1 TEST remains blocked until "
                    "all nine K1 cells are complete "
                    "and frozen."
                ),
        }

        (
            out / "completion.json"
        ).write_text(
            json.dumps(
                completion,
                indent=1,
                sort_keys=True,
                default=P6P._jsonable,
            )
        )

        state.update(
            {
                "status":
                    "COMPLETE",
                "best_epoch":
                    best["epoch"],
                "best_val_macro_f1":
                    best["metric"],
                "best_checkpoint_sha256":
                    ckpt_hash,
                "finished_at":
                    now(),
            }
        )

        (
            out / "state.json"
        ).write_text(
            json.dumps(
                state,
                indent=1,
                sort_keys=True,
                default=P6P._jsonable,
            )
        )

        log(
            f"{run_id}: COMPLETE "
            f"best={best['metric']:.4f}@{best['epoch']}",
            out,
        )

    except BaseException:

        state.update(
            {
                "status":
                    "FAILED",
                "error":
                    traceback.format_exc(),
                "failed_at":
                    now(),
            }
        )

        (
            out / "state.json"
        ).write_text(
            json.dumps(
                state,
                indent=1,
                sort_keys=True,
                default=P6P._jsonable,
            )
        )

        log(
            f"{run_id}: FAILED (state preserved)",
            out,
        )

        raise

    finally:
        hist.close()


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--mode",
        choices=("preflight", "run"),
        required=True,
    )

    ap.add_argument(
        "--fold",
        type=int,
        choices=(1, 2, 3),
        required=True,
    )

    ap.add_argument(
        "--seed",
        type=int,
        choices=(42, 1337, 2026),
        required=True,
    )

    ap.add_argument(
        "--device",
        default="cuda",
    )

    ap.add_argument(
        "--resume",
        action="store_true",
    )

    ap.add_argument(
        "--authorize",
        default="",
    )

    args = ap.parse_args()

    if args.mode == "preflight":
        if args.resume:
            raise SystemExit(
                "--resume is invalid for preflight"
            )

        preflight(
            args.fold,
            args.seed,
        )

        return

    run_cell(
        args.fold,
        args.seed,
        args.device,
        args.resume,
        args.authorize,
    )


if __name__ == "__main__":
    main()
