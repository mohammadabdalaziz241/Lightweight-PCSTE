"""Deterministic Part-6 run registry + seal machinery.

Run ids: {arm}_f{fold}_s{seed}  (arm in k1|c_small|k0 core; p1|dw_k1
optional; b1|b0|f1 push). 3 folds x 3 seeds x 3 core arms = 27 core rows.

Two build modes:
  * template  (require_checkpoints=False): all rows, teacher/init hashes
    "PENDING_PRIMARY" where the primary checkpoint does not exist yet;
    status "TEMPLATE_AWAITING_PRIMARY". Safe to (re)generate now.
  * final     (require_checkpoints=True): every referenced primary best.pt
    must exist, be COMPLETE and hash-verify; pending decisions must all be
    resolved; rows get status "REGISTERED"; the registry files are then
    hash-sealed (part6_hashes.csv + master hash) and verify fail-closed.

Every row carries: arm, fold, seed, architecture(+hash), init checkpoint
(+hash), teacher set (+hashes), split/window-manifest/representation
seals (Part-2/3B/4C/5B/5D), optimizer hash, loss hash, expected output
directory, enabled flag, status.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ..experiment.heads import head_seed
from ..experiment.registry import UPSTREAM, train_counts
from ..experiment.trainers import (DOWNSTREAM_EPOCHS, EFFECTIVE_BATCH,
                                   OPTIMIZER_SPEC, steps_per_epoch)
from ..integrity import sha256_file
from ..part3b_windows import PART3B_DIR
from .guards import Part6GuardError, primary_run_id, resolve_checkpoint
from .losses import LossConfig
from .protocol import (ALL_ARMS, ARCH_OF_ARM, CORE_ARMS, FOLDS, INIT_OF_ARM,
                       KD_ALPHA, KD_TEMPERATURE, LABEL_SMOOTHING_B0,
                       LOSS_OF_ARM, OPTIONAL_ARMS, PART6_DIR, PART6_RESULTS,
                       PART6_VERSION, PUSH_ARMS, SEEDS, TEACHER_SET_OF_ARM,
                       config_hash, master_hash, unresolved_pending)
from .student import (FULL_SPEC, STUDENT_D_SPEC, build_encoder, count_params,
                      half_4x1_spec, student_dw_spec)

REGISTRY_CSV = "part6_run_registry.csv"
HASHES_CSV = "part6_hashes.csv"
SEALED_SPEC_FILES = ("protocol.yaml", "student_spec.yaml",
                     "quantization_spec.yaml", "kd_spec.yaml",
                     "statistics_spec.yaml", "test_policy.yaml",
                     "measurement_spec.yaml", REGISTRY_CSV)


def run_id(arm: str, fold: int, seed: int) -> str:
    return f"{arm}_f{fold}_s{seed}"


def loss_config_for(arm: str, resolved: dict) -> LossConfig:
    kind = LOSS_OF_ARM[arm]
    if kind == "ce_hard":
        return LossConfig("ce_hard")
    if kind == "ce_label_smoothing_0.1":
        return LossConfig("ce_label_smoothing", label_smoothing=LABEL_SMOOTHING_B0)
    if kind == "kd_ensemble":
        return LossConfig("kd_ensemble", KD_TEMPERATURE, KD_ALPHA,
                          ensemble_rule=resolved.get("ensemble_rule"))
    if kind == "kd_ensemble+relational":
        return LossConfig("kd_ensemble+relational", KD_TEMPERATURE, KD_ALPHA,
                          relational_weight=resolved.get("relational_alpha_kl_weight"),
                          ensemble_rule=resolved.get("ensemble_rule"))
    if kind == "fewshot_kd":
        return LossConfig("fewshot_kd", KD_TEMPERATURE, KD_ALPHA,
                          ensemble_rule=resolved.get("ensemble_rule"))
    raise Part6GuardError(kind)


def spec_for(arm: str, resolved: dict):
    a = ARCH_OF_ARM[arm]
    if a == "student_d":
        # "student_d" = the compact student; the sealed variant comes from
        # the pre-registered Stage-2 rule outcome (compact_student_variant)
        if resolved.get("compact_student_variant") == "4x1":
            var = resolved.get("half_student_direction_variant") or {}
            return half_4x1_spec(var.get("residual", "mean_of_remaining"))
        return STUDENT_D_SPEC
    if a == "student_dw":
        return student_dw_spec(resolved.get("student_dw_stem_rank"))
    return FULL_SPEC


def _arch_hash_and_params(spec) -> tuple[str, int]:
    enc = build_encoder(spec, seed=0)
    return config_hash(spec.to_dict()), count_params(enc)


def _init_and_teacher_rows(arm: str, fold: int, seed: int,
                           require_checkpoints: bool) -> dict:
    """require_checkpoints here means: every referenced checkpoint MUST
    resolve and hash-verify (enforced for ENABLED rows of a final
    registry; disabled rows may carry PENDING markers)."""
    init_src = INIT_OF_ARM[arm]
    tset = TEACHER_SET_OF_ARM[arm]
    row = {"init_source": init_src or "random",
           "init_checkpoint": "none", "init_checkpoint_sha256": "none",
           "teacher_set": tset or "none", "teacher_checkpoints": "none",
           "teacher_sha256": "none"}
    pending = False
    if init_src in ("s1", "s0"):
        rid = primary_run_id(init_src, fold, seed, 100)
        row["init_checkpoint"] = f"results/methodology_v2/downstream/{rid}/best.pt"
        try:
            row["init_checkpoint_sha256"] = resolve_checkpoint(rid).sha256
        except Part6GuardError as e:
            if require_checkpoints:
                raise
            row["init_checkpoint_sha256"] = "PENDING_PRIMARY"
            pending = True
    elif init_src == "ssl":
        rid = f"ssl_f{fold}_s{seed}"
        row["init_checkpoint"] = f"results/methodology_v2/ssl/{rid}/best.pt"
        try:
            row["init_checkpoint_sha256"] = resolve_checkpoint(rid).sha256
        except Part6GuardError:
            if require_checkpoints:
                raise
            row["init_checkpoint_sha256"] = "PENDING_PRIMARY"
            pending = True
    if tset in ("s1", "s0"):
        rids = [primary_run_id(tset, fold, s, 100) for s in SEEDS]
        row["teacher_checkpoints"] = ";".join(rids)
        hs = []
        for r in rids:
            try:
                hs.append(resolve_checkpoint(r).sha256)
            except Part6GuardError:
                if require_checkpoints:
                    raise
                hs.append("PENDING_PRIMARY")
                pending = True
        row["teacher_sha256"] = ";".join(hs)
    elif tset == "s1_l010":
        rid = primary_run_id("s1", fold, seed, 10)
        row["teacher_checkpoints"] = rid
        try:
            row["teacher_sha256"] = resolve_checkpoint(rid).sha256
        except Part6GuardError:
            if require_checkpoints:
                raise
            row["teacher_sha256"] = "PENDING_FEWSHOT_REGISTRY"
            pending = True
    row["_pending"] = pending
    return row


def build_part6_registry(resolved: dict | None = None,
                         require_checkpoints: bool = False,
                         enabled_arms: tuple = CORE_ARMS,
                         include_optional: bool = True,
                         include_push: bool = True) -> pd.DataFrame:
    """Deterministic registry frame. Rows for optional/push arms are
    included (for completeness of the pre-registration) with
    enabled=False unless listed in `enabled_arms`."""
    resolved = resolved or {}
    if require_checkpoints:
        miss = unresolved_pending(resolved)
        if miss:
            raise Part6GuardError(f"cannot build a FINAL registry with "
                                  f"unresolved pending decisions: {miss}")
    counts = train_counts()
    man_hashes = {f: sha256_file(PART3B_DIR / f"window_manifest_fold_{f}.csv")
                  for f in FOLDS}
    arms = list(CORE_ARMS)
    if include_optional:
        arms += list(OPTIONAL_ARMS)
    if include_push:
        arms += list(PUSH_ARMS)
    arch_cache = {}
    rows = []
    for arm in arms:
        spec = spec_for(arm, resolved)
        if spec.name not in arch_cache:
            arch_cache[spec.name] = _arch_hash_and_params(spec)
        arch_hash, n_params = arch_cache[spec.name]
        loss = loss_config_for(arm, resolved)
        loss_hash = config_hash(loss.to_dict())
        for fold in FOLDS:
            for seed in SEEDS:
                rid = run_id(arm, fold, seed)
                enabled = arm in enabled_arms
                # a FINAL registry enforces checkpoint resolution only for
                # ENABLED rows; disabled optional/push rows are registered
                # with PENDING markers (e.g. F1's 10%-label teachers)
                extra = _init_and_teacher_rows(
                    arm, fold, seed, require_checkpoints and enabled)
                pending = extra.pop("_pending")
                if require_checkpoints:
                    if enabled:
                        status = "REGISTERED"
                    else:
                        status = ("REGISTERED_DISABLED_AWAITING_DEPS"
                                  if pending else "REGISTERED_DISABLED")
                else:
                    status = ("TEMPLATE_AWAITING_PRIMARY" if pending
                              else "TEMPLATE")
                rows.append({
                    "run_id": rid, "arm": arm, "fold": fold, "seed": seed,
                    "tier": ("core" if arm in CORE_ARMS else
                             "optional" if arm in OPTIONAL_ARMS else "push"),
                    "enabled": enabled,
                    "architecture": spec.name,
                    "architecture_config": json.dumps(spec.to_dict(), sort_keys=True),
                    "architecture_hash": arch_hash,
                    "encoder_params": n_params,
                    "surgery_mapping": (
                        json.dumps({"retained_layers": resolved.get(
                            "student_d_retained_layers", "PENDING")})
                        if spec.name == "student_d" and INIT_OF_ARM[arm]
                        else json.dumps({"kept_direction": (
                            resolved.get("half_student_direction_variant")
                            or {}).get("keep", "PENDING")})
                        if spec.name == "half_4x1" and INIT_OF_ARM[arm]
                        else "n/a"),
                    **extra,
                    "loss": loss.kind,
                    "loss_config": json.dumps(loss.to_dict(), sort_keys=True),
                    "loss_hash": loss_hash,
                    "head_init_seed": head_seed(fold, seed),
                    "sampler": "sup_dataset_class_group_window_16x4 (frozen; "
                               "same seed stream as the primary cell)",
                    "max_epochs": DOWNSTREAM_EPOCHS,
                    "steps_per_epoch": steps_per_epoch(counts[fold]),
                    "effective_batch": EFFECTIVE_BATCH,
                    "micro_batching": "dataset-bucketed 16x4 (exact mean of "
                                      "per-dataset means)",
                    "optimizer": json.dumps(OPTIMIZER_SPEC, sort_keys=True),
                    "optimizer_hash": config_hash(OPTIMIZER_SPEC),
                    "checkpoint_metric": "MacroDomainF1_val (maximize; exact "
                                         "tie -> earlier epoch)",
                    "primary_test_metric": "MacroDomainF1_test (Stage-5 "
                                           "sealed session only)",
                    "part2_hash": UPSTREAM["part2"],
                    "part3b_hash": UPSTREAM["part3b"],
                    "part4c_hash": UPSTREAM["part4c"],
                    "part5b_architecture_hash": UPSTREAM["part5b_architecture"],
                    "window_manifest_sha256": man_hashes[fold],
                    "output_location": f"results/methodology_v2/part6_compression/"
                                       f"runs/{rid}/",
                    "part6_version": PART6_VERSION,
                    "status": status})
    df = pd.DataFrame(rows)
    return df


def registry_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


def write_registry(df: pd.DataFrame, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or PART6_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / REGISTRY_CSV
    df.to_csv(p, index=False)
    return p


def dry_run_listing(df: pd.DataFrame, only_enabled: bool = True) -> str:
    """What WOULD run — no TEST, no training, no checkpoint loading."""
    sub = df[df["enabled"]] if only_enabled else df
    cols = ["run_id", "arm", "fold", "seed", "architecture", "encoder_params",
            "init_source", "teacher_set", "loss", "steps_per_epoch",
            "max_epochs", "status"]
    return sub[cols].to_string(index=False)


# ---------------------------------------------------------------------------
# seal
# ---------------------------------------------------------------------------
def seal_part6(out_dir: Path | None = None) -> str:
    out_dir = out_dir or PART6_DIR
    files = [out_dir / f for f in SEALED_SPEC_FILES]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise Part6GuardError(f"cannot seal, missing: {missing}")
    reg = pd.read_csv(out_dir / REGISTRY_CSV)
    if (reg["status"].str.startswith("TEMPLATE")).any():
        raise Part6GuardError("cannot seal a TEMPLATE registry (primary "
                              "checkpoints/pending decisions unresolved)")
    rec = pd.DataFrame([{"file": f.name, "sha256": sha256_file(f)}
                        for f in files])
    mh = master_hash(files)
    rec = pd.concat([rec, pd.DataFrame([{"file": "PART6_MASTER_HASH",
                                         "sha256": mh}])], ignore_index=True)
    rec.to_csv(out_dir / HASHES_CSV, index=False)
    return mh


def verify_part6_seal(out_dir: Path | None = None) -> str:
    out_dir = out_dir or PART6_DIR
    p = out_dir / HASHES_CSV
    if not p.exists():
        raise Part6GuardError("Part 6 is NOT sealed (part6_hashes.csv missing)")
    rec = pd.read_csv(p)
    stored = {r["file"]: r["sha256"] for _, r in rec.iterrows()}
    entries = []
    for name, expect in stored.items():
        if name == "PART6_MASTER_HASH":
            continue
        got = sha256_file(out_dir / name)
        if got != expect:
            raise Part6GuardError(f"SEALED PART-6 ARTIFACT CHANGED: {name}")
        entries.append((name, got))
    src = "".join(f"{n}:{h}\n" for n, h in sorted(entries))
    if hashlib.sha256(src.encode()).hexdigest() != stored["PART6_MASTER_HASH"]:
        raise Part6GuardError("Part-6 master hash mismatch (fail closed)")
    return stored["PART6_MASTER_HASH"]
