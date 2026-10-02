#!/usr/bin/env python3
"""Isolated TRAIN/VALIDATION follow-up; uses the unchanged canonical executor.

Run --help without scientific dependencies. Preflight is read-only. Screen and
train refuse an existing output directory. No TEST inference or resume path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def checked_ids(manifest, split):
    """Reject unexpected partitions/duplicate identities before representation I/O."""
    if split not in ("train", "validation"):
        raise ValueError("Only train and validation data may be requested")
    if manifest["window_id"].duplicated().any():
        raise ValueError("Duplicate window IDs")
    if not set(manifest["split"]).issubset({"train", "validation", "test"}):
        raise ValueError("Unrecognized split")
    ids = list(manifest.loc[manifest["split"] == split, "window_id"])
    if not ids:
        raise ValueError(f"Empty {split} partition")
    return ids


def assert_training_stream(stream, train_ids):
    allowed = set(train_ids)
    for batch in stream:
        if len(batch) != 64:
            raise ValueError("Unexpected effective batch size")
        for _, _, wid in batch:
            if wid not in allowed:
                raise ValueError(f"Non-TRAIN window in optimization stream: {wid}")


def load_base(root, plan):
    root = Path(root).resolve(strict=True)
    path = root / "analysis/pcste_v2_lightweight_v1/scripts/02_k1_production.py"
    if not path.is_file():
        raise RuntimeError(f"Canonical executor missing: {path}. Use the original scientific workspace, not only the publication checkout.")
    if sha(path) != plan["original_executor_sha256"]:
        raise RuntimeError("Canonical executor differs from the frozen source; refusing to run")
    # Import the canonical implementation, never a second copy of src.
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("architecture_followup_base", path)
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)
    if base.REPO.resolve() != root:
        raise RuntimeError("Executor resolved a different scientific repository")
    import src.methodology_v2.registry as registry
    if registry.REPO_ROOT.resolve() != root:
        raise RuntimeError("Imported src belongs to another repository")
    base.verify_frozen_contract()
    return base


def prepare(base, fold, seed):
    base.install_native12_gate(fold, check_bytes=True)
    man = base.manifest(fold)
    train_ids = checked_ids(man, "train")
    val_ids = checked_ids(man, "validation")
    base.install_final_v1_guard(man.set_index("window_id", drop=False))
    base.P6_GUARDS.assert_no_test_windows(train_ids + val_ids, "architecture follow-up")
    stream, stream_hash, epochs, steps = base.build_stream(man, fold, seed)
    assert_training_stream(stream, train_ids)
    if epochs != 50:
        raise RuntimeError("Recovery schedule changed")
    # Hash-check all ensemble members, as well as the matched initialization.
    teachers = base.teacher_set(fold)
    source, ck = base.source_checkpoint(base.registered_row(fold, seed))
    return man, train_ids, val_ids, stream, stream_hash, epochs, steps, teachers, source, ck


def provenance(base, root, plan, fold, seed, prepared):
    man, tr, va, stream, stream_hash, epochs, steps, teachers, source, ck = prepared
    normalizers = base.PDIR / "normalizers" / f"registry_fold_{fold}.csv"
    return {
        "study": plan["study"], "fold": fold, "seed": seed,
        "plan_sha256": sha(HERE / "plan.json"), "runner_sha256": sha(__file__),
        "canonical_git_head": base.git_head(), "executor_sha256": plan["original_executor_sha256"],
        "source_checkpoint": str(source), "source_checkpoint_sha256": sha(source),
        "source_epoch": int(ck["epoch"]), "teacher_hashes": teachers.hashes,
        "manifest_sha256": sha(base.PDIR / f"global_v2_fold_{fold}.csv"),
        "normalizer_registry_sha256": sha(normalizers), "stream_sha256": stream_hash,
        "train_windows": len(tr), "validation_windows": len(va),
        "epochs": epochs, "steps_per_epoch": steps,
        "new_test_inference": False, "previous_test_results_known": True,
        "python": platform.python_version(), "torch": base.torch.__version__,
        "created_utc": base.now(),
    }


def validation_store(base, man, fold):
    ids = checked_ids(man, "validation")
    index = man.set_index("window_id", drop=False)
    builder = base.ItemBuilder(base.PDIR, fold, grids=("v1",), channels={}, envelope=False)
    return {wid: builder.build(index.loc[wid])["streams"]["g0_c0"] for wid in ids}


def make_trainer(base, arm, fold, seed, ck, device, cache=None):
    from src.methodology_v2.compression.student import STUDENT_D_SPEC, FULL_SPEC
    spec = {"2x2": STUDENT_D_SPEC, "4x1": base.half_4x1_spec("mean_of_remaining"), "full": FULL_SPEC}[arm]
    loss = base.frozen_spec_and_loss()[1] if cache is not None else base.LossConfig("ce_hard")
    cfg = base.ArmConfig(
        arm=f"architecture_{arm}", fold=fold, seed=seed, spec=spec, loss=loss,
        init_source="s1", teacher_set="s1" if cache is not None else None,
        retained_layers=[0, 2] if arm == "2x2" else None,
        kept_direction="fwd" if arm == "4x1" else None,
        head_init_seed=base.head_seed(fold, seed),
    )
    base.torch.manual_seed(seed)
    base.np.random.seed(seed)
    if device.startswith("cuda"):
        base.torch.cuda.manual_seed_all(seed)
    trainer = base.Part6Trainer(cfg, device=device,
        init_encoder_state=base.old_encoder_state(ck), init_heads_state=ck["heads"], teacher_cache=cache)
    expected = {"full": 2382033, "2x2": 1375185, "4x1": 1375953}[arm]
    if sum(p.numel() for p in trainer.encoder.parameters()) != expected:
        raise RuntimeError("Unexpected architecture parameter count")
    return trainer


def score(base, trainer, store, man):
    reports = base.validation_reports(trainer, store, man)
    if set(reports) != set(base.DATASETS):
        raise RuntimeError("A validation dataset is missing")
    metric = float(base.macro_domain_f1(reports))
    per_ds = {d: float(reports[d]["macro_f1"]) for d in base.DATASETS}
    if not all(math.isfinite(v) for v in [metric, *per_ds.values()]):
        raise RuntimeError("Non-finite validation metric")
    return {"macro_domain_f1": metric, "per_dataset_macro_f1": per_ds}


def get_cache(base, experiment_root, prepared, fold, store, device):
    man, tr, va, *unused = prepared
    teachers = prepared[7]
    ids = tr + va
    existing = base.RESULTS / "teacher_cache" / f"teacher_s1_f{fold}.npz"
    existing_meta = existing.with_suffix(".json")
    if existing.is_file() and existing_meta.is_file():
        cache_root = base.RESULTS  # read-only reuse of verified production cache
    else:
        cache_root = experiment_root / f"cache_fold_{fold}"
        npz = cache_root / "teacher_cache" / f"teacher_s1_f{fold}.npz"
        meta = npz.with_suffix(".json")
        if npz.exists() != meta.exists():
            raise RuntimeError("Incomplete follow-up teacher cache; inspect before retrying")
        if not npz.exists():
            cache_root.mkdir(parents=True, exist_ok=True)
            lock = cache_root / "BUILDING.lock"
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            models = base.make_teacher_models(teachers, device)
            try:
                base.build_teacher_cache(teachers, man.set_index("window_id", drop=False),
                    lambda wid: store[wid], out_root=cache_root, device=device, chunk=64,
                    with_band_summaries=True, window_ids=ids, encoder_state_override=models)
            finally:
                models.clear()
                lock.unlink()
                if device.startswith("cuda"):
                    base.torch.cuda.empty_cache()
    cache = base.TeacherCache("s1", fold, root=cache_root, expected_hashes=teachers.hashes)
    mi = man.set_index("window_id", drop=False)
    expected = [(str(w), str(mi.loc[w, "dataset"]), str(mi.loc[w, "split"])) for w in ids]
    got = list(zip(map(str, cache.arrays["window_id"]), map(str, cache.arrays["dataset"]), map(str, cache.arrays["split"])))
    if got != expected or set(cache.meta["splits"]) != {"train", "validation"}:
        raise RuntimeError("Teacher cache membership/order differs from frozen TRAIN+VAL")
    return cache


def run_screen(base, prepared, fold, seed, device, out):
    man, _, _, _, _, _, _, _, _, ck = prepared
    store = validation_store(base, man, fold)
    scores = {}
    for arm in ("full", "2x2", "4x1"):
        trainer = make_trainer(base, arm, fold, seed, ck, device)
        scores[arm] = score(base, trainer, store, man)
        del trainer
        if device.startswith("cuda"):
            base.torch.cuda.empty_cache()
    write_json(out / "screen.json", {"status": "COMPLETE", "fold": fold, "seed": seed,
        "split": "validation", "recovery_training": False, "scores": scores})


def run_train(base, prepared, fold, seed, device, out, experiment_root):
    man, _, _, stream, stream_hash, epochs, steps, teachers, _, ck = prepared
    store = base.make_rep_store(man, fold, out)
    cache = get_cache(base, experiment_root, prepared, fold, store, device)
    trainer = make_trainer(base, "2x2", fold, seed, ck, device, cache)
    scheduler = base.Part6Trainer.scheduler_for(trainer.optimizer, steps)
    initial = score(base, trainer, store, man)
    write_json(out / "initial_validation.json", initial)
    best = None
    with (out / "epoch_metrics.jsonl").open("x") as log:
        for epoch in range(epochs):
            start = time.monotonic()
            losses = []
            for batch in stream[epoch * steps:(epoch + 1) * steps]:
                losses.append(trainer.train_step_bucketed([store[w] for _, _, w in batch], batch, scheduler))
            training_seconds = time.monotonic() - start
            if not all(math.isfinite(float(v)) for v in losses):
                raise RuntimeError("Non-finite training loss")
            result = score(base, trainer, store, man)
            if best is None or base.Part6Trainer.is_better(result["macro_domain_f1"], best["macro_domain_f1"]):
                best = dict(result, epoch=epoch)
                base.torch.save({"study": "architecture_compare_v1", "arm": "2x2", "fold": fold, "seed": seed,
                    "epoch": epoch, "validation": result, "stream_sha256": stream_hash,
                    "encoder": trainer.encoder_state(), "heads": trainer.heads_state()}, out / "best.pt")
            record = {"epoch": epoch, "train_loss_mean": statistics.mean(losses),
                "val_macro_domain_f1": result["macro_domain_f1"],
                "val_per_dataset_macro_f1": result["per_dataset_macro_f1"],
                "best_epoch": best["epoch"], "seconds_train": training_seconds,
                "lr": scheduler.get_last_lr()[0], "at": base.now()}
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            print(f"GF{fold} seed {seed} epoch {epoch+1}/50 validation={result['macro_domain_f1']:.6f}", flush=True)
    write_json(out / "completion.json", {"status": "COMPLETE", "arm": "2x2", "fold": fold, "seed": seed,
        "epochs": epochs, "best": best, "final_epoch": result,
        "best_checkpoint_sha256": sha(out / "best.pt"), "new_test_inference": False})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("preflight", "screen", "train"), required=True)
    ap.add_argument("--canonical-repo", type=Path, required=True)
    ap.add_argument("--fold", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--seed", type=int, choices=(42, 1337, 2026), required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    plan = json.loads((HERE / "plan.json").read_text())
    root = args.canonical_repo.resolve(strict=True)
    base = load_base(root, plan)
    if args.device.startswith("cuda") and not base.torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu explicitly if intended")
    prepared = prepare(base, args.fold, args.seed)
    metadata = provenance(base, root, plan, args.fold, args.seed, prepared)
    if args.mode == "preflight":
        print(json.dumps(dict(metadata, status="PREFLIGHT_PASSED_NO_INFERENCE"), indent=2))
        return
    experiment_root = root / plan["output_relative_path"]
    expected_root = root / "results/pcste_v2/architecture_compare_v1"
    if experiment_root.resolve() != expected_root.resolve():
        raise RuntimeError("Output path is outside the isolated experiment directory")
    # No overwrite, automatic rerun or resume, including interrupted runs.
    out = experiment_root / f"{args.mode}_gf{args.fold}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "provenance.json", metadata)
    write_json(out / "plan_snapshot.json", plan)
    try:
        if args.mode == "screen":
            run_screen(base, prepared, args.fold, args.seed, args.device, out)
        else:
            run_train(base, prepared, args.fold, args.seed, args.device, out, experiment_root)
    except BaseException as exc:
        write_json(out / "failure.json", {"status": "FAILED", "error": repr(exc), "at": base.now()})
        raise


if __name__ == "__main__":
    main()
