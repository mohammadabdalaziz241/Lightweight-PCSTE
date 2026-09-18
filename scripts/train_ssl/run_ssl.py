#!/usr/bin/env python
"""Masked-reconstruction SSL pretraining for the final matched S0/S1 stage (final_s0_s1.v1).

Accepted Part-5C objective on the v2 / N2b native12 representation: 60% of valid 16x8 patches masked (M1 random), masked patches
replaced by ONE learned mask token before the stem, X1 additive post-mixer context, per-token MLP decoder, masked valid-cell MSE
(per-window mean then batch mean). Label-free SSLSampler (label columns structurally inaccessible). TRAIN gradients only;
the frozen VAL split is used forward-only for checkpoint selection with FIXED masks. TEST is never loaded (asserted).
Checkpoint selector: the accepted mean_nmse.v2 (mean over datasets of masked MSE / masked-target energy), minimised,
strict improvement, earliest epoch on ties. The decoder and mask token are pretraining-only and are not part of the saved encoder.
"""
import argparse, hashlib, json, math, os, platform, socket, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from src.pcste_v2 import native12_freeze as N12, representation as REPR
from src.pcste_v2.final_s0_s1 import (SSLModuleV2, build_patch_mask_v2, collated_grid as FS_grid, mask_plan_hash, ssl_validation_metrics, ssl_selector_value,
                                      canonical_state_hash, SSL_SELECTOR, NATIVE12_GRIDS, FINAL_SPEC_VERSION)
from src.pcste_v2.model import collate_v2
from src.pcste_v2.protocol import verify_global
from src.methodology_v2.experiment.samplers import SSLSampler, DATASETS
from src.methodology_v2.experiment.trainers import SSL_EPOCHS, OPTIMIZER_SPEC, EFFECTIVE_BATCH, MASK_RATIO, MASK_GEOMETRY, lr_lambda
from src.pcste_v2.selector import SELECTORS
sys.path.insert(0, str(REPO / "scripts/pcste_v2"))
import run as R                                                   # reuse the supervised executor's preload/fingerprint helpers

RESULTS_ROOT = Path(os.environ.get("PCSTE_V2_RESULTS_ROOT", REPO / "results" / "pcste_v2"))
MICRO_BATCH = 32


def log(d: Path, msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"; print(line, flush=True)
    with open(d / "train_log.txt", "a") as f:
        f.write(line + "\n")


def run(args) -> None:
    protocol, fold, seed = args.protocol, args.fold, args.seed
    run_id = args.run_id or f"pcstev2_final_v1_ssl_gf{fold}_s{seed}"
    d = RESULTS_ROOT / args.stage / run_id
    pinfo = R.protocol_info(protocol); native12 = bool(pinfo.get("cwru_native12", False))
    if d.exists():
        st = (d / "STATUS").read_text().strip() if (d / "STATUS").exists() else "UNKNOWN"
        if st == "COMPLETE":
            print(f"{run_id}: already COMPLETE"); return
        raise SystemExit(f"{run_id}: directory exists with STATUS={st} (no silent overwrite)")
    if native12:
        REPR.NATIVE12_ALLOWLIST = N12.preflight(REPO, REPO / "pcste_v2/protocol" / protocol, fold, args.authorize_launch, check_bytes=True)
    d.mkdir(parents=True, exist_ok=False); (d / "STATUS").write_text("RUNNING")
    spe = int(pinfo["steps_per_epoch"][str(fold)]); epochs = SSL_EPOCHS
    cfg = {"run_id": run_id, "stage": args.stage, "phase": "ssl", "final_spec_version": FINAL_SPEC_VERSION, "protocol": protocol, "fold": fold, "seed": seed,
           "objective": "masked-reconstruction (Part-5C): mask token before the stem, X1 post-mixer context, per-token MLP decoder, masked valid-cell MSE (per-window mean then batch mean)",
           "mask": {"ratio": MASK_RATIO, "geometry": MASK_GEOMETRY, "unit": "16x8 valid patches", "grids": NATIVE12_GRIDS,
                    "determinism": "per-window RNG seeded by sha256(seed|epoch|window_id); TRAIN masks vary by epoch, VAL masks fixed via validation_mask_seed(seed) at epoch 0"},
           "target": "N2b-normalised log1p |STFT| patches (the same representation the downstream arms consume); completion-padding cells excluded from the loss",
           "sampler": "SSLSampler (label-free: dataset -> group -> window, 16 per dataset, label columns structurally inaccessible)",
           "optimizer": OPTIMIZER_SPEC, "epochs": epochs, "effective_batch": EFFECTIVE_BATCH, "micro_batch": MICRO_BATCH, "steps_per_epoch": spe,
           "selector": {"name": SSL_SELECTOR, "description": SELECTORS[SSL_SELECTOR], "direction": "minimise", "tie": "earliest epoch, strict improvement",
                        "denominator": "per-dataset mean of target^2 over exactly the masked valid cells of the FIXED validation masks (evaluation only; no fitting, no weight update on VAL)"},
           "test_policy": "TEST never read", "decoder": "pretraining-only (mask token, ctx_proj, MLP); discarded before S1"}
    try:
        (d / "CONFIG.yaml").write_text(R.yaml_dump(cfg)); (d / "ENVIRONMENT.json").write_text(json.dumps(R.env_json(), indent=1))
        fp = R.fingerprint(protocol, fold, cfg); (d / "fingerprint.json").write_text(json.dumps(fp, indent=1))
        log(d, f"{run_id}: START ssl fold={fold} seed={seed} protocol={protocol} host={socket.gethostname()} commit={fp['git_commit'][:12]}")
        pdir = REPO / "pcste_v2/protocol" / protocol; verify_global(pdir, fold)
        man = pd.read_csv(pdir / f"global_v2_fold_{fold}.csv", dtype={"fault_severity": str})
        train_ids = list(man.window_id[man.split == "train"]); val_ids = list(man.window_id[man.split == "validation"])
        assert not set(man.window_id[man.split == "test"]) & set(train_ids + val_ids)
        ds_of = dict(zip(man.window_id, man.dataset))
        torch.manual_seed(seed); np.random.seed(seed)
        lf = man[man.split == "train"][["dataset", "group_id", "window_id"]].copy()      # label-free view (sampler asserts this)
        sampler = SSLSampler(lf, seed); stream = [sampler.next_batch() for _ in range(epochs * spe)]
        sh = hashlib.sha256("".join("|".join(map(str, it)) + "\n" for b in stream for it in b).encode()).hexdigest()
        model = SSLModuleV2().to("cuda")                                                  # encoder built FIRST -> same initial state as S0
        init_enc_hash = canonical_state_hash(model.encoder_state())
        (d / "RUN_MANIFEST.yaml").write_text(R.yaml_dump({"run_id": run_id, "phase": "ssl", "fold": fold, "seed": seed, "protocol": protocol, "steps_per_epoch": spe, "epochs": epochs,
                                                          "n_train_windows": len(train_ids), "n_validation_windows": len(val_ids), "ssl_stream_sha256": sh,
                                                          "initial_encoder_canonical_sha256": init_enc_hash, "encoder_params": sum(p.numel() for p in model.encoder.parameters()),
                                                          "pretraining_only_params": sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in model.encoder.parameters()),
                                                          "val_mask_plan_sha256": mask_plan_hash([(ds_of[w], w) for w in val_ids], seed, 0, True),
                                                          "train_mask_plan_epoch0_sha256": mask_plan_hash([(ds_of[w], w) for w in train_ids[:512]], seed, 0, False),
                                                          "fingerprint": fp, "infrastructure": {"allocator": R.allocator_json()}}))
        params = list(model.parameters())
        opt = torch.optim.AdamW(params, lr=OPTIMIZER_SPEC["lr"], betas=OPTIMIZER_SPEC["betas"], eps=OPTIMIZER_SPEC["eps"], weight_decay=OPTIMIZER_SPEC["weight_decay"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(epochs, spe))
        store = R.preload(pdir, fold, man, train_ids + val_ids, ("v1",), {"MAFAULDA": 1}, False, {}, {}, args.workers, d)
        best = {"metric": float("inf"), "epoch": None}
        for epoch in range(epochs):
            model.train(); t0 = time.time(); losses = []
            for k in range(spe):
                batch = stream[epoch * spe + k]; opt.zero_grad(); total = 0.0
                for lo in range(0, len(batch), MICRO_BATCH):
                    chunk = batch[lo:lo + MICRO_BATCH]; w = len(chunk) / len(batch)
                    b = {kk: vv.to("cuda") for kk, vv in collate_v2([store[wid] for _, wid in chunk], 1, 1, False).items()}
                    pm = build_patch_mask_v2([(ds, wid) for ds, wid in chunk], seed, epoch, False, FS_grid(b)).to("cuda")
                    out = model(b, pm); (out["loss"] * w).backward(); total += float(out["loss"].detach()) * w
                    del out, b, pm
                torch.nn.utils.clip_grad_norm_(params, OPTIMIZER_SPEC["grad_clip_global_norm"]); opt.step(); sched.step(); losses.append(total)
            t_train = time.time() - t0; t0 = time.time()
            vm = ssl_validation_metrics(model, store, val_ids, ds_of, seed, "cuda"); metric = ssl_selector_value(vm)
            t_val = time.time() - t0
            if metric < best["metric"]:
                best = {"metric": metric, "epoch": epoch}
                torch.save({"run_id": run_id, "epoch": epoch, "selector": SSL_SELECTOR, "metric": metric, "encoder": model.encoder_state(),
                            "encoder_canonical_sha256": canonical_state_hash(model.encoder_state()), "per_dataset": vm, "cfg": cfg}, d / "best_encoder.pt")
            em = {"epoch": epoch, "train_masked_mse_mean": float(np.mean(losses)), "selector_value": metric, "selector": SSL_SELECTOR,
                  "per_dataset_val_mse": {k: v["mse"] for k, v in vm.items()}, "per_dataset_val_target_energy": {k: v["energy"] for k, v in vm.items()},
                  "per_dataset_val_nmse": {k: v["mse"] / max(v["energy"], 1e-12) for k, v in vm.items()}, "best_epoch": best["epoch"],
                  "lr": sched.get_last_lr()[0], "seconds_train": round(t_train, 1), "seconds_val": round(t_val, 1), "at": datetime.now(timezone.utc).isoformat()}
            with open(d / "epoch_metrics.jsonl", "a") as f:
                f.write(json.dumps(em) + "\n")
            log(d, f"epoch {epoch:02d} trainMSE {em['train_masked_mse_mean']:.5f} {SSL_SELECTOR} {metric:.6f} best {best['epoch']} ({best['metric']:.6f}) {t_train:.0f}s+{t_val:.0f}s")
        ck = torch.load(d / "best_encoder.pt", map_location="cpu", weights_only=False)
        (d / "SSL_SELECTION.json").write_text(json.dumps({"run_id": run_id, "selected_epoch": ck["epoch"], "selector": SSL_SELECTOR, "selector_value": ck["metric"],
                                                          "per_dataset": ck["per_dataset"], "encoder_canonical_sha256": ck["encoder_canonical_sha256"],
                                                          "best_encoder_file_sha256": R.sha256_file(d / "best_encoder.pt")}, indent=1))
        hashes = {p.name: R.sha256_file(p) for p in sorted(d.iterdir()) if p.is_file() and p.name not in ("hashes.json", "LOCK", "STATUS")}
        (d / "hashes.json").write_text(json.dumps(hashes, indent=1))
        (d / "STATUS").write_text("COMPLETE"); log(d, f"{run_id}: COMPLETE selected epoch {ck['epoch']} {SSL_SELECTOR} {ck['metric']:.6f}")
    except Exception:
        (d / "STATUS").write_text("FAILED"); log(d, "FAILED\n" + traceback.format_exc()); raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stage", default="final_v1"); ap.add_argument("--run-id", default=None); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--authorize-launch", default=None)
    run(ap.parse_args())
