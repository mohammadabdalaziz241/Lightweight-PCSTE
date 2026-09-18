#!/usr/bin/env python
"""PC-STE v2 run executor (Stage B screening and later stages).

One process = one run = one output directory owned atomically (mkdir + LOCK). Frozen four-domain recipe (AdamW 3e-4,
warm-up 5 -> cosine 1e-6, 50 epochs, effective batch 64 = 16 x 4 datasets, class-balanced sealed sampler, validation
MacroDomainF1 checkpoint selection with strict improvement) is reused unchanged; only the representation / model
configuration and the protocol manifests are new. TEST windows are never read by this executor (fail-closed assertion).

Outputs (run directory): RUN_MANIFEST.yaml, ENVIRONMENT.json, CONFIG.yaml, STATUS, LOCK, train_log.txt,
epoch_metrics.jsonl, best.pt, last.pt, validation_report.json, validation_predictions.csv, validation_embeddings.npz,
hashes.json, fingerprint.json.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, platform, socket, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.pcste_v2 import PCSTE_V2_CODE_VERSION  # noqa: E402
from src.pcste_v2.model import PCSTEv2, PCSTEv2Config, collate_v2, MODEL_VERSION  # noqa: E402
from src.pcste_v2.representation import ItemBuilder, REPRESENTATION_VERSION, N2B_VERSION, N2B_FLOOR, GRID_SETS, sha256_file  # noqa: E402
from src.pcste_v2.protocol import verify_global, label_subset_frame  # noqa: E402
from src.pcste_v2.protocol_cwru import verify_protocol as verify_cwru  # noqa: E402
from src.pcste_v2.metrics_v2 import evaluate_split  # noqa: E402
from src.pcste_v2.envelope import ENVELOPE_VERSION, ENVELOPE3_VERSION  # noqa: E402
from src.pcste_v2 import cwru_intervention as CI  # noqa: E402
from src.pcste_v2 import native12_freeze as N12  # noqa: E402
from src.pcste_v2 import representation as REPR  # noqa: E402
from src.pcste_v2.final_s0_s1 import canonical_state_hash as CI_HASH  # noqa: E402
from src.methodology_v2.experiment.heads import CLASS_ORDERS, DatasetHeads, head_seed  # noqa: E402
from src.methodology_v2.experiment.samplers import SupervisedSampler, DATASETS  # noqa: E402
from src.methodology_v2.experiment.trainers import OPTIMIZER_SPEC, DOWNSTREAM_EPOCHS, EFFECTIVE_BATCH, lr_lambda, steps_per_epoch  # noqa: E402

RESULTS_ROOT = Path(os.environ.get("PCSTE_V2_RESULTS_ROOT", REPO / "results" / "pcste_v2"))
VARIANTS = {   # name -> (grid set, envelope, n_channels, control)
    "n2b": ("single", False, 1, None),
    "env": ("single", True, 1, None),
    "multires": ("multires", False, 1, None),
    "env_rpmperm": ("single", True, 1, "rpm_perm"),
    "env_levelctrl": ("single", True, 1, "level_shuffle"),
    "n2b_2ch": ("single", False, 2, None),
    "env_2ch": ("single", True, 2, None),
    "env3": ("single", True, 1, None),                 # pre-registered 3-band envelope fallback (spec §2/§3, criteria failure_policy)
    "env3_rpmperm": ("single", True, 1, "rpm_perm"),
    "env3_levelctrl": ("single", True, 1, "level_shuffle"),
}
ENVELOPE_BANDS = {"env3": 3, "env3_rpmperm": 3, "env3_levelctrl": 3}   # default 1
# CWRU intervention (cwru_xspec.v1; analysis/pcste_v2_cwru_intervention_v1/PROTOCOL.md): C1 = class/physical-specimen-balanced CWRU
# sampling on the unchanged mixed stream; C2 = the identical C1 stream + cross-specimen supervised-contrastive loss on the CWRU group.
VARIANTS["n2b_c1"] = ("single", False, 1, None); VARIANTS["n2b_c2"] = ("single", False, 1, None)
CWRU_INTERVENTION = {"n2b_c1": {"sampling": True, "xspec": False}, "n2b_c2": {"sampling": True, "xspec": True}}   # default: off
MICRO_BATCH = {"single": 32, "multires": 8}
MICRO_BATCH_2CH = 16      # infrastructure setting for two-channel runs (memory); the accumulated objective is exact for any chunking
LEVEL_GAIN_RANGE = (0.25, 4.0)


def log(d: Path, msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    with open(d / "train_log.txt", "a") as f:
        f.write(line + "\n")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    out = subprocess.run(["git", "status", "--porcelain", "src/pcste_v2", "scripts/pcste_v2", "pcste_v2/protocol"], cwd=REPO, capture_output=True, text=True).stdout
    return any(line and not line.startswith("??") for line in out.splitlines())


def protocol_info(protocol: str) -> dict:
    p = REPO / "pcste_v2/protocol" / protocol / "global_v2_hashes.json"
    return json.loads(p.read_text()) if p.exists() else {}


def fingerprint(protocol: str, fold: int, cfg: dict) -> dict:
    pdir = REPO / "pcste_v2/protocol" / protocol; pinfo = protocol_info(protocol)
    cwru_dir = REPO / pinfo.get("cwru_protocol_dir", "pcste_v2/protocol/cwru_v2_PB_v1")     # old protocols: unchanged default
    reg = pdir / "normalizers" / f"registry_fold_{fold}.csv"
    code_files = sorted((REPO / "src/pcste_v2").glob("*.py")) + [REPO / "scripts/pcste_v2/run.py"] + sorted((REPO / "src/methodology_v2/encoder").glob("*.py")) + sorted((REPO / "src/methodology_v2/experiment").glob("*.py"))
    code_hash = hashlib.sha256(b"".join(p.read_bytes() for p in code_files)).hexdigest()
    return {"git_commit": git_commit(), "git_dirty_v2_paths": git_dirty(), "code_version": PCSTE_V2_CODE_VERSION, "code_sha256": code_hash,
            "model_version": MODEL_VERSION, "representation_version": REPRESENTATION_VERSION, "n2b": {"version": N2B_VERSION, "floor": N2B_FLOOR},
            "envelope_version": ENVELOPE_VERSION, "protocol": protocol, "global_manifest_sha256": verify_global(pdir, fold), "cwru_protocol_master_sha256": (__import__("src.pcste_v2.protocol_cwru_native12", fromlist=["verify_any_native12"]).verify_any_native12(cwru_dir) if pinfo.get("cwru_native12") else verify_cwru(cwru_dir)),
            **({"cwru_protocol_dir": pinfo["cwru_protocol_dir"], "native12_protocol_id": "pcste_v2_cwru_native12_specimen_v1", "native12_freeze_digest_sha256": N12.read_status(REPO, pinfo["cwru_protocol_dir"]).get("freeze_digest_sha256"), "native12_cwru_protocol_id": json.loads((cwru_dir / "cwru_v2_hashes.json").read_text())["protocol_id"], "native12_steps_per_epoch_frozen": pinfo.get("steps_per_epoch", {}).get(str(fold))} if pinfo.get("cwru_native12") else {}),
            "normalizer_registry_sha256": sha256_file(reg) if reg.exists() else None, "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
            **({"envelope3_registry_sha256": sha256_file(pdir / "normalizers" / f"registry_envelope3_fold_{fold}.csv"), "envelope3_version": ENVELOPE3_VERSION} if cfg.get("envelope_bands") == 3 else {}),
            **({"channel1_registry_sha256": sha256_file(pdir / "normalizers" / f"registry_channel1_fold_{fold}.csv"),
                "channel_map": {"MAFAULDA": {"c0": "CSV column index 2 = underhang radial (sealed v1 channel)", "c1": "CSV column index 1 = underhang axial"}, "CWRU|JNU|HIT": "single channel (c0 only; c1 masked)"}} if cfg.get("n_channels") == 2 else {}),
            "python": sys.version.split()[0], "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "hostname": socket.gethostname(), "platform": platform.platform(),
            "PCSTE_DATA_ROOT": os.environ.get("PCSTE_DATA_ROOT", "unset (repository-relative data/)"), "PCSTE_V2_RESULTS_ROOT": str(RESULTS_ROOT),
            "sealed_methodology_v2_untouched": True}


def env_json() -> dict:
    pkgs = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "torch", "einops"):
        try:
            pkgs[name] = __import__(name).__version__
        except Exception:
            pkgs[name] = None
    return {"hostname": socket.gethostname(), "pid": os.getpid(), "python": sys.version, "packages": pkgs, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "driver_nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
            "started_at": datetime.now(timezone.utc).isoformat(), "cwd": os.getcwd(), "argv": sys.argv, "allocator": allocator_json()}


def allocator_json() -> dict:
    """Infrastructure-only record of the CUDA caching-allocator configuration (no scientific effect: same model, data, batch
    stream, optimiser and checkpoint rule). Recorded so an OOM-relaunch with expandable segments is traceable in metadata."""
    settings = None
    try:
        settings = torch._C._accelerator_getAllocatorSettings()
    except Exception:
        pass
    return {"PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"), "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "torch_allocator_settings": settings, "torch_allocator_backend": torch.cuda.get_allocator_backend() if torch.cuda.is_available() else None,
            "note": "allocator/infrastructure setting only; not part of the scientific configuration or config_sha256"}


def yaml_dump(obj: dict) -> str:
    """Minimal YAML (JSON is valid YAML) — avoids a PyYAML dependency in the frozen venv."""
    return json.dumps(obj, indent=1, sort_keys=True, default=str)


def control_maps(man: pd.DataFrame, control: str | None, seed: int) -> tuple[dict, dict]:
    """rpm_perm: permute rpm within (dataset, split) deterministically. level_shuffle: fixed per-window log-uniform gain."""
    rpm_over, gains = {}, {}
    if control is None:
        return rpm_over, gains
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"control|{control}|{seed}".encode()).digest()[:4], "little"))
    if control == "rpm_perm":
        for (ds, sp), g in man.groupby(["dataset", "split"]):
            wids = g.window_id.values; rpms = g.rpm.values.astype(float); perm = rng.permutation(len(wids))
            for w, r in zip(wids, rpms[perm]):
                rpm_over[w] = float(r)
    elif control == "level_shuffle":
        lo, hi = np.log(LEVEL_GAIN_RANGE[0]), np.log(LEVEL_GAIN_RANGE[1])
        for w in man.window_id.values:
            gains[w] = float(np.exp(rng.uniform(lo, hi)))
    else:
        raise KeyError(control)
    return rpm_over, gains


_BUILDER = None


def _init_worker(pdir, fold, grids, channels, envelope, envelope_bands=1):
    global _BUILDER
    _BUILDER = ItemBuilder(pdir, fold, grids, channels, envelope, envelope_bands=envelope_bands)


def _build_one(args):
    row, rpm_o, gain = args
    return _BUILDER.build(pd.Series(row), rpm_override=rpm_o, gain=gain)


def preload(pdir: Path, fold: int, man: pd.DataFrame, wids: list[str], grids, channels, envelope, rpm_over, gains, workers: int, d: Path, envelope_bands: int = 1) -> dict:
    from multiprocessing import Pool
    sub = man.set_index("window_id").loc[wids]
    assert (sub.split != "test").all(), "TEST windows must never be loaded by this executor (fail closed)"
    order = sub.sort_values(["dataset", "source_file", "start_sample"])
    jobs = [(r._asdict() | {"window_id": r.Index}, rpm_over.get(r.Index), gains.get(r.Index, 1.0)) for r in order.itertuples()]
    t0 = time.time(); store = {}
    with Pool(workers, initializer=_init_worker, initargs=(pdir, fold, grids, channels, envelope, envelope_bands)) as pool:
        for (job, item) in zip(jobs, pool.imap(_build_one, jobs, chunksize=16)):
            store[job[0]["window_id"]] = item
    log(d, f"preloaded {len(store)} items in {time.time() - t0:.0f}s (workers={workers})")
    return store


def run(args) -> None:
    protocol, fold, seed, variant = args.protocol, args.fold, args.seed, args.variant
    grid_set, envelope, n_channels, control = VARIANTS[variant]; env_bands = ENVELOPE_BANDS.get(variant, 1)
    ci = CWRU_INTERVENTION.get(variant, {"sampling": False, "xspec": False}); w_cwru = 1.0 / len(CLASS_ORDERS)   # equal four-domain mean -> effective CWRU domain weight 1/4
    grids = GRID_SETS[grid_set]; channels = {"MAFAULDA": n_channels}
    if args.smoke:
        args.stage = "smoke_NOT_AN_EXPERIMENT"
    run_id = args.run_id or f"pcstev2_{args.stage}_{variant}_f{fold}_s{seed}"
    d = RESULTS_ROOT / args.stage / run_id
    pinfo = protocol_info(protocol); native12 = bool(pinfo.get("cwru_native12", False))
    if native12 and not args.smoke:                                     # hard launch gate: frozen protocol, matching digest, explicit authorization, allowlisted bytes
        REPR.NATIVE12_ALLOWLIST = N12.preflight(REPO, REPO / "pcste_v2/protocol" / protocol, fold, args.authorize_launch, check_bytes=True)
    elif native12:
        REPR.NATIVE12_ALLOWLIST = __import__("src.pcste_v2.protocol_cwru_native12", fromlist=["load_allowlist"]).load_allowlist(REPO / pinfo["cwru_protocol_dir"])
    # ---- atomic ownership --------------------------------------------------------------------------------------
    if d.exists():
        status = (d / "STATUS").read_text().strip() if (d / "STATUS").exists() else "UNKNOWN"
        if status == "COMPLETE":
            print(f"{run_id}: already COMPLETE"); return
        if not args.resume:
            raise SystemExit(f"{run_id}: directory exists with STATUS={status}; use --resume to continue (no silent overwrite)")
    else:
        d.mkdir(parents=True, exist_ok=False)
    lock = d / "LOCK"
    if lock.exists() and not args.resume:
        raise SystemExit(f"{run_id}: LOCK held by {lock.read_text()}")
    lock.write_text(f"{socket.gethostname()}:{os.getpid()}:{datetime.now(timezone.utc).isoformat()}")
    (d / "STATUS").write_text("RUNNING")
    cfg = {"run_id": run_id, "stage": args.stage, "variant": variant, "fold": fold, "seed": seed, "protocol": protocol, "grids": list(grids), "envelope_branch": envelope,
           "n_channels": n_channels, "control": control, "level_gain_range": LEVEL_GAIN_RANGE if control == "level_shuffle" else None,
           "model": PCSTEv2Config(envelope_branch=envelope, multires=(grid_set == "multires"), n_channels=n_channels, envelope_bands=env_bands).to_dict(),
           **({"envelope_bands": env_bands, "envelope_version": ENVELOPE3_VERSION} if env_bands == 3 else {}),   # Stage B config_sha256 values untouched
           **({"cwru_sampling": {"version": CI.CWRU_PLAN_VERSION, "items_per_step": CI.CWRU_ITEMS_PER_STEP, "class_quota_rotation": "5/5/6", "replaces_only_cwru_slots_of_original_stream": True}} if ci["sampling"] else {}),
           **({"cross_specimen_loss": {"version": CI.CWRU_XSPEC_VERSION, "tau": CI.XSPEC_TAU, "lambda": CI.XSPEC_LAMBDA, "cwru_domain_weight": w_cwru, "effective_coefficient": w_cwru * CI.XSPEC_LAMBDA,
                                        "embedding": "global_embedding (192-d encoder output, L2-normalised inside the loss)", "reduction": "anchor->specimen->class means", "same_class_same_specimen_pairs_excluded": True}} if ci["xspec"] else {}),
           "optimizer": OPTIMIZER_SPEC, "epochs": DOWNSTREAM_EPOCHS, "effective_batch": EFFECTIVE_BATCH, "micro_batch": MICRO_BATCH_2CH if n_channels == 2 else MICRO_BATCH[grid_set],
           "checkpoint_rule": "max validation MacroDomainF1 (strict >, earlier epoch on ties)", "init": "random (S0)", "label_fraction": 1.0, "test_policy": "TEST never read"}
    try:
        (d / "CONFIG.yaml").write_text(yaml_dump(cfg)); (d / "ENVIRONMENT.json").write_text(json.dumps(env_json(), indent=1))
        fp = fingerprint(protocol, fold, cfg); (d / "fingerprint.json").write_text(json.dumps(fp, indent=1))
        log(d, f"{run_id}: START variant={variant} fold={fold} seed={seed} protocol={protocol} host={socket.gethostname()} commit={fp['git_commit'][:12]}")
        pdir = REPO / "pcste_v2/protocol" / protocol
        man = pd.read_csv(pdir / f"global_v2_fold_{fold}.csv")
        train_ids = list(man.window_id[man.split == "train"]); val_ids = list(man.window_id[man.split == "validation"])
        if args.smoke:                                   # NOT_AN_EXPERIMENT: tiny subset, 2 steps, 1 epoch
            keep = man[man.split.isin(["train", "validation"])].groupby(["dataset", "split", "class", "physical_specimen"]).head(4)   # every class/specimen represented (smoke only)
            man = pd.concat([keep, man[man.split == "test"].head(0)]); train_ids = list(man.window_id[man.split == "train"]); val_ids = list(man.window_id[man.split == "validation"])
        subset = label_subset_frame(man)
        spe = 2 if args.smoke else steps_per_epoch(len(train_ids)); epochs = 1 if args.smoke else DOWNSTREAM_EPOCHS
        if native12 and not args.smoke:                                 # frozen per-fold optimiser-step count (preserves the baseline non-CWRU exposure and schedule)
            spe = int(pinfo["steps_per_epoch"][str(fold)])
        torch.manual_seed(seed); np.random.seed(seed)
        sampler = SupervisedSampler(subset, 1.0, seed); stream = [sampler.next_batch() for _ in range(epochs * spe)]
        stream_hash = lambda st: hashlib.sha256("".join("|".join(map(str, it)) + "\n" for b in st for it in b).encode()).hexdigest()
        sh_original = stream_hash(stream); universe = plan_info = None
        if ci["sampling"]:                                   # C1/C2: consume the original stream first, then replace only its CWRU slots
            universe = CI.cwru_train_universe(man); original_stream = stream; stream, plan_info = CI.balanced_stream(stream, universe, spe, epochs, seed)
            assert CI.non_cwru_preserved(original_stream, stream), "non-CWRU batch items changed (fail closed)"
        sh = stream_hash(stream)
        rpm_over, gains = control_maps(man, control, seed)
        hseed = head_seed(fold, seed)
        (d / "RUN_MANIFEST.yaml").write_text(yaml_dump({"run_id": run_id, "fold": fold, "seed": seed, "variant": variant, "protocol": protocol, "steps_per_epoch": spe, "epochs": epochs,
                                                       "n_train_windows": len(train_ids), "n_validation_windows": len(val_ids), "batch_stream_sha256": sh, "head_init_seed": hseed,
                                                       "control_rpm_permuted_windows": len(rpm_over), "control_gain_windows": len(gains), "fingerprint": fp,
                                                       "infrastructure": {"allocator": allocator_json()}, "init_encoder_arg": args.init_encoder,
                                                       **({"cwru_intervention": {"original_stream_sha256": sh_original, "balanced_stream_sha256": sh, "plan": plan_info, "cwru_train_windows": universe["n_windows"],
                                                                                 "windows_per_class_specimen": universe["windows_per_class_specimen"], "xspec_active": ci["xspec"]}} if ci["sampling"] else {})}))
        assert torch.cuda.is_available(), "GPU required"
        dev = "cuda"
        model = PCSTEv2(PCSTEv2Config(envelope_branch=envelope, multires=(grid_set == "multires"), n_channels=n_channels, envelope_bands=env_bands)).to(dev)
        init_encoder_provenance = {"kind": "random (S0)", "canonical_sha256": CI_HASH(model.state_dict())}
        if args.init_encoder:                                   # S1: load exactly the selected SSL encoder; decoder/mask token are NOT loaded
            ck_ssl = torch.load(REPO / args.init_encoder, map_location=dev, weights_only=False)
            enc_state = ck_ssl["encoder"]; missing, unexpected = model.load_state_dict(enc_state, strict=True), None
            assert not any(k.startswith("probe.") or "mask_token" in k or "decoder" in k or "ctx_proj" in k for k in enc_state), "pretraining-only parameters must not be loaded"
            loaded = CI_HASH(model.state_dict())
            assert loaded == ck_ssl["encoder_canonical_sha256"], f"loaded encoder hash {loaded[:12]} != selected SSL encoder {ck_ssl['encoder_canonical_sha256'][:12]}"
            assert all(p.requires_grad for p in model.parameters()), "S1 is FULL fine-tuning: every encoder parameter must be trainable"
            init_encoder_provenance = {"kind": "SSL-pretrained (S1)", "ssl_run_id": ck_ssl["run_id"], "ssl_selected_epoch": ck_ssl["epoch"], "ssl_selector": ck_ssl["selector"],
                                       "ssl_selector_value": ck_ssl["metric"], "canonical_sha256": loaded, "init_encoder_path": args.init_encoder,
                                       "init_encoder_file_sha256": sha256_file(REPO / args.init_encoder)}
            log(d, f"S1 init: encoder loaded from {args.init_encoder} (ssl epoch {ck_ssl['epoch']}, canonical {loaded[:12]}); decoder discarded; full fine-tuning")
        heads = DatasetHeads(init_seed=hseed).to(dev)
        init_head_hash = CI_HASH(heads.state_dict())
        (d / "INIT_PROVENANCE.json").write_text(json.dumps({"initial_encoder": init_encoder_provenance, "initial_heads_canonical_sha256": init_head_hash,
                                                            "head_init_seed": hseed, "batch_stream_sha256": sh, "note": "heads are initialised from head_seed(fold, seed) by an independent generator, so they are identical for the S0/S1 pair regardless of how many random draws pretraining consumed"}, indent=1))
        params = list(model.parameters()) + list(heads.parameters())
        opt = torch.optim.AdamW(params, lr=OPTIMIZER_SPEC["lr"], betas=OPTIMIZER_SPEC["betas"], eps=OPTIMIZER_SPEC["eps"], weight_decay=OPTIMIZER_SPEC["weight_decay"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(epochs, spe))
        store = preload(pdir, fold, man, train_ids + val_ids, grids, channels, envelope, rpm_over, gains, args.workers, d, envelope_bands=env_bands)
        mb = cfg["micro_batch"]; n_grids = len(grids)
        man_i = man.set_index("window_id")
        start_epoch, best = 0, {"metric": float("-inf"), "epoch": None}
        if args.resume and (d / "last.pt").exists():
            ck = torch.load(d / "last.pt", map_location=dev, weights_only=False)
            assert ck["config_sha256"] == fp["config_sha256"] and ck["stream_sha256"] == sh, "resume mismatch (fail closed)"
            model.load_state_dict(ck["model"]); heads.load_state_dict(ck["heads"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
            torch.set_rng_state(ck["torch_rng"].cpu()); torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
            start_epoch = ck["epoch"] + 1; best = ck["best"]; log(d, f"RESUMED at epoch {start_epoch}")

        def forward_loss(batch_items):
            items = [store[w] for _, _, w in batch_items]
            b = collate_v2(items, n_grids, n_channels, envelope, env_bands); b = {k: v.to(dev) for k, v in b.items()}
            z = model(**b)["global_embedding"]; losses = []
            for ds in sorted(CLASS_ORDERS):
                idx = [i for i, (dd, _, _) in enumerate(batch_items) if dd == ds]
                if not idx:
                    continue
                y = torch.tensor([CLASS_ORDERS[ds].index(batch_items[i][1]) for i in idx], device=dev)
                losses.append(F.cross_entropy(heads(z[idx], ds), y))
            aux = None
            if ci["xspec"]:
                pos, y_c, s_c = CI.cwru_group_tensors(batch_items, universe, dev)
                if pos:                                      # the auxiliary loss needs the COMPLETE logical CWRU group in one graph
                    if len(pos) != CI.CWRU_ITEMS_PER_STEP:
                        raise NotImplementedError(f"CWRU group split across micro-batches ({len(pos)} of {CI.CWRU_ITEMS_PER_STEP}); a separate CWRU-only auxiliary forward is not implemented")
                    aux = CI.cross_specimen_loss(z[pos], y_c, s_c, temperature=CI.XSPEC_TAU)
            return torch.stack(losses).mean(), aux

        @torch.no_grad()
        def validate() -> tuple[dict, pd.DataFrame, np.ndarray]:
            model.eval(); heads.eval(); rows, embs = [], []
            for ds in DATASETS:
                wids = [w for w in val_ids if man_i.loc[w, "dataset"] == ds]
                for lo in range(0, len(wids), 64):
                    chunk = wids[lo:lo + 64]; b = collate_v2([store[w] for w in chunk], n_grids, n_channels, envelope, env_bands); b = {k: v.to(dev) for k, v in b.items()}
                    z = model(**b)["global_embedding"]; lg = heads(z, ds); pr = torch.softmax(lg, -1).cpu().numpy(); lg = lg.cpu().numpy(); z = z.cpu().numpy()
                    for i, w in enumerate(chunk):
                        r = man_i.loc[w]; yt = str(r["class"]); yp = CLASS_ORDERS[ds][int(lg[i].argmax())]
                        rows.append({"dataset": ds, "window_id": w, "y_true": yt, "y_pred": yp, "p_max": float(pr[i].max()), "physical_specimen": r["physical_specimen"], "group_id": r["group_id"],
                                     "fault_severity": r["fault_severity"], "rpm": r["rpm"], "load": r["load"], "recording_id": r["recording_id"], "native_sampling_rate_hz": r["native_sampling_rate_hz"],
                                     **{f"prob__{c}": float(pr[i][j]) for j, c in enumerate(CLASS_ORDERS[ds])}, **{f"logit__{c}": float(lg[i][j]) for j, c in enumerate(CLASS_ORDERS[ds])}})
                        embs.append(z[i])
            model.train(); heads.train()
            df = pd.DataFrame(rows); return evaluate_split(df), df, np.stack(embs)

        for epoch in range(start_epoch, epochs):
            model.train(); heads.train(); t0 = time.time(); losses = []; aux_losses = []
            for k in range(spe):
                batch = stream[epoch * spe + k]; opt.zero_grad(); total = 0.0; n_ds = len({dd for dd, _, _ in batch})
                cnt_batch = {dd: sum(1 for x in batch if x[0] == dd) for dd in {x[0] for x in batch}}
                for lo in range(0, len(batch), mb):
                    chunk = batch[lo:lo + mb]; loss, aux = forward_loss(chunk)
                    # exact full-batch objective: L = mean_ds CE_ds; chunk loss = mean over datasets present of the chunk CE,
                    # so weight = sum_ds_in_chunk (n_ds_chunk / n_ds_batch) / n_ds  (equals k/n_ds when chunks hold whole datasets)
                    cnt = {dd: sum(1 for x in chunk if x[0] == dd) for dd in {x[0] for x in chunk}}
                    w = sum(cnt[dd] / cnt_batch[dd] for dd in cnt) / n_ds
                    objective = loss * w if aux is None else loss * w + (w_cwru * CI.XSPEC_LAMBDA) * aux      # L_total = L_existing + w_CWRU * lambda * L_cross-specimen
                    objective.backward(); total += float(loss.detach()) * w
                    if aux is not None:
                        aux_losses.append(float(aux.detach()))
                torch.nn.utils.clip_grad_norm_(params, OPTIMIZER_SPEC["grad_clip_global_norm"]); opt.step(); sched.step(); losses.append(total)
            t_train = time.time() - t0; t0 = time.time(); rep, _, _ = validate(); t_val = time.time() - t0
            metric = rep["macro_domain_f1"]
            improved = metric > best["metric"]
            if improved:
                best = {"metric": metric, "epoch": epoch}
                torch.save({"run_id": run_id, "epoch": epoch, "metric": metric, "model": {k: v.cpu() for k, v in model.state_dict().items()}, "heads": {k: v.cpu() for k, v in heads.state_dict().items()},
                            "config_sha256": fp["config_sha256"], "cfg": cfg}, d / "best.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(), "heads": heads.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "best": best,
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "config_sha256": fp["config_sha256"], "stream_sha256": sh}, d / "last.pt")
            em = {"epoch": epoch, "train_loss_mean": float(np.mean(losses)), "val_macro_domain_f1": metric, "val_macro3_f1": rep["macro3_f1_excl_cwru"],
                  "val_per_dataset_macro_f1": {ds: rep["per_dataset_reports"][ds]["macro_f1"] for ds in rep["per_dataset_reports"]},
                  "val_cwru_per_class_f1": rep.get("cwru_per_class_f1"), "val_cwru_per_specimen_recall": rep.get("cwru_per_specimen_recall"), "best_epoch": best["epoch"],
                  "lr": sched.get_last_lr()[0], "seconds_train": round(t_train, 1), "seconds_val": round(t_val, 1), "at": datetime.now(timezone.utc).isoformat(),
                  **({"train_xspec_loss_mean": float(np.mean(aux_losses)), "n_xspec_groups": len(aux_losses)} if ci["xspec"] else {})}
            with open(d / "epoch_metrics.jsonl", "a") as f:
                f.write(json.dumps(em) + "\n")
            log(d, f"epoch {epoch:02d} loss {em['train_loss_mean']:.4f} valMDF1 {metric:.4f} CWRU {em['val_per_dataset_macro_f1']['CWRU']:.3f} best {best['epoch']} ({best['metric']:.4f}) {t_train:.0f}s+{t_val:.0f}s")
        # ---- final validation outputs of the selected checkpoint ------------------------------------------------
        ck = torch.load(d / "best.pt", map_location=dev, weights_only=False); model.load_state_dict(ck["model"]); heads.load_state_dict(ck["heads"])
        rep, df, embs = validate()
        df.insert(0, "seed", seed); df.insert(0, "fold", fold); df.insert(0, "run_id", run_id); df["best_epoch"] = ck["epoch"]
        df.to_csv(d / "validation_predictions.csv", index=False)
        np.savez_compressed(d / "validation_embeddings.npz", window_id=df.window_id.values, embedding=embs.astype(np.float32), y_true=df.y_true.values, y_pred=df.y_pred.values, dataset=df.dataset.values)
        rep["best_epoch"] = ck["epoch"]; rep["run_id"] = run_id; rep["best_checkpoint_sha256"] = sha256_file(d / "best.pt")
        (d / "validation_report.json").write_text(json.dumps(rep, indent=1, default=float))
        hashes = {p.name: sha256_file(p) for p in sorted(d.iterdir()) if p.is_file() and p.name not in ("hashes.json", "LOCK", "STATUS", "last.pt")}
        (d / "hashes.json").write_text(json.dumps(hashes, indent=1))
        (d / "STATUS").write_text("COMPLETE"); log(d, f"{run_id}: COMPLETE best epoch {ck['epoch']} valMDF1 {rep['macro_domain_f1']:.4f}")
    except Exception:
        (d / "STATUS").write_text("FAILED"); log(d, "FAILED\n" + traceback.format_exc()); raise
    finally:
        if lock.exists():
            lock.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS)); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--protocol", default="global_v2_PB_sealedMAF_v1"); ap.add_argument("--stage", default="b"); ap.add_argument("--run-id", default=None)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="NOT_AN_EXPERIMENT: 2 steps x 1 epoch on a tiny subset")
    ap.add_argument("--authorize-launch", default=None, help="native12 protocols only: the freeze digest, given explicitly by the researcher at launch")
    ap.add_argument("--init-encoder", default=None, help="S1 only: repo-relative path to the selected SSL best_encoder.pt whose encoder state initialises this run")
    run(ap.parse_args())
