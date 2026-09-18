#!/usr/bin/env python
"""methodology_v2 PART 6 — PC-STE lightweight study: CLI / executor.

Epoch loops live HERE (outside src/methodology_v2/compression, which is
guard-tested to contain none). Every command below is one of:

  SAFE NOW (CPU, no TEST, no training):
    audit            repository/primary-state audit relevant to Part 6
    write-specs      (re)write protocol/spec files + TEMPLATE registry
    registry         --list / --dry-run listing (nothing loads TEST)
    scan-parity      synthetic parity suite (+ --real later, VAL windows)
    pending          print pending pre-registration decisions
    benchmark        four-axis harness (refuses latency while host busy)
  GATED (need the sealed registry AND the human LAUNCH_AUTHORIZED marker):
    seal             final registry + part6_hashes.csv (needs all primary
                     checkpoints + all pending decisions resolved)
    cache-teachers   Stage 2 teacher caching (TRAIN+VAL only)
    ptq              Stage 1 Q8 on frozen checkpoints (VALIDATION only)
    sensitivity      Stage 2 training-free maps (VALIDATION only)
    run / drive      Stage 3/4 training runs (frozen recipe)
    aggregate        validation aggregation + pairing checks
    pretest-ledger   Stage 5 step 1 (to be committed)
    test-session     Stage 5 step 2 — THE single sealed TEST session
    stats            pre-registered statistics on the sealed TEST outputs
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.methodology_v2.integrity import sha256_file  # noqa: E402
from src.methodology_v2.part2_builder import verify_frozen_hashes  # noqa: E402
from src.methodology_v2.part3b_windows import (PART3B_DIR,  # noqa: E402
                                               verify_part3b_hashes)
from src.methodology_v2.part4c_normalizers import verify_part4c_hashes  # noqa: E402
from src.methodology_v2.part4c_reader import get_representation  # noqa: E402
from src.methodology_v2.encoder import collate_representations  # noqa: E402
from src.methodology_v2.experiment.heads import (CLASS_ORDERS,  # noqa: E402
                                                 LABEL_FIELD, DatasetHeads,
                                                 head_seed)
from src.methodology_v2.experiment.label_subsets import FRACTIONS  # noqa: E402
from src.methodology_v2.experiment.metrics import (classification_report,  # noqa: E402
                                                   macro_domain_f1)
from src.methodology_v2.experiment.registry import (PART5D_DIR,  # noqa: E402
                                                    SUBSET_DIR,
                                                    verify_part5d_hash)
from src.methodology_v2.experiment.samplers import (SSLSampler,  # noqa: E402
                                                    SupervisedSampler)
from src.methodology_v2.experiment.trainers import (DOWNSTREAM_EPOCHS,  # noqa: E402
                                                    OPTIMIZER_SPEC)
from src.methodology_v2.compression import protocol as P  # noqa: E402
from src.methodology_v2.compression.guards import (  # noqa: E402
    Part6GuardError, assert_no_test_windows, assert_read_only_primary,
    assert_same_cell, load_checkpoint_payload, load_fold_manifest,
    primary_run_id, resolve_checkpoint)
from src.methodology_v2.compression.losses import LossConfig  # noqa: E402
from src.methodology_v2.compression.registry import (  # noqa: E402
    REGISTRY_CSV, build_part6_registry, dry_run_listing, loss_config_for,
    registry_hash, seal_part6, spec_for, verify_part6_seal, write_registry)
from src.methodology_v2.compression.student import (  # noqa: E402
    FULL_SPEC, STUDENT_D_SPEC, architecture_summary, build_encoder,
    build_heads, half_4x1_spec, student_dw_spec)
from src.methodology_v2.compression.teachers import (  # noqa: E402
    TeacherCache, build_teacher_cache, cache_paths, discover_teacher_set,
    saturation_diagnostic)
from src.methodology_v2.compression.trainer import ArmConfig, Part6Trainer  # noqa: E402
from src.methodology_v2.compression import quantization as Q  # noqa: E402
from src.methodology_v2.compression import sensitivity as S  # noqa: E402
from src.methodology_v2.compression import scan_fast as SF  # noqa: E402
from src.methodology_v2.compression import stats as ST  # noqa: E402
from src.methodology_v2.compression import test_policy as TP  # noqa: E402
from src.methodology_v2.compression import benchmark as BM  # noqa: E402
from src.methodology_v2.compression import workqueue as WQ  # noqa: E402
from src.methodology_v2.compression import assignment as ASG  # noqa: E402

PART6_DIR = P.PART6_DIR
RESULTS6 = P.PART6_RESULTS
RUNS = RESULTS6 / "runs"
LAUNCH_MARKER = PART6_DIR / "LAUNCH_AUTHORIZED"
DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")
EXECUTOR_VERSION = "part6-cli-1.0"


def git_head() -> str | None:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def device_provenance(device: str) -> dict:
    return {"host": socket.gethostname(),
            "gpu_name": (torch.cuda.get_device_name(0)
                         if device.startswith("cuda")
                         and torch.cuda.is_available() else None),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "git_head": git_head(),
            "registry_sha256": (sha256_file(PART6_DIR / REGISTRY_CSV)
                                if (PART6_DIR / REGISTRY_CSV).exists()
                                else None),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}",
          flush=True)


def _jsonable(o):
    return P._jsonable(o)


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_jsonable))


# ---------------------------------------------------------------------------
# resolved pending decisions
# ---------------------------------------------------------------------------
def load_resolved(path: Path | None = None) -> dict:
    """resolved_decisions.yaml (JSON) — written by the human before seal;
    values are validated against the alternatives (or must be explicitly
    marked custom)."""
    path = path or (PART6_DIR / "resolved_decisions.yaml")
    if not path.exists():
        return {}
    d = P.load_spec(path)
    out = {}
    for k, v in d.items():
        if k not in P.PENDING_DECISIONS:
            raise Part6GuardError(f"unknown decision key {k}")
        out[k] = v
    return out


def require_seal_and_authorization(cmd: str) -> None:
    verify_part6_seal()
    if not LAUNCH_MARKER.exists():
        raise SystemExit(f"{cmd}: refused — {LAUNCH_MARKER} missing. Part-6 "
                         "execution needs explicit human authorization.")


def verify_primary_seals() -> dict:
    verify_frozen_hashes()
    verify_part3b_hashes()
    verify_part4c_hashes()
    verify_part5d_hash()
    return {"part2/3b/4c/5d": "verified", "at": datetime.now(timezone.utc).isoformat()}


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def audit() -> None:
    log("PART 6 AUDIT (read-only; never opens test_report.json)")
    rows = []
    for arm in ("s0", "s1"):
        for f in P.FOLDS:
            for s in P.SEEDS:
                rid = primary_run_id(arm, f, s, 100)
                d = P.PRIMARY_DOWNSTREAM / rid
                st = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() else None
                seal = (d / "test_seal.json").exists()
                rows.append({"run_id": rid, "status": st["status"] if st else "REGISTERED",
                             "best_epoch": (st or {}).get("best_epoch"),
                             "checkpoint_frozen": seal})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    n_done = int((df["status"] == "COMPLETE").sum())
    print(f"\nprimary downstream COMPLETE: {n_done}/18 — Part-6 final registry "
          f"{'CAN' if n_done == 18 else 'CANNOT yet'} be sealed")
    print(f"Part-6 protocol dir: {PART6_DIR} "
          f"({'sealed' if (PART6_DIR / 'part6_hashes.csv').exists() else 'NOT sealed'})")
    print(f"launch marker: {'PRESENT' if LAUNCH_MARKER.exists() else 'absent'}")
    for spec_name, spec in (("full", FULL_SPEC), ("student_d", STUDENT_D_SPEC),
                            ("student_dw", student_dw_spec())):
        enc = build_encoder(spec, seed=0)
        hd = build_heads(spec, 0)
        a = architecture_summary(enc, hd)
        print(f"{spec_name}: params enc {a['encoder_params']:,} heads "
              f"{a['head_params']:,} blocks {a['n_blocks']} scan steps "
              f"{a['scan_steps_T23']}/{a['scan_steps_T24']}")


# ---------------------------------------------------------------------------
# specs + template registry
# ---------------------------------------------------------------------------
def write_specs(resolved: dict | None = None, final: bool = False) -> None:
    resolved = resolved if resolved is not None else load_resolved()
    PART6_DIR.mkdir(parents=True, exist_ok=True)
    if (PART6_DIR / "part6_hashes.csv").exists() and not final:
        raise SystemExit("Part 6 is SEALED — refusing to rewrite spec files "
                         "(delete the seal only by an explicit, documented "
                         "decision)")
    P.dump_spec(P.protocol_document(resolved), PART6_DIR / "protocol.yaml")
    st = {}
    for name, spec in (("full", FULL_SPEC), ("student_d", STUDENT_D_SPEC),
                       ("student_dw", student_dw_spec(
                           resolved.get("student_dw_stem_rank")))):
        enc = build_encoder(spec, seed=0)
        hd = build_heads(spec, 0)
        st[name] = {"spec": spec.to_dict(),
                    "architecture_hash": P.config_hash(spec.to_dict()),
                    **architecture_summary(enc, hd)}
    var = resolved.get("compact_student_variant")
    if var:
        cs = spec_for("k1", resolved)
        enc = build_encoder(cs, seed=0)
        st["compact_student_RESOLVED"] = {
            "variant": var, "spec": cs.to_dict(),
            "architecture_hash": P.config_hash(cs.to_dict()),
            **architecture_summary(enc, build_heads(cs, 0)),
            "chosen_by": "pre-registered Stage-2 rule (median validation "
                         "MacroDomainF1 after training-free removal), "
                         "executed before sealing — see "
                         "half_student_decision.json",
            "surgery": ("student layer i <- teacher layer retained[i]"
                        if var == "2x2" else
                        "keep the fwd direction of every layer + all norms/"
                        "stem/coords/mixer/heads verbatim; drop bwd blocks")}
    st["student_d_surgery"] = {
        "retained_layers": resolved.get("student_d_retained_layers", "PENDING"),
        "copied_verbatim": ["stem", "coords", "temporal.norm (final LN)",
                            "mixer", "heads"],
        "mapping": "student temporal.layers.i <- teacher temporal.layers."
                   "retained[i]; every tensor shape-checked; strict load"}
    from src.methodology_v2.compression.student import half_4x1_spec
    h41 = build_encoder(half_4x1_spec(), seed=0)
    st["half_student_alternative_4x1"] = {
        "spec": half_4x1_spec().to_dict(),
        **architecture_summary(h41, build_heads(half_4x1_spec(), 0)),
        "note": "equal-cost comparator for the pre-registered Stage-2 rule "
                "(median validation MacroDomainF1 after training-free "
                "removal); never the a-priori primary"}
    P.dump_spec(st, PART6_DIR / "student_spec.yaml")
    P.dump_spec({
        "recipe": "Q8", "weights": "per-output-channel symmetric int8, "
                                   "zero-point 0, scale=max|w_row|/127",
        "allowlist_patterns": list(Q.ALLOW_PATTERNS),
        "denylist_patterns": list(Q.DENY_PATTERNS),
        "fp32_tensor_patterns": list(Q.FP32_TENSOR_PATTERNS),
        "stays_fp32": ["dt_proj", "A_log", "D", "conv1d", "all LayerNorms",
                       "recurrent state / exp(delta*A) path (not a "
                       "parameterised module — untouched by construction)"],
        "deployment_representations": {
            "sim": "int8 weight-only, dequantised fp32 compute (CPU=GPU numerics; "
                   "the accuracy representation)",
            "cpu_dynamic": "torch.ao dynamic quantization, per-channel int8 "
                           "weights + dynamic int8 activations, allowlisted "
                           "modules only (qualified-name qconfig dict)"},
        "size_measurement": "actual torch.save bytes of the compact int8 state "
                            "and of the torch.ao state (never an estimate)",
        "latency_claim": "NONE for Q8 (weight-only int8 does not speed the "
                         "reference scan)",
        "ni_margin": P.NI_MARGIN_PTQ,
        "exploratory_val_only": ["fp16_all_but_sensitive", "w4_per_group_in_out_proj",
                                 "static_w8a8_train_calibrated",
                                 "leave_one_tensor_sensitivity"],
        "plan_module": "src/methodology_v2/compression/quantization.py"},
        PART6_DIR / "quantization_spec.yaml")
    P.dump_spec({
        "loss": "(1-alpha)*CE + alpha*T^2*KL(teacher_T || student_T) on the "
                "window's OWN dataset head",
        "T": P.KD_TEMPERATURE, "alpha": P.KD_ALPHA, "fixed_a_priori": True,
        "kl": "forward KL(teacher||student), sum over classes, per window",
        "reduction": "per-dataset window mean, then mean over datasets present "
                     "(identical to primary L_sup)",
        "relational_term": {"form": "KL(alpha_teacher || alpha_student) over "
                                    "VALID mixer bands, same-cell single "
                                    "teacher (cached alpha), renormalised over "
                                    "valid bands, padded bands excluded",
                            "weight": resolved.get("relational_alpha_kl_weight",
                                                   "PENDING_PREREG")},
        "teacher_sets": {"k1": "S1(f,42)+S1(f,1337)+S1(f,2026)",
                         "k0": "S0(f,42)+S0(f,1337)+S0(f,2026)",
                         "b1": "same as k1 (full-size student)",
                         "f1": "registered 10%-label S1(f,s) (single)"},
        "ensemble_rule": resolved.get("ensemble_rule", "PENDING_PREREG"),
        "arms": {a: {"architecture": P.ARCH_OF_ARM[a], "init": P.INIT_OF_ARM[a],
                     "teacher_set": P.TEACHER_SET_OF_ARM[a],
                     "loss": P.LOSS_OF_ARM[a],
                     "tier": ("core" if a in P.CORE_ARMS else
                              "optional" if a in P.OPTIONAL_ARMS else "push")}
                 for a in P.ALL_ARMS},
        "b0_label_smoothing": P.LABEL_SMOOTHING_B0,
        "cache": "TRAIN+VAL only; per-seed raw logits of all 4 heads, 192-d "
                 "embedding, mixer alpha, band summaries (fp16); keyed by "
                 "window id + dataset + split + teacher hashes + encoder "
                 "config hash; TEST ids refused structurally"},
        PART6_DIR / "kd_spec.yaml")
    P.dump_spec({
        "paired_unit": "fold x seed (9 cells)",
        "report": ["all 9 deltas", "mean", "median", "SD", "fraction positive",
                   "effect size mean/SD"],
        "sign_flip": "exact, all 512 patterns",
        "non_inferiority": "one-sided exact sign-flip on MARGIN-SHIFTED deltas "
                           "(delta + m); H0: mean(delta) <= -m",
        "confirmatory_family_holm_m3": {"H1": "K1 vs S1: NI 0.02 then two-sided "
                                              "superiority (hierarchical)",
                                        "H2": "K1 vs C_small: two-sided",
                                        "H3": "Q8(K1) vs K1: NI 0.01"},
        "secondary_family_holm_m4": ["K0 vs S0 NI 0.02", "K1 vs K0 two-sided",
                                     "Q8(S1) vs S1 NI 0.01", "Q8(S0) vs S0 NI 0.01"],
        "push_arms": {"rule": "mean delta >= 0.02 AND two-sided p < 0.05 else "
                              "'not distinguishable'",
                      "contrasts": ["B1 vs S1", "F1 vs S1-10%"]},
        "sensitivity": ["per-dataset deltas", "MacroDomainF1 excluding CWRU"],
        "power_note": "n=9: smallest two-sided p 0.0039; Holm superiority needs "
                      "~8-9/9 positive cells; nulls are legitimate results",
        "module": "src/methodology_v2/compression/stats.py"},
        PART6_DIR / "statistics_spec.yaml")
    P.dump_spec({
        "policy": "ONE sealed TEST session after all Part-6 training",
        "steps": ["all registered runs COMPLETE", "pretest-ledger written AND "
                  "committed", "test-session: test_seal.json BEFORE TEST load",
                  "one evaluation per ledger model", "test_touch_ledger.csv "
                  "append-only", "session_summary.json"],
        "stage_0_4_guard": "guards.assert_no_test_windows on every entry; TEST "
                           "manifest access needs a TestSessionToken minted "
                           "only by test_policy.open_test_session",
        "second_touch": "forbidden unless integrity_failures.json documents it",
        "primary_results_visible_at_sealing": "some Phase-B TEST values exist "
                                              "in results/methodology_v2/"
                                              "downstream/*/test_report.json; "
                                              "Part 6 never opens them and no "
                                              "Part-6 rule derives from them "
                                              "(disclosed)"},
        PART6_DIR / "test_policy.yaml")
    P.dump_spec({
        "axes": {"size": ["encoder params", "head params", "total", "fp32 bytes",
                          "int8 bytes (measured)"],
                 "compute": ["FlopCounter GFLOP per dataset shape",
                             "analytic scan GFLOP", "scan steps", "R"],
                 "latency": {"cpu": {"batches": [1, 16], "threads": [1, 4],
                                     "warmup": 5, "timed": 20,
                                     "stat": "median + IQR", "abab": True},
                             "gpu": {"batches": [16, 64], "sync": True,
                                     "only_when_idle": True}},
                 "memory": "VmHWM of a fresh subprocess (3 forwards); "
                           "cuda.max_memory_allocated on GPU"},
        "energy": "not measured (RAPL root-only); CPU-seconds flagged proxy",
        "busy_host_rule": "refuse latency when load1>1.5 or GPU util>20% unless "
                          "--allow-busy (tagged)",
        "caveat_on_every_table": BM.BACKEND_CAVEAT},
        PART6_DIR / "measurement_spec.yaml")
    reg = build_part6_registry(resolved, require_checkpoints=final)
    write_registry(reg)
    P.dump_spec({"pending": P.pending_summary(),
                 "resolved": resolved,
                 "unresolved": P.unresolved_pending(resolved)},
                PART6_DIR / "pending_decisions.yaml")
    log(f"specs written to {PART6_DIR} ; registry rows={len(reg)} "
        f"(enabled {int(reg['enabled'].sum())}) hash={registry_hash(reg)[:12]} "
        f"final={final}")


def cmd_registry(a) -> None:
    resolved = load_resolved()
    reg = build_part6_registry(resolved, require_checkpoints=a.final)
    if a.list or a.dry_run:
        print(dry_run_listing(reg, only_enabled=not a.all))
        print(f"\nrows: {len(reg)} enabled: {int(reg['enabled'].sum())} "
              f"registry hash: {registry_hash(reg)}")
        st = reg["status"].value_counts().to_dict()
        print(f"status counts: {st}")
        print("NOTHING was trained, no TEST loaded, no checkpoint opened "
              "beyond hash verification.")


def cmd_seal(a) -> None:
    resolved = load_resolved()
    miss = P.unresolved_pending(resolved)
    if miss:
        raise SystemExit(f"cannot seal: unresolved pending decisions {miss} — "
                         f"write {PART6_DIR / 'resolved_decisions.yaml'}")
    verify_primary_seals()
    write_specs(resolved, final=True)
    mh = seal_part6()
    log(f"PART 6 SEALED master hash {mh}")


# ---------------------------------------------------------------------------
# rep store (mirrors the primary executor's in-RAM store, bit-equality)
# ---------------------------------------------------------------------------
class RepStore:
    def __init__(self, fold: int, manifest: pd.DataFrame):
        self.fold, self.man, self.store = fold, manifest, {}

    def preload(self, window_ids: list[str], allow_test: bool = False) -> float:
        if not allow_test:
            assert_no_test_windows(window_ids, "RepStore.preload")
        todo = [w for w in window_ids if w not in self.store]
        sub = self.man.loc[todo]
        order = sub.sort_values(["dataset", "source_file", "start_sample"]).index
        t0 = time.time()
        for wid in order:
            x, meta = get_representation(wid, self.fold)
            self.store[wid] = (x, np.asarray(meta["frequency_hz"], np.float32),
                               np.asarray(meta["time_seconds"], np.float32))
        return time.time() - t0

    def rep(self, wid: str) -> tuple:
        return self.store[wid]

    def equivalence_check(self, k: int = 16, seed: int = 0) -> int:
        wids = sorted(self.store)
        rng = np.random.default_rng(seed)
        for i in rng.choice(len(wids), size=min(k, len(wids)), replace=False):
            fresh, meta = get_representation(wids[i], self.fold)
            x, f, t = self.store[wids[i]]
            assert np.array_equal(fresh, x)
            assert np.array_equal(np.asarray(meta["frequency_hz"], np.float32), f)
        return min(k, len(wids))


def ids_by_dataset(man: pd.DataFrame, ids: list[str]) -> dict:
    sub = man.loc[ids]
    return {ds: sorted(sub.index[sub["dataset"] == ds]) for ds in DATASETS}


def split_ids(man: pd.DataFrame, split: str) -> list[str]:
    return list(man.index[man["split"] == split])


# ---------------------------------------------------------------------------
# Stage 0: scan parity
# ---------------------------------------------------------------------------
def cmd_scan_parity(a) -> None:
    out = {"synthetic": SF.synthetic_parity_suite(chunk=a.chunk),
           "cpu_timing_full_shape": SF.time_backends(528, 24, 384, 16, reps=2,
                                                     chunk=a.chunk),
           "compile_probe": SF.maybe_compile(SF.selective_scan_reference)[1],
           "at": datetime.now(timezone.utc).isoformat()}
    if a.real:
        require_seal_and_authorization("scan-parity --real")
        # 1,000 VALIDATION windows through finished checkpoints:
        # max|dy| of the full model + prediction agreement.
        out["real"] = real_scan_parity(a.chunk, a.n_windows, a.device)
    p = RESULTS6 / "stage0" / "scan_parity.json"
    dump_json(out, p)
    print(json.dumps({k: v for k, v in out.items() if k != "synthetic"},
                     indent=1, default=_jsonable))
    print("synthetic all_pass:", out["synthetic"]["all_pass"])
    log(f"written {p} (approval file must be written by a human: "
        f"{SF.APPROVAL_FILE})")


def real_scan_parity(chunk: int, n_windows: int, device: str) -> dict:
    res = {"backends": {}}
    rng = np.random.default_rng(0)
    for name in ("chunked", "compiled_reference"):
        worst, agree, n = 0.0, 0, 0
        for f in P.FOLDS:
            man = load_fold_manifest(f)
            val = split_ids(man, "validation")
            pick = [val[i] for i in rng.choice(len(val), size=min(
                n_windows // 3, len(val)), replace=False)]
            store = RepStore(f, man)
            store.preload(pick)
            for s in P.SEEDS:
                ref = resolve_checkpoint(primary_run_id("s1", f, s, 100))
                ck = load_checkpoint_payload(ref)
                enc = build_encoder(FULL_SPEC)
                enc.load_state_dict(ck["encoder"])
                hd = DatasetHeads()
                hd.load_state_dict(ck["heads"])
                enc.to(device).eval()
                hd.to(device).eval()
                by = ids_by_dataset(man, pick)
                for ds in DATASETS:
                    for lo in range(0, len(by[ds]), 32):
                        sub = by[ds][lo:lo + 32]
                        b = collate_representations([store.rep(w) for w in sub])
                        b = {k: v.to(device) for k, v in b.items()}
                        with torch.no_grad():
                            z0 = enc(**b)["global_embedding"]
                            l0 = hd(z0, ds)
                            with SF.use_scan_backend(name, chunk=chunk,
                                                     require_approval=False):
                                z1 = enc(**b)["global_embedding"]
                                l1 = hd(z1, ds)
                        worst = max(worst, float((z1 - z0).abs().max()))
                        agree += int((l0.argmax(-1) == l1.argmax(-1)).sum())
                        n += len(sub)
        res["backends"][name] = {"max_abs_embedding_diff": worst,
                                 "prediction_agreement": agree / max(n, 1),
                                 "n_windows_x_models": n,
                                 "pass": worst < SF.PARITY_MAX_ABS and agree == n}
    return res


# ---------------------------------------------------------------------------
# Stage 2: teacher caching
# ---------------------------------------------------------------------------
def cmd_cache_teachers(a) -> None:
    require_seal_and_authorization("cache-teachers")
    verify_primary_seals()
    resolved = P.load_spec(PART6_DIR / "protocol.yaml")["pending_decisions"]
    rule = resolved["ensemble_rule"]["value"]
    for fold in a.folds:
        for tset_name in a.teacher_sets:
            tset = discover_teacher_set(tset_name, fold, rule)
            man = load_fold_manifest(fold)               # TRAIN+VAL only
            ids = split_ids(man, "train") + split_ids(man, "validation")
            store = RepStore(fold, man)
            t = store.preload(ids)
            eq = store.equivalence_check(seed=fold)
            log(f"fold {fold} {tset_name}: preloaded {len(ids)} in {t:.0f}s "
                f"(bit-equal {eq}); teachers {[r.run_id for r in tset.refs]}")
            npz, js = build_teacher_cache(
                tset, man, store.rep, device=a.device, chunk=a.chunk,
                with_band_summaries=not a.no_band_summaries)
            cache = TeacherCache(tset_name, fold, expected_hashes=tset.hashes)
            diag = saturation_diagnostic(cache, rule)
            dump_json(diag, RESULTS6 / "teacher_cache" /
                      f"saturation_{tset_name}_f{fold}.json")
            log(f"cache written {npz} ({npz.stat().st_size / 2**20:.0f} MB); "
                f"saturation diagnostic written")


# ---------------------------------------------------------------------------
# Stage 1: PTQ (validation only here; TEST via Stage 5 ledger)
# ---------------------------------------------------------------------------
def load_model_for(model_id: str, variant: str, device: str = "cpu"):
    """model_id = primary run id (s0/s1_f_s_l100) or Part-6 run id."""
    if model_id.startswith(("s0_", "s1_")):
        ref = resolve_checkpoint(model_id)
        ck = load_checkpoint_payload(ref)
        spec, sha = FULL_SPEC, ref.sha256
    else:
        d = RUNS / model_id
        st = json.loads((d / "state.json").read_text())
        if st["status"] != "COMPLETE":
            raise Part6GuardError(f"{model_id} not COMPLETE")
        ck = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
        sha = sha256_file(d / "best.pt")
        arch = st["arm_config"]["architecture"]
        name = arch["name"]
        if name == "full":
            spec = FULL_SPEC
        elif name == "student_d":
            spec = STUDENT_D_SPEC
        elif name == "half_4x1":
            spec = half_4x1_spec(arch.get("uni_residual", "mean_of_remaining"))
        elif name == "student_dw":
            spec = student_dw_spec(arch.get("stem_rank"))
        else:
            raise Part6GuardError(f"unknown architecture {name}")
    enc = build_encoder(spec)
    enc.load_state_dict(ck["encoder"], strict=True)
    hd = build_heads(spec, 0)
    hd.load_state_dict(ck["heads"], strict=True)
    if variant == "q8":
        enc = Q.apply_q8_simulated(enc).model
        hd = Q.apply_q8_simulated(hd).model
    elif variant != "fp32":
        raise Part6GuardError(f"unknown variant {variant}")
    return enc.to(device).eval(), hd.to(device).eval(), sha, spec


def cmd_ptq(a) -> None:
    require_seal_and_authorization("ptq")
    out_dir = RESULTS6 / "ptq"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_id in a.models:
        enc, hd, sha, spec = load_model_for(model_id, "fp32", a.device)
        rep = Q.q8_report(enc)
        rep_h = Q.q8_report(hd, include_dynamic=False)
        rec = {"model_id": model_id, "checkpoint_sha256": sha,
               "encoder": rep, "heads": rep_h,
               "int8_total_bytes": rep["int8_compact_state_bytes"]
               + rep_h["int8_compact_state_bytes"],
               "fp32_total_bytes": rep["fp32_state_bytes"] + rep_h["fp32_state_bytes"]}
        if a.validate:
            fold = int(model_id.split("_f")[1][0])
            man = load_fold_manifest(fold)
            val = split_ids(man, "validation")
            store = RepStore(fold, man)
            store.preload(val)
            by = ids_by_dataset(man, val)
            r32 = S.evaluate_split(enc, hd, store.rep, by, man, a.device)
            q_enc, q_hd, _, _ = load_model_for(model_id, "q8", a.device)
            r8 = S.evaluate_split(q_enc, q_hd, store.rep, by, man, a.device)
            rec["validation"] = {"fp32": r32["macro_domain_f1"],
                                 "q8": r8["macro_domain_f1"],
                                 "delta": r8["macro_domain_f1"] - r32["macro_domain_f1"],
                                 "per_dataset_fp32": r32["per_dataset_macro_f1"],
                                 "per_dataset_q8": r8["per_dataset_macro_f1"]}
            if a.exploratory:
                rec["exploratory_VAL_ONLY"] = exploratory_sweep(
                    enc, hd, store, by, man, a.device, split_ids(man, "train"))
        dump_json(rec, out_dir / f"ptq_{model_id}.json")
        rows.append({"model_id": model_id, "fp32_bytes": rec["fp32_total_bytes"],
                     "int8_bytes": rec["int8_total_bytes"],
                     **({"val_fp32": rec["validation"]["fp32"],
                         "val_q8": rec["validation"]["q8"]} if a.validate else {})})
        log(f"ptq {model_id}: fp32 {rec['fp32_total_bytes'] / 2**20:.2f} MB -> "
            f"int8 {rec['int8_total_bytes'] / 2**20:.2f} MB")
    pd.DataFrame(rows).to_csv(out_dir / "PTQ_TABLE.csv", index=False)


def exploratory_sweep(enc, hd, store, by, man, device, train_ids) -> dict:
    """VAL-only exploratory PTQ variants (never TEST, never registered)."""
    out = {"tag": Q.EXPLORATORY_TAG}
    m16 = Q.apply_fp16_all_but_sensitive(enc)
    out["fp16_all_but_sensitive"] = S.evaluate_split(
        m16, hd, store.rep, by, man, device)["macro_domain_f1"]
    m4 = Q.apply_w4_inout_proj(enc)
    out["w4_in_out_proj"] = S.evaluate_split(
        m4, hd, store.rep, by, man, device)["macro_domain_f1"]
    obs = Q.StaticActObserver()
    plan = Q.q8_module_plan(enc)
    obs.attach(enc, plan)
    assert_no_test_windows(train_ids, "static calibration")
    calib = [w for w in train_ids if man.loc[w, "split"] == "train"][:256]
    store.preload(calib)
    with torch.no_grad():
        for lo in range(0, len(calib), 32):
            b = collate_representations([store.rep(w) for w in calib[lo:lo + 32]])
            enc(**{k: v.to(device) for k, v in b.items()})
    obs.detach()
    m88 = Q.apply_static_w8a8(enc, obs.amax)
    out["static_w8a8_train_calibrated"] = S.evaluate_split(
        m88, hd, store.rep, by, man, device)["macro_domain_f1"]
    lot = {}
    for name, m1 in Q.leave_one_tensor_variants(enc):
        lot[name] = S.evaluate_split(m1, hd, store.rep, by, man,
                                     device)["macro_domain_f1"]
    out["leave_one_tensor_quantized"] = lot
    return out


# ---------------------------------------------------------------------------
# Stage 2: sensitivity maps
# ---------------------------------------------------------------------------
def cmd_sensitivity(a) -> None:
    require_seal_and_authorization("sensitivity")
    proto = P.load_spec(PART6_DIR / "protocol.yaml")["pending_decisions"]
    retained = proto["student_d_retained_layers"]["value"]
    dirvar = proto["half_student_direction_variant"]["value"]
    out_dir = RESULTS6 / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    half22, half41 = [], []
    for fold in a.folds:
        man = load_fold_manifest(fold)
        val = split_ids(man, "validation")
        train = split_ids(man, "train")
        store = RepStore(fold, man)
        store.preload(val)
        by = ids_by_dataset(man, val)
        train_by = {ds: v[:a.train_stat_windows] for ds, v in
                    ids_by_dataset(man, train).items()}
        store.preload([w for v in train_by.values() for w in v])
        for seed in P.SEEDS:
            rid = primary_run_id("s1", fold, seed, 100)
            enc, hd, sha, _ = load_model_for(rid, "fp32", a.device)
            rec = {"model": rid, "sha256": sha, "split": "validation"}
            base = S.evaluate_split(enc, hd, store.rep, by, man, a.device)
            rec["baseline"] = base["macro_domain_f1"]
            rec["drop_one_layer"] = {
                i: S.evaluate_split(S.drop_layer(enc, i), hd, store.rep, by,
                                    man, a.device)["macro_domain_f1"]
                for i in range(4)}
            rec["drop_one_direction"] = {
                f"L{i}_{d}_{res}": S.evaluate_split(
                    S.drop_direction(enc, i, d, res), hd, store.rep, by, man,
                    a.device)["macro_domain_f1"]
                for i in range(4) for d in ("fwd", "bwd")
                for res in ("mean_of_remaining", "keep_half_scale")}
            scores = S.accumulate_train_stats(enc, hd, store.rep, train_by, man,
                                              a.device)
            rec["channel_pruning"] = {}
            for sc in ("taylor", "abs_D", "mean_softplus_delta",
                       "activation_variance"):
                for kf in (0.75, 0.5):
                    m = S.prune_channels(enc, scores, sc, keep_fraction=kf)
                    rec["channel_pruning"][f"{sc}_keep{int(kf * 100)}"] = \
                        S.evaluate_split(m, hd, store.rep, by, man,
                                         a.device)["macro_domain_f1"]
                m = S.prune_channels(enc, scores, sc, keep_n=320)
                rec["channel_pruning"][f"{sc}_fallback320"] = S.evaluate_split(
                    m, hd, store.rep, by, man, a.device)["macro_domain_f1"]
            act = S.state_decay_activity(enc, store.rep, train_by, a.device)
            rec["d_state"] = {
                "keep8": S.evaluate_split(S.prune_states(enc, act, 8), hd,
                                          store.rep, by, man, a.device)["macro_domain_f1"],
                "fallback12": S.evaluate_split(S.prune_states(enc, act, 12), hd,
                                               store.rep, by, man, a.device)["macro_domain_f1"]}
            rec["time_merge_2to1"] = S.evaluate_split(
                S.merge_time_tokens_2to1(enc), hd, store.rep, by, man,
                a.device)["macro_domain_f1"]
            rec["band_occlusion"] = {
                b: S.evaluate_split(S.occlude_band(enc, b), hd, store.rep, by,
                                    man, a.device)["macro_domain_f1"]
                for b in range(33)}
            rec["low_rank_stem"] = {}
            for r in (16, 32, 64, 128):
                m, info = S.low_rank_stem(enc, r)
                rec["low_rank_stem"][r] = {
                    "val_f1": S.evaluate_split(m, hd, store.rep, by, man,
                                               a.device)["macro_domain_f1"],
                    "energy_captured": info["energy_captured"]}
            h22 = S.half_student_2x2(enc, retained)
            h41 = S.half_student_4x1(enc, dirvar["keep"], dirvar["residual"])
            v22 = S.evaluate_split(h22, hd, store.rep, by, man, a.device)["macro_domain_f1"]
            v41 = S.evaluate_split(h41, hd, store.rep, by, man, a.device)["macro_domain_f1"]
            rec["half_students_training_free"] = {"2x2": v22, "4x1": v41}
            half22.append(v22)
            half41.append(v41)
            dump_json(rec, out_dir / f"sensitivity_{rid}.json")
            log(f"sensitivity {rid}: base {base['macro_domain_f1']:.4f} "
                f"2x2 {v22:.4f} 4x1 {v41:.4f}")
    choice = S.choose_half_student(half22, half41)
    thr = proto["stage2_fallback_threshold"]["value"]
    thr_val = thr["max_drop"] if isinstance(thr, dict) else float(thr)
    base_med = float(np.median([json.loads((out_dir / f"sensitivity_{primary_run_id('s1', f, s, 100)}.json").read_text())["baseline"]
                                for f in a.folds for s in P.SEEDS]))
    chosen_med = choice["median_2x2"] if choice["chosen"] == "2x2" else choice["median_4x1"]
    fb = S.fallback_trigger(base_med - chosen_med, thr_val)
    dump_json({"half_student_choice": choice, "fallback": fb},
              out_dir / "half_student_decision.json")
    log(f"half-student rule: {choice} ; fallback: {fb}")


# ---------------------------------------------------------------------------
# Stage 3/4: run
# ---------------------------------------------------------------------------
def build_sup_stream(subset, fraction, seed, n_steps):
    s = SupervisedSampler(subset, fraction, seed)
    return [s.next_batch() for _ in range(n_steps)]


def build_ssl_stream(view, seed, n_steps):
    s = SSLSampler(view, seed)
    return [s.next_batch() for _ in range(n_steps)]


def stream_hash(stream) -> str:
    import hashlib
    h = hashlib.sha256()
    for batch in stream:
        for item in batch:
            h.update(("|".join(str(v) for v in item) + "\n").encode())
    return h.hexdigest()


def run_part6(run_id: str, resume: bool = False, device: str = "cuda") -> None:
    require_seal_and_authorization("run")
    seals = verify_primary_seals()
    reg = pd.read_csv(PART6_DIR / REGISTRY_CSV).set_index("run_id")
    row = reg.loc[run_id]
    if not bool(row["enabled"]) or row["status"] != "REGISTERED":
        raise SystemExit(f"{run_id}: not an enabled REGISTERED row (fail closed)")
    arm, fold, seed = str(row["arm"]), int(row["fold"]), int(row["seed"])
    spe, epochs = int(row["steps_per_epoch"]), int(row["max_epochs"])
    assert epochs == DOWNSTREAM_EPOCHS
    if arm == "f1":
        raise SystemExit("f1: batch composition is a pending decision and the "
                         "10%-label teachers are unregistered — not runnable")
    d = RUNS / run_id
    assert_read_only_primary(d)
    st = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() else None
    if st and st["status"] == "COMPLETE":
        log(f"{run_id}: already COMPLETE — skipping")
        return
    if st and not resume:
        raise SystemExit(f"{run_id}: state={st['status']}; restart needs --resume")
    proto = P.load_spec(PART6_DIR / "protocol.yaml")["pending_decisions"]
    resolved = {k: v["value"] for k, v in proto.items()}
    spec = spec_for(arm, resolved)
    loss = loss_config_for(arm, resolved)
    cfg_hash = P.config_hash({**{k: (v.item() if hasattr(v, "item") else v)
                                 for k, v in row.to_dict().items()
                                 if k != "status"}, "cli": EXECUTOR_VERSION})
    log(f"{run_id}: START arm={arm} fold={fold} seed={seed} spe={spe} cfg={cfg_hash[:12]}")

    # ---- init + teacher provenance (same-cell / same-fold enforced) ----
    init_state, init_heads, init_prov = None, None, None
    if row["init_source"] in ("s1", "s0"):
        ref = resolve_checkpoint(primary_run_id(row["init_source"], fold, seed, 100))
        assert_same_cell(fold, seed, ref, f"{run_id} init")
        if ref.sha256 != row["init_checkpoint_sha256"]:
            raise SystemExit("init checkpoint hash != registry (fail closed)")
        ck = load_checkpoint_payload(ref)
        init_state, init_heads = ck["encoder"], ck["heads"]
        init_prov = {"run_id": ref.run_id, "sha256": ref.sha256,
                     "best_epoch": ref.best_epoch}
    elif row["init_source"] == "ssl":
        ref = resolve_checkpoint(f"ssl_f{fold}_s{seed}")
        assert_same_cell(fold, seed, ref, f"{run_id} ssl init")
        if ref.sha256 != row["init_checkpoint_sha256"]:
            raise SystemExit("ssl checkpoint hash != registry (fail closed)")
        init_state = load_checkpoint_payload(ref)["encoder"]
        init_prov = {"run_id": ref.run_id, "sha256": ref.sha256}
    cache = None
    if row["teacher_set"] in ("s1", "s0"):
        expected = dict(zip(row["teacher_checkpoints"].split(";"),
                            row["teacher_sha256"].split(";")))
        cache = TeacherCache(row["teacher_set"], fold, expected_hashes=expected)
    hseed = int(row["head_init_seed"])
    assert hseed == head_seed(fold, seed)
    acfg = ArmConfig(arm, fold, seed, spec, loss, row["init_source"] if
                     row["init_source"] != "random" else None,
                     row["teacher_set"] if row["teacher_set"] != "none" else None,
                     retained_layers=resolved.get("student_d_retained_layers")
                     if spec.name == "student_d" and init_state is not None else None,
                     kept_direction=(resolved.get("half_student_direction_variant")
                                     or {}).get("keep")
                     if spec.name == "half_4x1" and init_state is not None else None,
                     head_init_seed=hseed)
    # heads: K1/K0/P1 inherit the primary heads (plan: 'heads' copied);
    # C_small / B* / DW use the paired deterministic head seed.
    inherit_heads = arm in ("k1", "k0", "p1")
    trainer = Part6Trainer(acfg, device=device, init_encoder_state=init_state,
                           init_heads_state=init_heads if inherit_heads else None,
                           teacher_cache=cache)
    sched = Part6Trainer.scheduler_for(trainer.optimizer, spe)

    # ---- data (TRAIN+VAL only), frozen sampler stream of the SAME cell ----
    reg5d = pd.read_csv(PART5D_DIR / "main_run_registry.csv").set_index("run_id")
    prow = reg5d.loc[primary_run_id("s1", fold, seed, 100)]
    spath = SUBSET_DIR / f"label_subset_f{fold}_s{seed}.csv"
    assert sha256_file(spath) == prow["label_subset_hash"], "label subset hash"
    subset = pd.read_csv(spath)
    man = load_fold_manifest(fold)
    train_ids, val_ids = split_ids(man, "train"), split_ids(man, "validation")
    assert math.ceil(len(train_ids) / 64) == spe
    stream = build_sup_stream(subset, 1.0, seed, epochs * spe)
    sh = stream_hash(stream)
    store = RepStore(fold, man)
    pre_s = store.preload(train_ids + val_ids)
    eq = store.equivalence_check(seed=seed)
    val_by = ids_by_dataset(man, val_ids)
    log(f"{run_id}: preloaded {len(store.store)} reps in {pre_s:.0f}s (bit-equal {eq})")

    start_epoch, best = 0, {"metric": float("-inf"), "epoch": None, "reports": None}
    if resume and (d / "last.pt").exists():
        ck = torch.load(d / "last.pt", map_location=device, weights_only=False)
        assert ck["config_hash"] == cfg_hash and ck["stream_hash"] == sh
        trainer.encoder.load_state_dict(ck["encoder"])
        trainer.heads.load_state_dict(ck["heads"])
        trainer.optimizer.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        log(f"{run_id}: RESUMED at epoch {start_epoch}")

    state = {"run_id": run_id, "kind": "part6", "status": "RUNNING", "arm": arm,
             "fold": fold, "seed": seed, "config_hash": cfg_hash,
             "stream_sha256": sh, "seals": seals,
             "arm_config": acfg.to_dict(), "init_provenance": init_prov,
             "inherit_heads": inherit_heads,
             "teacher_cache": cache.meta["teacher_set"] if cache else None,
             "surgery_report": ({k: v for k, v in trainer.surgery_report.items()
                                 if k != "mapping"} if trainer.surgery_report else None),
             "frozen_settings": Part6Trainer.frozen_settings(),
             "encoder_params": sum(p.numel() for p in trainer.encoder.parameters()),
             "head_params": sum(p.numel() for p in trainer.heads.parameters()),
             "preload_windows": len(store.store), "preload_equivalence": eq,
             "started_at": datetime.now(timezone.utc).isoformat(),
             "pid": os.getpid(), "host": socket.gethostname(),
             "device_provenance": device_provenance(device),
             "part6_master_hash_at_start": verify_part6_seal(),
             "worker_claim_token_present": bool(
                 os.environ.get(WQ.CLAIM_TOKEN_ENV)),
             "scan_backend": "reference"}
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps(state, indent=1, default=_jsonable, sort_keys=True))
    hist = open(d / "epoch_metrics.jsonl", "a")
    try:
        for epoch in range(start_epoch, epochs):
            t0 = time.time()
            losses = []
            for k in range(spe):
                batch = stream[epoch * spe + k]
                reps = [store.rep(w) for _, _, w in batch]
                losses.append(trainer.train_step_bucketed(reps, batch, sched))
            t_tr = time.time() - t0
            t0 = time.time()
            reports = validation_reports(trainer, store, val_by, man)
            macro = macro_domain_f1(reports)
            t_val = time.time() - t0
            if Part6Trainer.is_better(macro, best["metric"]):
                best = {"metric": macro, "epoch": epoch, "reports": reports}
                torch.save({"run_id": run_id, "config_hash": cfg_hash, "epoch": epoch,
                            "macro_f1_val": macro, "val_reports": reports,
                            "encoder": trainer.encoder_state(),
                            "heads": trainer.heads_state()}, d / "best.pt")
            torch.save({"run_id": run_id, "config_hash": cfg_hash, "stream_hash": sh,
                        "epoch": epoch, "best": best,
                        "encoder": trainer.encoder_state(), "heads": trainer.heads_state(),
                        "optimizer": trainer.optimizer.state_dict(),
                        "scheduler": sched.state_dict(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all()
                        if device.startswith("cuda") else []}, d / "last.pt")
            rec = {"epoch": epoch, "train_loss_mean": float(np.mean(losses)),
                   "val_per_dataset_macro_f1": {ds: reports[ds]["macro_f1"] for ds in DATASETS},
                   "val_macro_domain_f1": macro, "best_epoch": best["epoch"],
                   "lr": sched.get_last_lr()[0], "seconds_train": round(t_tr, 1),
                   "seconds_val": round(t_val, 1),
                   "at": datetime.now(timezone.utc).isoformat()}
            hist.write(json.dumps(rec, default=_jsonable) + "\n")
            hist.flush()
            # ownership heartbeat: when launched by a queue worker, verify
            # we still hold the claim; abort loudly if the lock was broken
            WQ.epoch_heartbeat_from_env(run_id, extra={"epoch": epoch})
            log(f"{run_id}: epoch {epoch + 1}/{epochs} loss={rec['train_loss_mean']:.4f} "
                f"valF1={macro:.4f} best={best['metric']:.4f}@{best['epoch']} "
                f"({t_tr:.0f}s+{t_val:.0f}s)")
        ckpt_hash = sha256_file(d / "best.pt")
        completion = {"run_id": run_id, "best_epoch": best["epoch"],
                      "best_val_macro_f1": best["metric"],
                      "best_checkpoint_sha256": ckpt_hash, "config_hash": cfg_hash,
                      "epochs_completed": epochs,
                      "note": "NO TEST evaluation here — Stage 5 sealed session only"}
        (d / "completion.json").write_text(json.dumps(completion, indent=1, default=_jsonable, sort_keys=True))
        state.update({"status": "COMPLETE", "best_epoch": best["epoch"],
                      "best_val_macro_f1": best["metric"],
                      "best_checkpoint_sha256": ckpt_hash,
                      "finished_at": datetime.now(timezone.utc).isoformat()})
        (d / "state.json").write_text(json.dumps(state, indent=1, default=_jsonable, sort_keys=True))
        log(f"{run_id}: COMPLETE best={best['metric']:.4f}@{best['epoch']}")
    except BaseException:
        state.update({"status": "FAILED", "error": traceback.format_exc(),
                      "failed_at": datetime.now(timezone.utc).isoformat()})
        (d / "state.json").write_text(json.dumps(state, indent=1, default=_jsonable, sort_keys=True))
        log(f"{run_id}: FAILED (state preserved)")
        raise
    finally:
        hist.close()


@torch.no_grad()
def validation_reports(trainer, store, by_ds, man) -> dict:
    reports = {}
    for ds in DATASETS:
        wids = by_ds[ds]
        assert_no_test_windows(wids, "validation")
        y_true = [str(man.loc[w, LABEL_FIELD[ds]]) for w in wids]
        y_pred = []
        for lo in range(0, len(wids), 64):
            chunk = wids[lo:lo + 64]
            y_pred.extend(trainer.predict([store.rep(w) for w in chunk],
                                          [ds] * len(chunk)))
        reports[ds] = classification_report(y_true, y_pred, ds)
    return reports


def drive(stage: str, device: str) -> None:
    require_seal_and_authorization("drive")
    reg = pd.read_csv(PART6_DIR / REGISTRY_CSV)
    tiers = {"3": ("core", "optional"), "4": ("push",)}[stage]
    rows = reg[reg["enabled"] & reg["tier"].isin(tiers)]
    for rid in rows["run_id"]:
        d = RUNS / rid
        for attempt in (1, 2):
            st = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() else None
            if st and st["status"] == "COMPLETE":
                break
            cmd = [sys.executable, str(Path(__file__).resolve()), "run",
                   "--run-id", rid, "--device", device]
            if st is not None:
                cmd.append("--resume")
            log(f"driver: launching {rid} (attempt {attempt})")
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "run_log.txt", "a") as f:
                import subprocess
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
        st = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() else None
        if not (st and st["status"] == "COMPLETE"):
            raise SystemExit(f"driver: {rid} failed twice — stopping (state preserved)")
    log(f"driver: stage {stage} done")


def status() -> None:
    if not (PART6_DIR / REGISTRY_CSV).exists():
        print("no registry")
        return
    reg = pd.read_csv(PART6_DIR / REGISTRY_CSV)
    rows = []
    for _, r in reg.iterrows():
        d = RUNS / r["run_id"]
        st = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() else None
        rows.append({"run_id": r["run_id"], "tier": r["tier"], "enabled": r["enabled"],
                     "registry_status": r["status"],
                     "run_status": st["status"] if st else "NOT_STARTED",
                     "best_epoch": (st or {}).get("best_epoch")})
    print(pd.DataFrame(rows).to_string(index=False))


# ---------------------------------------------------------------------------
# aggregation (validation-level; pairing checks) + pre-test ledger
# ---------------------------------------------------------------------------
def aggregate() -> dict:
    reg = pd.read_csv(PART6_DIR / REGISTRY_CSV)
    rows = []
    for _, r in reg[reg["enabled"]].iterrows():
        d = RUNS / r["run_id"]
        if not (d / "completion.json").exists():
            rows.append({"run_id": r["run_id"], "status": "INCOMPLETE"})
            continue
        c = json.loads((d / "completion.json").read_text())
        st = json.loads((d / "state.json").read_text())
        rows.append({"run_id": r["run_id"], "arm": r["arm"], "fold": r["fold"],
                     "seed": r["seed"], "status": "COMPLETE",
                     "best_epoch": c["best_epoch"],
                     "best_val_macro_f1": c["best_val_macro_f1"],
                     "checkpoint_sha256": c["best_checkpoint_sha256"],
                     "stream_sha256": st["stream_sha256"]})
    df = pd.DataFrame(rows)
    out = RESULTS6 / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "PART6_VALIDATION_TABLE.csv", index=False)
    # pairing: every arm of a cell must share the frozen batch stream hash
    pairing_ok = True
    if "stream_sha256" in df.columns:
        for (f, s), g in df[df["status"] == "COMPLETE"].groupby(["fold", "seed"]):
            if g["stream_sha256"].nunique() > 1:
                pairing_ok = False
    summary = {"n_registered_enabled": int(reg["enabled"].sum()),
               "n_complete": int((df["status"] == "COMPLETE").sum()),
               "pairing_streams_identical_within_cell": pairing_ok}
    dump_json(summary, out / "aggregate_summary.json")
    print(json.dumps(summary, indent=1))
    return summary


def cmd_pretest_ledger(a) -> None:
    require_seal_and_authorization("pretest-ledger")
    agg = aggregate()
    if agg["n_complete"] != agg["n_registered_enabled"]:
        raise SystemExit("not all enabled Part-6 runs are COMPLETE — the ledger "
                         "must cover the whole registered set (drop whole arms "
                         "only by an explicit dated decision, never cells)")
    models = []
    df = pd.read_csv(RESULTS6 / "aggregate" / "PART6_VALIDATION_TABLE.csv")
    for _, r in df.iterrows():
        for variant in ("fp32", "q8"):
            models.append({"model_id": f"{r['run_id']}__{variant}",
                           "run_id": r["run_id"], "variant": variant,
                           "best_epoch": int(r["best_epoch"]),
                           "val_macro_domain_f1": float(r["best_val_macro_f1"]),
                           "checkpoint_sha256": r["checkpoint_sha256"],
                           "fold": int(r["fold"]), "seed": int(r["seed"])})
    # Q8 variants of the 18 primary checkpoints (Stage 1) are TEST-scored
    # in the same session (their fp32 TEST values already exist upstream).
    for arm in ("s0", "s1"):
        for f in P.FOLDS:
            for s in P.SEEDS:
                rid = primary_run_id(arm, f, s, 100)
                ref = resolve_checkpoint(rid)
                models.append({"model_id": f"{rid}__q8", "run_id": rid,
                               "variant": "q8", "best_epoch": ref.best_epoch,
                               "val_macro_domain_f1": ref.best_val_macro_f1,
                               "checkpoint_sha256": ref.sha256, "fold": f,
                               "seed": s})
    p = TP.write_pre_test_ledger(models)
    log(f"pre-test ledger written: {p} ({len(models)} models) — COMMIT IT "
        f"before running test-session")


# ---------------------------------------------------------------------------
# Stage 5: THE sealed TEST session
# ---------------------------------------------------------------------------
def cmd_test_session(a) -> None:
    require_seal_and_authorization("test-session")
    verify_primary_seals()
    token = TP.open_test_session(require_committed=not a.allow_uncommitted_ledger)
    log(f"TEST SESSION OPEN {token.session_id} — seal written before any TEST load")
    ledger = pd.read_csv(PART6_DIR / TP.PRE_TEST_LEDGER)
    sd = TP.session_dir()
    stores = {}
    for _, m in ledger.sort_values(["fold", "model_id"]).iterrows():
        mid, fold = str(m["model_id"]), int(m["fold"])
        TP.assert_touch_allowed(mid, token)
        enc, hd, sha, _ = load_model_for(m["run_id"], m["variant"], a.device)
        if sha != m["checkpoint_sha256"]:
            TP.document_integrity_failure(mid, f"checkpoint sha {sha} != ledger")
            raise SystemExit(f"{mid}: checkpoint hash != ledger (fail closed)")
        if fold not in stores:
            man = load_fold_manifest(fold, allow_test=True, token=token)
            store = RepStore(fold, man)
            store.preload(split_ids(man, "test"), allow_test=True)
            stores[fold] = (man, store)
        man, store = stores[fold]
        test_ids = split_ids(man, "test")
        by = ids_by_dataset(man, test_ids)
        reports, rows = {}, []
        for ds in DATASETS:
            wids = by[ds]
            y_true = [str(man.loc[w, LABEL_FIELD[ds]]) for w in wids]
            y_pred = []
            for lo in range(0, len(wids), 64):
                sub = wids[lo:lo + 64]
                b = collate_representations([store.rep(w) for w in sub])
                b = {k: v.to(a.device) for k, v in b.items()}
                with torch.no_grad():
                    z = enc(**b)["global_embedding"]
                    y_pred.extend(CLASS_ORDERS[ds][int(i)] for i in hd(z, ds).argmax(-1))
            reports[ds] = classification_report(y_true, y_pred, ds)
            rows.extend({"window_id": w, "dataset": ds, "y_true": t, "y_pred": p_,
                         "correct": t == p_} for w, t, p_ in zip(wids, y_true, y_pred))
        macro = macro_domain_f1(reports)
        od = sd / mid
        od.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(od / "test_predictions.csv", index=False)
        dump_json({"model_id": mid, "run_id": m["run_id"], "variant": m["variant"],
                   "fold": fold, "seed": int(m["seed"]), "checkpoint_sha256": sha,
                   "macro_domain_f1_test": macro, "per_dataset_reports": reports,
                   "session_id": token.session_id,
                   "evaluated_at": datetime.now(timezone.utc).isoformat()},
                  od / "test_report.json")
        TP.record_touch(mid, sha, fold, len(test_ids), macro, token)
        log(f"{mid}: TEST MacroDomainF1 = {macro:.4f} (single sealed touch)")
    summary = TP.close_test_session(token)
    log(f"TEST SESSION CLOSED: {summary}")


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def _test_values(model_ids: list[str]) -> tuple[list[float], list[dict]]:
    sd = TP.session_dir()
    vals, per_ds = [], []
    for mid in model_ids:
        rep = json.loads((sd / mid / "test_report.json").read_text())
        vals.append(rep["macro_domain_f1_test"])
        per_ds.append({ds: rep["per_dataset_reports"][ds]["macro_f1"] for ds in DATASETS})
    return vals, per_ds


def _primary_test_values(arm: str) -> tuple[list[float], list[dict]]:
    """Primary S0/S1 TEST values — read ONLY at Stage 5 statistics time
    (after the Part-6 session), from the primary reports."""
    vals, per_ds = [], []
    for f in P.FOLDS:
        for s in P.SEEDS:
            rep = json.loads((P.PRIMARY_DOWNSTREAM / primary_run_id(arm, f, s, 100)
                              / "test_report.json").read_text())
            vals.append(rep["macro_domain_f1_test"])
            per_ds.append({ds: rep["per_dataset_reports"][ds]["macro_f1"] for ds in DATASETS})
    return vals, per_ds


def cmd_stats(a) -> None:
    require_seal_and_authorization("stats")
    if not (TP.session_dir() / "session_summary.json").exists():
        raise SystemExit("TEST session not closed — statistics refuse to run")
    cells = [(f, s) for f in P.FOLDS for s in P.SEEDS]
    ids = lambda arm, v: [f"{arm}_f{f}_s{s}__{v}" for f, s in cells]  # noqa: E731
    k1, k1_ds = _test_values(ids("k1", "fp32"))
    c_small, cs_ds = _test_values(ids("c_small", "fp32"))
    k0, k0_ds = _test_values(ids("k0", "fp32"))
    q8k1, _ = _test_values(ids("k1", "q8"))
    s1, s1_ds = _primary_test_values("s1")
    s0, s0_ds = _primary_test_values("s0")
    q8s1, _ = _test_values([f"{primary_run_id('s1', f, s, 100)}__q8" for f, s in cells])
    q8s0, _ = _test_values([f"{primary_run_id('s0', f, s, 100)}__q8" for f, s in cells])
    out = {"confirmatory": ST.confirmatory_family(k1, s1, c_small, q8k1),
           "secondary": ST.secondary_family(k0, s0, k1, q8s1, s1, q8s0),
           "sensitivity": {
               "K1_vs_S1_per_dataset": ST.per_dataset_deltas(k1_ds, s1_ds),
               "K1_vs_S1_excl_CWRU": ST.excluding_cwru_contrast(
                   "K1 vs S1", k1_ds, s1_ds, "ni_then_superiority", P.NI_MARGIN_ARCH),
               "K1_vs_Csmall_per_dataset": ST.per_dataset_deltas(k1_ds, cs_ds),
               "K0_vs_S0_per_dataset": ST.per_dataset_deltas(k0_ds, s0_ds)}}
    reg = pd.read_csv(PART6_DIR / REGISTRY_CSV)
    push = {}
    if bool(reg[(reg["arm"] == "b1") & reg["enabled"]].shape[0]):
        b1, _ = _test_values(ids("b1", "fp32"))
        push["B1_vs_S1"] = ST.contrast("B1 vs S1", b1, s1, "push")
    out["push"] = push
    dump_json(out, RESULTS6 / "statistics" / "part6_statistics.json")
    log("statistics written")


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------
def cmd_benchmark(a) -> None:
    out_dir = RESULTS6 / "benchmark"
    models = {}
    for name, spec in (("full", FULL_SPEC), ("student_d", STUDENT_D_SPEC),
                       ("student_dw", student_dw_spec())):
        enc = build_encoder(spec, seed=0).eval()
        hd = build_heads(spec, 0).eval()
        models[name] = (enc, hd)
    rec = {"backend_caveat": BM.BACKEND_CAVEAT, "size": {}, "compute": {},
           "memory": {}, "at": datetime.now(timezone.utc).isoformat()}
    for name, (enc, hd) in models.items():
        rec["size"][name] = BM.size_axis(enc, hd)
        rec["compute"][name] = BM.compute_axis(enc, hd)
        if a.memory:
            spec = {"full": FULL_SPEC, "student_d": STUDENT_D_SPEC,
                    "student_dw": student_dw_spec()}[name]
            rec["memory"][name] = {f"b{b}": BM.memory_axis(spec.to_dict(), "JNU", b, 1)
                                   for b in (1, 16)}
    if a.latency:
        rec["latency"] = BM.latency_axis(models, device=a.device,
                                         allow_busy=a.allow_busy,
                                         timed=a.timed, warmup=a.warmup)
    dump_json(rec, out_dir / "benchmark.json")
    print(json.dumps({k: rec[k] for k in ("size", "compute")}, indent=1, default=_jsonable))
    if a.latency:
        print(json.dumps(rec["latency"]["settings"], indent=1, default=_jsonable))
    log(f"benchmark written to {out_dir / 'benchmark.json'}")




# ---------------------------------------------------------------------------
# multi-machine Stage-3 worker (shared atomic queue) + global status
# ---------------------------------------------------------------------------
def _enabled_rows() -> pd.DataFrame:
    reg = pd.read_csv(PART6_DIR / REGISTRY_CSV)
    rows = reg[reg["enabled"] & (reg["status"] == "REGISTERED")]
    if rows.empty:
        raise SystemExit("no enabled REGISTERED rows — seal the registry first")
    return rows


def _run_state(rid: str) -> dict | None:
    p = RUNS / rid / "state.json"
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def cmd_worker(a) -> None:
    """Assigned-set worker: claims ONLY the runs the static execution
    assignment gives to THIS host (exactly 9), one at a time (single
    GPU), and EXITS after its last assigned run completes — it can never
    claim a run beyond its assignment (the claim universe is structurally
    restricted, stale reclaim included). Hardlink-atomic claims,
    heartbeats, intra-host stale recovery, resume support unchanged."""
    require_seal_and_authorization("worker")
    verify_primary_seals()
    mh = verify_part6_seal()
    all_rows = _enabled_rows()
    regh = sha256_file(PART6_DIR / REGISTRY_CSV)
    assign = ASG.load_assignment(all_rows, regh, mh)     # fail closed
    host = ASG.local_host()
    my_ids = assign["hosts"].get(host, [])
    if not my_ids:
        raise SystemExit(f"host {host!r} has no assigned runs "
                         f"(assigned hosts: {sorted(assign['hosts'])})")
    rows = all_rows.set_index("run_id").loc[my_ids].reset_index()
    prov = device_provenance(a.device)
    meta = {"arm": None, "fold": None, "seed": None,
            "gpu": prov["gpu_name"], "cuda_version": prov["cuda_version"],
            "git_commit": prov["git_head"], "registry_sha256": regh,
            "part6_master_hash": mh, "assigned_host": host,
            "cuda_visible_devices": prov["cuda_visible_devices"]}
    log(f"worker START host={socket.gethostname()} gpu={prov['gpu_name']} "
        f"registry={regh[:12]} seal={mh[:12]} ASSIGNED {len(my_ids)} runs: "
        f"{my_ids}")
    import subprocess
    while True:
        # janitor: archive stale locks left on COMPLETE assigned runs
        for _, r in rows.iterrows():
            st = _run_state(r["run_id"])
            if st and st.get("status") == "COMPLETE":
                if WQ.janitor_archive_stale_complete(r["run_id"]):
                    log(f"worker: archived orphaned lock of COMPLETE "
                        f"{r['run_id']}")
        done = [r for _, r in rows.iterrows()
                if (_run_state(r["run_id"]) or {}).get("status") == "COMPLETE"]
        failed = [r for _, r in rows.iterrows()
                  if WQ.failure_recorded(r["run_id"])]
        if len(done) + len(failed) == len(rows):
            log(f"worker DONE: assigned set complete — "
                f"{len(done)}/{len(rows)} COMPLETE, {len(failed)} FAILED "
                f"(human review); EXITING, no further claims")
            return
        # single-GPU serialization: if one of MY assigned runs already has
        # a fresh lock on THIS host (e.g. a run outliving a prior worker),
        # wait for it instead of claiming a second run
        busy = None
        for _, r in rows.iterrows():
            st = _run_state(r["run_id"])
            if st and st.get("status") == "COMPLETE":
                continue
            lock = WQ.read_lock(r["run_id"])
            if lock is not None and not WQ.is_stale(lock)                     and str(lock.get("host", "")).split(".")[0] == host:
                busy = r["run_id"]
                break
        if busy is not None and not a.allow_concurrent:
            log(f"worker: {busy} is active on this host — waiting "
                f"({a.poll_s}s) instead of claiming a second run")
            time.sleep(a.poll_s)
            continue
        handle, row = None, None
        for _, r in rows.iterrows():
            rid = r["run_id"]
            st = _run_state(rid)
            if st and st.get("status") == "COMPLETE":
                continue
            if WQ.failure_recorded(rid):
                continue                       # needs human review
            lock = WQ.read_lock(rid)
            if lock is not None:
                if WQ.is_stale(lock):
                    if not WQ.reclaim_stale(rid):
                        continue               # lost the reclaim race
                    log(f"worker: broke STALE lock on {rid} "
                        f"(last heartbeat {int(time.time() - lock['heartbeat_at'])}s ago)")
                else:
                    # fresh lock on an assigned run held elsewhere: the
                    # assignment forbids claiming it; never redistribute
                    log(f"worker: {rid} carries a fresh foreign lock "
                        f"(host {lock.get('host')}) — skipping")
                    continue
            m = dict(meta)
            m.update({"arm": r["arm"], "fold": int(r["fold"]),
                      "seed": int(r["seed"])})
            h = WQ.claim_run(rid, m)
            if h is not None:
                handle, row = h, r
                break
        if handle is None:
            log(f"worker: nothing claimable right now "
                f"({len(done)}/{len(rows)} complete) — poll in {a.poll_s}s")
            if a.exit_when_empty:
                return
            time.sleep(a.poll_s)
            continue

        rid = handle.run_id
        log(f"worker: CLAIMED {rid} (token {handle.token[:8]})")
        outcome = "failed"
        try:
            for attempt in (1, 2):
                st = _run_state(rid)
                cmd = [sys.executable, str(Path(__file__).resolve()), "run",
                       "--run-id", rid, "--device", a.device]
                if st is not None:
                    cmd.append("--resume")
                env = dict(os.environ)
                env[WQ.CLAIM_TOKEN_ENV] = handle.token
                d = RUNS / rid
                d.mkdir(parents=True, exist_ok=True)
                WQ.heartbeat(handle, extra={"attempt": attempt,
                                            "phase": "starting"})
                with open(d / "run_log.txt", "a") as f:
                    f.write(f"\n===== worker {socket.gethostname()} attempt "
                            f"{attempt} {datetime.now(timezone.utc).isoformat()}"
                            f" =====\n")
                    f.flush()
                    proc = subprocess.Popen(cmd, stdout=f,
                                            stderr=subprocess.STDOUT,
                                            cwd=REPO, env=env)
                    while proc.poll() is None:
                        time.sleep(WQ.HEARTBEAT_INTERVAL_S)
                        try:
                            WQ.heartbeat(handle, extra={"phase": "running",
                                                        "run_pid": proc.pid})
                        except Part6GuardError:
                            log(f"worker: LOST the lock on {rid} — killing "
                                f"the run (never train without the lock)")
                            proc.kill()
                            proc.wait()
                            return
                st = _run_state(rid)
                if st and st.get("status") == "COMPLETE":
                    outcome = "complete"
                    break
                log(f"worker: {rid} attempt {attempt} did not complete "
                    f"(status={st.get('status') if st else None})")
        finally:
            try:
                WQ.release(handle, outcome,
                           extra={"final_status": (_run_state(rid) or {})
                                  .get("status")})
            except Part6GuardError as e:
                log(f"worker: release skipped ({e})")
        if outcome == "failed":
            log(f"worker: {rid} FAILED twice — recorded; state preserved; "
                f"continuing with the next run")


def cmd_queue_status(a) -> None:
    rows = _enabled_rows()
    now = time.time()
    counts = {"COMPLETE": 0, "RUNNING": 0, "WAITING": 0, "FAILED": 0,
              "STALE": 0}
    active = []
    for _, r in rows.iterrows():
        rid = r["run_id"]
        st = _run_state(rid)
        lock = WQ.read_lock(rid)
        if st and st.get("status") == "COMPLETE":
            counts["COMPLETE"] += 1
            continue
        if WQ.failure_recorded(rid):
            counts["FAILED"] += 1
            continue
        if lock is None:
            counts["WAITING"] += 1
            continue
        if WQ.is_stale(lock, now):
            counts["STALE"] += 1
            continue
        counts["RUNNING"] += 1
        epoch, latest, best = None, None, None
        mp = RUNS / rid / "epoch_metrics.jsonl"
        if mp.exists():
            lines = mp.read_text().splitlines()
            if lines:
                last = json.loads(lines[-1])
                epoch = last["epoch"] + 1
                latest = last["val_macro_domain_f1"]
                best = max(json.loads(l)["val_macro_domain_f1"]
                           for l in lines)
        active.append({
            "run_id": rid, "host": lock.get("host"), "gpu": lock.get("gpu"),
            "epoch": f"{epoch or 0}/50",
            "latest_valF1": round(latest, 4) if latest is not None else None,
            "best_valF1": round(best, 4) if best is not None else None,
            "elapsed_min": round((now - lock["claimed_at"]) / 60),
            "heartbeat_s_ago": int(now - lock["heartbeat_at"])})
    total = len(rows)
    print(f"COMPLETE: {counts['COMPLETE']} / {total}")
    for k in ("RUNNING", "WAITING", "FAILED", "STALE"):
        print(f"{k}: {counts[k]}")
    if active:
        print()
        print(pd.DataFrame(active).to_string(index=False))
    if (PART6_DIR / REGISTRY_CSV).exists():
        print(f"\nregistry sha256: {sha256_file(PART6_DIR / REGISTRY_CSV)}")
        try:
            print(f"part6 master seal: {verify_part6_seal()}")
        except Part6GuardError as e:
            print(f"seal: {e}")


# ---------------------------------------------------------------------------
# SMOKE (NOT_AN_EXPERIMENT): CPU end-to-end proof on a few TRAIN/VAL windows
# ---------------------------------------------------------------------------
def cmd_smoke(a) -> None:
    """Real fold-1 primary checkpoints -> surgery -> real (tiny) teacher
    cache -> K1/K0/C_small/P1 steps -> validation checkpoint rule -> Q8.
    Uses n TRAIN + m VAL windows per dataset from the sealed reader; writes
    ONLY under results/.../smoke_NOT_AN_EXPERIMENT/; never touches TEST;
    results are discarded from any analysis. Recommended pending values
    are used HERE ONLY (labelled) — nothing is sealed by this command."""
    torch.set_num_threads(a.threads)
    out = RESULTS6 / "smoke_NOT_AN_EXPERIMENT"
    out.mkdir(parents=True, exist_ok=True)
    fold = 1
    verify_primary_seals()
    man = load_fold_manifest(fold)                       # TRAIN+VAL only
    rng = np.random.default_rng(0)
    tr_by, va_by = {}, {}
    for ds in DATASETS:
        tr = sorted(man.index[(man["dataset"] == ds) & (man["split"] == "train")])
        va = sorted(man.index[(man["dataset"] == ds) & (man["split"] == "validation")])
        tr_by[ds] = [tr[i] for i in rng.choice(len(tr), a.n_train, replace=False)]
        va_by[ds] = [va[i] for i in rng.choice(len(va), a.n_val, replace=False)]
    ids = [w for ds in DATASETS for w in tr_by[ds] + va_by[ds]]
    assert_no_test_windows(ids, "smoke")
    store = RepStore(fold, man)
    t = store.preload(ids)
    log(f"smoke: preloaded {len(ids)} TRAIN/VAL windows in {t:.1f}s")
    smoke_resolved = {"ensemble_rule": "mean_prob_at_T",
                      "relational_alpha_kl_weight": 1.0,
                      "student_d_retained_layers": [0, 2]}
    report = {"label": "NOT_AN_EXPERIMENT", "fold": fold,
              "n_train_per_ds": a.n_train, "n_val_per_ds": a.n_val,
              "smoke_only_pending_values": smoke_resolved, "arms": {}}
    caches = {}
    for tset_name in ("s1", "s0"):
        tset = discover_teacher_set(tset_name, fold, smoke_resolved["ensemble_rule"])
        build_teacher_cache(tset, man, store.rep, out_root=out, device="cpu",
                            chunk=16, with_band_summaries=True, window_ids=ids)
        caches[tset_name] = TeacherCache(tset_name, fold, root=out,
                                         expected_hashes=tset.hashes)
        report[f"teacher_cache_{tset_name}"] = {
            "members": [r.run_id for r in tset.refs],
            "content_sha256": caches[tset_name].meta["content_sha256"],
            "saturation": saturation_diagnostic(caches[tset_name],
                                                smoke_resolved["ensemble_rule"])}
        log(f"smoke: {tset_name} cache built from {[r.run_id for r in tset.refs]}")
    seed = 42
    hseed = head_seed(fold, seed)
    for arm in ("k1", "c_small", "k0", "p1"):
        loss = loss_config_for(arm, smoke_resolved)
        init_state = init_heads = None
        src = P.INIT_OF_ARM[arm]
        if src:
            ref = resolve_checkpoint(primary_run_id(src, fold, seed, 100))
            assert_same_cell(fold, seed, ref, "smoke")
            ck = load_checkpoint_payload(ref)
            init_state, init_heads = ck["encoder"], ck["heads"]
        tset = P.TEACHER_SET_OF_ARM[arm]
        cfg = ArmConfig(arm, fold, seed, STUDENT_D_SPEC, loss, src, tset,
                        retained_layers=[0, 2] if src else None,
                        head_init_seed=hseed)
        trainer = Part6Trainer(cfg, device="cpu", init_encoder_state=init_state,
                               init_heads_state=init_heads if arm in ("k1", "k0", "p1") else None,
                               teacher_cache=caches[tset] if tset else None)
        sched = Part6Trainer.scheduler_for(trainer.optimizer, a.steps)
        best = {"metric": float("-inf"), "epoch": None}
        hist = []
        for epoch in range(a.epochs):
            losses = []
            for _ in range(a.steps):
                batch = []
                for ds in DATASETS:
                    pool = tr_by[ds]
                    for _ in range(16):
                        w = pool[int(rng.integers(len(pool)))]
                        batch.append((ds, str(man.loc[w, LABEL_FIELD[ds]]), w))
                reps = [store.rep(w) for _, _, w in batch]
                losses.append(trainer.train_step_bucketed(reps, batch, sched))
            reports = validation_reports(trainer, store, va_by, man)
            macro = macro_domain_f1(reports)
            if Part6Trainer.is_better(macro, best["metric"]):
                best = {"metric": macro, "epoch": epoch}
                torch.save({"encoder": trainer.encoder_state(),
                            "heads": trainer.heads_state(), "epoch": epoch},
                           out / f"smoke_{arm}_best.pt")
            hist.append({"epoch": epoch, "loss": float(np.mean(losses)),
                         "val_macro_f1": macro})
            log(f"smoke {arm}: epoch {epoch} loss {np.mean(losses):.4f} val {macro:.4f}")
        report["arms"][arm] = {"loss": loss.to_dict(), "history": hist, "best": best,
                               "surgery": ({k: v for k, v in trainer.surgery_report.items()
                                            if k != "mapping"} if trainer.surgery_report else None),
                               "encoder_params": sum(p.numel() for p in trainer.encoder.parameters()),
                               "all_finite": all(math.isfinite(h["loss"]) for h in hist)}
    # Q8 on the real fold-1 S1 seed-42 checkpoint + a Student-D smoke checkpoint
    enc, hd, sha, _ = load_model_for(primary_run_id("s1", fold, seed, 100), "fp32")
    q = Q.q8_report(enc)
    r32 = S.evaluate_split(enc, hd, store.rep, va_by, man)
    q_enc, q_hd, _, _ = load_model_for(primary_run_id("s1", fold, seed, 100), "q8")
    r8 = S.evaluate_split(q_enc, q_hd, store.rep, va_by, man)
    report["q8_smoke_primary_s1_f1_s42"] = {
        "checkpoint_sha256": sha, "fp32_bytes": q["fp32_state_bytes"],
        "int8_bytes": q["int8_compact_state_bytes"],
        "ratio": q["compression_ratio_bytes"],
        "val_tiny_fp32": r32["macro_domain_f1"], "val_tiny_q8": r8["macro_domain_f1"],
        "n_val_windows": sum(len(v) for v in va_by.values())}
    dump_json(report, out / "smoke_report.json")
    log(f"SMOKE DONE -> {out / 'smoke_report.json'} (NOT_AN_EXPERIMENT)")

# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit")
    sub.add_parser("pending")
    p = sub.add_parser("write-specs")
    p.add_argument("--final", action="store_true")
    p = sub.add_parser("registry")
    p.add_argument("--list", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--final", action="store_true")
    sub.add_parser("seal")
    p = sub.add_parser("scan-parity")
    p.add_argument("--chunk", type=int, default=SF.DEFAULT_CHUNK)
    p.add_argument("--real", action="store_true")
    p.add_argument("--n-windows", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("cache-teachers")
    p.add_argument("--folds", type=int, nargs="+", default=list(P.FOLDS))
    p.add_argument("--teacher-sets", nargs="+", default=["s1", "s0"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--no-band-summaries", action="store_true")
    p = sub.add_parser("ptq")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--exploratory", action="store_true")
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("sensitivity")
    p.add_argument("--folds", type=int, nargs="+", default=list(P.FOLDS))
    p.add_argument("--device", default="cpu")
    p.add_argument("--train-stat-windows", type=int, default=256)
    p = sub.add_parser("run")
    p.add_argument("--run-id", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("drive")
    p.add_argument("--stage", choices=("3", "4"), required=True)
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("worker")
    p.add_argument("--device", default="cuda")
    p.add_argument("--poll-s", type=int, default=300)
    p.add_argument("--exit-when-empty", action="store_true")
    p.add_argument("--allow-concurrent", action="store_true",
                   help="permit claiming while another assigned run is "
                        "active on this host (multi-GPU hosts only)")
    sub.add_parser("write-assignment")
    sub.add_parser("queue-status")
    sub.add_parser("status")
    sub.add_parser("aggregate")
    sub.add_parser("pretest-ledger")
    p = sub.add_parser("test-session")
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-uncommitted-ledger", action="store_true",
                   help="tests/dry only — the real session needs a committed ledger")
    sub.add_parser("stats")
    p = sub.add_parser("smoke")
    p.add_argument("--n-train", type=int, default=8)
    p.add_argument("--n-val", type=int, default=4)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--threads", type=int, default=2)
    p = sub.add_parser("benchmark")
    p.add_argument("--latency", action="store_true")
    p.add_argument("--memory", action="store_true")
    p.add_argument("--allow-busy", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--timed", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()
    if a.cmd == "audit":
        audit()
    elif a.cmd == "pending":
        print(json.dumps(P.pending_summary(), indent=1, default=_jsonable))
        print("resolved:", load_resolved())
    elif a.cmd == "write-specs":
        write_specs(final=a.final)
    elif a.cmd == "registry":
        cmd_registry(a)
    elif a.cmd == "seal":
        cmd_seal(a)
    elif a.cmd == "scan-parity":
        cmd_scan_parity(a)
    elif a.cmd == "cache-teachers":
        cmd_cache_teachers(a)
    elif a.cmd == "ptq":
        cmd_ptq(a)
    elif a.cmd == "sensitivity":
        cmd_sensitivity(a)
    elif a.cmd == "run":
        run_part6(a.run_id, resume=a.resume, device=a.device)
    elif a.cmd == "drive":
        drive(a.stage, a.device)
    elif a.cmd == "worker":
        cmd_worker(a)
    elif a.cmd == "write-assignment":
        mh = verify_part6_seal()
        rows = _enabled_rows()
        regh = sha256_file(PART6_DIR / REGISTRY_CSV)
        path = ASG.write_assignment(rows, regh, mh)
        doc = ASG.load_assignment(rows, regh, mh)      # verify roundtrip
        for h in ASG.HOSTS:
            print(f"{h}: {doc['hosts'][h]}")
        log(f"execution assignment written + verified: {path}")
    elif a.cmd == "queue-status":
        cmd_queue_status(a)
    elif a.cmd == "status":
        status()
    elif a.cmd == "aggregate":
        aggregate()
    elif a.cmd == "pretest-ledger":
        cmd_pretest_ledger(a)
    elif a.cmd == "test-session":
        cmd_test_session(a)
    elif a.cmd == "stats":
        cmd_stats(a)
    elif a.cmd == "benchmark":
        cmd_benchmark(a)
    elif a.cmd == "smoke":
        cmd_smoke(a)


if __name__ == "__main__":
    main()
