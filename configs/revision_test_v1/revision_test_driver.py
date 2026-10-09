#!/usr/bin/env python
"""
Fail-closed TEST driver for the supervisor-review ablations
(protocol lightweight_pcste_revision_ablation_test_v1).

Lifecycle (each step is a separate, explicit command):

  --freeze      after ALL training is complete: resolve every checkpoint, verify it,
                write the model table, the machine-readable plan and the barrier.
                Never reads TEST data. Refuses if a barrier already exists.
  --preflight   re-verify the barrier, plan, model table, code and input hashes,
                and strict-load every checkpoint on CPU. Never reads TEST data.
  --execute     the ONLY path that opens TEST manifests/samples. Needs the exact
                barrier SHA256 and the authorization token. Runs once; refuses to
                overwrite an existing output directory.

Execution order inside --execute (driver-correctness gate first):
  1. re-evaluate the already-sealed Full-S1 and K1 checkpoints (their TEST
     results are already published, so this is no new exposure) and require
     their reports to reproduce the sealed reports exactly;
  2. only then evaluate S0 and the five ablation arms;
  3. compute the pre-specified analyses with the frozen statistics module.

The scientific content (families, comparisons, tests, margins, labels) is fixed
in ANALYSES below and therefore pinned by this file's SHA256 in the barrier.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def find_repo() -> Path:
    env = os.environ.get("PCSTE_REPO")
    if env:
        return Path(env).resolve()
    for p in Path(__file__).resolve().parents:
        if (p / "src/methodology_v2").is_dir() and (p / "src/pcste_v2").is_dir():
            return p
    raise SystemExit("cannot locate repository root (set PCSTE_REPO)")


REPO = find_repo()
sys.path.insert(0, str(REPO))

PROTOCOL_ID = "lightweight_pcste_revision_ablation_test_v1"
AUTHORIZATION_TOKEN = "AUTHORIZE_LIGHTWEIGHT_PCSTE_REVISION_ABLATION_TEST_V1"
DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")
FOLDS = (1, 2, 3)
SEEDS = (42, 1337, 2026)

OUT = REPO / "analysis/pcste_v2_revision_v1/revision_test_v1"
TABLE = OUT / "REVISION_TEST_MODEL_TABLE.csv"
PLAN = OUT / "REVISION_TEST_PLAN.json"
BARRIER = OUT / "REVISION_TEST_BARRIER.json"

PUB_TABLE = "analysis/pcste_v2_lightweight_v1/final_test_v1/PUBLICATION_FROZEN_MODEL_TABLE.csv"
SEALED_REPORTS = "results/pcste_v2/lightweight_v1/publication_final_test_v1/per_model_reports"
S0_TABLE = "analysis/pcste_v2_final_s0_s1_v1/FINAL_SELECTED_CHECKPOINTS.csv"
COLLECTED = "results/pcste_v2/revision_v1_collected"
GLOBAL_PDIR = "pcste_v2/protocol/global_v2_final_s0s1_v1"
CWRU_PDIR = "pcste_v2/protocol/cwru_native12_load_v1"
CWRU_FREEZE_INDEX = f"{CWRU_PDIR}/FREEZE_BUNDLE_INDEX.txt"
CWRU_FREEZE_DIGEST = "ddf4d32573c016b08da65b4f878ee0511a9bae4b15444626e1d932d24ebf39e4"

# family -> (architecture, checkpoint format)
FAMILIES = {
    "full_s1":   ("full",      "pcstev2"),
    "k1":        ("half_4x1",  "student"),
    "s0":        ("full",      "pcstev2"),
    "c_small":   ("half_4x1",  "student"),
    "p1":        ("half_4x1",  "student"),
    "k1_2x2":    ("student_d", "student"),
    "k1_single": ("half_4x1",  "student"),
    "k0":        ("half_4x1",  "student"),
}
SEALED_FAMILIES = ("full_s1", "k1")          # TEST already published -> driver-correctness gate
NEW_FAMILIES = ("s0", "c_small", "p1", "k1_2x2", "k1_single", "k0")
EXPECTED_ENCODER_PARAMS = {"full": None, "half_4x1": 1_375_953, "student_d": 1_375_185}

# ---------------------------------------------------------------------------
# PRE-SPECIFIED ANALYSES (fixed before any ablation TEST access)
# delta = a - b per (fold, seed) cell; endpoint Macro-4 Macro-F1 unless stated
# ---------------------------------------------------------------------------
ANALYSES = {
    "A_ssl_full_scale": {
        "status": "executes the frozen dissertation-era plan FINAL_STATISTICAL_PLAN.yaml "
                  "(final_s0_s1.v1, FROZEN_BEFORE_TRAINING) without change",
        "question": "Does self-supervised pretraining improve the full model? (S1 vs S0)",
        "contrasts": [{"name": "S1 vs S0 (Macro-4 F1)", "a": "full_s1", "b": "s0",
                       "endpoint": "macro_4_macro_f1", "kind": "two_sided"},
                      {"name": "S1 vs S0 (Macro-4 AUC)", "a": "full_s1", "b": "s0",
                       "endpoint": "macro_4_macro_auc", "kind": "two_sided"}],
        "multiplicity": "Holm over the two endpoints (m=2), alpha 0.05",
        "labels": "SSL_BENEFICIAL / SSL_COMPETITIVE (+/-0.01 band on both endpoints) / "
                  "SSL_UNFAVOURABLE / SSL_INCONCLUSIVE_OR_MIXED, as defined in the frozen plan",
        "deviation": "S0 gf3_s2026 is a retrained replacement (original lost); a sensitivity "
                     "analysis on the 8 cells with original S0 checkpoints is reported",
    },
    "B_original_compression_contrasts": {
        "status": "pre-specified in the original compression protocol (statistics_spec.yaml: "
                  "H2 K1 vs C_small; secondary K1 vs K0 and K0 vs S0) but not executed in the "
                  "publication stage; executed here with the same tests",
        "question": "Is K1's performance explained by the small architecture alone, and does "
                    "SSL pretraining matter at the student level?",
        "contrasts": [{"name": "K1 vs C_small", "a": "k1", "b": "c_small",
                       "endpoint": "macro_4_macro_f1", "kind": "two_sided"},
                      {"name": "K1 vs K0", "a": "k1", "b": "k0",
                       "endpoint": "macro_4_macro_f1", "kind": "two_sided"},
                      {"name": "K0 vs S0", "a": "k0", "b": "s0",
                       "endpoint": "macro_4_macro_f1", "kind": "ni", "margin": 0.02}],
        "multiplicity": "Holm over the three contrasts (m=3), alpha 0.05",
        "deviation": "the original families also contained Q8 contrasts (reported separately "
                     "in the published study / not executable); the three executable "
                     "contrasts form one Holm family here",
    },
    "C_review_ablations": {
        "status": "POST-HOC: specified in October 2026 in response to supervisor review, after "
                  "the primary sealed TEST, before any ablation TEST access",
        "question": "Does distillation add value beyond initialisation; is removing a direction "
                    "better than removing depth; is the three-teacher ensemble necessary?",
        "contrasts": [{"name": "K1 vs P1 (no distillation)", "a": "k1", "b": "p1",
                       "endpoint": "macro_4_macro_f1", "kind": "two_sided"},
                      {"name": "K1 vs K1-2x2 (depth instead of direction)", "a": "k1",
                       "b": "k1_2x2", "endpoint": "macro_4_macro_f1", "kind": "two_sided"},
                      {"name": "K1 vs K1-single (one teacher)", "a": "k1", "b": "k1_single",
                       "endpoint": "macro_4_macro_f1", "kind": "two_sided"}],
        "multiplicity": "Holm over the three contrasts (m=3), alpha 0.05",
        "equivalence": "for each contrast also an exact TOST at +/-0.02 (the frozen "
                       "architecture non-inferiority margin), reported descriptively",
    },
    "descriptive": ["all nine deltas, mean, median, SD (ddof=1), wins/ties/losses",
                    "per-dataset Macro-F1 deltas", "Macro-3 excluding CWRU",
                    "Macro-4 AUC deltas (except where an AUC test is specified above)"],
}


class BarrierError(RuntimeError):
    pass


def fail(msg: str) -> None:
    raise BarrierError(msg)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rp(rel: str) -> Path:
    """Repository-relative path; lexical containment check (symlinked sub-trees allowed)."""
    if Path(rel).is_absolute():
        fail(f"absolute path not allowed: {rel}")
    p = Path(os.path.normpath(REPO / rel))
    if not str(p).startswith(str(REPO) + os.sep):
        fail(f"path escapes repository: {rel}")
    return p


def verify_file(rel: str, expected: str, label: str) -> Path:
    p = rp(rel)
    if not p.is_file():
        fail(f"{label} missing: {rel}")
    got = sha256_file(p)
    if got != expected:
        fail(f"{label} SHA256 mismatch: {rel}; expected {expected}; got {got}")
    return p


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_files() -> list[str]:
    files = [os.path.relpath(os.path.abspath(__file__), REPO),
             "src/methodology_v2/compression/stats.py",
             "src/methodology_v2/compression/student.py",
             "src/methodology_v2/experiment/heads.py",
             "src/methodology_v2/experiment/metrics.py",
             "src/pcste_v2/metrics_v2.py", "src/pcste_v2/model.py",
             "src/pcste_v2/representation.py"]
    files += sorted(str(p.relative_to(REPO)) for p in (REPO / "src/methodology_v2/encoder").glob("*.py"))
    return files


# ---------------------------------------------------------------------------
# model construction / checkpoint loading (CPU or GPU)
# ---------------------------------------------------------------------------
def build_model(family: str, ck: dict, device: str):
    import torch  # noqa: F401
    from src.pcste_v2.model import PCSTEv2, PCSTEv2Config
    from src.methodology_v2.experiment.heads import DatasetHeads
    from src.methodology_v2.compression.student import (STUDENT_D_SPEC, build_encoder,
                                                        half_4x1_spec)
    arch, fmt = FAMILIES[family]
    if fmt == "pcstev2":
        enc = PCSTEv2(PCSTEv2Config())
        enc.load_state_dict(ck["model"], strict=True)
    else:
        spec = half_4x1_spec("mean_of_remaining") if arch == "half_4x1" else STUDENT_D_SPEC
        enc = build_encoder(spec, seed=0)
        n = sum(p.numel() for p in enc.parameters())
        if n != EXPECTED_ENCODER_PARAMS[arch]:
            fail(f"{family}: encoder parameter identity changed ({n})")
        enc.load_state_dict(ck["encoder"], strict=True)
    heads = DatasetHeads()
    heads.load_state_dict(ck["heads"], strict=True)
    return enc.to(device).eval(), heads.to(device).eval()


# ---------------------------------------------------------------------------
# FREEZE: build model table + plan + barrier (no TEST access)
# ---------------------------------------------------------------------------
def resolve_models() -> list[dict]:
    import pandas as pd
    import torch
    pub = pd.read_csv(rp(PUB_TABLE))
    s0t = pd.read_csv(rp(S0_TABLE))
    col = rp(COLLECTED)
    reps = {}
    for f in col.glob("pcstev2_final_v1_s0_*_retrain/S0_REPLACEMENTS.json"):
        reps.update(json.loads(f.read_text()))
    rows = []
    for fold in FOLDS:
        for seed in SEEDS:
            q = pub[(pub.fold == fold) & (pub.seed == seed)].iloc[0]
            rows.append(dict(family="full_s1", fold=fold, seed=seed, run_id=q.full_s1_run_id,
                             checkpoint_path=q.full_s1_checkpoint, checkpoint_sha256=q.full_s1_sha256,
                             provenance="published Full-S1 (sealed TEST)"))
            rows.append(dict(family="k1", fold=fold, seed=seed, run_id=q.k1_run_id,
                             checkpoint_path=q.k1_checkpoint, checkpoint_sha256=q.k1_sha256,
                             provenance="published K1 (sealed TEST)"))
            r = s0t[(s0t.strategy == "S0") & (s0t.global_fold == fold) & (s0t.seed == seed)].iloc[0]
            if r.run_id in reps:
                rec = reps[r.run_id]
                rid = f"{r.run_id}_retrain"
                path = f"{COLLECTED}/{rid}/best.pt"
                rows.append(dict(family="s0", fold=fold, seed=seed, run_id=rid,
                                 checkpoint_path=path, checkpoint_sha256=rec["checkpoint_sha256"],
                                 provenance="S0 REPLACEMENT (retrained; original lost)"))
            else:
                rows.append(dict(family="s0", fold=fold, seed=seed, run_id=r.run_id,
                                 checkpoint_path=r.checkpoint_path, checkpoint_sha256=r.checkpoint_sha256,
                                 provenance="original pre-registered S0"))
            for fam in ("c_small", "p1", "k1_2x2", "k1_single", "k0"):
                rid = f"pcstev2_revision_v1_{fam}_gf{fold}_s{seed}"
                d = col / rid
                st = d / "state.json"
                if not st.exists():
                    fail(f"missing collected run {rid}")
                state = json.loads(st.read_text())
                if state.get("status") != "COMPLETE" or state.get("test_used") is not False:
                    fail(f"{rid} is not COMPLETE/TEST-free")
                rows.append(dict(family=fam, fold=fold, seed=seed, run_id=rid,
                                 checkpoint_path=f"{COLLECTED}/{rid}/best.pt",
                                 checkpoint_sha256=state["best_checkpoint_sha256"],
                                 provenance="revision ablation (post-hoc)"))
    for row in rows:
        p = verify_file(row["checkpoint_path"], row["checkpoint_sha256"], f"{row['family']} checkpoint")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        build_model(row["family"], ck, "cpu")
        row["selected_epoch"] = int(ck["epoch"])
        ck_run = str(ck.get("run_id", ""))
        if row["family"] in ("full_s1", "k1", "c_small", "p1", "k1_2x2", "k1_single", "k0") \
                and ck_run != row["run_id"]:
            fail(f"checkpoint run_id {ck_run} != {row['run_id']}")
    return rows


SEALED_NAMES = [f"gf{f}_s{s}_{fam}_report.json" for f in FOLDS for s in SEEDS for fam in ("full_s1", "k1")]


def find_sealed_reports_dir() -> str:
    """Locate the folder holding the 18 sealed publication per-model reports.

    Accepts a folder only if it contains all 18 expected file names and every report
    declares protocol lightweight_pcste_final_test_v1. Searches the expected location
    first, then the whole results/ tree. Fails closed if zero or several qualify."""
    cands = [SEALED_REPORTS] + sorted({os.path.relpath(p.parent, REPO)
                                        for p in rp("results").rglob("gf1_s42_k1_report.json")})
    good, seen = [], []
    for rel in dict.fromkeys(cands):
        d = rp(rel)
        if not d.is_dir():
            continue
        seen.append(rel)
        if not all((d / n).is_file() for n in SEALED_NAMES):
            continue
        try:
            ok = all(json.loads((d / n).read_text()).get("protocol_identity") == "lightweight_pcste_final_test_v1"
                     for n in SEALED_NAMES)
        except (OSError, ValueError):
            ok = False
        if ok:
            good.append(rel)
    if len(good) != 1:
        fail(f"need exactly one folder with the 18 sealed publication reports; found {good} "
             f"(searched {seen})")
    return good[0]


def cmd_freeze() -> None:
    if BARRIER.exists():
        fail(f"barrier already exists in {OUT}; refusing to re-freeze")
    import pandas as pd
    sealed_dir = find_sealed_reports_dir()          # validate everything BEFORE writing anything
    rows = resolve_models()
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["family", "fold", "seed", "run_id", "checkpoint_path", "checkpoint_sha256",
            "selected_epoch", "provenance"]
    pd.DataFrame(rows)[cols].to_csv(TABLE, index=False)
    man = {f"GF{f}": {"path": f"{GLOBAL_PDIR}/global_v2_fold_{f}.csv",
                      "sha256": sha256_file(rp(f"{GLOBAL_PDIR}/global_v2_fold_{f}.csv"))} for f in FOLDS}
    plan = {
        "schema_version": "lightweight_pcste_revision_test_plan.v1",
        "protocol_identity": PROTOCOL_ID, "created_at_utc": now(),
        "test_access_at_freeze": {"ablation_test_loader_invoked": False,
                                  "ablation_test_inference_performed": False,
                                  "ablation_test_metrics_computed": False},
        "families": list(FAMILIES), "sealed_families_for_driver_check": list(SEALED_FAMILIES),
        "cells": [[f, s] for f in FOLDS for s in SEEDS],
        "analyses": ANALYSES,
        "statistics_module": "src/methodology_v2/compression/stats.py",
        "model_table": {"path": str(TABLE.relative_to(REPO)), "sha256": sha256_file(TABLE)},
        "test_membership": {"global_fold_manifests": man,
                            "global_hash_registry": {"path": f"{GLOBAL_PDIR}/global_v2_hashes.json",
                                                     "sha256": sha256_file(rp(f"{GLOBAL_PDIR}/global_v2_hashes.json"))},
                            "cwru_freeze_index": {"path": CWRU_FREEZE_INDEX, "sha256": CWRU_FREEZE_DIGEST}},
        "scientific_source_hashes": {f: sha256_file(rp(f)) for f in source_files()},
        "sealed_reports_dir": sealed_dir,
        "sealed_report_hashes": {n: sha256_file(rp(sealed_dir) / n) for n in SEALED_NAMES},
        "execution": {"authorization_token": AUTHORIZATION_TOKEN, "runs": "once",
                      "order": "sealed families first (must reproduce), then new families"},
    }
    if len(plan["sealed_report_hashes"]) != 18:
        fail(f"expected 18 sealed publication reports, found {len(plan['sealed_report_hashes'])}")
    PLAN.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    barrier = {"schema_version": "lightweight_pcste_revision_barrier.v1",
               "protocol_identity": PROTOCOL_ID, "status": "FROZEN_BEFORE_TEST",
               "evaluation_plan_sha256": sha256_file(PLAN),
               "model_table_sha256": sha256_file(TABLE),
               "driver_sha256": sha256_file(Path(__file__).resolve()), "frozen_at_utc": now()}
    BARRIER.write_text(json.dumps(barrier, indent=1, sort_keys=True) + "\n")
    counts = {}
    for r in rows:
        counts[r["family"]] = counts.get(r["family"], 0) + 1
    print("FROZEN:", json.dumps(counts))
    print("replacement S0 cells:", [f"gf{r['fold']}_s{r['seed']}" for r in rows
                                     if r["family"] == "s0" and "REPLACEMENT" in r["provenance"]])
    print("sealed publication reports:", sealed_dir)
    print("BARRIER SHA256:", sha256_file(BARRIER))
    print("No TEST data was opened.")


# ---------------------------------------------------------------------------
# PREFLIGHT
# ---------------------------------------------------------------------------
def verify_all(expected_barrier: str, load_checkpoints: bool = True) -> tuple[dict, list[dict]]:
    import pandas as pd
    if not BARRIER.exists():
        fail("barrier missing")
    if sha256_file(BARRIER) != expected_barrier:
        fail(f"barrier SHA256 mismatch (got {sha256_file(BARRIER)})")
    barrier = json.loads(BARRIER.read_text())
    if barrier.get("status") != "FROZEN_BEFORE_TEST" or barrier.get("protocol_identity") != PROTOCOL_ID:
        fail("barrier not in FROZEN_BEFORE_TEST state")
    if sha256_file(PLAN) != barrier["evaluation_plan_sha256"]:
        fail("plan differs from barrier commitment")
    if sha256_file(TABLE) != barrier["model_table_sha256"]:
        fail("model table differs from barrier commitment")
    if sha256_file(Path(__file__).resolve()) != barrier["driver_sha256"]:
        fail("driver differs from barrier commitment")
    plan = json.loads(PLAN.read_text())
    if plan["analyses"] != json.loads(json.dumps(ANALYSES)):
        fail("analyses in plan differ from the driver's pre-specified analyses")
    for rel, h in plan["scientific_source_hashes"].items():
        verify_file(rel, h, "scientific source")
    for k, rec in plan["test_membership"]["global_fold_manifests"].items():
        verify_file(rec["path"], rec["sha256"], f"manifest {k}")
    reg = plan["test_membership"]["global_hash_registry"]
    verify_file(reg["path"], reg["sha256"], "global hash registry")
    idx = plan["test_membership"]["cwru_freeze_index"]
    index = verify_file(idx["path"], idx["sha256"], "CWRU freeze index")
    for n, raw in enumerate(index.read_text().splitlines(), 1):
        if raw.strip():
            rel, h = raw.strip().rsplit(None, 1)
            verify_file(rel, h, f"CWRU frozen input line {n}")
    for name, h in plan["sealed_report_hashes"].items():
        verify_file(f"{plan['sealed_reports_dir']}/{name}", h, "sealed publication report")
    rows = pd.read_csv(TABLE).to_dict("records")
    if len(rows) != 9 * len(FAMILIES):
        fail(f"model table has {len(rows)} rows, expected {9 * len(FAMILIES)}")
    import torch
    for r in rows:
        p = verify_file(r["checkpoint_path"], r["checkpoint_sha256"], f"{r['family']} checkpoint")
        if load_checkpoints:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            build_model(r["family"], ck, "cpu")
            if int(ck["epoch"]) != int(r["selected_epoch"]):
                fail(f"selected epoch changed for {r['run_id']}")
    return plan, rows


# ---------------------------------------------------------------------------
# EXECUTE (the only TEST path)
# ---------------------------------------------------------------------------
def add_macro_fields(report: dict) -> dict:
    import numpy as np
    for ds in DATASETS:
        rep = report["per_dataset_reports"][ds]
        rep["macro_precision"] = float(np.mean(rep["per_class_precision"]))
        rep["macro_recall"] = float(np.mean(rep["per_class_recall"]))
    aucs = [float(report["per_dataset_reports"][d]["macro_ovr_auc"]) for d in DATASETS]
    report["macro_4_macro_f1"] = float(report["macro_domain_f1"])
    report["macro_4_macro_auc"] = float(sum(aucs) / 4.0) if all(math.isfinite(x) for x in aucs) else None
    return report


def evaluate(row: dict, items_by_ds: dict, manifest_idx, device: str):
    import pandas as pd
    import torch
    from src.pcste_v2.model import collate_v2
    from src.pcste_v2.metrics_v2 import evaluate_split
    from src.methodology_v2.encoder import collate_representations
    from src.methodology_v2.experiment.heads import CLASS_ORDERS

    p = rp(row["checkpoint_path"])
    before = sha256_file(p)
    if before != row["checkpoint_sha256"]:
        fail(f"checkpoint changed before inference: {row['run_id']}")
    ck = torch.load(p, map_location=device, weights_only=False)
    enc, heads = build_model(row["family"], ck, device)
    fmt = FAMILIES[row["family"]][1]
    out_rows = []
    with torch.no_grad():
        for ds in DATASETS:
            ids, items = items_by_ds[ds]
            for lo in range(0, len(ids), 64):
                chunk_ids, chunk = ids[lo:lo + 64], items[lo:lo + 64]
                if fmt == "pcstev2":
                    b = collate_v2(chunk, 1, 1, False)
                else:
                    b = collate_representations([it["streams"]["g0_c0"] for it in chunk])
                b = {k: v.to(device) for k, v in b.items()}
                logits = heads(enc(**b)["global_embedding"], ds)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                pred = logits.argmax(dim=-1).cpu().numpy()
                for i, wid in enumerate(chunk_ids):
                    s = manifest_idx.loc[wid]
                    r = {"protocol_identity": PROTOCOL_ID, "family": row["family"], "run_id": row["run_id"],
                         "fold": int(row["fold"]), "seed": int(row["seed"]), "dataset": ds, "window_id": wid,
                         "y_true": str(s["class"]), "y_pred": CLASS_ORDERS[ds][int(pred[i])],
                         "p_max": float(probs[i].max()), "physical_specimen": s["physical_specimen"],
                         "group_id": s["group_id"], "fault_severity": s["fault_severity"],
                         "rpm": s["rpm"], "load": s["load"]}
                    r.update({f"prob__{c}": float(probs[i, j]) for j, c in enumerate(CLASS_ORDERS[ds])})
                    out_rows.append(r)
    pred_df = pd.DataFrame(out_rows)
    n_expected = sum(len(v[0]) for v in items_by_ds.values())
    if len(pred_df) != n_expected or pred_df.window_id.duplicated().any():
        fail(f"inexact TEST membership for {row['run_id']}")
    if sha256_file(p) != before:
        fail(f"checkpoint changed during inference: {row['run_id']}")
    rep = add_macro_fields(evaluate_split(pred_df))
    rep.update({"protocol_identity": PROTOCOL_ID, "family": row["family"], "run_id": row["run_id"],
                "fold": int(row["fold"]), "seed": int(row["seed"]),
                "checkpoint_path": row["checkpoint_path"], "checkpoint_sha256": before,
                "selected_epoch": int(row["selected_epoch"]), "provenance": row["provenance"],
                "partition": "test"})
    return pred_df, rep


def compare_with_sealed(rep: dict, sealed: dict) -> list[str]:
    bad = []
    if abs(rep["macro_4_macro_f1"] - sealed["macro_4_macro_f1"]) > 1e-9:
        bad.append(f"Macro-4 F1 {rep['macro_4_macro_f1']} vs sealed {sealed['macro_4_macro_f1']}")
    if (rep["macro_4_macro_auc"] is None) != (sealed.get("macro_4_macro_auc") is None) or (
            rep["macro_4_macro_auc"] is not None
            and abs(rep["macro_4_macro_auc"] - sealed["macro_4_macro_auc"]) > 1e-6):
        bad.append(f"Macro-4 AUC {rep['macro_4_macro_auc']} vs sealed {sealed.get('macro_4_macro_auc')}")
    for ds in DATASETS:
        a = rep["per_dataset_reports"][ds]["confusion_matrix"]
        b = sealed["per_dataset_reports"][ds]["confusion_matrix"]
        if a != b:
            bad.append(f"{ds} confusion matrix differs")
    return bad


def tost(deltas: list[float], margin: float) -> dict:
    from src.methodology_v2.compression.stats import non_inferiority
    lo = non_inferiority(deltas, margin)
    hi = non_inferiority([-x for x in deltas], margin)
    p = max(lo["p_one_sided"], hi["p_one_sided"])
    return {"margin": margin, "p": p, "equivalent": bool(lo["passes"] and hi["passes"]),
            "method": "exact TOST: two one-sided margin-shifted sign-flip tests"}


def run_analyses(scores: dict) -> dict:
    """scores[family][(fold, seed)] = report"""
    from src.methodology_v2.compression.stats import contrast, descriptives, holm
    cells = [(f, s) for f in FOLDS for s in SEEDS]

    def vec(fam, key, cs=cells):
        return [float(scores[fam][c][key]) for c in cs]

    out = {}
    for fam_name in ("A_ssl_full_scale", "B_original_compression_contrasts", "C_review_ablations"):
        spec = ANALYSES[fam_name]
        res = {}
        for c in spec["contrasts"]:
            a, b = vec(c["a"], c["endpoint"]), vec(c["b"], c["endpoint"])
            r = contrast(c["name"], a, b, c["kind"], c.get("margin"))
            d = [x - y for x, y in zip(a, b)]
            r["per_dataset_f1_deltas"] = {
                ds: descriptives([scores[c["a"]][k]["per_dataset_reports"][ds]["macro_f1"]
                                  - scores[c["b"]][k]["per_dataset_reports"][ds]["macro_f1"] for k in cells])
                for ds in DATASETS}
            r["macro3_excl_cwru_deltas"] = descriptives([scores[c["a"]][k]["macro3_f1_excl_cwru"]
                                                         - scores[c["b"]][k]["macro3_f1_excl_cwru"] for k in cells])
            if fam_name == "C_review_ablations":
                r["tost_0.02"] = tost(d, 0.02)
            if c["endpoint"] == "macro_4_macro_f1" and fam_name != "A_ssl_full_scale":
                ra, rb = vec(c["a"], "macro_4_macro_auc"), vec(c["b"], "macro_4_macro_auc")
                r["macro_4_auc_deltas"] = descriptives([x - y for x, y in zip(ra, rb)])
            res[c["name"]] = r
        res["holm"] = holm({k: v["p_for_holm"] for k, v in res.items()})
        out[fam_name] = res

    # frozen interpretation labels for S1 vs S0
    A = out["A_ssl_full_scale"]
    f1, auc = A["S1 vs S0 (Macro-4 F1)"], A["S1 vs S0 (Macro-4 AUC)"]
    pf, pa = A["holm"]["p_adjusted"]["S1 vs S0 (Macro-4 F1)"], A["holm"]["p_adjusted"]["S1 vs S0 (Macro-4 AUC)"]
    if pf < 0.05 and pa < 0.05 and (f1["mean"] > 0) != (auc["mean"] > 0):
        label = "SSL_INCONCLUSIVE_OR_MIXED"
    elif pf < 0.05 and f1["mean"] > 0:
        label = "SSL_BENEFICIAL"
    elif pf < 0.05 and f1["mean"] < 0:
        label = "SSL_UNFAVOURABLE"
    elif abs(f1["mean"]) <= 0.01 and abs(auc["mean"]) <= 0.01 and not (pa < 0.05):
        label = "SSL_COMPETITIVE"
    else:
        label = "SSL_INCONCLUSIVE_OR_MIXED"
    A["interpretation_label"] = label

    # sensitivity: S1 vs S0 on cells with ORIGINAL S0 checkpoints only
    orig = [c for c in cells if "REPLACEMENT" not in scores["s0"][c]["provenance"]]
    if len(orig) < len(cells):
        A["sensitivity_original_s0_only"] = {
            "cells": [f"gf{f}_s{s}" for f, s in orig],
            "f1": contrast("S1 vs S0 F1 (original S0 only)", vec("full_s1", "macro_4_macro_f1", orig),
                           vec("s0", "macro_4_macro_f1", orig), "two_sided"),
            "auc": contrast("S1 vs S0 AUC (original S0 only)", vec("full_s1", "macro_4_macro_auc", orig),
                            vec("s0", "macro_4_macro_auc", orig), "two_sided")}

    out["per_family_summary"] = {
        fam: {"macro_4_f1": descriptives(vec(fam, "macro_4_macro_f1")),
              "macro_4_auc": descriptives(vec(fam, "macro_4_macro_auc")),
              "per_dataset_f1": {ds: descriptives([scores[fam][c]["per_dataset_reports"][ds]["macro_f1"]
                                                   for c in cells]) for ds in DATASETS}}
        for fam in FAMILIES}
    return out


def write_report(out_dir: Path, analyses: dict) -> None:
    names = {"full_s1": "Full-S1", "k1": "K1", "s0": "S0", "c_small": "C_small (scratch)",
             "p1": "P1 (no KD)", "k1_2x2": "K1-2x2", "k1_single": "K1-single", "k0": "K0 (no SSL)"}
    L = ["# Revision ablation TEST results", "",
         f"Protocol `{PROTOCOL_ID}`. Nine matched cells per model; mean +/- SD (ddof=1).", "",
         "| Model | Macro-4 F1 | Macro-4 AUC | CWRU | JNU | HIT | MaFaulDa |", "|---|---|---|---|---|---|---|"]
    for fam in FAMILIES:
        s = analyses["per_family_summary"][fam]
        cell = lambda d: f"{d['mean']:.4f} +/- {d['sd']:.4f}"
        L.append(f"| {names[fam]} | {cell(s['macro_4_f1'])} | {cell(s['macro_4_auc'])} | "
                 + " | ".join(f"{s['per_dataset_f1'][ds]['mean']:.4f}" for ds in DATASETS) + " |")
    for fam_name in ("A_ssl_full_scale", "B_original_compression_contrasts", "C_review_ablations"):
        res = analyses[fam_name]
        L += ["", f"## {fam_name}", "", ANALYSES[fam_name]["status"], "",
              "| Contrast | Mean delta | W/T/L | raw p | Holm p |"
              + (" TOST 0.02 p |" if fam_name == "C_review_ablations" else ""),
              "|---|---|---|---|---|" + ("---|" if fam_name == "C_review_ablations" else "")]
        for k, v in res.items():
            if not isinstance(v, dict) or "contrast" not in v:
                continue
            line = (f"| {k} | {v['mean']:+.4f} | {v['n_positive']}/{v['n_ties']}/{v['n_negative']} | "
                    f"{v['p_for_holm']:.4f} | {res['holm']['p_adjusted'][k]:.4f} |")
            if fam_name == "C_review_ablations":
                line += f" {v['tost_0.02']['p']:.4f} |"
            L.append(line)
        if fam_name == "A_ssl_full_scale":
            L += ["", f"Frozen interpretation label: **{res['interpretation_label']}**"]
    (out_dir / "REVISION_TEST_REPORT.md").write_text("\n".join(L) + "\n")


def cmd_execute(expected_barrier: str, out_dir: Path, device: str) -> None:
    import pandas as pd
    import torch
    from src.pcste_v2 import representation as REPR
    from src.pcste_v2.protocol_cwru_native12 import load_allowlist
    from src.pcste_v2.representation import ItemBuilder

    plan, rows = verify_all(expected_barrier, load_checkpoints=True)
    if out_dir.exists():
        fail(f"refusing to overwrite an existing TEST result area: {out_dir}")
    out_dir.mkdir(parents=True)
    (out_dir / "STATUS").write_text("RUNNING\n")
    device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    pdir = rp(GLOBAL_PDIR)
    REPR.NATIVE12_ALLOWLIST = load_allowlist(rp(CWRU_PDIR))
    scores = {fam: {} for fam in FAMILIES}
    by_fold = {f: [r for r in rows if int(r["fold"]) == f] for f in FOLDS}

    def items_for(fold):
        man = pd.read_csv(pdir / f"global_v2_fold_{fold}.csv", dtype={"fault_severity": str})
        test = man.loc[man["split"] == "test"]
        idx = man.set_index("window_id", drop=False)
        builder = ItemBuilder(pdir, fold, ("v1",), {}, False)
        items = {}
        for ds in DATASETS:
            ids = list(test.loc[test["dataset"] == ds, "window_id"])
            items[ds] = (ids, [builder.build(idx.loc[w]) for w in ids])
        return idx, items

    def run_rows(sel):
        for fold in FOLDS:
            todo = [r for r in by_fold[fold] if r["family"] in sel]
            if not todo:
                continue
            idx, items = items_for(fold)
            for r in todo:
                pred, rep = evaluate(r, items, idx, device)
                stem = f"gf{fold}_s{r['seed']}_{r['family']}"
                pred.to_csv(out_dir / f"{stem}_predictions.csv", index=False)
                (out_dir / f"{stem}_report.json").write_text(
                    json.dumps(rep, indent=1, sort_keys=True, allow_nan=False) + "\n")
                scores[r["family"]][(fold, int(r["seed"]))] = rep
                print(f"[{now()}] evaluated {stem}: Macro-4 F1 {rep['macro_4_macro_f1']:.4f}", flush=True)

    # 1. driver-correctness gate on already-sealed models
    run_rows(SEALED_FAMILIES)
    problems = []
    for fam in SEALED_FAMILIES:
        for (f, s), rep in scores[fam].items():
            sealed = json.loads(rp(f"{plan['sealed_reports_dir']}/gf{f}_s{s}_{fam}_report.json").read_text())
            for msg in compare_with_sealed(rep, sealed):
                problems.append(f"{fam} gf{f}_s{s}: {msg}")
    (out_dir / "DRIVER_CHECK.json").write_text(json.dumps({"reproduced": not problems,
                                                           "problems": problems}, indent=1) + "\n")
    if problems:
        (out_dir / "STATUS").write_text("DRIVER_CHECK_FAILED\n")
        fail("driver did not reproduce the sealed publication results; NO new family was "
             "evaluated:\n  " + "\n  ".join(problems))
    print("DRIVER CHECK PASS: all 18 sealed Full-S1/K1 results reproduced exactly", flush=True)

    # 2. new families
    run_rows(NEW_FAMILIES)
    for fam in FAMILIES:
        if len(scores[fam]) != 9:
            fail(f"{fam}: {len(scores[fam])} cells evaluated, expected 9")

    # 3. pre-specified analyses
    analyses = run_analyses(scores)
    analyses.update({"protocol_identity": PROTOCOL_ID, "created_at_utc": now(),
                     "barrier_sha256": sha256_file(BARRIER), "evaluation_plan_sha256": sha256_file(PLAN),
                     "device": device})
    (out_dir / "ANALYSIS_RESULTS.json").write_text(json.dumps(analyses, indent=1, sort_keys=True,
                                                              default=float) + "\n")
    with (out_dir / "matched_cells_all.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "seed"] + [f"{fam}_macro_4_f1" for fam in FAMILIES]
                   + [f"{fam}_macro_4_auc" for fam in FAMILIES])
        for f in FOLDS:
            for s in SEEDS:
                w.writerow([f, s] + [scores[fam][(f, s)]["macro_4_macro_f1"] for fam in FAMILIES]
                           + [scores[fam][(f, s)]["macro_4_macro_auc"] for fam in FAMILIES])
    write_report(out_dir, analyses)
    hashes = {p.name: sha256_file(p) for p in sorted(out_dir.iterdir()) if p.is_file() and p.name != "STATUS"}
    (out_dir / "OUTPUT_HASHES.json").write_text(json.dumps(hashes, indent=1) + "\n")
    (out_dir / "STATUS").write_text("COMPLETE\n")
    print((out_dir / "REVISION_TEST_REPORT.md").read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--freeze", action="store_true")
    m.add_argument("--preflight", action="store_true")
    m.add_argument("--execute", action="store_true")
    ap.add_argument("--expected-barrier-sha256")
    ap.add_argument("--authorization")
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    try:
        if a.freeze:
            cmd_freeze()
            return
        if not a.expected_barrier_sha256:
            fail("--expected-barrier-sha256 required")
        if a.preflight:
            verify_all(a.expected_barrier_sha256, load_checkpoints=True)
            print("PREFLIGHT PASS: barrier, plan, model table, code, inputs and all 72 checkpoints verified")
            print("TEST loader not invoked; TEST samples remain unopened")
            return
        if a.authorization != AUTHORIZATION_TOKEN:
            fail("TEST execution refused: exact authorization token required")
        if a.output_dir is None:
            fail("--output-dir required")
        cmd_execute(a.expected_barrier_sha256, a.output_dir.resolve(), a.device)
    except BarrierError as exc:
        raise SystemExit(f"REVISION TEST REFUSED: {exc}") from exc


if __name__ == "__main__":
    main()
