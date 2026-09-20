#!/usr/bin/env python
"""Fail-closed driver for `lightweight_pcste_q8_v1` — the secondary Q8 extension.

Q8 is derived deterministically from each already-frozen K1 selected checkpoint
using the historical Part-6 implementation, which is imported and never edited.

This driver never retrains, never reruns Full-S1 or K1 inference, never changes
TEST membership, and never tunes anything on TEST. K1 reference numbers are read
from the already-frozen primary TEST artifacts.

Stages:
    convert   quantize all nine cells, write the conversion manifest (no TEST)
    evaluate  Q8 TEST inference for all nine cells + statistics vs frozen K1
    size      standardized storage comparison on the representative cell
    agreement sim vs cpu_dynamic prediction agreement on VALIDATION (never TEST)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PUB = HERE.parents[1]
sys.path.insert(0, str(PUB))

CANON = Path("/scratch/ma06314/Standard Project/rotating_machinery_fault_diagnosis")
PROTOCOL_ID = "lightweight_pcste_q8_v1"
DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")
NI_MARGIN = 0.01          # verified from quantization_spec.yaml and protocol.py
K1_RESULTS = PUB / "results/final_test/publication_final_test_v1"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg: str):
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# barrier
# ---------------------------------------------------------------------------
def verify_barrier(expected_barrier_sha256: str) -> dict:
    bpath = HERE / "Q8_EVALUATION_BARRIER.json"
    got = sha256_file(bpath)
    if got != expected_barrier_sha256:
        fail(f"barrier SHA mismatch: expected {expected_barrier_sha256}, got {got}")
    barrier = json.loads(bpath.read_text())
    if barrier.get("status") != "FROZEN_BEFORE_Q8_EVALUATION":
        fail("barrier is not in frozen-before-Q8 state")
    for flag in ("q8_evaluated_before_freeze", "q8_result_observed_before_freeze",
                 "test_membership_changed", "k1_inference_rerun",
                 "full_s1_inference_rerun", "q8_tuned_on_test"):
        if barrier.get(flag) is not False:
            fail(f"barrier declaration not sealed: {flag}")

    plan = json.loads((HERE / "Q8_EVALUATION_PLAN.json").read_text())
    for key, rel in (("evaluation_plan_sha256", "Q8_EVALUATION_PLAN.json"),
                     ("human_plan_sha256", "Q8_EVALUATION_PLAN.md"),
                     ("model_table_sha256", "Q8_FROZEN_MODEL_TABLE.csv")):
        if barrier[key] != sha256_file(HERE / rel):
            fail(f"barrier no longer pins {rel}")

    # every pinned source file must be byte-identical
    for rel, want in plan["source_hashes"].items():
        got = sha256_file(PUB / rel)
        if got != want:
            fail(f"pinned source changed: {rel}\n  expected {want}\n  got      {got}")

    # frozen K1 reference artifacts must be untouched
    for rel, want in plan["evaluation"]["k1_reference_artifacts"].items():
        if sha256_file(K1_RESULTS / rel) != want:
            fail(f"frozen K1 reference artifact changed: {rel}")

    # TEST membership manifests must be unchanged
    for key, rec in plan["evaluation"]["test_membership"]["global_fold_manifests"].items():
        if sha256_file(PUB / rec["path"]) != rec["sha256"]:
            fail(f"TEST membership manifest changed: {key}")

    if plan["statistics"]["provenance"][0]["value"] != NI_MARGIN:
        fail("NI margin in the plan does not match the driver constant")
    return plan


def verify_cells(plan: dict) -> list[dict]:
    cells = plan["source_models"]["cells"]
    if len(cells) != 9:
        fail(f"expected 9 cells, got {len(cells)}")
    for c in cells:
        p = CANON / c["k1_checkpoint"]
        if not p.name == "best.pt":
            fail(f"refusing a non-selected checkpoint: {c['k1_checkpoint']}")
        got = sha256_file(p)
        if got != c["k1_sha256"]:
            fail(f"K1 checkpoint changed for GF{c['fold']}/S{c['seed']}")
    return cells


# ---------------------------------------------------------------------------
# model construction and Q8 conversion
# ---------------------------------------------------------------------------
def load_k1(cell: dict, device="cpu"):
    from src.methodology_v2.compression.student import build_encoder, half_4x1_spec
    from src.methodology_v2.experiment.heads import DatasetHeads
    ck = torch.load(CANON / cell["k1_checkpoint"], map_location="cpu", weights_only=False)
    enc = build_encoder(half_4x1_spec("mean_of_remaining"), seed=0)
    r = enc.load_state_dict(ck["encoder"], strict=True)
    assert not r.missing_keys and not r.unexpected_keys
    hd = DatasetHeads()
    r = hd.load_state_dict(ck["heads"], strict=True)
    assert not r.missing_keys and not r.unexpected_keys
    n = sum(p.numel() for p in enc.parameters())
    if n != 1_375_953:
        fail(f"K1 encoder parameter count changed: {n}")
    return enc.to(device).eval(), hd.to(device).eval()


def quantize_cell(cell: dict):
    """Apply the exact historical Q8 recipe. Returns (enc_q8, heads_q8, record)."""
    from src.methodology_v2.compression import quantization as Q
    enc, hd = load_k1(cell, "cpu")
    q_enc = Q.apply_q8_simulated(enc)
    q_hd = Q.apply_q8_simulated(hd)

    def tensor_digest(state: dict) -> str:
        h = hashlib.sha256()
        for k in sorted(state):
            v = state[k]
            h.update(k.encode())
            h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
        return h.hexdigest()

    scales = {}
    for tag, res in (("encoder", q_enc), ("heads", q_hd)):
        for k, v in res.int8_state.items():
            if k.endswith(".scale"):
                s = v.detach().cpu().numpy()
                scales[f"{tag}.{k}"] = {"n_channels": int(s.size),
                                        "min": float(s.min()), "max": float(s.max()),
                                        "mean": float(s.mean())}
    record = {
        "fold": cell["fold"], "seed": cell["seed"], "k1_run_id": cell["k1_run_id"],
        "k1_checkpoint": cell["k1_checkpoint"], "k1_sha256": cell["k1_sha256"],
        "calibration": "none",
        "int8_modules": sorted([k for k, v in q_enc.plan.items() if v == "int8"]
                               + [f"heads::{k}" for k, v in q_hd.plan.items() if v == "int8"]),
        "fp32_modules": sorted([k for k, v in q_enc.plan.items() if v == "fp32"]
                               + [f"heads::{k}" for k, v in q_hd.plan.items() if v == "fp32"]),
        "n_int8_params": q_enc.n_int8_params + q_hd.n_int8_params,
        "n_fp32_params": q_enc.n_fp32_params + q_hd.n_fp32_params,
        "fraction_int8": (q_enc.n_int8_params + q_hd.n_int8_params)
                         / (q_enc.n_int8_params + q_hd.n_int8_params
                            + q_enc.n_fp32_params + q_hd.n_fp32_params),
        "max_weight_abs_err": max(q_enc.max_weight_abs_err, q_hd.max_weight_abs_err),
        "n_scale_tensors": len(scales),
        "per_channel_scales": scales,
        # deterministic identity of the produced Q8 artifact
        "q8_encoder_state_digest": tensor_digest(q_enc.int8_state),
        "q8_heads_state_digest": tensor_digest(q_hd.int8_state),
    }
    return q_enc.model, q_hd.model, record


# ---------------------------------------------------------------------------
# evaluation (the only path that reads TEST)
# ---------------------------------------------------------------------------
def add_macro_fields(report: dict) -> dict:
    """Identical aggregation to the primary driver — frozen per-class metrics are
    aggregated, never recomputed."""
    for ds in DATASETS:
        rep = report["per_dataset_reports"][ds]
        rep["macro_precision"] = float(np.mean(rep["per_class_precision"]))
        rep["macro_recall"] = float(np.mean(rep["per_class_recall"]))
    aucs = [float(report["per_dataset_reports"][d]["macro_ovr_auc"]) for d in DATASETS]
    report["macro_4_macro_f1"] = float(report["macro_domain_f1"])
    report["macro_4_macro_auc"] = (float(sum(aucs) / 4.0)
                                   if all(math.isfinite(x) for x in aucs) else None)
    return report


def predict_q8(cell, enc, hd, manifest, test_rows, builder, device):
    from src.pcste_v2.metrics_v2 import evaluate_split
    from src.methodology_v2.encoder import collate_representations
    from src.methodology_v2.experiment.heads import CLASS_ORDERS
    enc = enc.to(device).eval()
    hd = hd.to(device).eval()
    indexed = manifest.set_index("window_id", drop=False)
    rows = []
    with torch.no_grad():
        for ds in DATASETS:
            ids = list(test_rows.loc[test_rows["dataset"] == ds, "window_id"])
            for lo in range(0, len(ids), 64):
                chunk = ids[lo:lo + 64]
                items = [builder.build(indexed.loc[w]) for w in chunk]
                batch = collate_representations([it["streams"]["g0_c0"] for it in items])
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = hd(enc(**batch)["global_embedding"], ds)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                pred_idx = logits.argmax(dim=-1).cpu().numpy()
                for i, wid in enumerate(chunk):
                    src = indexed.loc[wid]
                    row = {"protocol_identity": PROTOCOL_ID, "family": "Q8",
                           "run_id": cell["k1_run_id"], "fold": int(cell["fold"]),
                           "seed": int(cell["seed"]), "dataset": ds, "window_id": wid,
                           "y_true": str(src["class"]),
                           "y_pred": CLASS_ORDERS[ds][int(pred_idx[i])],
                           "p_max": float(probs[i].max()),
                           "physical_specimen": src["physical_specimen"],
                           "group_id": src["group_id"], "fault_severity": src["fault_severity"],
                           "rpm": src["rpm"], "load": src["load"]}
                    row.update({f"prob__{n}": float(probs[i, j])
                                for j, n in enumerate(CLASS_ORDERS[ds])})
                    rows.append(row)
    pred = pd.DataFrame(rows)
    if (len(pred) != len(test_rows) or pred["window_id"].duplicated().any()
            or set(pred["window_id"]) != set(test_rows["window_id"])):
        fail(f"inexact TEST membership for {cell['k1_run_id']}")
    report = add_macro_fields(evaluate_split(pred))
    report.update({"protocol_identity": PROTOCOL_ID, "family": "Q8",
                   "run_id": cell["k1_run_id"], "fold": int(cell["fold"]),
                   "seed": int(cell["seed"]), "partition": "test",
                   "source_k1_checkpoint": cell["k1_checkpoint"],
                   "source_k1_sha256": cell["k1_sha256"]})
    return pred, report


def k1_reference() -> dict:
    """Already-frozen K1 TEST numbers. K1 is NOT rerun."""
    out = {}
    with (K1_RESULTS / "matched_cells.csv").open() as fh:
        for r in csv.DictReader(fh):
            out[(int(r["fold"]), int(r["seed"]))] = {
                "macro_4_macro_f1": float(r["k1_macro_4_f1"]),
                "macro_4_macro_auc": float(r["k1_macro_4_auc"])}
    for (f, s), rec in out.items():
        rep = json.loads((K1_RESULTS / f"per_model_reports/gf{f}_s{s}_k1_report.json").read_text())
        if abs(rep["macro_4_macro_f1"] - rec["macro_4_macro_f1"]) > 1e-12:
            fail(f"frozen K1 artifacts disagree for GF{f}/S{s}")
        rec["per_dataset"] = {ds: {"macro_f1": float(rep["per_dataset_reports"][ds]["macro_f1"]),
                                   "macro_ovr_auc": float(rep["per_dataset_reports"][ds]["macro_ovr_auc"])}
                              for ds in DATASETS}
    return out


def evaluate(plan: dict, out_dir: Path, device: str):
    from src.pcste_v2.representation import ItemBuilder
    from src.pcste_v2.protocol_cwru_native12 import load_allowlist
    from src.pcste_v2 import representation as REPR
    from src.methodology_v2.compression import stats as ST
    if out_dir.exists() and any(out_dir.iterdir()):
        fail(f"refusing to overwrite an existing Q8 result area: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    REPR.NATIVE12_ALLOWLIST = load_allowlist(PUB / "protocols/datasets/cwru_native12_load_v1")

    cells = verify_cells(plan)
    k1ref = k1_reference()
    pdir = CANON / "pcste_v2/protocol/global_v2_final_s0s1_v1"
    conv, reports, matched = [], [], []

    for fold in (1, 2, 3):
        manifest = pd.read_csv(pdir / f"global_v2_fold_{fold}.csv", dtype={"fault_severity": str})
        test_rows = manifest.loc[manifest["split"] == "test"].copy()
        if test_rows.empty:
            fail(f"empty TEST partition for GF{fold}")
        builder = ItemBuilder(pdir, fold, ("v1",), {}, False)
        for cell in [c for c in cells if c["fold"] == fold]:
            enc_q, hd_q, rec = quantize_cell(cell)
            conv.append(rec)
            pred, report = predict_q8(cell, enc_q, hd_q, manifest, test_rows, builder, device)
            after = sha256_file(CANON / cell["k1_checkpoint"])
            if after != cell["k1_sha256"]:
                fail(f"K1 checkpoint changed during Q8 evaluation: {cell['k1_run_id']}")
            reports.append(report)
            (out_dir / f"gf{fold}_s{cell['seed']}_q8_report.json").write_text(
                json.dumps(report, indent=1) + "\n")
            k1 = k1ref[(fold, cell["seed"])]
            matched.append({
                "fold": fold, "seed": cell["seed"],
                "k1_macro_4_f1": k1["macro_4_macro_f1"], "q8_macro_4_f1": report["macro_4_macro_f1"],
                "delta_f1": report["macro_4_macro_f1"] - k1["macro_4_macro_f1"],
                "k1_macro_4_auc": k1["macro_4_macro_auc"], "q8_macro_4_auc": report["macro_4_macro_auc"],
                "delta_auc": report["macro_4_macro_auc"] - k1["macro_4_macro_auc"]})
            print(f"  GF{fold}/S{cell['seed']:<5} K1 {k1['macro_4_macro_f1']:.6f} -> Q8 "
                  f"{report['macro_4_macro_f1']:.6f}  delta {matched[-1]['delta_f1']:+.6f}", flush=True)

    matched.sort(key=lambda r: (r["fold"], r["seed"]))
    with (out_dir / "q8_matched_cells.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(matched[0].keys())); w.writeheader(); w.writerows(matched)

    order = [(r["fold"], r["seed"]) for r in matched]
    q8_f1 = [r["q8_macro_4_f1"] for r in matched]
    k1_f1 = [r["k1_macro_4_f1"] for r in matched]
    q8_auc = [r["q8_macro_4_auc"] for r in matched]
    k1_auc = [r["k1_macro_4_auc"] for r in matched]

    # EXACT historical call, as used by stats.confirmatory_family H3
    h3 = ST.contrast("H3 Q8(K1) vs K1", q8_f1, k1_f1, "ni", NI_MARGIN)

    rep_by_cell = {(r["fold"], r["seed"]): r for r in reports}
    per_ds_rows, per_ds = [], {}
    for ds in DATASETS:
        for metric, key in (("macro_f1", "macro_f1"), ("macro_auc", "macro_ovr_auc")):
            q = [float(rep_by_cell[c]["per_dataset_reports"][ds][key]) for c in order]
            k = [float(k1ref[c]["per_dataset"][ds][key]) for c in order]
            d = [a - b for a, b in zip(q, k)]
            desc = ST.descriptives(d)
            per_ds[f"{ds}|{metric}"] = {"k1_mean": float(np.mean(k)), "k1_sd": float(np.std(k, ddof=1)),
                                        "q8_mean": float(np.mean(q)), "q8_sd": float(np.std(q, ddof=1)),
                                        "delta": desc}
            per_ds_rows.append({"dataset": ds, "metric": metric,
                                "k1_mean": np.mean(k), "k1_sd": np.std(k, ddof=1),
                                "q8_mean": np.mean(q), "q8_sd": np.std(q, ddof=1),
                                "delta_mean": desc["mean"], "delta_sd": desc["sd"],
                                "q8_wins": desc["n_positive"], "ties": desc["n_ties"],
                                "q8_losses": desc["n_negative"]})
    with (out_dir / "q8_per_dataset_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_ds_rows[0].keys())); w.writeheader(); w.writerows(per_ds_rows)

    def d(v):
        return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
    agg = {
        "protocol_identity": PROTOCOL_ID,
        "schema_version": "lightweight_pcste_q8_aggregate.v1",
        "stage": "SECONDARY post-primary extension",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device, "n_cells": 9,
        "k1_reference": "already-frozen primary TEST results; K1 inference was NOT rerun",
        "macro_4_macro_f1": {"k1": d(k1_f1), "q8": d(q8_f1),
                             "paired_delta_q8_minus_k1": ST.descriptives([a - b for a, b in zip(q8_f1, k1_f1)])},
        "macro_4_macro_auc": {"k1": d(k1_auc), "q8": d(q8_auc),
                              "paired_delta_q8_minus_k1": ST.descriptives([a - b for a, b in zip(q8_auc, k1_auc)])},
        "non_inferiority": {
            "defined_by_historical_protocol": True,
            "margin": NI_MARGIN,
            "margin_provenance": "quantization_spec.yaml ni_margin AND protocol.py NI_MARGIN_PTQ",
            "call": "stats.contrast('H3 Q8(K1) vs K1', q8, k1, 'ni', 0.01)",
            "result": h3,
            "holm_family_applied": False,
            "holm_note": "The historical Holm m=3 family is not executable (C_small never evaluated) and the "
                         "primary study used a single confirmatory contrast. H3 is run standalone as a secondary "
                         "contrast and carries no family-wise error control from that family.",
            "primary_margin_does_not_apply": -0.02},
        "per_dataset": per_ds,
        "conversion": {"calibration": "none",
                       "n_int8_params": conv[0]["n_int8_params"],
                       "n_fp32_params": conv[0]["n_fp32_params"],
                       "fraction_int8": conv[0]["fraction_int8"]},
    }
    (out_dir / "q8_aggregate_summary.json").write_text(json.dumps(agg, indent=1) + "\n")
    (out_dir / "q8_conversion_manifest.json").write_text(json.dumps(
        {"protocol_identity": PROTOCOL_ID, "calibration": "none",
         "quantization_implementation_sha256": plan["source_hashes"]["src/methodology_v2/compression/quantization.py"],
         "cells": conv}, indent=1) + "\n")
    print("\nQ8 Macro-4 Macro-F1: K1 %.6f +/- %.6f -> Q8 %.6f +/- %.6f" % (
        np.mean(k1_f1), np.std(k1_f1, ddof=1), np.mean(q8_f1), np.std(q8_f1, ddof=1)))
    print("paired delta %.6f +/- %.6f | NI margin %.2f | p=%s | passes=%s" % (
        h3["mean"], h3["sd"], NI_MARGIN,
        h3["non_inferiority"]["p_one_sided"], h3["non_inferiority"]["passes"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("verify", "evaluate"))
    ap.add_argument("--expected-barrier-sha256", required=True)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    try:
        plan = verify_barrier(a.expected_barrier_sha256)
        if a.stage == "verify":
            verify_cells(plan)
            print("Q8 BARRIER OK: plan, sources, checkpoints, TEST manifests and frozen K1 "
                  "reference artifacts all match. TEST not opened.")
            return
        dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        if a.output_dir is None:
            fail("--output-dir is required for evaluate")
        evaluate(plan, a.output_dir.resolve(), dev)
    except Exception as exc:
        raise SystemExit(f"Q8 EXTENSION REFUSED: {exc}") from exc


if __name__ == "__main__":
    main()
