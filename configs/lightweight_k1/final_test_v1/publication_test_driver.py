#!/usr/bin/env python
"""Fail-closed driver for lightweight_pcste_final_test_v1.

The default/preflight and self-test paths perform hash, schema, and
checkpoint-structure checks only. They never read a TEST manifest as a
dataframe, instantiate ItemBuilder, load a TEST waveform, or run inference.
Real TEST access requires both --execute and the frozen authorization token.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL_ID = "lightweight_pcste_final_test_v1"
AUTHORIZATION_TOKEN = "AUTHORIZE_LIGHTWEIGHT_PCSTE_FINAL_TEST_V1"
DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")
FOLDS = (1, 2, 3)
SEEDS = (42, 1337, 2026)

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
PLAN = HERE / "PUBLICATION_FINAL_EVALUATION_PLAN.json"
BARRIER = HERE / "PUBLICATION_FINAL_EVALUATION_BARRIER.json"


class BarrierError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def fail(message: str) -> None:
    raise BarrierError(message)


def resolve_repo(rel: str) -> Path:
    p = (REPO / rel).resolve()
    try:
        p.relative_to(REPO.resolve())
    except ValueError:
        fail(f"path escapes repository: {rel}")
    return p


def verify_file(rel: str, expected: str, label: str) -> Path:
    p = resolve_repo(rel)
    if not p.is_file():
        fail(f"{label} missing: {rel}")
    got = sha256_file(p)
    if got != expected:
        fail(f"{label} SHA256 mismatch: {rel}; expected {expected}; got {got}")
    return p


def validate_plan_schema(plan: dict) -> None:
    required = {
        "schema_version", "protocol_identity", "created_at_utc",
        "test_access_at_freeze", "model_families", "excluded_families",
        "cells", "authoritative_registries", "test_membership",
        "metrics", "statistics", "scientific_source_hashes",
        "software_environment", "execution", "historical_barrier_provenance",
    }
    missing = sorted(required - set(plan))
    if missing:
        fail(f"plan schema missing fields: {missing}")
    if plan["protocol_identity"] != PROTOCOL_ID:
        fail("wrong publication protocol identity")
    if plan["model_families"] != ["Full-S1", "K1"]:
        fail("model families must be exactly Full-S1 and K1")
    if plan["excluded_families"] != ["S0", "Q8"]:
        fail("excluded families must be exactly S0 and Q8")
    if plan["test_access_at_freeze"] != {
        "test_loader_invoked": False,
        "test_inference_performed": False,
        "test_metrics_computed": False,
        "test_predictions_accessed": False,
        "test_samples_accessed": False,
    }:
        fail("freeze-time TEST-access declaration is not fully sealed")
    cells = plan["cells"]
    if len(cells) != 9:
        fail(f"expected exactly 9 matched cells, got {len(cells)}")
    keys = [(int(x["fold"]), int(x["seed"])) for x in cells]
    expected = [(f, s) for f in FOLDS for s in SEEDS]
    if keys != expected or len(set(keys)) != 9:
        fail(f"fold/seed matrix differs from frozen ordering: {keys}")
    for cell in cells:
        for family in ("full_s1", "k1"):
            model = cell.get(family, {})
            if set(("run_id", "checkpoint_path", "checkpoint_sha256", "selected_epoch")) - set(model):
                fail(f"cell {cell['fold']}/{cell['seed']} has incomplete {family} record")
    metrics = plan["metrics"]
    if metrics["datasets"] != list(DATASETS):
        fail("dataset order changed")
    if metrics["macro_4"]["formula"] != "(CWRU + JNU + HIT + MaFaulDa) / 4":
        fail("Macro-4 formula changed")
    if metrics["primary_endpoint"] != "Macro-4 Macro-F1":
        fail("primary endpoint changed")
    if metrics["primary_matched_difference"] != "K1 - Full-S1":
        fail("matched difference direction changed")
    stats = plan["statistics"]
    if float(stats["non_inferiority"]["margin"]) != -0.02:
        fail("non-inferiority margin changed")
    if stats["non_inferiority"]["implementation_function"] != "src.methodology_v2.compression.stats.contrast":
        fail("statistical implementation changed")
    if plan["execution"]["authorization_token"] != AUTHORIZATION_TOKEN:
        fail("execution authorization token changed")


def verify_cwru_freeze_index(index_rel: str, expected_digest: str) -> None:
    index = verify_file(index_rel, expected_digest, "CWRU freeze index")
    for line_number, raw in enumerate(index.read_text().splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            rel, expected = raw.rsplit(None, 1)
        except ValueError as exc:
            raise BarrierError(f"invalid CWRU freeze index line {line_number}") from exc
        verify_file(rel, expected, f"CWRU frozen input at index line {line_number}")


def verify_checkpoint_metadata(plan: dict) -> None:
    """CPU-only strict state loading; no dataset or representation access."""
    import torch
    from src.pcste_v2.model import PCSTEv2, PCSTEv2Config
    from src.methodology_v2.experiment.heads import DatasetHeads
    from src.methodology_v2.compression.student import (
        build_encoder, build_heads, half_4x1_spec,
    )

    for cell in plan["cells"]:
        fold, seed = int(cell["fold"]), int(cell["seed"])
        full = cell["full_s1"]
        ck = torch.load(resolve_repo(full["checkpoint_path"]), map_location="cpu", weights_only=False)
        if str(ck.get("run_id")) != full["run_id"] or int(ck.get("epoch", -1)) != int(full["selected_epoch"]):
            fail(f"Full-S1 checkpoint metadata mismatch for GF{fold}/S{seed}")
        model = PCSTEv2(PCSTEv2Config())
        heads = DatasetHeads()
        model.load_state_dict(ck["model"], strict=True)
        heads.load_state_dict(ck["heads"], strict=True)
        del ck, model, heads

        k1 = cell["k1"]
        ck = torch.load(resolve_repo(k1["checkpoint_path"]), map_location="cpu", weights_only=False)
        if str(ck.get("run_id")) != k1["run_id"] or int(ck.get("epoch", -1)) != int(k1["selected_epoch"]):
            fail(f"K1 checkpoint metadata mismatch for GF{fold}/S{seed}")
        spec = half_4x1_spec("mean_of_remaining")
        encoder = build_encoder(spec, seed=0)
        heads = build_heads(spec, init_seed=seed)
        if sum(p.numel() for p in encoder.parameters()) != 1_375_953:
            fail("K1 encoder parameter identity changed")
        encoder.load_state_dict(ck["encoder"], strict=True)
        heads.load_state_dict(ck["heads"], strict=True)
        del ck, encoder, heads


def verify_plan_and_barrier(expected_barrier_sha256: str, *, checkpoint_metadata: bool = True) -> dict:
    if not PLAN.is_file() or not BARRIER.is_file():
        fail("publication plan or barrier is missing")
    barrier_bytes = BARRIER.read_bytes()
    got_barrier = sha256_bytes(barrier_bytes)
    if got_barrier != expected_barrier_sha256:
        fail(f"barrier SHA256 mismatch; expected {expected_barrier_sha256}; got {got_barrier}")
    barrier = json.loads(barrier_bytes)
    if barrier.get("schema_version") != "lightweight_pcste_publication_barrier.v1":
        fail("barrier schema/version mismatch")
    if barrier.get("protocol_identity") != PROTOCOL_ID:
        fail("barrier protocol identity mismatch")
    if barrier.get("status") != "FROZEN_BEFORE_TEST":
        fail("barrier is not in frozen-before-TEST state")

    plan_bytes = PLAN.read_bytes()
    if sha256_bytes(plan_bytes) != barrier.get("evaluation_plan_sha256"):
        fail("machine-readable evaluation plan differs from barrier commitment")
    plan = json.loads(plan_bytes)
    validate_plan_schema(plan)

    if barrier.get("publication_driver_sha256") != plan["scientific_source_hashes"][barrier["publication_driver_path"]]:
        fail("driver digest disagreement between plan and barrier")
    if barrier.get("model_table_sha256") != plan["authoritative_registries"]["publication_frozen_model_table"]["sha256"]:
        fail("model-table digest disagreement between plan and barrier")

    for rel, expected in plan["scientific_source_hashes"].items():
        verify_file(rel, expected, "scientific source")
    for record in plan["authoritative_registries"].values():
        verify_file(record["path"], record["sha256"], "authoritative registry")
    for fold_key, record in plan["test_membership"]["global_fold_manifests"].items():
        verify_file(record["path"], record["sha256"], f"global fold manifest {fold_key}")
    reg = plan["test_membership"]["global_hash_registry"]
    verify_file(reg["path"], reg["sha256"], "global manifest hash registry")
    global_registry = json.loads(resolve_repo(reg["path"]).read_text())
    if global_registry.get("cwru_protocol_id") != plan["test_membership"]["cwru"]["protocol_identity"]:
        fail("CWRU protocol identity changed in global hash registry")
    for fold in FOLDS:
        expected = plan["test_membership"]["global_fold_manifests"][f"GF{fold}"]["sha256"]
        if global_registry.get(f"fold_{fold}", {}).get("sha256") != expected:
            fail(f"global hash registry disagrees for GF{fold}")
    cwru = plan["test_membership"]["cwru"]
    verify_cwru_freeze_index(cwru["freeze_index_path"], cwru["freeze_digest_sha256"])

    for cell in plan["cells"]:
        for family, label in (("full_s1", "Full-S1"), ("k1", "K1")):
            rec = cell[family]
            verify_file(rec["checkpoint_path"], rec["checkpoint_sha256"], f"{label} checkpoint")
    if checkpoint_metadata:
        verify_checkpoint_metadata(plan)
    return plan


def self_test(expected_barrier_sha256: str) -> None:
    plan = verify_plan_and_barrier(expected_barrier_sha256, checkpoint_metadata=True)
    original = PLAN.read_bytes()
    changed = bytearray(original)
    changed[len(changed) // 2] ^= 1
    committed = json.loads(BARRIER.read_text())["evaluation_plan_sha256"]
    if sha256_bytes(bytes(changed)) == committed:
        fail("in-memory altered-plan rejection probe failed")
    altered = copy.deepcopy(plan)
    altered["statistics"]["non_inferiority"]["margin"] = -0.019
    rejected = False
    try:
        validate_plan_schema(altered)
    except BarrierError:
        rejected = True
    if not rejected:
        fail("in-memory altered-scientific-field rejection probe failed")
    print("PREFLIGHT PASS: all frozen paths, hashes, schemas, and checkpoint state dicts are valid")
    print("NEGATIVE SELF-TEST PASS: altered plan bytes and altered margin are rejected")
    print("TEST loader not invoked; TEST samples/predictions remain unopened")


def add_macro_fields(report: dict) -> dict:
    """Aggregate unchanged frozen per-class metrics; do not recompute classes."""
    import numpy as np
    for ds in DATASETS:
        rep = report["per_dataset_reports"][ds]
        rep["macro_precision"] = float(np.mean(rep["per_class_precision"]))
        rep["macro_recall"] = float(np.mean(rep["per_class_recall"]))
    aucs = [float(report["per_dataset_reports"][d]["macro_ovr_auc"]) for d in DATASETS]
    report["macro_4_macro_f1"] = float(report["macro_domain_f1"])
    report["macro_4_macro_auc"] = float(sum(aucs) / 4.0) if all(math.isfinite(x) for x in aucs) else None
    return report


def predict_one(cell: dict, family: str, manifest, test_rows, builder, device: str):
    """Forward-only inference using frozen model/representation/metric modules."""
    import numpy as np
    import pandas as pd
    import torch
    from src.pcste_v2.model import PCSTEv2, PCSTEv2Config, collate_v2
    from src.pcste_v2.metrics_v2 import evaluate_split
    from src.methodology_v2.encoder import collate_representations
    from src.methodology_v2.experiment.heads import CLASS_ORDERS, DatasetHeads
    from src.methodology_v2.compression.student import build_encoder, build_heads, half_4x1_spec

    rec = cell[family]
    before = sha256_file(resolve_repo(rec["checkpoint_path"]))
    ck = torch.load(resolve_repo(rec["checkpoint_path"]), map_location=device, weights_only=False)
    if family == "full_s1":
        encoder = PCSTEv2(PCSTEv2Config()).to(device)
        encoder.load_state_dict(ck["model"], strict=True)
    else:
        spec = half_4x1_spec("mean_of_remaining")
        encoder = build_encoder(spec, seed=0).to(device)
        encoder.load_state_dict(ck["encoder"], strict=True)
    heads = (DatasetHeads() if family == "full_s1" else build_heads(half_4x1_spec("mean_of_remaining"), init_seed=int(cell["seed"]))).to(device)
    heads.load_state_dict(ck["heads"], strict=True)
    encoder.eval(); heads.eval()
    rows = []
    indexed = manifest.set_index("window_id", drop=False)
    with torch.no_grad():
        for ds in DATASETS:
            ids = list(test_rows.loc[test_rows["dataset"] == ds, "window_id"])
            for lo in range(0, len(ids), 64):
                chunk = ids[lo:lo + 64]
                items = [builder.build(indexed.loc[w]) for w in chunk]
                if family == "full_s1":
                    batch = collate_v2(items, 1, 1, False)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    z = encoder(**batch)["global_embedding"]
                else:
                    batch = collate_representations([it["streams"]["g0_c0"] for it in items])
                    batch = {k: v.to(device) for k, v in batch.items()}
                    z = encoder(**batch)["global_embedding"]
                logits = heads(z, ds)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                pred_idx = logits.argmax(dim=-1).cpu().numpy()
                for i, wid in enumerate(chunk):
                    source = indexed.loc[wid]
                    row = {
                        "protocol_identity": PROTOCOL_ID, "family": "Full-S1" if family == "full_s1" else "K1",
                        "run_id": rec["run_id"], "fold": int(cell["fold"]), "seed": int(cell["seed"]),
                        "dataset": ds, "window_id": wid, "y_true": str(source["class"]),
                        "y_pred": CLASS_ORDERS[ds][int(pred_idx[i])], "p_max": float(probs[i].max()),
                        "physical_specimen": source["physical_specimen"], "group_id": source["group_id"],
                        "fault_severity": source["fault_severity"], "rpm": source["rpm"], "load": source["load"],
                    }
                    row.update({f"prob__{name}": float(probs[i, j]) for j, name in enumerate(CLASS_ORDERS[ds])})
                    rows.append(row)
    pred = pd.DataFrame(rows)
    if len(pred) != len(test_rows) or pred["window_id"].duplicated().any() or set(pred["window_id"]) != set(test_rows["window_id"]):
        fail(f"inexact TEST membership for {rec['run_id']}")
    after = sha256_file(resolve_repo(rec["checkpoint_path"]))
    if before != rec["checkpoint_sha256"] or after != before:
        fail(f"checkpoint changed during inference for {rec['run_id']}")
    report = add_macro_fields(evaluate_split(pred))
    report.update({
        "protocol_identity": PROTOCOL_ID, "family": "Full-S1" if family == "full_s1" else "K1",
        "run_id": rec["run_id"], "fold": int(cell["fold"]), "seed": int(cell["seed"]),
        "checkpoint_path": rec["checkpoint_path"], "checkpoint_sha256": before,
        "selected_epoch": int(rec["selected_epoch"]), "partition": "test",
    })
    return pred, report


def execute(plan: dict, output_dir: Path, device: str) -> None:
    """The only code path that opens TEST manifests or TEST samples."""
    import pandas as pd
    import torch
    from src.pcste_v2 import representation as REPR
    from src.pcste_v2.protocol_cwru_native12 import load_allowlist
    from src.pcste_v2.representation import ItemBuilder
    from src.methodology_v2.compression.stats import contrast, descriptives

    if output_dir.exists():
        fail(f"refusing to overwrite an existing TEST result area: {output_dir}")
    output_dir.mkdir(parents=True)
    device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    pdir = resolve_repo(plan["test_membership"]["global_protocol_directory"])
    cwru_dir = resolve_repo(plan["test_membership"]["cwru"]["protocol_directory"])
    REPR.NATIVE12_ALLOWLIST = load_allowlist(cwru_dir)
    reports = []
    for cell in plan["cells"]:
        fold = int(cell["fold"])
        manifest = pd.read_csv(pdir / f"global_v2_fold_{fold}.csv", dtype={"fault_severity": str})
        test_rows = manifest.loc[manifest["split"] == "test"].copy()
        if test_rows.empty:
            fail(f"empty TEST partition for GF{fold}")
        builder = ItemBuilder(pdir, fold, ("v1",), {}, False)
        for family in ("full_s1", "k1"):
            pred, report = predict_one(cell, family, manifest, test_rows, builder, device)
            stem = f"gf{fold}_s{cell['seed']}_{family}"
            pred.to_csv(output_dir / f"{stem}_predictions.csv", index=False)
            (output_dir / f"{stem}_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
            reports.append(report)
    by_cell = {(r["fold"], r["seed"], r["family"]): r for r in reports}
    rows = []
    full_f1, k1_f1, full_auc, k1_auc = [], [], [], []
    for cell in plan["cells"]:
        key = (int(cell["fold"]), int(cell["seed"]))
        a, b = by_cell[key + ("Full-S1",)], by_cell[key + ("K1",)]
        row = {"fold": key[0], "seed": key[1], "full_s1_macro_4_f1": a["macro_4_macro_f1"],
               "k1_macro_4_f1": b["macro_4_macro_f1"], "delta_f1": b["macro_4_macro_f1"] - a["macro_4_macro_f1"],
               "full_s1_macro_4_auc": a["macro_4_macro_auc"], "k1_macro_4_auc": b["macro_4_macro_auc"],
               "delta_auc": None if a["macro_4_macro_auc"] is None or b["macro_4_macro_auc"] is None else b["macro_4_macro_auc"] - a["macro_4_macro_auc"]}
        rows.append(row); full_f1.append(row["full_s1_macro_4_f1"]); k1_f1.append(row["k1_macro_4_f1"])
        full_auc.append(row["full_s1_macro_4_auc"]); k1_auc.append(row["k1_macro_4_auc"])
    paired_f1 = contrast("K1 vs Full-S1", k1_f1, full_f1, "ni_then_superiority", 0.02)
    aggregate = {"protocol_identity": PROTOCOL_ID, "cells": rows, "paired_macro_4_f1": paired_f1,
                 "full_s1_macro_4_f1": descriptives(full_f1), "k1_macro_4_f1": descriptives(k1_f1),
                 "macro_4_auc": None if any(x is None for x in full_auc + k1_auc) else {
                     "full_s1": descriptives(full_auc), "k1": descriptives(k1_auc),
                     "paired_delta": descriptives([b - a for a, b in zip(full_auc, k1_auc)])},
                 "created_at_utc": datetime.now(timezone.utc).isoformat(),
                 "barrier_sha256": sha256_file(BARRIER), "evaluation_plan_sha256": sha256_file(PLAN)}
    with (output_dir / "matched_cells.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output_dir / "STATUS").write_text("COMPLETE\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-barrier-sha256", required=True)
    parser.add_argument("--authorization")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test(args.expected_barrier_sha256)
            return
        plan = verify_plan_and_barrier(args.expected_barrier_sha256, checkpoint_metadata=True)
        if args.preflight:
            print("PREFLIGHT PASS: publication barrier and all pinned inputs match")
            print("TEST loader not invoked; TEST samples/predictions remain unopened")
            return
        if args.authorization != AUTHORIZATION_TOKEN:
            fail("TEST execution refused: exact frozen authorization token required")
        if args.output_dir is None:
            fail("TEST execution refused: --output-dir is required")
        execute(plan, args.output_dir.resolve(), args.device)
    except BarrierError as exc:
        raise SystemExit(f"PUBLICATION TEST REFUSED: {exc}") from exc


if __name__ == "__main__":
    main()
