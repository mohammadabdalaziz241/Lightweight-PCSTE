#!/usr/bin/env python3
"""Describe all nine validation comparisons; never reads TEST results."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from run import HERE, sha, write_json


def describe(values):
    return {"n": len(values), "mean": statistics.mean(values),
        "median": statistics.median(values), "sample_sd": statistics.stdev(values),
        "positive": sum(x > 0 for x in values), "zero": sum(x == 0 for x in values),
        "negative": sum(x < 0 for x in values)}


def collect(root, phase):
    reference = json.loads((HERE / "reference_validation.json").read_text())
    expected = {(f, s) for f in (1, 2, 3) for s in (42, 1337, 2026)}
    if {(x["fold"], x["seed"]) for x in reference["cells"]} != expected:
        raise ValueError("Incomplete or duplicate reference matrix")
    rows, data_deltas = [], {}
    for ref in reference["cells"]:
        fold, seed = ref["fold"], ref["seed"]
        folder = root / f"{phase}_gf{fold}_s{seed}"
        meta = json.loads((folder / "provenance.json").read_text())
        if (meta["fold"], meta["seed"]) != (fold, seed):
            raise ValueError("Provenance cell mismatch")
        if meta["plan_sha256"] != sha(HERE / "plan.json") or meta["runner_sha256"] != sha(HERE / "run.py"):
            raise ValueError("Mixed experiment versions; inspect before aggregating")
        if meta["new_test_inference"] is not False:
            raise ValueError("Unexpected TEST usage")
        row = {"fold": fold, "seed": seed}
        if phase == "screen":
            result = json.loads((folder / "screen.json").read_text())
            if result["status"] != "COMPLETE" or result["split"] != "validation":
                raise ValueError("Incomplete/non-validation screening result")
            if (result["fold"], result["seed"]) != (fold, seed):
                raise ValueError("Screen result cell mismatch")
            scores = result["scores"]
            for arm in ("full", "2x2", "4x1"):
                row[arm] = scores[arm]["macro_domain_f1"]
            two_ds = scores["2x2"]["per_dataset_macro_f1"]
            four_ds = scores["4x1"]["per_dataset_macro_f1"]
        else:
            result = json.loads((folder / "completion.json").read_text())
            history = [json.loads(x) for x in (folder / "epoch_metrics.jsonl").read_text().splitlines() if x.strip()]
            if result["status"] != "COMPLETE" or result["epochs"] != 50 or len(history) != 50:
                raise ValueError("Incomplete recovery run")
            if [x["epoch"] for x in history] != list(range(50)):
                raise ValueError("Missing/duplicate recovery epochs")
            if (result["fold"], result["seed"], result["arm"]) != (fold, seed, "2x2"):
                raise ValueError("Recovery result cell/arm mismatch")
            best = max(history, key=lambda x: x["val_macro_domain_f1"])
            if best["epoch"] != result["best"]["epoch"] or best["val_macro_domain_f1"] != result["best"]["macro_domain_f1"]:
                raise ValueError("Best-checkpoint summary disagrees with epoch history")
            if sha(folder / "best.pt") != result["best_checkpoint_sha256"]:
                raise ValueError("Best checkpoint changed")
            row.update({"2x2": result["best"]["macro_domain_f1"], "4x1": ref["best_val_macro_domain_f1"],
                "2x2_best_epoch": result["best"]["epoch"], "4x1_best_epoch": ref["best_epoch"],
                "2x2_final_epoch": history[-1]["val_macro_domain_f1"], "4x1_final_epoch": ref["final_val_macro_domain_f1"]})
            row["final_epoch_delta_2x2_minus_4x1"] = row["2x2_final_epoch"] - row["4x1_final_epoch"]
            two_ds = result["best"]["per_dataset_macro_f1"]
            four_ds = ref["best_val_per_dataset_macro_f1"]
        if set(two_ds) != {"CWRU", "JNU", "HIT", "MAFAULDA"} or set(two_ds) != set(four_ds):
            raise ValueError("Missing dataset")
        row["delta_2x2_minus_4x1"] = row["2x2"] - row["4x1"]
        for ds in sorted(two_ds):
            row[f"{ds}_2x2"] = two_ds[ds]
            row[f"{ds}_4x1"] = four_ds[ds]
            data_deltas.setdefault(ds, []).append(two_ds[ds] - four_ds[ds])
        if not all(math.isfinite(v) for v in row.values()):
            raise ValueError("Non-finite result")
        rows.append(row)
    summary = {"phase": phase, "cells": 9, "endpoint": "validation MacroDomainF1",
        "delta_definition": "2x2 minus 4x1; positive favors 2x2",
        "paired_delta": describe([r["delta_2x2_minus_4x1"] for r in rows]),
        "per_dataset_paired_deltas": {d: describe(v) for d, v in data_deltas.items()},
        "inference": "Descriptive only. Shared data/teachers; known prior TEST results; validation-selected checkpoints.",
        "latency_measured": False, "plan_sha256": sha(HERE / "plan.json"),
        "reference_sha256": sha(HERE / "reference_validation.json")}
    if phase == "train":
        summary["final_epoch_paired_delta"] = describe([r["final_epoch_delta_2x2_minus_4x1"] for r in rows])
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-repo", type=Path, required=True)
    ap.add_argument("--phase", choices=("screen", "train"), required=True)
    args = ap.parse_args()
    root = args.canonical_repo.resolve(strict=True) / "results/pcste_v2/architecture_compare_v1"
    rows, summary = collect(root, args.phase)
    out = root / f"summary_{args.phase}"
    out.mkdir(exist_ok=False)
    with (out / "paired_validation.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
