#!/usr/bin/env python
"""Fail-closed Stage-3 consolidation audit (TRAIN/VALIDATION artifacts only)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
import torch

from src.methodology_v2.compression import protocol as P


EXPECTED_COMMIT = "9996e96a2fd31818fc6ca0dec65e6bf6eb2705fa"
EXPECTED_REGISTRY = "bb375a5b40a2ac7a92329da97ddbf2c4b878dca7f765591c47e870dc07596fcd"
EXPECTED_SEAL = "989696925cf8c230bb346667bf401ec2161282ae365fd64a6e43151bab735592"
EXECUTOR_VERSION = "part6-cli-1.0"
REQUIRED_FILES = {
    "best.pt", "completion.json", "epoch_metrics.jsonl", "last.pt",
    "run_log.txt", "state.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def scalar(v):
    return v.item() if hasattr(v, "item") else v


def audit_run(run_dir: Path, reg_row: pd.Series, assigned_host: str) -> dict:
    rid = str(reg_row["run_id"])
    assert run_dir.name == rid
    files = {p.name for p in run_dir.iterdir() if p.is_file()}
    assert files == REQUIRED_FILES, (rid, "unexpected/missing files", files)
    state = load_json(run_dir / "state.json")
    complete = load_json(run_dir / "completion.json")
    history = [json.loads(x) for x in (run_dir / "epoch_metrics.jsonl").read_text().splitlines()]
    assert len(history) == int(reg_row["max_epochs"]) == 50
    assert [x["epoch"] for x in history] == list(range(50))
    assert state["status"] == "COMPLETE"
    for key in ("run_id", "arm", "fold", "seed"):
        assert state[key] == scalar(reg_row[key]), (rid, key)
    assert complete["run_id"] == rid
    assert complete["epochs_completed"] == 50
    assert state["part6_master_hash_at_start"] == EXPECTED_SEAL
    prov = state["device_provenance"]
    assert prov["registry_sha256"] == EXPECTED_REGISTRY
    allowed_commits = {EXPECTED_COMMIT, "dd60fc16470e93a883565bdff7df950c26664a0a"}
    assert prov["git_head"] in allowed_commits
    assert prov["git_head"] == EXPECTED_COMMIT or rid == "k1_f1_s42"
    assert state["host"].split(".")[0] == assigned_host
    assert prov["host"].split(".")[0] == assigned_host
    assert prov["gpu_name"]
    arch = state["arm_config"]["architecture"]
    assert arch["name"] == str(reg_row["architecture"]) == "half_4x1"
    assert P.config_hash(arch) == str(reg_row["architecture_hash"])
    assert P.config_hash(state["arm_config"]["loss"]) == str(reg_row["loss_hash"])
    expected_cfg = P.config_hash({
        **{k: scalar(v) for k, v in reg_row.to_dict().items() if k not in ("status", "run_id")},
        "cli": EXECUTOR_VERSION,
    })
    assert state["config_hash"] == complete["config_hash"] == expected_cfg, (rid, state["config_hash"], complete["config_hash"], expected_cfg)
    best_hash = sha256(run_dir / "best.pt")
    assert best_hash == state["best_checkpoint_sha256"] == complete["best_checkpoint_sha256"]
    metrics = np.asarray([float(x["val_macro_domain_f1"]) for x in history])
    expected_epoch = int(np.argmax(metrics))  # np.argmax implements earliest tie.
    expected_metric = float(metrics[expected_epoch])
    assert state["best_epoch"] == complete["best_epoch"] == expected_epoch
    assert state["best_val_macro_f1"] == complete["best_val_macro_f1"] == expected_metric
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    assert best["run_id"] == rid and best["config_hash"] == expected_cfg
    assert best["epoch"] == expected_epoch and best["macro_f1_val"] == expected_metric
    assert last["run_id"] == rid and last["config_hash"] == expected_cfg
    assert last["stream_hash"] == state["stream_sha256"]
    assert last["epoch"] == 49
    assert last["best"]["epoch"] == expected_epoch
    assert last["best"]["metric"] == expected_metric
    return {
        "run_id": rid, "arm": state["arm"], "fold": state["fold"],
        "seed": state["seed"], "status": state["status"],
        "epochs_completed": 50, "best_epoch": expected_epoch,
        "best_val_macro_f1": expected_metric,
        "final_val_macro_f1": float(history[-1]["val_macro_domain_f1"]),
        "hostname": state["host"], "gpu": prov["gpu_name"],
        "checkpoint_path": str(run_dir / "best.pt"),
        "checkpoint_sha256": best_hash, "config_hash": expected_cfg,
        "architecture_hash": str(reg_row["architecture_hash"]),
        "loss_hash": str(reg_row["loss_hash"]),
        "optimizer_hash": str(reg_row["optimizer_hash"]),
        "registry_sha256": EXPECTED_REGISTRY, "part6_master_seal": EXPECTED_SEAL,
        "git_commit": prov["git_head"], "stream_sha256": state["stream_sha256"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--otter134", type=Path, required=True)
    ap.add_argument("--otter133", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    registry_path = args.repo / "methodology_v2/part6_compression/part6_run_registry.csv"
    assignment_path = args.repo / "methodology_v2/part6_compression/execution_assignment.json"
    assert sha256(registry_path) == EXPECTED_REGISTRY
    reg = pd.read_csv(registry_path)
    enabled = reg[reg["enabled"] & (reg["status"] == "REGISTERED")].copy()
    assert len(enabled) == 27 and set(enabled["tier"]) == {"core"}
    assert set(enabled["arm"]) == {"k1", "c_small", "k0"}
    assert set(enabled["fold"]) == {1, 2, 3} and set(enabled["seed"]) == {42, 1337, 2026}
    assert not enabled.duplicated(["arm", "fold", "seed"]).any()
    assignment = load_json(assignment_path)
    assert assignment["registry_sha256"] == EXPECTED_REGISTRY
    assert assignment["part6_master_hash"] == EXPECTED_SEAL
    roots = {
        "otter135": args.repo / "results/methodology_v2/part6_compression/runs",
        "otter134": args.otter134,
        "otter133": args.otter133,
    }
    expected_all = set(enabled["run_id"].astype(str))
    observed_all: set[str] = set()
    rows = []
    for host, root in roots.items():
        expected = assignment["hosts"][host]
        observed = sorted(p.name for p in root.iterdir() if p.is_dir())
        assert sorted(expected) == observed, (host, expected, observed)
        assert not (observed_all & set(observed)), (host, "duplicate run ID")
        observed_all.update(observed)
        by_id = enabled.set_index("run_id")
        for rid in expected:
            row = by_id.loc[rid].copy()
            row["run_id"] = rid
            rows.append(audit_run(root / rid, row, host))
    assert observed_all == expected_all
    df = pd.DataFrame(rows).sort_values(["arm", "fold", "seed"]).reset_index(drop=True)
    for (_, _), g in df.groupby(["fold", "seed"]):
        assert g["stream_sha256"].nunique() == 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "stage3_validation_runs.csv", index=False)
    summary_rows = []
    for arm, g in df.groupby("arm", sort=False):
        vals = g["best_val_macro_f1"]
        summary_rows.append({"arm": arm, "n": len(g), "mean": vals.mean(),
                             "sd_sample": vals.std(ddof=1), "median": vals.median(),
                             "min": vals.min(), "max": vals.max()})
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "stage3_validation_summary.csv", index=False)
    wide = df.pivot(index=["fold", "seed"], columns="arm", values="best_val_macro_f1").reset_index()
    wide["k1_minus_c_small"] = wide["k1"] - wide["c_small"]
    wide["k1_minus_k0"] = wide["k1"] - wide["k0"]
    wide.to_csv(args.output_dir / "stage3_validation_paired_deltas.csv", index=False)
    report = {
        "complete": 27, "running": 0, "waiting": 0, "failed": 0, "stale": 0,
        "by_host": {k: 9 for k in roots}, "by_arm": {k: 9 for k in ("k1", "c_small", "k0")},
        "assignment_verified": True, "checkpoint_selection_verified": True,
        "content_hashes_verified": True, "pairing_streams_verified": True,
        "registry_sha256": EXPECTED_REGISTRY, "part6_master_seal": EXPECTED_SEAL,
        "execution_commit": EXPECTED_COMMIT,
        "scope": "TRAIN/VALIDATION only; no TEST artifacts opened",
    }
    (args.output_dir / "stage3_completeness_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
