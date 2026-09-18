#!/usr/bin/env python
"""Sealed-TEST evaluation for the final matched S0/S1 stage. Forward-only inference with a frozen selected checkpoint.

Hard barrier: refuses unless FINAL_SELECTED_CHECKPOINTS.csv exists with exactly 18 supervised rows, is committed, and its digest
matches the recorded evaluation-plan digest. No optimiser step, no normaliser fitting, no calibration, no threshold search,
no test-time augmentation, no ensembling. Checkpoint bytes are verified before and after inference.
"""
import argparse, hashlib, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np, pandas as pd, torch

REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/pcste_v2"))
from src.pcste_v2 import native12_freeze as N12, representation as REPR, protocol_cwru_native12 as P
from src.pcste_v2.model import PCSTEv2, PCSTEv2Config, collate_v2
from src.pcste_v2.representation import ItemBuilder
from src.pcste_v2.metrics_v2 import evaluate_split
from src.methodology_v2.experiment.heads import CLASS_ORDERS, DatasetHeads
from src.pcste_v2.final_s0_s1 import canonical_state_hash
import run as R

BARRIER = REPO / "analysis/pcste_v2_final_s0_s1_v1/FINAL_SELECTED_CHECKPOINTS.csv"
PLAN_DIGEST = REPO / "analysis/pcste_v2_final_s0_s1_v1/FINAL_EVALUATION_PLAN_DIGEST.json"


def check_barrier(run_id: str) -> pd.Series:
    if not BARRIER.exists(): raise PermissionError("TEST barrier: FINAL_SELECTED_CHECKPOINTS.csv does not exist")
    df = pd.read_csv(BARRIER)
    if len(df) != 18: raise PermissionError(f"TEST barrier: FINAL_SELECTED_CHECKPOINTS.csv has {len(df)} rows, expected exactly 18")
    if not PLAN_DIGEST.exists(): raise PermissionError("TEST barrier: FINAL_EVALUATION_PLAN_DIGEST.json missing")
    dg = json.loads(PLAN_DIGEST.read_text())
    if hashlib.sha256(BARRIER.read_bytes()).hexdigest() != dg["final_selected_checkpoints_sha256"]:
        raise PermissionError("TEST barrier: the selected-checkpoint table changed after its digest was frozen")
    row = df[df.run_id == run_id]
    if len(row) != 1: raise PermissionError(f"TEST barrier: {run_id} is not one of the 18 frozen supervised rows")
    return row.iloc[0]


def main(args) -> None:
    row = check_barrier(args.run_id)
    protocol, fold = row["protocol"], int(row["global_fold"]); pinfo = R.protocol_info(protocol)
    ck_path = REPO / row["checkpoint_path"]
    pre = hashlib.sha256(ck_path.read_bytes()).hexdigest()
    if pre != row["checkpoint_sha256"]: raise PermissionError(f"TEST: checkpoint bytes differ from the frozen record for {args.run_id}")
    out = REPO / "results/pcste_v2/final_v1_test" / args.run_id; out.mkdir(parents=True, exist_ok=True)
    if (out / "STATUS").exists() and (out / "STATUS").read_text().strip() == "COMPLETE":
        print(f"{args.run_id}: TEST already COMPLETE"); return
    (out / "STATUS").write_text("RUNNING")
    if pinfo.get("cwru_native12"):
        REPR.NATIVE12_ALLOWLIST = P.load_allowlist(REPO / pinfo["cwru_protocol_dir"])
    pdir = REPO / "pcste_v2/protocol" / protocol; man = pd.read_csv(pdir / f"global_v2_fold_{fold}.csv", dtype={"fault_severity": str})
    te = man[man.split == "test"]; assert len(te) > 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    model = PCSTEv2(PCSTEv2Config()).to(dev); model.load_state_dict(ck["model"]); heads = DatasetHeads().to(dev); heads.load_state_dict(ck["heads"])
    model.eval(); heads.eval()
    ib = ItemBuilder(pdir, fold, ("v1",), {"MAFAULDA": 1}, False); mi = man.set_index("window_id"); rows, embs = [], []
    with torch.no_grad():
        for ds in ("CWRU", "JNU", "HIT", "MAFAULDA"):
            wids = list(te.window_id[te.dataset == ds])
            for lo in range(0, len(wids), 64):
                chunk = wids[lo:lo + 64]
                b = {k: v.to(dev) for k, v in collate_v2([ib.build(pd.Series(mi.loc[w].to_dict() | {"window_id": w})) for w in chunk], 1, 1, False).items()}
                z = model(**b)["global_embedding"]; lg = heads(z, ds); pr = torch.softmax(lg, -1).cpu().numpy(); lg = lg.cpu().numpy(); zc = z.cpu().numpy()
                for i, w in enumerate(chunk):
                    r = mi.loc[w]
                    rows.append({"run_id": args.run_id, "strategy": row["strategy"], "global_fold": fold, "seed": int(row["seed"]), "dataset": ds, "window_id": w,
                                 "y_true": str(r["class"]), "y_pred": CLASS_ORDERS[ds][int(lg[i].argmax())], "p_max": float(pr[i].max()),
                                 "physical_specimen": r["physical_specimen"], "group_id": r["group_id"], "recording_id": r["recording_id"], "fault_severity": r["fault_severity"],
                                 "rpm": r["rpm"], "load": r["load"], "native_sampling_rate_hz": r["native_sampling_rate_hz"],
                                 **{f"prob__{c}": float(pr[i][j]) for j, c in enumerate(CLASS_ORDERS[ds])}, **{f"logit__{c}": float(lg[i][j]) for j, c in enumerate(CLASS_ORDERS[ds])}})
                    embs.append(zc[i])
    df = pd.DataFrame(rows)
    assert set(df.window_id) == set(te.window_id) and df.window_id.is_unique, "TEST membership must be exact, with no omission or duplicate"
    post = hashlib.sha256(ck_path.read_bytes()).hexdigest(); assert post == pre, "checkpoint bytes changed during evaluation"
    rep = evaluate_split(df); rep["run_id"] = args.run_id; rep["strategy"] = row["strategy"]; rep["global_fold"] = fold; rep["seed"] = int(row["seed"])
    rep["partition"] = "test"; rep["checkpoint_sha256"] = pre; rep["selected_epoch"] = int(row["selected_epoch"]); rep["protocol"] = protocol
    df.to_csv(out / "test_predictions.csv", index=False)
    np.savez_compressed(out / "test_embeddings.npz", window_id=df.window_id.values, embedding=np.stack(embs).astype(np.float32), y_true=df.y_true.values, y_pred=df.y_pred.values, dataset=df.dataset.values)
    (out / "test_report.json").write_text(json.dumps(rep, indent=1, default=float))
    (out / "PROVENANCE.json").write_text(json.dumps({"run_id": args.run_id, "checkpoint_path": row["checkpoint_path"], "checkpoint_sha256_before": pre, "checkpoint_sha256_after": post,
                                                     "protocol": protocol, "global_fold": fold, "manifest_sha256": json.loads((pdir / "global_v2_hashes.json").read_text())[f"fold_{fold}"]["sha256"],
                                                     "barrier_sha256": hashlib.sha256(BARRIER.read_bytes()).hexdigest(), "n_test_windows": len(df),
                                                     "mode": "model.eval(), torch.no_grad(); no optimiser step, no normaliser fitting, no calibration, no TTA, no ensembling",
                                                     "seconds": round(time.time() - t0, 1), "at": datetime.now(timezone.utc).isoformat()}, indent=1))
    (out / "STATUS").write_text("COMPLETE")
    print(f"{args.run_id}: TEST COMPLETE  {len(df)} windows  " + " ".join(f"{d}={rep['per_dataset_reports'][d]['macro_f1']:.4f}" for d in ("CWRU", "JNU", "HIT", "MAFAULDA")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run-id", required=True); main(ap.parse_args())
