"""Global PC-STE v2 fold manifests and the MaFaulDa severity-stratified protocol.

A global v2 manifest for fold k = the CWRU v2 P_B windows (protocol_cwru) + the SEALED methodology_v2 windows of JNU, HIT
and MaFaulDa for the same fold k (read by reference, hash-verified, never modified). The MaFaulDa rows can alternatively
come from the v2 severity-stratified protocol (Stage C factorial arm). Every manifest is hashed; runs verify the hash.
"""
from __future__ import annotations

import hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd

from src.methodology_v2.registry import REPO_ROOT
from src.methodology_v2.part3b_windows import PART3B_DIR, verify_part3b_hashes
from src.methodology_v2.part2_builder import verify_frozen_hashes
from src.methodology_v2.experiment.heads import LABEL_FIELD
from .protocol_cwru import verify_protocol as verify_cwru_protocol

GLOBAL_COLUMNS = ["protocol_version", "fold_id", "dataset", "split", "window_id", "group_id", "recording_id", "source_file", "mat_variable",
                  "original_label", "class", "fault_type", "fault_severity", "bearing_factor_set", "channel", "native_sampling_rate_hz",
                  "start_sample", "end_sample", "rpm", "load", "session", "speed_group", "temporal_block_id", "physical_specimen"]
MAF_V2_VERSION = "mafaulda_v2.stratified.v1"


def sealed_windows(fold: int, datasets: tuple[str, ...]) -> pd.DataFrame:
    verify_frozen_hashes(); verify_part3b_hashes()
    w = pd.read_csv(PART3B_DIR / f"window_manifest_fold_{fold}.csv")
    return w[w.dataset.isin(datasets)].copy()


def _harmonise(df: pd.DataFrame, protocol_version: str) -> pd.DataFrame:
    out = pd.DataFrame({c: df[c] if c in df.columns else "" for c in GLOBAL_COLUMNS})
    out["protocol_version"] = protocol_version
    out["class"] = [str(r[LABEL_FIELD[r["dataset"]]]) for _, r in df.iterrows()]
    out["physical_specimen"] = df["physical_specimen"] if "physical_specimen" in df.columns else df["group_id"]
    out["bearing_factor_set"] = df["bearing_factor_set"] if "bearing_factor_set" in df.columns else np.where(df.dataset == "MAFAULDA", "MAFAULDA", "UNKNOWN")
    out["mat_variable"] = df["mat_variable"] if "mat_variable" in df.columns else ""
    return out


# ------------------------------------------------------------------------------------------ MaFaulDa v2 (Stage C)
MAF_V2_FOLDS = {   # held-out severity levels per class family; everything else is TRAIN (see docstring in write_mafaulda_v2)
    1: {"bearing": {"validation": "added_mass_6g", "test": "added_mass_20g"}, "hm": {"validation": "1.5mm", "test": "1.0mm"},
        "vm": {"validation": "1.27mm", "test": "1.40mm"}, "imb": {"validation": "15g", "test": "25g"}},
    2: {"bearing": {"validation": "added_mass_20g", "test": "added_mass_6g"}, "hm": {"validation": "1.0mm", "test": "1.5mm"},
        "vm": {"validation": "1.40mm", "test": "1.27mm"}, "imb": {"validation": "25g", "test": "15g"}},
    3: {"bearing": {"validation": "added_mass_0g", "test": "added_mass_35g"}, "hm": {"validation": "0.5mm", "test": "2.0mm"},
        "vm": {"validation": "0.51mm", "test": "1.90mm"}, "imb": {"validation": "6g", "test": "35g"}},
}


def mafaulda_v2_split(row: pd.Series, fold: int, sealed_normal_split: str) -> str:
    lab, sev = str(row["original_label"]), str(row["fault_severity"])
    if lab == "normal":
        return sealed_normal_split                          # normal recordings keep their sealed per-recording assignment
    fam = "bearing" if "/" in lab else {"horizontal-misalignment": "hm", "vertical-misalignment": "vm", "imbalance": "imb"}[lab]
    rule = MAF_V2_FOLDS[fold][fam]
    return "validation" if sev == rule["validation"] else ("test" if sev == rule["test"] else "train")


def write_mafaulda_v2(out_dir: Path) -> dict:
    """Severity-stratified MaFaulDa protocol: folds 1-2 hold out INTERIOR severity levels (held-out level inside the class's
    TRAIN range), fold 3 holds out the EXTREMES (explicit extrapolation fold, disclosed). Configuration-disjoint as before;
    windows re-cut from the sealed identity tables with the sealed window geometry (1 s, stride 0.5/1.0)."""
    out_dir.mkdir(parents=True, exist_ok=True); stats = []
    for fold in (1, 2, 3):
        w = sealed_windows(fold, ("MAFAULDA",))
        rec = w.drop_duplicates("recording_id")[["recording_id", "original_label", "fault_severity", "split"]].set_index("recording_id")
        new_split = {rid: mafaulda_v2_split(r, fold, r["split"]) for rid, r in rec.iterrows()}
        rows = []
        for r in w.drop_duplicates("recording_id").itertuples():
            sp = new_split[r.recording_id]; rate = int(r.native_sampling_rate_hz); wlen = rate; stride = rate // 2 if sp == "train" else rate
            n = int(w[w.recording_id == r.recording_id].end_sample.max())          # recording length as cut by the sealed manifest
            for s in range(0, n - wlen + 1, stride):
                d = r._asdict(); d.update(split=sp, start_sample=s, end_sample=s + wlen, stride_seconds=0.5 if sp == "train" else 1.0,
                                          window_id=f"v2f{fold}:MAFAULDA:{r.recording_id}:col3:{sp}:{s}-{s + wlen}")
                d.pop("Index", None); rows.append(d)
        df = pd.DataFrame(rows); df["protocol_version"] = MAF_V2_VERSION
        df.to_csv(out_dir / f"mafaulda_v2_windows_fold_{fold}.csv", index=False)
        for (lab, sp), g in df.groupby(["original_label", "split"]):
            stats.append(dict(fold=fold, label=lab, split=sp, n_recordings=g.recording_id.nunique(), n_windows=len(g), severities="|".join(sorted(set(map(str, g.fault_severity))))))
    pd.DataFrame(stats).to_csv(out_dir / "mafaulda_v2_fold_summary.csv", index=False)
    (out_dir / "mafaulda_v2_folds.json").write_text(json.dumps({"version": MAF_V2_VERSION, "folds": MAF_V2_FOLDS}, indent=1))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out_dir.glob("mafaulda_v2_*"))}
    (out_dir / "mafaulda_v2_hashes.json").write_text(json.dumps(hashes, indent=1)); return hashes


# ------------------------------------------------------------------------------------------ global manifests
def build_global(fold: int, cwru_dir: Path, maf_protocol: str = "sealed", maf_dir: Path | None = None) -> pd.DataFrame:
    cwru_master = verify_cwru_protocol(cwru_dir)
    cw = pd.read_csv(cwru_dir / f"cwru_v2_windows_fold_{fold}.csv")
    others = sealed_windows(fold, ("JNU", "HIT") + (("MAFAULDA",) if maf_protocol == "sealed" else ()))
    parts = [_harmonise(cw, f"cwru:{cw.protocol_version.iloc[0]}"), _harmonise(others, "sealed:methodology_v2.part3b.v1")]
    if maf_protocol != "sealed":
        mf = pd.read_csv(maf_dir / f"mafaulda_v2_windows_fold_{fold}.csv"); parts.append(_harmonise(mf, f"mafaulda:{MAF_V2_VERSION}"))
    df = pd.concat(parts, ignore_index=True)
    assert df.window_id.is_unique
    df.attrs["cwru_master_hash"] = cwru_master
    return df


def write_global(out_dir: Path, cwru_dir: Path, maf_protocol: str = "sealed", maf_dir: Path | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True); info = {}
    for fold in (1, 2, 3):
        df = build_global(fold, cwru_dir, maf_protocol, maf_dir); p = out_dir / f"global_v2_fold_{fold}.csv"
        df.to_csv(p, index=False)
        info[f"fold_{fold}"] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "n_windows": len(df),
                               "counts": {f"{d}/{s}": int(n) for (d, s), n in df.groupby(["dataset", "split"]).size().items()}}
    info["cwru_protocol_master"] = verify_cwru_protocol(cwru_dir); info["mafaulda_protocol"] = maf_protocol
    info["cwru_protocol_dir"] = str(Path(cwru_dir).resolve().relative_to(REPO_ROOT.resolve())); info["cwru_native12"] = bool(json.loads((Path(cwru_dir) / "cwru_v2_hashes.json").read_text()).get("native12", False))
    (out_dir / "global_v2_hashes.json").write_text(json.dumps(info, indent=1, sort_keys=True)); return info


def verify_global(out_dir: Path, fold: int) -> str:
    info = json.loads((out_dir / "global_v2_hashes.json").read_text()); p = out_dir / f"global_v2_fold_{fold}.csv"
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha == info[f"fold_{fold}"]["sha256"], f"global v2 manifest fold {fold} changed (fail closed)"
    return sha


def label_subset_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Full-label TRAIN subset in the shape the sealed SupervisedSampler expects (frac_100 only)."""
    tr = df[df.split == "train"]
    return pd.DataFrame({"dataset": tr.dataset.values, "class": tr["class"].values, "group_id": tr.group_id.values, "window_id": tr.window_id.values, "frac_100": True})
