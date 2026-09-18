"""CWRU v2 specimen-rich protocol P_B (analysis/pcste_v2_diagnostics/CWRU_EXPANDED_PROTOCOL_PROPOSAL.yaml).

Specimen pool (20 physical specimens): drive-end bearing (SKF 6205) faults IR/B/OR at 7/14/21 mil recorded at 48 kHz AND
re-recorded at 12 kHz (same specimens), the 12 kHz-only 28-mil DE specimens IR028 and B028, and the fan-end bearing
(SKF 6203) faults IR/B/OR at 7/14/21 mil recorded at 12 kHz (downloaded from the CWRU site, audited in
analysis/pcste_v2_execution/CWRU_FE12K_DOWNLOAD_AUDIT.csv). Channel = accelerometer on the faulted bearing housing
(DE_time for DE specimens, FE_time for FE specimens). Outer-race clock positions are merged into the specimen.

Folds are specimen-disjoint at the physical-specimen level (all loads, positions and sampling-rate re-recordings of a
specimen move together). Every TRAIN class holds 3 specimens; VALIDATION and TEST hold DE and FE specimens of every class;
12 kHz and 48 kHz recordings occur in every class of every split (rate is not a class shortcut).
"""
from __future__ import annotations

import hashlib, json, re
from pathlib import Path

import numpy as np
import pandas as pd

from src.methodology_v2.registry import REPO_ROOT

PROTOCOL_VERSION = "cwru_v2.P_B.v1"
WINDOW_S = 1.0
STRIDE_S = {"train": 0.5, "validation": 1.0, "test": 1.0}
CLASS_OF = {"IR": "inner_race", "B": "ball", "OR": "outer_race"}
FOLDS = {   # specimen ids: DE_<label>, FE_<label>  (label = IR007 etc.)
    1: {"test": ["DE_IR021", "FE_IR007", "DE_B007", "FE_B021", "DE_OR014", "FE_OR007"],
        "validation": ["DE_IR014", "FE_IR014", "DE_B021", "FE_B014", "DE_OR007"]},
    2: {"test": ["DE_IR014", "FE_IR021", "DE_B021", "FE_B007", "DE_OR007", "FE_OR014"],
        "validation": ["DE_IR007", "FE_IR007", "DE_B014", "FE_B021", "DE_OR021"]},
    3: {"test": ["DE_IR007", "FE_IR014", "DE_B014", "FE_B014", "DE_OR021", "FE_OR021"],
        "validation": ["DE_IR021", "FE_IR021", "DE_B007", "FE_B007", "DE_OR014"]},
}
NOMINAL_RPM_BY_LOAD = {"0": 1797.0, "1": 1772.0, "2": 1750.0, "3": 1730.0}   # CWRU site table (files without an RPM variable)
FNAME = re.compile(r"(IR|B|OR)(\d{3})(@\d+)?_(\d)_(DE|FE)(12|48)k")


def _mat_meta(path: Path) -> dict:
    import scipy.io as sio
    m = sio.loadmat(path); ks = [k for k in m if not k.startswith("__")]
    tk = [k for k in ks if k.endswith("_time")]
    rpm = [float(np.ravel(m[k])[0]) for k in ks if k.endswith("RPM")]
    return {"time_vars": tk, "rpm": rpm[0] if rpm else np.nan, "n": {k: int(m[k].shape[0]) for k in tk}}


def _sealed_48k_ids() -> dict[str, str]:
    """source_file -> internal id as fixed by the sealed Part-1/2 audit (resolves the two known multi-ID 48 kHz files)."""
    g = pd.read_csv(REPO_ROOT / "methodology_v2/part2_splits/global_fold_1.csv")
    g = g[g.dataset == "CWRU"]
    return {sf: rid.replace("cwru_", "") for sf, rid in zip(g.source_file, g.recording_id)}


def build_recording_table() -> pd.DataFrame:
    rows = []
    sealed = _sealed_48k_ids()
    sources = [("data/raw_cwru_48k", "DE"), ("data/raw", "DE"), ("data/raw_cwru_12k_fe", "FE")]
    for root, bearing in sources:
        for p in sorted((REPO_ROOT / root).rglob("*.mat")):
            mm = FNAME.match(p.stem)
            if not mm:
                continue                                    # normal baselines are not part of the 3-class task
            loc, size, pos, load, chan, rate = mm.groups(); rate = int(rate) * 1000
            meta = _mat_meta(p)
            var = [k for k in meta["time_vars"] if k.endswith(f"_{chan}_time")]
            assert var, f"{p}: no {chan}_time channel"
            rel = str(p.relative_to(REPO_ROOT))
            if rel in sealed:                                   # keep the dissertation identity for the 48 kHz files
                var = f"{sealed[rel]}_{chan}_time"; assert var in meta["time_vars"], (rel, var)
            else:
                ids = sorted({v.split("_")[0] for v in var}); assert len(ids) == 1, (rel, ids)
                var = f"{ids[0]}_{chan}_time"
            pid = var.split("_")[0]
            spec = f"{bearing}_{loc}{size}"
            rpm = meta["rpm"]; rpm_source = "file_RPM_variable"
            if not np.isfinite(rpm):
                rpm, rpm_source = NOMINAL_RPM_BY_LOAD[load], "nominal_by_load(site_table)"
            rows.append(dict(rpm_source=rpm_source, recording_id=f"cwru_{rate // 1000}k_{chan}_{pid}", source_file=str(p.relative_to(REPO_ROOT)), mat_variable=var,
                             internal_id=pid, physical_specimen=spec, group_id=f"cwruv2_{spec}", bearing=bearing,
                             bearing_factor_set="CWRU_DE_6205" if bearing == "DE" else "CWRU_FE_6203", fault_type=CLASS_OF[loc],
                             fault_severity=f"{int(size)} mil", or_clock_position=(pos or "").lstrip("@"), load=f"{load}hp",
                             rpm=rpm, native_sampling_rate_hz=rate, channel=chan, n_samples=meta["n"][var], duration_s=round(meta["n"][var] / rate, 3)))
    df = pd.DataFrame(rows)
    assert df.recording_id.is_unique, "recording ids must be unique"
    return df.sort_values(["physical_specimen", "native_sampling_rate_hz", "or_clock_position", "load"]).reset_index(drop=True)


def assign_split(spec: str, fold: int) -> str:
    if spec in FOLDS[fold]["test"]: return "test"
    if spec in FOLDS[fold]["validation"]: return "validation"
    return "train"


def build_windows(rec: pd.DataFrame, fold: int) -> pd.DataFrame:
    rows = []
    for r in rec.itertuples():
        split = assign_split(r.physical_specimen, fold); rate = int(r.native_sampling_rate_hz)
        w = int(round(WINDOW_S * rate)); stride = int(round(STRIDE_S[split] * rate))
        for s in range(0, r.n_samples - w + 1, stride):
            rows.append(dict(protocol_version=PROTOCOL_VERSION, fold_id=fold, dataset="CWRU", split=split,
                             window_id=f"v2f{fold}:CWRU:{r.recording_id}:{r.channel}:{split}:{s}-{s + w}", group_id=r.group_id,
                             physical_specimen=r.physical_specimen, recording_id=r.recording_id, source_file=r.source_file, mat_variable=r.mat_variable,
                             original_label=r.physical_specimen.split("_")[1] + (f"@{r.or_clock_position}" if r.or_clock_position else ""),
                             fault_type=r.fault_type, fault_severity=r.fault_severity, bearing=r.bearing, bearing_factor_set=r.bearing_factor_set,
                             channel=r.channel, native_sampling_rate_hz=rate, window_duration_seconds=WINDOW_S, stride_seconds=STRIDE_S[split],
                             start_sample=s, end_sample=s + w, rpm=r.rpm, load=r.load, or_clock_position=r.or_clock_position))
    return pd.DataFrame(rows)


def validate(rec: pd.DataFrame, win: dict[int, pd.DataFrame]) -> list[dict]:
    """Fail-closed protocol checks; returns the fold summary rows."""
    out = []
    for fold, w in win.items():
        for split in ("train", "validation", "test"):
            sp = w[w.split == split]
            for cls in ("inner_race", "ball", "outer_race"):
                c = sp[sp.fault_type == cls]
                specs = sorted(c.physical_specimen.unique()); rates = sorted(c.native_sampling_rate_hz.unique())
                out.append(dict(fold=fold, split=split, fault_type=cls, n_specimens=len(specs), specimens="|".join(specs), n_recordings=c.recording_id.nunique(),
                                n_windows=len(c), rates_hz="|".join(map(str, rates)), sizes="|".join(sorted(c.fault_severity.unique())), bearings="|".join(sorted(c.bearing.unique())),
                                loads="|".join(sorted(c.load.unique()))))
                if split == "train": assert len(specs) >= 3, (fold, cls, specs)
                assert len(rates) == 2, f"rate shortcut check failed: fold {fold} {split} {cls} rates {rates}"
                assert {"DE", "FE"} <= set(c.bearing.unique()) or (split == "validation" and cls == "outer_race"), (fold, split, cls)
        s_tr, s_va, s_te = (set(w[w.split == k].physical_specimen) for k in ("train", "validation", "test"))
        assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), f"fold {fold}: specimen overlap"
        assert s_tr | s_va | s_te == set(rec.physical_specimen)
    tests = [set(FOLDS[f]["test"]) for f in FOLDS]
    assert not (tests[0] & tests[1]) and not (tests[0] & tests[2]) and not (tests[1] & tests[2]), "test specimens must be disjoint across folds"
    return out


def write_protocol(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = build_recording_table(); rec.to_csv(out_dir / "cwru_v2_recordings.csv", index=False)
    win = {f: build_windows(rec, f) for f in FOLDS}
    summary = validate(rec, win)
    for f, w in win.items():
        w.to_csv(out_dir / f"cwru_v2_windows_fold_{f}.csv", index=False)
    pd.DataFrame(summary).to_csv(out_dir / "cwru_v2_fold_summary.csv", index=False)
    (out_dir / "cwru_v2_folds.json").write_text(json.dumps({"protocol_version": PROTOCOL_VERSION, "folds": FOLDS, "window_s": WINDOW_S, "stride_s": STRIDE_S}, indent=1))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out_dir.glob("cwru_v2_*"))}
    master = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(hashes.items())).encode()).hexdigest()
    (out_dir / "cwru_v2_hashes.json").write_text(json.dumps({"files": hashes, "master": master}, indent=1))
    return {"recordings": len(rec), "windows": {f: len(w) for f, w in win.items()}, "master_hash": master}


def verify_protocol(out_dir: Path) -> str:
    h = json.loads((out_dir / "cwru_v2_hashes.json").read_text())
    for name, sha in h["files"].items():
        assert hashlib.sha256((out_dir / name).read_bytes()).hexdigest() == sha, f"CWRU v2 protocol file {name} changed (fail closed)"
    return h["master"]
