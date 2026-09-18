"""CWRU strictly native 12 kHz LOAD-BASED diagnostic protocol: pcste_v2_cwru_native12_load_v1 (cwru_native12.load.v1).

Purpose: a split-difficulty diagnostic. Splits are decided by MOTOR LOAD at the RECORDING level, not by physical specimen:
TRAIN = 0 hp + 1 hp, VALIDATION = 2 hp, TEST = 3 hp. The same physical specimen therefore appears in more than one split through
different load recordings. This is intentional and is the scientific point of the diagnostic:

    recording-disjoint = YES        (every recording's windows live in exactly one split)
    load-disjoint      = YES        (a load appears in exactly one split)
    specimen-disjoint  = NO         (by design; this is NOT an unseen-specimen protocol)

Source eligibility, the physical-specimen registry, the faulted-bearing sensor policy, the 1.0 s / 12000-sample window geometry,
the TRAIN 0.5 s / VAL-TEST 1.0 s strides and the CWRU12 STFT grid are inherited unchanged from pcste_v2_cwru_native12_specimen_v1:
the same byte-verified allowlist of the 105 official 12k DE/FE fault acquisitions, and the same 48k denylist. Nothing is discovered
by directory scan. TEST (3 hp) is built structurally so the protocol is complete; no TEST waveform is read by this builder.
"""
from __future__ import annotations

import hashlib, json
from pathlib import Path

import pandas as pd

from src.methodology_v2.registry import REPO_ROOT
from .protocol_cwru_native12 import (build_source_registry, de_fe_tables, sha256_file, NATIVE_RATE_HZ, STFT_CWRU12,
                                     WINDOW_S, STRIDE_S, INVENTORY_SHA256)

PROTOCOL_ID = "pcste_v2_cwru_native12_load_v1"
PROTOCOL_VERSION = "cwru_native12.load.v1"
SPLIT_BY_LOAD = {"0hp": "train", "1hp": "train", "2hp": "validation", "3hp": "test"}
INHERITS = "pcste_v2_cwru_native12_specimen_v1 (allowlist, denylist, specimen registry, sensor policy, window geometry, STFT grid)"


def assign_split(load: str) -> str:
    assert load in SPLIT_BY_LOAD, f"unexpected load {load}"
    return SPLIT_BY_LOAD[load]


def build_windows(rec: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    rows, excl = [], []; w = int(round(WINDOW_S * NATIVE_RATE_HZ))
    for r in rec.itertuples():
        split = assign_split(r.load); stride = int(round(STRIDE_S[split] * NATIVE_RATE_HZ))
        if r.n_samples < w:
            excl.append(dict(recording_id=r.recording_id, reason=f"n_samples {r.n_samples} < window {w}")); continue
        for s in range(0, r.n_samples - w + 1, stride):
            rows.append(dict(protocol_version=PROTOCOL_VERSION, fold_id=1, dataset="CWRU", split=split,
                             window_id=f"n12L:CWRU:{r.recording_id}:{r.sensor_location}:{split}:{s}-{s + w}", group_id=r.group_id,
                             physical_specimen=r.physical_specimen, recording_id=r.recording_id, source_file=r.local_path, mat_variable=r.mat_variable,
                             source_sha256=r.sha256, official_file_number=r.official_file_number, original_label=r.official_label.split("_")[0],
                             fault_type=r.fault_type, fault_severity=r.fault_severity, bearing=r.fault_bearing_end, fault_bearing_end=r.fault_bearing_end,
                             sensor_location=r.sensor_location, bearing_factor_set=r.bearing_factor_set, channel=r.sensor_location,
                             native_sampling_rate_hz=NATIVE_RATE_HZ, window_duration_seconds=WINDOW_S, stride_seconds=STRIDE_S[split],
                             start_sample=s, end_sample=s + w, rpm=r.rpm, load=r.load, or_clock_position=r.or_clock_position))
    return pd.DataFrame(rows), excl


def validate(rec: pd.DataFrame, win: pd.DataFrame, deny: pd.DataFrame) -> list[dict]:
    """Fail-closed checks for the load protocol: native rate, recording/load disjointness, class coverage, denylist separation."""
    out = []; deny_sha = {x for v in deny.local_sha256.fillna("") for x in v.split("|") if x}
    assert (rec.native_sampling_rate_hz == NATIVE_RATE_HZ).all() and (rec.channel_rate_hz == NATIVE_RATE_HZ).all()
    assert (win.native_sampling_rate_hz == NATIVE_RATE_HZ).all() and (win.end_sample - win.start_sample == int(WINDOW_S * NATIVE_RATE_HZ)).all()
    assert not (set(win.source_sha256) & deny_sha) and not (set(rec.sha256) & deny_sha), "denylisted (48k) ancestry present"
    assert win.window_id.is_unique
    for k in ("recording_id", "source_sha256", "source_file"):
        g = win.groupby(k).split.nunique(); assert (g == 1).all(), f"a {k} appears in more than one split (recording-disjointness violated)"
    g = win.groupby("load").split.nunique(); assert (g == 1).all(), "a load appears in more than one split (load-disjointness violated)"
    assert set(win[win.split == "train"].load) == {"0hp", "1hp"} and set(win[win.split == "validation"].load) == {"2hp"} and set(win[win.split == "test"].load) == {"3hp"}
    for split in ("train", "validation", "test"):
        sp = win[win.split == split]
        for cls in ("inner_race", "ball", "outer_race"):
            c = sp[sp.fault_type == cls]; assert len(c) > 0, f"{split}/{cls} empty"
            out.append(dict(split=split, fault_type=cls, n_specimens=c.physical_specimen.nunique(), specimens="|".join(sorted(c.physical_specimen.unique())),
                            n_recordings=c.recording_id.nunique(), n_windows=len(c), loads="|".join(sorted(c.load.unique())),
                            sizes="|".join(sorted(c.fault_severity.unique())), fault_ends="|".join(sorted(c.fault_bearing_end.unique())),
                            sensors="|".join(sorted(c.sensor_location.unique()))))
    # specimen-disjointness is deliberately NOT required; record the actual overlap as evidence
    s_tr, s_va, s_te = (set(win[win.split == k].physical_specimen) for k in ("train", "validation", "test"))
    assert s_tr & s_va, "expected specimen overlap between TRAIN and VALIDATION (same-specimen/different-load design)"
    return out


def disjointness_audit(win: pd.DataFrame) -> dict:
    s = {k: set(win[win.split == k].physical_specimen) for k in ("train", "validation", "test")}
    r = {k: set(win[win.split == k].recording_id) for k in ("train", "validation", "test")}
    l = {k: set(win[win.split == k].load) for k in ("train", "validation", "test")}
    return {"recording_disjoint": bool(not (r["train"] & r["validation"]) and not (r["train"] & r["test"]) and not (r["validation"] & r["test"])),
            "load_disjoint": bool(not (l["train"] & l["validation"]) and not (l["train"] & l["test"]) and not (l["validation"] & l["test"])),
            "specimen_disjoint": bool(not (s["train"] & s["validation"]) and not (s["train"] & s["test"]) and not (s["validation"] & s["test"])),
            "window_id_unique": bool(win.window_id.is_unique),
            "specimens_shared_train_validation": sorted(s["train"] & s["validation"]), "n_specimens_shared_train_validation": len(s["train"] & s["validation"]),
            "loads": {k: sorted(v) for k, v in l.items()}, "n_recordings": {k: len(v) for k, v in r.items()}, "n_specimens": {k: len(v) for k, v in s.items()},
            "intent": "specimen_disjoint=False is the deliberate design of this diagnostic: it asks whether a KNOWN physical fault specimen is recognised under a DIFFERENT operating load."}


def write_protocol(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rec, deny = build_source_registry()                                   # identical allowlist/denylist machinery as the specimen protocol
    rec = rec.assign(split=[assign_split(x) for x in rec.load])
    rec.to_csv(out_dir / "cwru_v2_recordings.csv", index=False); deny.to_csv(out_dir / "cwru_n12_denylist.csv", index=False)
    allow = {r.local_path: {"sha256": r.sha256, "official_filename": r.official_filename, "source_url": r.source_url, "mat_variable": r.mat_variable, "native_sampling_rate_hz": NATIVE_RATE_HZ} for r in rec.itertuples()}
    (out_dir / "cwru_n12_allowlist.json").write_text(json.dumps({"protocol_id": PROTOCOL_ID, "inventory_sha256": INVENTORY_SHA256, "n": len(allow), "files": allow}, indent=1, sort_keys=True))
    win, excl = build_windows(rec)
    summary = validate(rec, win, deny)
    win.to_csv(out_dir / "cwru_v2_windows_fold_1.csv", index=False)       # single split set; fold_id 1 keeps the executor's naming
    pd.DataFrame(summary).to_csv(out_dir / "cwru_load_split_summary.csv", index=False)
    de_fe_tables(rec, {1: win}).to_csv(out_dir / "cwru_n12_de_fe_tables.csv", index=False)
    pd.DataFrame(excl, columns=["recording_id", "reason"]).to_csv(out_dir / "cwru_n12_structural_exclusions.csv", index=False)
    audit = disjointness_audit(win); (out_dir / "cwru_load_disjointness_audit.json").write_text(json.dumps(audit, indent=1))
    rec[["recording_id", "physical_specimen", "fault_type", "fault_severity", "fault_bearing_end", "sensor_location", "bearing_family", "load", "rpm", "rpm_source",
         "native_sampling_rate_hz", "channel_rate_hz", "official_filename", "official_file_number", "source_category", "source_url", "sha256", "local_path", "mat_variable",
         "n_samples", "split"]].assign(n_windows=[int((win.recording_id == r).sum()) for r in rec.recording_id]).to_csv(out_dir / "cwru_load_source_audit.csv", index=False)
    (out_dir / "cwru_v2_folds.json").write_text(json.dumps({"protocol_id": PROTOCOL_ID, "protocol_version": PROTOCOL_VERSION, "inherits": INHERITS,
        "split_authority": "recording/load level", "split_by_load": SPLIT_BY_LOAD, "folds": {"1": {"validation": ["load:2hp"], "test": ["load:3hp"]}},
        "window_s": WINDOW_S, "stride_s": STRIDE_S, "native_rate_hz": NATIVE_RATE_HZ, "stft_cwru12": STFT_CWRU12,
        "sensor_policy": "accelerometer on the faulted bearing housing (DE_time for DE-fault, FE_time for FE-fault); class-independent; unchanged",
        "disjointness": {"recording": True, "load": True, "specimen": False},
        "design_note": "specimen-disjointness is deliberately NOT enforced; this protocol tests same-specimen/different-load recognition and is not comparable as a matched experiment to the specimen-disjoint protocol"}, indent=1))
    hashes = {p.name: sha256_file(p) for p in sorted(out_dir.iterdir()) if p.is_file() and p.name not in ("cwru_v2_hashes.json", "PROTOCOL_STATUS.json", "FREEZE_DIGEST.json", "FREEZE_BUNDLE_INDEX.txt")}
    master = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(hashes.items())).encode()).hexdigest()
    (out_dir / "cwru_v2_hashes.json").write_text(json.dumps({"protocol_id": PROTOCOL_ID, "files": hashes, "master": master, "native12": True}, indent=1))
    return {"recordings": len(rec), "denylisted": len(deny), "windows": {k: int((win.split == k).sum()) for k in ("train", "validation", "test")},
            "structural_exclusions": excl, "master_hash": master, "audit": audit}


def verify_protocol(out_dir: Path) -> str:
    h = json.loads((out_dir / "cwru_v2_hashes.json").read_text()); assert h.get("native12") is True and h.get("protocol_id") == PROTOCOL_ID
    for name, sha in h["files"].items():
        assert sha256_file(out_dir / name) == sha, f"native12 load protocol file {name} changed (fail closed)"
    return h["master"]
