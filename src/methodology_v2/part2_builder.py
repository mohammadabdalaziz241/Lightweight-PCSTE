"""Part-2 split builder — methodology_v2.

Builds the three frozen global folds (identity/region assignment ONLY) from
the Part-1 recording manifest, under the frozen constants of
part2_protocol.py. Strictly audit-level: no signals are read, no windows,
no preprocessing, no training artefacts. Deterministic: rerunning
reproduces byte-identical manifests.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .integrity import sha256_file
from .registry import OUTPUT_DIR as PART1_DIR, REPO_ROOT
from . import part2_protocol as P

PART2_DIR = REPO_ROOT / "methodology_v2" / "part2_splits"

MANIFEST_COLUMNS = [
    "methodology_version", "fold_id", "dataset", "split",
    "recording_id", "group_id", "original_label",
    "mapped_label_if_applicable", "grouping_type", "source_file",
    "sampling_rate_hz", "rpm", "load", "fault_type", "fault_severity",
    "bearing_position", "session", "experiment_id",
    "temporal_block_id", "temporal_start_sample", "temporal_end_sample",
    "guard_before", "guard_after", "is_usable", "notes",
]

MAFAULDA_CLASSES = (
    "normal", "imbalance", "horizontal-misalignment",
    "vertical-misalignment",
    "underhang/ball_fault", "underhang/cage_fault", "underhang/outer_race",
    "overhang/ball_fault", "overhang/cage_fault", "overhang/outer_race",
)


class Part2ProtocolError(AssertionError):
    """Raised loudly on any violation of the frozen Part-2 protocol."""


def _load_part1_manifest() -> pd.DataFrame:
    return pd.read_csv(PART1_DIR / "recording_manifest.csv")


def _base_row(fold: int, dataset: str, split: str) -> dict:
    return {c: None for c in MANIFEST_COLUMNS} | {
        "methodology_version": P.METHODOLOGY_VERSION,
        "fold_id": fold, "dataset": dataset, "split": split,
        "is_usable": True,
    }


# ---------------------------------------------------------------------------
# CWRU
# ---------------------------------------------------------------------------

def build_cwru_rows(m: pd.DataFrame, fold: int) -> list[dict]:
    sub = m[(m["dataset"] == "CWRU")
            & (m["original_file"].str.startswith("data/raw_cwru_48k/"))]
    if len(sub) != 52:
        raise Part2ProtocolError(
            f"expected 52 retained CWRU 48k fault recordings, got {len(sub)}")
    rot = P.CWRU_ROTATION[fold]
    spec_to_split = {s: sp for sp, specs in rot.items() for s in specs}
    rows = []
    for _, r in sub.sort_values("recording_id").iterrows():
        label = r["original_label"]
        if label == "Normal" or "028" in label:
            raise Part2ProtocolError(f"excluded CWRU label {label} present")
        spec = label.split("@")[0]
        row = _base_row(fold, "CWRU", spec_to_split[spec])
        row.update({
            "recording_id": r["recording_id"],
            "group_id": f"cwru48k_{spec}",
            "original_label": label,
            "mapped_label_if_applicable": r["mapped_label_candidate"],
            "grouping_type": "physical_specimen_across_loads",
            "source_file": r["original_file"],
            "sampling_rate_hz": int(r["sampling_rate_hz"]),
            "rpm": r["rpm"], "load": r["load"],
            "fault_type": r["fault_type"],
            "fault_severity": r["fault_severity"],
            "bearing_position": r["bearing_position"],
            "experiment_id": r["experiment_id"],
            "notes": "OR clock positions merged into specimen (approved "
                     "conservative rule, CWRU_GROUPING_RECHECK.md)",
        })
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# JNU
# ---------------------------------------------------------------------------

def build_jnu_rows(m: pd.DataFrame, fold: int) -> list[dict]:
    sub = m[m["dataset"] == "JNU"]
    if len(sub) != 12:
        raise Part2ProtocolError(f"expected 12 JNU recordings, got {len(sub)}")
    rot = P.JNU_ROTATION[fold]
    block_to_split = {b: sp for sp, blocks in rot.items() for b in blocks}
    rows = []
    for _, r in sub.sort_values("recording_id").iterrows():
        n = int(r["n_samples"])
        bounds = [i * n // 5 for i in range(6)]
        common = {
            "recording_id": r["recording_id"],
            "group_id": r["recording_id"],
            "original_label": r["original_label"],
            "mapped_label_if_applicable": r["mapped_label_candidate"],
            "source_file": r["original_file"],
            "sampling_rate_hz": int(r["sampling_rate_hz"]),
            "rpm": r["rpm"], "load": r["load"],
            "fault_type": r["fault_type"],
            "fault_severity": r["fault_severity"],
            "experiment_id": r["experiment_id"],
        }
        for i, blk in enumerate(P.JNU_BLOCKS):
            row = _base_row(fold, "JNU", block_to_split[blk])
            row.update(common)
            row.update({
                "grouping_type": "within_recording_temporal_macro_block",
                "temporal_block_id": blk,
                "temporal_start_sample": bounds[i],
                "temporal_end_sample": bounds[i + 1],
                "guard_before": i > 0,
                "guard_after": i < 4,
                "notes": (f"{P.JNU_EVALUATION_LABEL}; nominal slot "
                          f"boundaries pre-guard; edges flagged guard_* "
                          f"shrink by ceil(G/2) at Part-3 instantiation, "
                          f"G >= effective window span"),
            })
            rows.append(row)
        for j in range(1, 5):
            row = _base_row(fold, "JNU", "guard")
            row.update(common)
            row.update({
                "grouping_type": "guard_region",
                "temporal_block_id":
                    f"GUARD_{P.JNU_BLOCKS[j-1]}{P.JNU_BLOCKS[j]}",
                "temporal_start_sample": bounds[j],
                "temporal_end_sample": bounds[j],
                "is_usable": False,
                "notes": ("symbolic guard anchored at nominal boundary "
                          f"{bounds[j]}; expands to [b-ceil(G/2), "
                          "b+ceil(G/2)) once Part 3 freezes "
                          "G >= effective window span; never usable"),
            })
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# HIT
# ---------------------------------------------------------------------------

def _lp_tertile(lp: float) -> int:
    return min(2, int((lp - 1000.0) / ((5000.0 - 1000.0) / 3)))


def _hit_acceptance(assign: dict[str, str], sub: pd.DataFrame) -> list[str]:
    """Return list of violated frozen criteria (empty == accepted)."""
    v = []
    by = {sp: sub[sub["recording_id"].map(assign) == sp] for sp in P.SPLITS}
    for sp, part in by.items():
        for cls in ("0", "1", "2"):
            ng = (part["original_label"].astype(str) == cls).sum()
            if ng < P.HIT_MIN_GROUPS_PER_CLASS:
                v.append(f"H2: class {cls} has {ng} groups in {sp}")
        for sess in sorted(sub["session_id"].unique()):
            if (part["session_id"] == sess).sum() < 1:
                v.append(f"H3: session {sess} absent from {sp}")
        terts = {_lp_tertile(x) for x in part["rpm"]}
        need = (P.HIT_TRAIN_MIN_LP_TERTILES if sp == "train"
                else P.HIT_VALTEST_MIN_LP_TERTILES)
        if len(terts) < need:
            v.append(f"H4: {sp} covers LP tertiles {sorted(terts)} < {need}")
    return v


def build_hit_rows(m: pd.DataFrame, fold: int,
                   rejections: list[dict]) -> tuple[list[dict], int]:
    sub = (m[m["dataset"] == "HIT"]
           .sort_values("recording_id").reset_index(drop=True))
    if len(sub) != 134:
        raise Part2ProtocolError(f"expected 134 HIT groups, got {len(sub)}")

    seed_used = None
    assign: dict[str, str] = {}
    for attempt in range(P.MAX_SEED_ATTEMPTS):
        seed = P.HIT_SEEDS[fold] + attempt * P.SEED_REPLACEMENT_INCREMENT
        rng = np.random.default_rng(seed)
        cand: dict[str, str] = {}
        for sess in sorted(sub["session_id"].unique()):
            ids = sub.loc[sub["session_id"] == sess, "recording_id"].tolist()
            n = len(ids)
            n_val = max(1, round(P.HIT_TARGET["validation"] * n))
            n_test = max(1, round(P.HIT_TARGET["test"] * n))
            perm = rng.permutation(n)
            for k, idx in enumerate(perm):
                cand[ids[idx]] = ("validation" if k < n_val else
                                  "test" if k < n_val + n_test else "train")
        violations = _hit_acceptance(cand, sub)
        if not violations:
            assign, seed_used = cand, seed
            break
        rejections.append({"dataset": "HIT", "fold_id": fold, "seed": seed,
                           "reasons": violations,
                           "replacement_seed":
                               seed + P.SEED_REPLACEMENT_INCREMENT})
    if seed_used is None:
        raise Part2ProtocolError(
            f"HIT fold {fold}: no seed passed structural criteria in "
            f"{P.MAX_SEED_ATTEMPTS} attempts")

    rows = []
    for _, r in sub.iterrows():
        row = _base_row(fold, "HIT", assign[r["recording_id"]])
        row.update({
            "recording_id": r["recording_id"],
            "group_id": r["recording_id"],
            "original_label": str(r["original_label"]),
            "mapped_label_if_applicable": r["mapped_label_candidate"],
            "grouping_type": "session_x_speed_group_recording",
            "source_file": r["original_file"],
            "sampling_rate_hz": int(r["sampling_rate_hz"]),
            "rpm": r["rpm"], "load": r["load"],
            "fault_type": r["fault_type"],
            "fault_severity": r["fault_severity"],
            "bearing_position": r["bearing_position"],
            "session": r["session_id"],
            "experiment_id": r["experiment_id"],
            "notes": (f"full Drive release only; split seed {seed_used}; "
                      "rpm column = LP speed, HP in experiment_id; all "
                      "channels/series of this group inherit this split"),
        })
        rows.append(row)
    return rows, seed_used


# ---------------------------------------------------------------------------
# MaFaulDa
# ---------------------------------------------------------------------------

def _rpm_tertile(rpm: float, lo: float, hi: float) -> int:
    return min(2, int((rpm - lo) / ((hi - lo) / 3)))


def _mafaulda_acceptance(unit_split: dict[str, str],
                         sub: pd.DataFrame) -> list[str]:
    v = []
    splits = sub["unit_id"].map(unit_split)
    for sp in P.SPLITS:
        part = sub[splits == sp]
        for cls in MAFAULDA_CLASSES:
            if (part["original_label"] == cls).sum() < 1:
                v.append(f"M3: class {cls} absent from {sp}")
    normal = sub[sub["original_label"] == "normal"]
    lo, hi = normal["rpm"].min(), normal["rpm"].max()
    for sp in ("validation", "test"):
        part = normal[normal["unit_id"].map(unit_split) == sp]
        terts = {_rpm_tertile(x, lo, hi) for x in part["rpm"]}
        if len(terts) < P.MAFAULDA_VALTEST_MIN_NORMAL_RPM_TERTILES:
            v.append(f"M4: normal {sp} spans RPM tertiles {sorted(terts)}")
    return v


def build_mafaulda_rows(m: pd.DataFrame, fold: int,
                        rejections: list[dict]) -> tuple[list[dict], int]:
    sub = (m[m["dataset"] == "MAFAULDA"]
           .sort_values("recording_id").reset_index(drop=True)).copy()
    if len(sub) != 1951:
        raise Part2ProtocolError(
            f"expected 1951 MaFaulDa recordings, got {len(sub)}")
    # atomic unit: fault configuration; recording for the single-config
    # Normal class (explicit weaker unit, documented limitation)
    sub["unit_id"] = np.where(sub["original_label"] == "normal",
                              sub["recording_id"],
                              sub["group_id_candidate"])
    n_fault_cfg = sub.loc[sub["original_label"] != "normal",
                          "unit_id"].nunique()
    if n_fault_cfg != 41:
        raise Part2ProtocolError(
            f"expected 41 fault configurations, got {n_fault_cfg}")

    seed_used = None
    unit_split: dict[str, str] = {}
    for attempt in range(P.MAX_SEED_ATTEMPTS):
        seed = P.MAFAULDA_SEEDS[fold] + attempt * P.SEED_REPLACEMENT_INCREMENT
        rng = np.random.default_rng(seed)
        cand: dict[str, str] = {}
        for cls in MAFAULDA_CLASSES:  # fixed order
            units = sorted(sub.loc[sub["original_label"] == cls,
                                   "unit_id"].unique())
            c = len(units)
            n_val = max(1, round(P.MAFAULDA_TARGET["validation"] * c))
            n_test = max(1, round(P.MAFAULDA_TARGET["test"] * c))
            perm = rng.permutation(c)
            for k, idx in enumerate(perm):
                cand[units[idx]] = ("validation" if k < n_val else
                                    "test" if k < n_val + n_test else "train")
        violations = _mafaulda_acceptance(cand, sub)
        if not violations:
            unit_split, seed_used = cand, seed
            break
        rejections.append({"dataset": "MAFAULDA", "fold_id": fold,
                           "seed": seed, "reasons": violations,
                           "replacement_seed":
                               seed + P.SEED_REPLACEMENT_INCREMENT})
    if seed_used is None:
        raise Part2ProtocolError(
            f"MaFaulDa fold {fold}: no seed passed structural criteria in "
            f"{P.MAX_SEED_ATTEMPTS} attempts")

    rows = []
    for _, r in sub.iterrows():
        is_normal = r["original_label"] == "normal"
        row = _base_row(fold, "MAFAULDA", unit_split[r["unit_id"]])
        row.update({
            "recording_id": r["recording_id"],
            "group_id": r["unit_id"],
            "original_label": r["original_label"],
            "mapped_label_if_applicable": r["mapped_label_candidate"],
            "grouping_type": ("recording_normal_exception" if is_normal
                              else "fault_configuration"),
            "source_file": r["original_file"],
            "sampling_rate_hz": int(r["sampling_rate_hz"]),
            "rpm": r["rpm"], "load": r["load"],
            "fault_type": r["fault_type"],
            "fault_severity": r["fault_severity"],
            "bearing_position": r["bearing_position"],
            "experiment_id": r["experiment_id"],
            "notes": (f"split seed {seed_used}; operational folder taxonomy "
                      "kept verbatim (naming caveat M2 not remapped)"
                      + ("; Normal grouped per recording — weaker unit, "
                         "seen-setup limitation" if is_normal else "")),
        })
        rows.append(row)
    return rows, seed_used


# ---------------------------------------------------------------------------
# assembly, statistics, sealing
# ---------------------------------------------------------------------------

def build_fold(m: pd.DataFrame, fold: int,
               rejections: list[dict]) -> tuple[pd.DataFrame, dict]:
    cwru = build_cwru_rows(m, fold)
    jnu = build_jnu_rows(m, fold)
    hit, hit_seed = build_hit_rows(m, fold, rejections)
    maf, maf_seed = build_mafaulda_rows(m, fold, rejections)
    df = pd.DataFrame(cwru + jnu + hit + maf, columns=MANIFEST_COLUMNS)
    seeds = {"HIT": hit_seed, "MAFAULDA": maf_seed}

    # leakage assertions at group level (guards excluded)
    usable = df[df["split"].isin(P.SPLITS)]
    for ds in usable["dataset"].unique():
        part = usable[usable["dataset"] == ds]
        key = ("recording_id" if ds != "JNU" else None)
        if ds == "JNU":
            # a JNU group is (recording, block); recordings legitimately
            # span splits via disjoint temporal regions
            span = part.groupby(["recording_id",
                                 "temporal_block_id"])["split"].nunique()
        else:
            span = part.groupby("group_id")["split"].nunique()
        if (span > 1).any():
            raise Part2ProtocolError(f"{ds}: group crosses partitions")
    return df, seeds


def compute_fold_stats(df: pd.DataFrame, part1: pd.DataFrame) -> dict:
    p1 = part1.set_index("recording_id")["duration_seconds"]
    out: dict = {}
    usable = df[df["is_usable"] == True]  # noqa: E712
    for ds in ("CWRU", "JNU", "HIT", "MAFAULDA"):
        sub = usable[usable["dataset"] == ds]
        out[ds] = {}
        for sp in P.SPLITS:
            part = sub[sub["split"] == sp]
            if ds == "JNU":
                dur = ((part["temporal_end_sample"]
                        - part["temporal_start_sample"])
                       / part["sampling_rate_hz"]).sum()
                n_units = len(part)  # (recording, block) regions
            else:
                dur = p1.reindex(part["recording_id"]).sum()
                n_units = part["group_id"].nunique()
            per_class = {}
            for cls, cpart in part.groupby("original_label"):
                if ds == "JNU":
                    cdur = ((cpart["temporal_end_sample"]
                             - cpart["temporal_start_sample"])
                            / cpart["sampling_rate_hz"]).sum()
                else:
                    cdur = p1.reindex(cpart["recording_id"]).sum()
                per_class[str(cls)] = {
                    "groups": int(cpart["group_id"].nunique()
                                  if ds != "JNU" else len(cpart)),
                    "recordings": int(cpart["recording_id"].nunique()),
                    "duration_s": round(float(cdur), 2),
                }
            out[ds][sp] = {
                "groups": int(n_units),
                "recordings": int(part["recording_id"].nunique()),
                "duration_s": round(float(dur), 2),
                "per_class": per_class,
            }
    return out


def write_outputs(out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir) if out_dir else PART2_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    m = _load_part1_manifest()

    rejections: list[dict] = []
    seeds_used: dict = {}
    stats: dict = {}
    fold_paths, test_paths = [], []
    for fold in P.FOLD_IDS:
        df, seeds = build_fold(m, fold, rejections)
        seeds_used[str(fold)] = seeds
        stats[str(fold)] = compute_fold_stats(df, m)
        fp = out_dir / f"global_fold_{fold}.csv"
        df.to_csv(fp, index=False)
        fold_paths.append(fp)
        tp = out_dir / f"test_identity_fold_{fold}.csv"
        df[df["split"] == "test"].to_csv(tp, index=False)
        test_paths.append(tp)

    protocol = {
        "methodology_version": P.METHODOLOGY_VERSION,
        "fold_ids": list(P.FOLD_IDS),
        "cwru": {"specimens": list(P.CWRU_SPECIMENS),
                 "rotation": {str(k): {sp: list(v) for sp, v in d.items()}
                              for k, d in P.CWRU_ROTATION.items()},
                 "grouping": "physical specimen across all loads, OR clock "
                             "positions merged; 48 kHz DE family only; "
                             "Healthy/0.028\"/12 kHz excluded"},
        "jnu": {"blocks": list(P.JNU_BLOCKS),
                "rotation": {str(k): {sp: list(v) for sp, v in d.items()}
                             for k, d in P.JNU_ROTATION.items()},
                "evaluation_label": P.JNU_EVALUATION_LABEL,
                "guard_rule": P.JNU_GUARD_RULE},
        "hit": {"predeclared_seeds": {str(k): v
                                      for k, v in P.HIT_SEEDS.items()},
                "target": P.HIT_TARGET,
                "allocation_rule": P.HIT_ALLOCATION_RULE,
                "acceptance_criteria": ["H1 group-disjoint (construction)",
                                        "H2 >=2 groups/class/partition",
                                        "H3 every session in every partition",
                                        "H4 LP tertiles 3/2/2"],
                "github_release_split": "REJECTED as split authority "
                                        "(Part-1 finding H2)"},
        "mafaulda": {"predeclared_seeds": {str(k): v for k, v in
                                           P.MAFAULDA_SEEDS.items()},
                     "target": P.MAFAULDA_TARGET,
                     "allocation_rule": P.MAFAULDA_ALLOCATION_RULE,
                     "acceptance_criteria": [
                         "M1 config-disjoint (construction)",
                         "M2 normal-recording-disjoint (construction)",
                         "M3 all 10 classes in every partition",
                         "M4 normal val/test span >=2 RPM tertiles"],
                     "classes": list(MAFAULDA_CLASSES)},
        "seed_governance": {
            "replacement_increment": P.SEED_REPLACEMENT_INCREMENT,
            "max_attempts": P.MAX_SEED_ATTEMPTS,
            "rule": "structural criteria only; never model performance"},
        "seeds_used": seeds_used,
        "usage_rules": P.USAGE_RULES,
    }
    proto_path = out_dir / "split_protocol.json"
    with open(proto_path, "w") as f:
        json.dump(protocol, f, indent=1, sort_keys=True)

    with open(out_dir / "rejected_split_seeds.json", "w") as f:
        json.dump({"rejections": rejections}, f, indent=1)

    with open(out_dir / "fold_statistics.json", "w") as f:
        json.dump(stats, f, indent=1, sort_keys=True)

    # ---- sealing ---------------------------------------------------------
    hash_rows = []
    for p in [*fold_paths, *test_paths, proto_path]:
        hash_rows.append({"file": p.name, "sha256": sha256_file(p),
                          "bytes": p.stat().st_size})
    master_src = "".join(f"{r['file']}:{r['sha256']}\n"
                         for r in sorted(hash_rows, key=lambda r: r["file"]))
    import hashlib
    master = hashlib.sha256(master_src.encode()).hexdigest()
    hash_rows.append({"file": "MASTER_PROTOCOL_HASH", "sha256": master,
                      "bytes": None})
    pd.DataFrame(hash_rows).to_csv(out_dir / "split_hashes.csv", index=False)

    return {"out_dir": out_dir, "rejections": rejections,
            "seeds_used": seeds_used, "master_hash": master, "stats": stats}


def verify_frozen_hashes(out_dir: Path | None = None) -> None:
    """FAIL CLOSED if any sealed manifest differs from its frozen hash.

    Future methodology stages must call this before consuming any fold
    manifest.
    """
    out_dir = Path(out_dir) if out_dir else PART2_DIR
    rec = pd.read_csv(out_dir / "split_hashes.csv")
    stored = {r["file"]: r["sha256"] for _, r in rec.iterrows()}
    import hashlib
    entries = []
    for name, expect in stored.items():
        if name == "MASTER_PROTOCOL_HASH":
            continue
        p = out_dir / name
        if not p.exists():
            raise Part2ProtocolError(f"sealed file missing: {name}")
        got = sha256_file(p)
        if got != expect:
            raise Part2ProtocolError(
                f"FROZEN MANIFEST CHANGED: {name}\n expected {expect}\n"
                f" got      {got}\nRefusing to proceed (fail closed).")
        entries.append((name, got))
    master_src = "".join(f"{n}:{h}\n" for n, h in sorted(entries))
    master = hashlib.sha256(master_src.encode()).hexdigest()
    if master != stored["MASTER_PROTOCOL_HASH"]:
        raise Part2ProtocolError("master protocol hash mismatch (fail closed)")
