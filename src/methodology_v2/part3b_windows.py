"""Part-3B window-manifest builder — methodology_v2.

Converts the frozen Part-2 legal signal regions into deterministic
1-second raw-window IDENTITIES (manifest-first; signals are served lazily
by part3b_reader). No signal values are transformed; no resampling, no
filtering, no STFT, no normalization. Fail-closed on any Part-2 seal
violation or frozen-protocol contradiction.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .integrity import sha256_file
from .part2_builder import PART2_DIR
from .registry import OUTPUT_DIR as PART1_DIR, REPO_ROOT
from . import part3b_protocol as P

PART3B_DIR = REPO_ROOT / "methodology_v2" / "part3_windows"

WINDOW_COLUMNS = [
    "methodology_version", "fold_id", "dataset", "split", "window_id",
    "group_id", "recording_id", "source_file", "source_fragment_ids",
    "original_label", "mapped_label", "channel", "native_sampling_rate_hz",
    "window_duration_seconds", "stride_seconds", "start_sample",
    "end_sample", "start_time_seconds", "end_time_seconds", "rpm", "load",
    "fault_type", "fault_severity", "session", "speed_group",
    "temporal_block_id", "crosses_source_fragment_boundary",
    "fragment_boundaries_crossed", "parent_part2_identity", "notes",
]

SEALED_FILES = ["window_manifest_fold_1.csv", "window_manifest_fold_2.csv",
                "window_manifest_fold_3.csv", "jnu_guards_1s.csv",
                "hit_logical_stream_manifest.csv"]


class Part3BProtocolError(AssertionError):
    """Raised loudly on any frozen-protocol violation."""


def _win_id(fold: int, ds: str, rec: str, channel: str, split: str,
            start: int, end: int) -> str:
    return f"f{fold}:{ds}:{rec}:{channel}:{split}:{start}-{end}"


def _starts(n_usable_from: int, n_usable_to: int, w: int,
            stride: int) -> list[int]:
    """All legal start samples for w-length windows fully inside
    [n_usable_from, n_usable_to). Remainders are discarded, never padded."""
    if n_usable_to - n_usable_from < w:
        return []
    return list(range(n_usable_from, n_usable_to - w + 1, stride))


def _base(fold: int, ds: str, r: pd.Series, split: str, rate: int,
          stride_s: float) -> dict:
    return {
        "methodology_version": P.METHODOLOGY_VERSION, "fold_id": fold,
        "dataset": ds, "split": split, "group_id": r["group_id"],
        "recording_id": r["recording_id"], "source_file": r["source_file"],
        "source_fragment_ids": None, "original_label": r["original_label"],
        "mapped_label": r["mapped_label_if_applicable"],
        "channel": P.CHANNELS[ds], "native_sampling_rate_hz": rate,
        "window_duration_seconds": P.WINDOW_S, "stride_seconds": stride_s,
        "rpm": r["rpm"], "load": r["load"], "fault_type": r["fault_type"],
        "fault_severity": r["fault_severity"], "session": r["session"],
        "speed_group": None, "temporal_block_id": None,
        "crosses_source_fragment_boundary": False,
        "fragment_boundaries_crossed": None, "notes": None,
    }


def _rate_of(r: pd.Series, ds: str) -> int:
    rate = int(r["sampling_rate_hz"])
    if rate != P.EXPECTED_NATIVE_RATE[ds]:
        raise Part3BProtocolError(
            f"{ds} {r['recording_id']}: rate {rate} != frozen native "
            f"{P.EXPECTED_NATIVE_RATE[ds]}")
    return rate


def _emit(rows: list, base: dict, fold: int, ds: str, rec: str,
          split: str, rate: int, s: int, e: int) -> None:
    rows.append(base | {
        "window_id": _win_id(fold, ds, rec, P.CHANNELS[ds], split, s, e),
        "start_sample": s, "end_sample": e,
        "start_time_seconds": s / rate, "end_time_seconds": e / rate,
    })


# ---------------------------------------------------------------------------
# CWRU / MaFaulDa — plain per-recording windowing
# ---------------------------------------------------------------------------

def build_plain_windows(fold: int, ds: str, part2: pd.DataFrame,
                        n_samples: pd.Series) -> list[dict]:
    sub = part2[(part2["dataset"] == ds)
                & (part2["is_usable"] == True)]  # noqa: E712
    if ds == "CWRU":
        bad = sub[(sub["original_label"] == "Normal")
                  | sub["original_label"].str.contains("028")
                  | ~sub["source_file"].str.startswith("data/raw_cwru_48k/")]
        if len(bad):
            raise Part3BProtocolError(f"CWRU frozen subset violated: {bad}")
    rows: list[dict] = []
    for _, r in sub.sort_values("recording_id").iterrows():
        rate = _rate_of(r, ds)
        w, n = int(round(P.WINDOW_S * rate)), int(n_samples[r["recording_id"]])
        stride_s = P.STRIDE_S[r["split"]]
        base = _base(fold, ds, r, r["split"], rate, stride_s)
        base["parent_part2_identity"] = f"{ds}:{r['recording_id']}"
        for s in _starts(0, n, w, int(round(stride_s * rate))):
            _emit(rows, base, fold, ds, r["recording_id"], r["split"],
                  rate, s, s + w)
    return rows


# ---------------------------------------------------------------------------
# JNU — guard instantiation + within-block windowing
# ---------------------------------------------------------------------------

def jnu_guard_table(part2: pd.DataFrame) -> pd.DataFrame:
    """Instantiate the frozen symbolic guards at G = 1.0 s (fold-invariant;
    derived from Part-2 anchors without modifying them)."""
    jn = part2[(part2["dataset"] == "JNU")
               & (part2["split"] == "guard")].sort_values(
                   ["recording_id", "temporal_start_sample"])
    rows = []
    for _, r in jn.iterrows():
        rate = _rate_of(r, "JNU")
        g = int(round(P.JNU_GUARD_S * rate))
        half = -(-g // 2)  # ceil
        b = int(r["temporal_start_sample"])
        blk = r["temporal_block_id"]  # e.g. GUARD_AB
        rows.append({
            "recording_id": r["recording_id"], "anchor_sample": b,
            "native_sampling_rate_hz": rate,
            "guard_start_sample": max(0, b - half),
            "guard_end_sample": b + half,
            "guard_duration_samples": (b + half) - max(0, b - half),
            "guard_duration_seconds":
                ((b + half) - max(0, b - half)) / rate,
            "affected_blocks": f"{blk[-2]},{blk[-1]}",
            "part2_anchor_id": blk,
        })
    return pd.DataFrame(rows)


def build_jnu_windows(fold: int, part2: pd.DataFrame,
                      guards: pd.DataFrame) -> list[dict]:
    jn = part2[(part2["dataset"] == "JNU")
               & (part2["is_usable"] == True)]  # noqa: E712
    g_by_rec = {rec: grp.sort_values("anchor_sample")
                for rec, grp in guards.groupby("recording_id")}
    rows: list[dict] = []
    for _, r in jn.sort_values(["recording_id",
                                "temporal_start_sample"]).iterrows():
        rate = _rate_of(r, "JNU")
        w = int(round(P.WINDOW_S * rate))
        b0, b1 = int(r["temporal_start_sample"]), int(r["temporal_end_sample"])
        # shrink the block by every instantiated guard that overlaps it
        lo, hi = b0, b1
        for _, g in g_by_rec[r["recording_id"]].iterrows():
            gs, ge = g["guard_start_sample"], g["guard_end_sample"]
            if gs < hi and ge > lo:            # guard overlaps this block
                if g["anchor_sample"] == b0:
                    lo = max(lo, ge)
                elif g["anchor_sample"] == b1:
                    hi = min(hi, gs)
                else:
                    raise Part3BProtocolError(
                        f"guard at {g['anchor_sample']} inside block "
                        f"{r['recording_id']}:{r['temporal_block_id']}")
        stride_s = P.STRIDE_S[r["split"]]
        base = _base(fold, "JNU", r, r["split"], rate, stride_s)
        base["temporal_block_id"] = r["temporal_block_id"]
        base["parent_part2_identity"] = (
            f"JNU:{r['recording_id']}:{r['temporal_block_id']}")
        base["notes"] = ("within-recording temporal holdout; usable "
                        f"[{lo},{hi}) after 1.0 s guard instantiation")
        rec_blk = f"{r['recording_id']}:{r['temporal_block_id']}"
        for s in _starts(lo, hi, w, int(round(stride_s * rate))):
            _emit(rows, base, fold, "JNU", rec_blk, r["split"], rate,
                  s, s + w)
    return rows


# ---------------------------------------------------------------------------
# HIT — logical streams + windowing with fragment provenance
# ---------------------------------------------------------------------------

def hit_stream_table(part2: pd.DataFrame) -> pd.DataFrame:
    """One row per audited session x speed-group logical stream
    (fold-invariant identity), with ordered fragment ids and offsets."""
    hit = part2[part2["dataset"] == "HIT"].sort_values("recording_id")
    rows = []
    for _, r in hit.iterrows():
        rate = _rate_of(r, "HIT")
        k = int(r["recording_id"].rsplit("rec", 1)[1])
        f0 = P.HIT_FRAGMENTS_PER_STREAM * k
        frags = list(range(f0, f0 + P.HIT_FRAGMENTS_PER_STREAM))
        rows.append({
            "logical_group_id": r["recording_id"],
            "session": r["session"],
            "speed_group": r["experiment_id"],
            "label": r["original_label"], "channel": P.CHANNELS["HIT"],
            "source_file": r["source_file"],
            "ordered_fragment_ids": ",".join(map(str, frags)),
            "fragment_length_samples": P.HIT_FRAGMENT_SAMPLES,
            "n_fragments": len(frags),
            "cumulative_offsets": ",".join(
                str(i * P.HIT_FRAGMENT_SAMPLES)
                for i in range(len(frags))),
            "total_stream_samples":
                len(frags) * P.HIT_FRAGMENT_SAMPLES,
            "native_sampling_rate_hz": rate,
        })
    return pd.DataFrame(rows)


def build_hit_windows(fold: int, part2: pd.DataFrame,
                      streams: pd.DataFrame) -> list[dict]:
    hit = part2[part2["dataset"] == "HIT"]
    st = streams.set_index("logical_group_id")
    frag = P.HIT_FRAGMENT_SAMPLES
    rows: list[dict] = []
    for _, r in hit.sort_values("recording_id").iterrows():
        rate = _rate_of(r, "HIT")
        w = int(round(P.WINDOW_S * rate))
        s_row = st.loc[r["recording_id"]]
        total = int(s_row["total_stream_samples"])
        frag_ids = [int(x) for x in
                    s_row["ordered_fragment_ids"].split(",")]
        stride_s = P.STRIDE_S[r["split"]]
        base = _base(fold, "HIT", r, r["split"], rate, stride_s)
        base["speed_group"] = r["experiment_id"]
        base["parent_part2_identity"] = f"HIT:{r['recording_id']}"
        base["notes"] = ("start/end are LOGICAL-STREAM samples; stream = "
                        "ordered concatenation of audited fragments")
        for s in _starts(0, total, w, int(round(stride_s * rate))):
            e = s + w
            lf, le = s // frag, (e - 1) // frag
            crossed = [str((i + 1) * frag) for i in range(lf, le)]
            rows.append(base | {
                "window_id": _win_id(fold, "HIT", r["recording_id"],
                                     P.CHANNELS["HIT"], r["split"], s, e),
                "start_sample": s, "end_sample": e,
                "start_time_seconds": s / rate,
                "end_time_seconds": e / rate,
                "source_fragment_ids":
                    ",".join(str(frag_ids[i]) for i in range(lf, le + 1)),
                "crosses_source_fragment_boundary": le > lf,
                "fragment_boundaries_crossed": ",".join(crossed) or None,
            })
    return rows


# ---------------------------------------------------------------------------
# assembly, statistics, sealing
# ---------------------------------------------------------------------------

def build_all(out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir) if out_dir else PART3B_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    part1 = pd.read_csv(PART1_DIR / "recording_manifest.csv")
    n_samples = part1.set_index("recording_id")["n_samples"]
    folds = {k: pd.read_csv(PART2_DIR / f"global_fold_{k}.csv")
             for k in (1, 2, 3)}

    guards = jnu_guard_table(folds[1])
    guards.to_csv(out_dir / "jnu_guards_1s.csv", index=False)
    streams = hit_stream_table(folds[1])
    streams.to_csv(out_dir / "hit_logical_stream_manifest.csv", index=False)

    manifests: dict[int, pd.DataFrame] = {}
    for fold, p2 in folds.items():
        rows = (build_plain_windows(fold, "CWRU", p2, n_samples)
                + build_jnu_windows(fold, p2, guards)
                + build_hit_windows(fold, p2, streams)
                + build_plain_windows(fold, "MAFAULDA", p2, n_samples))
        df = pd.DataFrame(rows, columns=WINDOW_COLUMNS)
        if df["window_id"].duplicated().any():
            raise Part3BProtocolError(f"fold {fold}: duplicate window ids")
        # eval windows must tile disjointly within each source stream
        ev = df[df["split"].isin(["validation", "test"])]
        for (rec, blk), grp in ev.groupby(["recording_id",
                                           "temporal_block_id"],
                                          dropna=False):
            g = grp.sort_values("start_sample")
            if (g["start_sample"].values[1:]
                    < g["end_sample"].values[:-1]).any():
                raise Part3BProtocolError(
                    f"fold {fold}: overlapping eval windows in {rec}")
        df.to_csv(out_dir / f"window_manifest_fold_{fold}.csv", index=False)
        manifests[fold] = df

    stats = compute_statistics(manifests, folds)
    with open(out_dir / "window_statistics.json", "w") as f:
        json.dump(stats, f, indent=1, sort_keys=True)

    hash_rows = []
    for name in SEALED_FILES:
        p = out_dir / name
        hash_rows.append({"file": name, "sha256": sha256_file(p),
                          "bytes": p.stat().st_size})
    master_src = "".join(f"{r['file']}:{r['sha256']}\n"
                         for r in sorted(hash_rows, key=lambda r: r["file"]))
    master = hashlib.sha256(master_src.encode()).hexdigest()
    hash_rows.append({"file": "PART3B_MASTER_HASH", "sha256": master,
                      "bytes": None})
    pd.DataFrame(hash_rows).to_csv(out_dir / "window_hashes.csv",
                                   index=False)
    return {"out_dir": out_dir, "manifests": manifests, "stats": stats,
            "master_hash": master}


def compute_statistics(manifests: dict[int, pd.DataFrame],
                       folds: dict[int, pd.DataFrame]) -> dict:
    est = pd.read_csv(REPO_ROOT / "methodology_v2" / "part3_input_design"
                      / "window_count_estimates.csv")
    est = est[(est["counting_basis"] == "standard")
              & (est["window_s"] == P.WINDOW_S)]
    stats: dict = {"counts": {}, "estimate_comparison": [],
                   "hit_boundary": {}, "coverage": {}}
    for fold, df in manifests.items():
        fs: dict = {}
        for ds in ("CWRU", "JNU", "HIT", "MAFAULDA"):
            sub = df[df["dataset"] == ds]
            fs[ds] = {}
            for sp in ("train", "validation", "test"):
                part = sub[sub["split"] == sp]
                fs[ds][sp] = {
                    "windows": int(len(part)),
                    "by_class": {str(k): int(v) for k, v in
                                 part.groupby("original_label").size()
                                 .items()},
                }
                ov = 50 if sp == "train" else 0
                e = est[(est["fold_id"] == fold) & (est["dataset"] == ds)
                        & (est["split"] == sp)
                        & (est["overlap_pct"] == ov)]
                expected = int(e["n_windows"].iloc[0])
                stats["estimate_comparison"].append({
                    "fold": fold, "dataset": ds, "split": sp,
                    "estimated": expected, "actual": int(len(part)),
                    "difference": int(len(part)) - expected,
                })
        stats["counts"][str(fold)] = fs

        hit = df[df["dataset"] == "HIT"]
        nb = hit["fragment_boundaries_crossed"].fillna("").map(
            lambda s: len(s.split(",")) if s else 0)
        stats["hit_boundary"][str(fold)] = {
            "total_windows": int(len(hit)),
            "crossing_ge1_pct": round(100.0 * float(
                hit["crosses_source_fragment_boundary"].mean()), 2),
            "crossing_2_boundaries": int((nb == 2).sum()),
            "crossing_1_boundary": int((nb == 1).sum()),
        }

        cw = df[df["dataset"] == "CWRU"]
        stats["coverage"].setdefault(str(fold), {})["cwru_load"] = {
            sp: {str(k): int(v) for k, v in
                 cw[cw["split"] == sp].groupby("load").size().items()}
            for sp in ("train", "validation", "test")}
        l0 = cw[cw["load"] == "0hp"]
        stats["coverage"][str(fold)]["cwru_load0_recordings_with_windows"] \
            = int(l0["recording_id"].nunique())
        jn = df[df["dataset"] == "JNU"]
        stats["coverage"][str(fold)]["jnu_class_speed"] = {
            sp: int(jn[jn["split"] == sp]
                    .groupby(["original_label", "rpm"]).ngroups)
            for sp in ("train", "validation", "test")}
        mf = df[df["dataset"] == "MAFAULDA"]
        stats["coverage"][str(fold)]["mafaulda_configs"] = {
            sp: int(mf[mf["split"] == sp]["group_id"].nunique())
            for sp in ("train", "validation", "test")}
    return stats


def verify_part3b_hashes(out_dir: Path | None = None) -> None:
    """FAIL CLOSED if any sealed Part-3B manifest changed."""
    out_dir = Path(out_dir) if out_dir else PART3B_DIR
    rec = pd.read_csv(out_dir / "window_hashes.csv")
    stored = {r["file"]: r["sha256"] for _, r in rec.iterrows()}
    entries = []
    for name, expect in stored.items():
        if name == "PART3B_MASTER_HASH":
            continue
        got = sha256_file(out_dir / name)
        if got != expect:
            raise Part3BProtocolError(
                f"FROZEN PART-3B MANIFEST CHANGED: {name} (fail closed)")
        entries.append((name, got))
    master_src = "".join(f"{n}:{h}\n" for n, h in sorted(entries))
    if hashlib.sha256(master_src.encode()).hexdigest() \
            != stored["PART3B_MASTER_HASH"]:
        raise Part3BProtocolError("Part-3B master hash mismatch")
