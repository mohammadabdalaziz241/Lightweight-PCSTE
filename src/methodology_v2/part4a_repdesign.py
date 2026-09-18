"""Part 4A — training-only time-frequency representation design study.

Computes, on a deterministic FOLD-1 TRAIN development subset only:
  - the candidate analysis grid (n_fft x hop) with exact physical
    time/frequency resolution per native sampling rate;
  - value-transform comparisons (magnitude, power, log1p, dB);
  - sharpness / information-preservation diagnostics (descriptive only —
    never a single optimised score, never label-driven);
  - the physical frequency-coordinate audit across native rates;
  - the HIT fragment-joint audit;
  - representation memory estimates.

Time-frequency convention (exact, no library defaults):
  frame_t[k] = x[t*hop : t*hop + n_fft] * w,  w = periodic Hann
  w[n] = 0.5 - 0.5*cos(2*pi*n / n_fft),  n = 0..n_fft-1
  X[t, :] = numpy.fft.rfft(frame_t)  (one-sided, n_fft/2 + 1 bins)
  center = False; NO padding; only complete frames are taken:
  n_frames = floor((N - n_fft)/hop) + 1;  f_k = k * fs / n_fft.
(torch.stft / scipy conventions were reviewed; both default to centred,
padded frames — rejected here so every frame contains only real signal.)

Data-access rule: every raw value read here comes from a Fold-1 TRAIN
window served by the frozen Part-3B lazy reader; a fail-closed guard
refuses anything else. No validation/test signal content is ever read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .part3b_windows import PART3B_DIR
from .registry import REPO_ROOT

PART4A_DIR = REPO_ROOT / "methodology_v2" / "part4_stft_design"

N_FFT_GRID = (256, 512, 1024, 2048, 4096)
# 4096 added beyond the required minimum grid with explicit engineering
# reason: at 48-50 kHz even n_fft=2048 gives ~24 Hz bins, coarser than
# the slowest shaft frequencies (JNU 600 rpm -> 10 Hz), so one longer
# candidate is needed to characterise the resolution ceiling.
HOP_RATIOS = (0.25, 0.50)
TRANSFORMS = ("magnitude", "power", "log1p", "db")
DB_EPS = 1e-10        # dB convention: 20*log10(|X| + 1e-10), absolute
                      # reference (no per-window reference -> no hidden
                      # normalization)
DEV_PER_CLASS = 3     # windows per dataset x class in the dev subset
HIT_AUDIT_STEP = 50   # every 50th fold-1 TRAIN HIT window


class Part4ADataAccessError(AssertionError):
    """Raised if any non-Fold-1-TRAIN window would be read."""


# ---------------------------------------------------------------------------
# development-window selection (deterministic, label-blind rules)
# ---------------------------------------------------------------------------

def load_fold1_train() -> pd.DataFrame:
    df = pd.read_csv(PART3B_DIR / "window_manifest_fold_1.csv")
    tr = df[df["split"] == "train"].copy()
    if tr.empty or (tr["fold_id"] != 1).any():
        raise Part4ADataAccessError("fold-1 train manifest malformed")
    return tr


def assert_fold1_train(row: pd.Series) -> None:
    if int(row["fold_id"]) != 1 or row["split"] != "train":
        raise Part4ADataAccessError(
            f"{row['window_id']}: not Fold-1 TRAIN — sealed for Part 4A")


def select_dev_windows(train: pd.DataFrame) -> pd.DataFrame:
    """Per dataset x class: spread over recordings by taking the first,
    middle and last recording (sorted by recording_id), then the
    temporally middle train window of each. Fixed rule, no visual
    cherry-picking, no signal values involved."""
    rows = []
    for (ds, cls), grp in train.groupby(["dataset", "original_label"]):
        recs = sorted(grp["recording_id"].unique())
        picks = sorted({recs[0], recs[len(recs) // 2], recs[-1]})
        for rec in picks[:DEV_PER_CLASS]:
            g = grp[grp["recording_id"] == rec].sort_values("start_sample")
            w = g.iloc[len(g) // 2]  # temporally middle window
            assert_fold1_train(w)
            rows.append({
                "window_id": w["window_id"], "dataset": ds, "class": cls,
                "group_id": w["group_id"], "rpm": w["rpm"],
                "load": w["load"], "severity": w["fault_severity"],
                "recording_id": w["recording_id"],
                "native_sampling_rate_hz": w["native_sampling_rate_hz"],
                "temporal_block_id": w["temporal_block_id"],
                "reason": ("deterministic: recording at sorted position "
                           "{first|middle|last} of its class, temporally "
                           "middle train window"),
            })
    return pd.DataFrame(rows).sort_values("window_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# explicit framed one-sided DFT (convention in module docstring)
# ---------------------------------------------------------------------------

def periodic_hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def tf_map(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Complex one-sided time-frequency map, frames x bins."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n_frames = (x.size - n_fft) // hop + 1
    if n_frames < 1:
        raise AssertionError("window shorter than analysis frame")
    w = periodic_hann(n_fft)
    frames = np.stack([x[t * hop: t * hop + n_fft] * w
                       for t in range(n_frames)])
    return np.fft.rfft(frames, axis=1)


def apply_transform(z: np.ndarray, kind: str) -> np.ndarray:
    mag = np.abs(z)
    if kind == "magnitude":
        return mag
    if kind == "power":
        return mag ** 2
    if kind == "log1p":
        return np.log1p(mag)
    if kind == "db":
        return 20.0 * np.log10(mag + DB_EPS)
    raise AssertionError(f"unknown transform {kind}")


# ---------------------------------------------------------------------------
# candidate grid / resolution tables
# ---------------------------------------------------------------------------

RATES = {"CWRU": 48_000, "JNU": 50_000, "HIT": 25_000, "MAFAULDA": 50_000}


def candidate_grid() -> pd.DataFrame:
    rows = []
    for ds, fs in RATES.items():
        for n_fft in N_FFT_GRID:
            for ratio in HOP_RATIOS:
                hop = int(n_fft * ratio)
                frames = (fs - n_fft) // hop + 1
                rows.append({
                    "dataset": ds, "sampling_rate_hz": fs, "n_fft": n_fft,
                    "window_samples": n_fft,
                    "window_ms": round(1000 * n_fft / fs, 3),
                    "hop_samples": hop,
                    "hop_ms": round(1000 * hop / fs, 3),
                    "overlap_pct": int(round((1 - ratio) * 100)),
                    "freq_bins": n_fft // 2 + 1,
                    "time_frames_1s": frames,
                    "bin_spacing_hz": round(fs / n_fft, 3),
                    "nyquist_hz": fs / 2,
                    "tf_shape_1s": f"({n_fft // 2 + 1}, {frames})",
                    "convention": "periodic-Hann, center=False, no "
                                  "padding, one-sided numpy.fft.rfft",
                })
    return pd.DataFrame(rows)


def frequency_coordinate_table() -> pd.DataFrame:
    """Same row index -> different Hz across native rates (n_fft=1024)."""
    n_fft = 1024
    rows = []
    for k in (5, 10, 21, 43, 85, 171, 256, 341, 512):
        row = {"bin_index": k, "n_fft": n_fft}
        for ds, fs in RATES.items():
            row[f"{ds}_hz"] = round(k * fs / n_fft, 1)
            row[f"{ds}_f_over_nyquist"] = round(k / (n_fft // 2), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def memory_table() -> pd.DataFrame:
    rows = []
    for ds, fs in RATES.items():
        for n_fft in N_FFT_GRID:
            for ratio in HOP_RATIOS:
                hop = int(n_fft * ratio)
                bins = n_fft // 2 + 1
                frames = (fs - n_fft) // hop + 1
                elems = bins * frames
                mb = elems * 4 / 2 ** 20
                rows.append({
                    "dataset": ds, "n_fft": n_fft,
                    "hop_ratio": ratio, "freq_bins": bins,
                    "time_frames": frames, "elements": elems,
                    "float32_mb_per_window": round(mb, 3),
                    "batch16_mb": round(16 * mb, 1),
                    "batch32_mb": round(32 * mb, 1),
                    "batch64_mb": round(64 * mb, 1),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# diagnostics (descriptive only)
# ---------------------------------------------------------------------------

def _frame_peaks(v: np.ndarray) -> tuple[int, float]:
    """Count of resolvable local maxima above median + 3*MAD, and mean
    peak-to-neighbour contrast (value units of the given transform)."""
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-12
    thr = med + 3.0 * mad
    inner = v[1:-1]
    is_peak = (inner > v[:-2]) & (inner > v[2:]) & (inner > thr)
    idx = np.where(is_peak)[0] + 1
    if idx.size == 0:
        return 0, 0.0
    lo = np.clip(idx - 2, 0, v.size - 1)
    hi = np.clip(idx + 2, 0, v.size - 1)
    contrast = float(np.mean(v[idx] - 0.5 * (v[lo] + v[hi])))
    return int(idx.size), contrast


def sharpness_metrics(rep: np.ndarray) -> dict:
    """Descriptive sharpness/information measures on a (frames, bins)
    representation. No single combined score is ever formed."""
    p = rep - rep.min() + 1e-12
    p = p / p.sum(axis=1, keepdims=True)
    entropy = float(np.mean(-np.sum(p * np.log(p), axis=1))
                    / np.log(rep.shape[1]))
    scale = float(np.std(rep)) + 1e-12
    grad_f = float(np.mean(np.abs(np.diff(rep, axis=1)))) / scale
    grad_t = float(np.mean(np.abs(np.diff(rep, axis=0)))) / scale
    peaks, contrast = zip(*(_frame_peaks(rep[t])
                            for t in range(rep.shape[0])))
    l1l2 = float(np.mean(np.abs(rep).sum(axis=1)
                         / (np.linalg.norm(rep, axis=1) + 1e-12))
                 / np.sqrt(rep.shape[1]))
    return {
        "spectral_entropy_norm": round(entropy, 4),
        "grad_energy_freq": round(grad_f, 4),
        "grad_energy_time": round(grad_t, 4),
        "mean_resolvable_peaks_per_frame": round(float(np.mean(peaks)), 2),
        "mean_peak_neighbour_contrast": round(float(np.mean(contrast)), 4),
        "l1_l2_concentration": round(l1l2, 4),
    }


def dynamic_range_metrics(rep: np.ndarray) -> dict:
    q = np.percentile(rep, [0.1, 1, 50, 99, 99.9])
    return {
        "min": float(rep.min()), "p0_1": float(q[0]), "p1": float(q[1]),
        "median": float(q[2]), "p99": float(q[3]), "p99_9": float(q[4]),
        "max": float(rep.max()),
        "dynamic_range_p999_p01": float(q[4] - q[0]),
        "finite": bool(np.isfinite(rep).all()),
    }


# ---------------------------------------------------------------------------
# HIT fragment-joint audit (fold-1 TRAIN only)
# ---------------------------------------------------------------------------

def hit_boundary_audit(train: pd.DataFrame, read_window) -> pd.DataFrame:
    hit = (train[train["dataset"] == "HIT"]
           .sort_values("window_id").iloc[::HIT_AUDIT_STEP])
    n_fft, hop = 1024, 256
    rows = []
    for _, w in hit.iterrows():
        assert_fold1_train(w)
        x = read_window(w)
        z = apply_transform(tf_map(x, n_fft, hop), "log1p")
        frame_energy = z.mean(axis=1)
        mu, sd = frame_energy.mean(), frame_energy.std() + 1e-12
        flat = np.exp(np.mean(np.log(np.abs(z) + 1e-9), axis=1)) \
            / (np.mean(np.abs(z), axis=1) + 1e-12)
        bounds = [int(b) - int(w["start_sample"])
                  for b in str(w["fragment_boundaries_crossed"]).split(",")]
        diffs = np.abs(np.diff(x))
        p999 = np.percentile(diffs, 99.9)
        for b in bounds:
            jump = float(np.abs(x[b] - x[b - 1])) if 0 < b < x.size \
                else float("nan")
            fr = [t for t in range(z.shape[0])
                  if t * hop <= b - 1 and b <= t * hop + n_fft]
            zsc = [float((frame_energy[t] - mu) / sd) for t in fr]
            flz = [float(flat[t]) for t in fr]
            rows.append({
                "window_id": w["window_id"],
                "boundary_sample_in_window": b,
                "abs_jump": jump,
                "window_p999_absdiff": float(p999),
                "jump_ratio": jump / (float(p999) + 1e-12),
                "n_frames_touching_boundary": len(fr),
                "max_frame_energy_zscore": (round(max(zsc), 3)
                                            if zsc else None),
                "mean_frame_energy_zscore": (round(float(np.mean(zsc)), 3)
                                             if zsc else None),
                "mean_boundary_frame_flatness": (round(float(np.mean(flz)),
                                                       4) if flz else None),
                "window_mean_flatness": round(float(np.mean(flat)), 4),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# orchestration of the numeric studies
# ---------------------------------------------------------------------------

def run_numeric_studies(out_dir: Path, read_window) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = load_fold1_train()
    dev = select_dev_windows(train)
    dev.to_csv(out_dir / "part4a_development_windows.csv", index=False)

    candidate_grid().to_csv(out_dir / "stft_candidate_grid.csv",
                            index=False)
    grid = candidate_grid()
    grid[["dataset", "n_fft", "overlap_pct", "window_ms", "hop_ms",
          "bin_spacing_hz", "time_frames_1s",
          "tf_shape_1s"]].to_csv(out_dir / "stft_resolution_table.csv",
                                 index=False)
    frequency_coordinate_table().to_csv(
        out_dir / "frequency_coordinate_study.csv", index=False)
    memory_table().to_csv(out_dir / "stft_memory_estimates.csv",
                          index=False)

    train_by_id = train.set_index("window_id")
    sharp_rows, dyn_rows = [], []
    signals: dict[str, np.ndarray] = {}
    for _, d in dev.iterrows():
        w = train_by_id.loc[d["window_id"]]
        assert_fold1_train(w)
        x = read_window(w)
        signals[d["window_id"]] = x
        fs = int(d["native_sampling_rate_hz"])
        for n_fft in N_FFT_GRID:
            for ratio in HOP_RATIOS:
                z = tf_map(x, n_fft, int(n_fft * ratio))
                for kind in TRANSFORMS:
                    rep = apply_transform(z, kind)
                    key = {"window_id": d["window_id"],
                           "dataset": d["dataset"], "class": d["class"],
                           "rpm": d["rpm"], "sampling_rate_hz": fs,
                           "n_fft": n_fft, "hop_ratio": ratio,
                           "transform": kind}
                    if kind == "log1p":
                        sharp_rows.append(key | sharpness_metrics(rep))
                    dyn_rows.append(key | dynamic_range_metrics(rep))
    pd.DataFrame(sharp_rows).to_csv(out_dir / "stft_sharpness_metrics.csv",
                                    index=False)
    pd.DataFrame(dyn_rows).to_csv(
        out_dir / "stft_dynamic_range_metrics.csv", index=False)

    audit = hit_boundary_audit(train, read_window)
    audit.to_csv(out_dir / "hit_boundary_audit.csv", index=False)

    access_log = {
        "rule": "Fold-1 TRAIN only; guard fail-closed",
        "dev_window_ids": dev["window_id"].tolist(),
        "hit_audit_window_ids":
            sorted(audit["window_id"].unique().tolist()),
    }
    with open(out_dir / "part4a_signal_access_log.json", "w") as f:
        json.dump(access_log, f, indent=1)
    return {"dev": dev, "signals": signals, "boundary_audit": audit,
            "train_by_id": train_by_id}
