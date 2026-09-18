"""PC-STE v2 representation layer: raw readers (CWRU v2 pool incl. 12 kHz and fan-end files; MaFaulDa second channel),
STFT grids (frozen v1 grid plus the pre-registered multi-resolution pair), log1p, N2b normalisers (TRAIN-only, floor 0.05,
versioned), and the envelope/order feature cache. Nothing here touches the sealed methodology_v2 artefacts.
"""
from __future__ import annotations

import hashlib, io, json, zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.methodology_v2.registry import REPO_ROOT
from src.methodology_v2.part4a_repdesign import apply_transform, tf_map
from src.methodology_v2.part3b_reader import read_window as read_window_v1
from .envelope import ENVELOPE_BANDS_HZ, BEARING_FACTORS, envelope_order_vector, N_ORDER_CELLS, ENVELOPE_VERSION, envelope3_order_matrix, N_ENVELOPE3_BANDS, ENVELOPE3_VERSION

REPRESENTATION_VERSION = "pcste_v2.representation.v1"
N2B_FLOOR = 0.05            # diagnostics recommendation (N2_DIAGNOSTIC_COMPARISON.csv): max(std_train, 0.05)
N2B_VERSION = "n2b.v1"
V2_ROOT = REPO_ROOT / "pcste_v2"          # all v2 artefacts (protocol tables, normalisers, caches) live here
RATES = {"CWRU48": 48000, "CWRU12": 12000, "JNU": 50000, "HIT": 25000, "MAFAULDA": 50000}
# physically matched grids: (n_fft, hop) -> frame ~21 ms, hop ~5.3 ms, bin ~47-49 Hz on the v1 grid
STFT_GRIDS = {
    "v1":       {"CWRU48": (1024, 256), "CWRU12": (256, 64),  "JNU": (1024, 256), "HIT": (512, 128),  "MAFAULDA": (1024, 256)},
    "mr_short": {"CWRU48": (512, 128),  "CWRU12": (128, 32),  "JNU": (512, 128),  "HIT": (256, 64),   "MAFAULDA": (512, 128)},
    "mr_long":  {"CWRU48": (4096, 1024), "CWRU12": (1024, 256), "JNU": (4096, 1024), "HIT": (2048, 512), "MAFAULDA": (4096, 1024)},
}
GRID_SETS = {"single": ("v1",), "multires": ("mr_short", "mr_long")}


def rate_key(dataset: str, rate_hz: int) -> str:
    return f"CWRU{rate_hz // 1000}" if dataset == "CWRU" else dataset


def frequency_hz(key: str, grid: str) -> np.ndarray:
    n_fft, _ = STFT_GRIDS[grid][key]
    return np.arange(n_fft // 2 + 1) * RATES[key] / n_fft


def stft_rep(x: np.ndarray, key: str, grid: str) -> np.ndarray:
    """log1p(|STFT|) as (frames, bins) float64 — identical construction to the sealed rep_of for the v1 grid."""
    n_fft, hop = STFT_GRIDS[grid][key]
    return apply_transform(tf_map(x, n_fft, hop), "log1p")


# ------------------------------------------------------------------------------------------------------ raw readers
NATIVE12_ALLOWLIST: dict | None = None      # set by the executor for native12 protocols: every CWRU read must pass the allowlist gate


@lru_cache(maxsize=16)
def _cwru_channel(source_file: str, var: str) -> np.ndarray:
    import scipy.io as sio
    if NATIVE12_ALLOWLIST is not None:
        from .protocol_cwru_native12 import assert_native12_source
        assert_native12_source(source_file, var, NATIVE12_ALLOWLIST, check_bytes=False)   # bytes verified once at preflight
    return np.ascontiguousarray(sio.loadmat(str(REPO_ROOT / source_file))[var]).ravel().astype(np.float64)


@lru_cache(maxsize=32)
def _mafaulda_col(source_file: str, col: int) -> np.ndarray:
    return pd.read_csv(REPO_ROOT / source_file, header=None, usecols=[col], dtype=np.float64).to_numpy().ravel()


def read_raw(row: pd.Series, channel_index: int = 0) -> np.ndarray:
    """Raw samples for one v2 window-manifest row. channel_index 0 = the row's primary channel;
    channel_index 1 = MaFaulDa underhang axial (CSV column 1). CWRU v2 rows carry 'mat_variable'."""
    ds = row["dataset"]
    s, e = int(row["start_sample"]), int(row["end_sample"])
    if ds == "CWRU" and "mat_variable" in row and isinstance(row["mat_variable"], str) and row["mat_variable"]:
        return _cwru_channel(row["source_file"], row["mat_variable"])[s:e]
    if ds == "MAFAULDA" and channel_index == 1:
        return _mafaulda_col(row["source_file"], 1)[s:e]
    if channel_index != 0:
        raise ValueError(f"{ds}: no channel {channel_index}")
    return np.asarray(read_window_v1(row), dtype=np.float64)


# ------------------------------------------------------------------------------------------------------ N2b
class BinStats:
    def __init__(self, n_bins: int):
        self.n = 0; self.mean = np.zeros(n_bins); self.m2 = np.zeros(n_bins); self.n_windows = 0

    def add(self, rep: np.ndarray) -> None:
        nb = rep.shape[0]; mb = rep.mean(axis=0); m2b = ((rep - mb) ** 2).sum(axis=0)
        if self.n == 0:
            self.n, self.mean, self.m2 = nb, mb, m2b
        else:
            delta = mb - self.mean; tot = self.n + nb
            self.mean = self.mean + delta * (nb / tot); self.m2 = self.m2 + m2b + delta ** 2 * (self.n * nb / tot); self.n = tot
        self.n_windows += 1

    def finalize(self) -> dict:
        std = np.sqrt(self.m2 / self.n)
        return {"mean": self.mean, "std_raw": std, "std_denominator": np.maximum(std, N2B_FLOOR),
                "floored_bins": np.where(std < N2B_FLOOR)[0].astype(np.int64), "n_frames": self.n, "n_windows": self.n_windows}


def deterministic_savez(path: Path, arrays: dict) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        for name in sorted(arrays):
            buf = io.BytesIO(); np.save(buf, np.asarray(arrays[name]), allow_pickle=False)
            z.writestr(zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)), buf.getvalue())


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def norm_path(protocol_dir: Path, fold: int, grid: str, key: str, channel_index: int) -> Path:
    return protocol_dir / "normalizers" / f"fold_{fold}" / f"{grid}__{key}__c{channel_index}.npz"


def env_norm_path(protocol_dir: Path, fold: int, dataset: str) -> Path:
    return protocol_dir / "normalizers" / f"fold_{fold}" / f"envelope__{dataset}.npz"


def env3_norm_path(protocol_dir: Path, fold: int, dataset: str) -> Path:
    return protocol_dir / "normalizers" / f"fold_{fold}" / f"envelope3__{dataset}.npz"


def fit_envelope3_normalizers(protocol_dir: Path, fold: int, manifest: pd.DataFrame, workers: int = 8) -> list[dict]:
    """Per-(dataset, band, cell) TRAIN statistics for the 3-band fallback (N2b floor rule); absent bands are excluded from the
    statistics of their rows (a band that exists for no TRAIN row of a dataset keeps mean 0 and the floor as denominator)."""
    from multiprocessing import Pool
    tr = manifest[manifest["split"] == "train"]
    assert (tr["split"] == "train").all() and (tr["fold_id"] == fold).all()
    jobs = [(protocol_dir, fold, "ENVELOPE3", ds, 0, tr[tr["dataset"] == ds].sort_values(["recording_id", "start_sample"]).to_dict("records")) for ds in sorted(tr["dataset"].unique())]
    with Pool(min(workers, len(jobs))) as pool:
        return pool.map(_fit_job, jobs)


def fit_normalizers(protocol_dir: Path, fold: int, manifest: pd.DataFrame, grids: tuple[str, ...], channels: dict[str, int],
                    envelope: bool, workers: int = 8) -> list[dict]:
    """Fit N2b for every (grid, rate-key, channel) and the envelope-cell stats from TRAIN rows only. Deterministic."""
    from multiprocessing import Pool
    tr = manifest[manifest["split"] == "train"]
    assert (tr["split"] == "train").all() and (tr["fold_id"] == fold).all()
    jobs = []
    for ds in sorted(tr["dataset"].unique()):
        sub = tr[tr["dataset"] == ds]
        for key in sorted(sub.apply(lambda r: rate_key(ds, int(r["native_sampling_rate_hz"])), axis=1).unique()):
            rows = sub[sub.apply(lambda r: rate_key(ds, int(r["native_sampling_rate_hz"])) == key, axis=1)]
            for ci in range(channels.get(ds, 1)):
                for grid in grids:
                    jobs.append((protocol_dir, fold, grid, key, ci, rows.sort_values(["recording_id", "start_sample"]).to_dict("records")))
        if envelope:
            jobs.append((protocol_dir, fold, "ENVELOPE", ds, 0, sub.sort_values(["recording_id", "start_sample"]).to_dict("records")))
    with Pool(min(workers, len(jobs))) as pool:
        return pool.map(_fit_job, jobs)


def _fit_job(args) -> dict:
    protocol_dir, fold, grid, key, ci, rows = args
    if grid == "ENVELOPE":
        st = BinStats(N_ORDER_CELLS)
        for r in rows:
            st.add(envelope_vector_for_row(pd.Series(r))[None, :].astype(np.float64))
        out = st.finalize(); p = env_norm_path(protocol_dir, fold, key)
    elif grid == "ENVELOPE3":
        sts = [BinStats(N_ORDER_CELLS) for _ in range(N_ENVELOPE3_BANDS)]
        for r in rows:
            m, valid = envelope3_matrix_for_row(pd.Series(r))
            for bi in range(N_ENVELOPE3_BANDS):
                if valid[bi]:
                    sts[bi].add(m[bi][None, :].astype(np.float64))
        outs = []
        for bi, st in enumerate(sts):
            if st.n_windows == 0:                                # band absent for every TRAIN row of this dataset
                outs.append({"mean": np.zeros(N_ORDER_CELLS), "std_raw": np.zeros(N_ORDER_CELLS), "std_denominator": np.full(N_ORDER_CELLS, N2B_FLOOR),
                             "floored_bins": np.arange(N_ORDER_CELLS, dtype=np.int64), "n_frames": 0, "n_windows": 0})
            else:
                outs.append(st.finalize())
        out = {"mean": np.stack([o["mean"] for o in outs]), "std_raw": np.stack([o["std_raw"] for o in outs]), "std_denominator": np.stack([o["std_denominator"] for o in outs]),
               "floored_bins": np.concatenate([o["floored_bins"] + bi * N_ORDER_CELLS for bi, o in enumerate(outs)]).astype(np.int64),
               "n_frames": int(sum(o["n_frames"] for o in outs)), "n_windows": int(max(o["n_windows"] for o in outs)),
               "n_windows_per_band": np.array([o["n_windows"] for o in outs], dtype=np.int64)}
        p = env3_norm_path(protocol_dir, fold, key)
    else:
        n_fft, _ = STFT_GRIDS[grid][key]; st = BinStats(n_fft // 2 + 1)
        for r in rows:
            st.add(stft_rep(read_raw(pd.Series(r), ci), key, grid))
        out = st.finalize(); p = norm_path(protocol_dir, fold, grid, key, ci); out["frequency_hz"] = frequency_hz(key, grid)
    p.parent.mkdir(parents=True, exist_ok=True)
    deterministic_savez(p, {k: v for k, v in out.items()} | {"fold": np.int64(fold), "floor": np.float64(N2B_FLOOR)})
    return {"fold": fold, "grid": grid, "key": key, "channel": ci, "file": str(p.relative_to(REPO_ROOT)), "n_windows": out["n_windows"],
            "n_frames": out["n_frames"], "n_floored_bins": int(out["floored_bins"].size), "min_std_raw": float(out["std_raw"].min()),
            "sha256": sha256_file(p), "rule": f"denominator = max(std_train, {N2B_FLOOR})", "version": N2B_VERSION}


@lru_cache(maxsize=256)
def _load_norm(path: str) -> dict:
    z = np.load(path); return {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------------------ envelope
def envelope_vector_for_row(row: pd.Series, rpm_override: float | None = None, gain: float = 1.0) -> np.ndarray:
    ds = row["dataset"]; fs = float(row["native_sampling_rate_hz"])
    rpm = float(rpm_override) if rpm_override is not None else float(row["rpm"])
    x = read_raw(row, 0) * gain
    return envelope_order_vector(x, fs, rpm, ENVELOPE_BANDS_HZ[ds])


def envelope3_matrix_for_row(row: pd.Series, rpm_override: float | None = None, gain: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    fs = float(row["native_sampling_rate_hz"])
    rpm = float(rpm_override) if rpm_override is not None else float(row["rpm"])
    x = read_raw(row, 0) * gain
    return envelope3_order_matrix(x, fs, rpm)


def bearing_factor_info(row: pd.Series) -> tuple[float, int]:
    """(BPFO multiplier, known flag) for the recording's bearing; unknown -> (1.0, 0)."""
    fs = row.get("bearing_factor_set", None)
    if isinstance(fs, str) and fs in BEARING_FACTORS:
        return float(BEARING_FACTORS[fs]["BPFO"]), 1
    if row["dataset"] == "MAFAULDA":
        return float(BEARING_FACTORS["MAFAULDA"]["BPFO"]), 1
    return 1.0, 0


# ------------------------------------------------------------------------------------------------------ item builder
class ItemBuilder:
    """Builds the collate_v2 item for a window row under one representation configuration, applying N2b from the
    protocol's sealed TRAIN normalisers. Optional controls: rpm_override (order permutation control) and gain (level control)."""

    def __init__(self, protocol_dir: Path, fold: int, grids: tuple[str, ...], channels: dict[str, int], envelope: bool, envelope_bands: int = 1):
        self.protocol_dir, self.fold, self.grids, self.channels, self.envelope = Path(protocol_dir), fold, grids, channels, envelope
        self.envelope_bands = int(envelope_bands)

    def build(self, row: pd.Series, rpm_override: float | None = None, gain: float = 1.0) -> dict:
        ds = row["dataset"]; key = rate_key(ds, int(row["native_sampling_rate_hz"])); streams = {}
        for ci in range(self.channels.get(ds, 1)):
            x = read_raw(row, ci) * gain
            for gi, grid in enumerate(self.grids):
                rep = stft_rep(x, key, grid)
                nm = _load_norm(str(norm_path(self.protocol_dir, self.fold, grid, key, ci)))
                z = (rep - nm["mean"]) / nm["std_denominator"]
                n_fft, hop = STFT_GRIDS[grid][key]
                streams[f"g{gi}_c{ci}"] = (np.ascontiguousarray(z.T, dtype=np.float32), nm["frequency_hz"].astype(np.float32),
                                           (np.arange(rep.shape[0]) * hop / RATES[key]).astype(np.float32))
        item = {"streams": streams}
        if self.envelope and self.envelope_bands == 1:
            v = envelope_vector_for_row(row, rpm_override=rpm_override, gain=gain)
            nm = _load_norm(str(env_norm_path(self.protocol_dir, self.fold, ds)))
            v = ((v - nm["mean"]) / nm["std_denominator"]).astype(np.float32)
            bpfo, known = bearing_factor_info(row)
            item["env"] = (v, bpfo, known)
        elif self.envelope:
            m, valid = envelope3_matrix_for_row(row, rpm_override=rpm_override, gain=gain)
            nm = _load_norm(str(env3_norm_path(self.protocol_dir, self.fold, ds)))
            z = ((m - nm["mean"]) / nm["std_denominator"]).astype(np.float32)
            z[~valid] = 0.0                                          # absent band: zero channel + band_valid flag
            bpfo, known = bearing_factor_info(row)
            item["env"] = (z, bpfo, known, valid.astype(np.bool_))
        return item
