"""Deterministic lazy window reader — methodology_v2 Part 3B.

Serves the exact RAW acceleration values (original numerical
representation, no transformation, no normalization) for any frozen
window-manifest row. HIT windows are assembled by ordered concatenation
of the audited source fragments before slicing, exactly per the frozen
concatenation policy.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import REPO_ROOT
from . import part3b_protocol as P


@lru_cache(maxsize=8)
def _cwru_channel(source_file: str, recording_id: str) -> np.ndarray:
    import scipy.io as sio
    mat = sio.loadmat(str(REPO_ROOT / source_file))
    pid = recording_id.removeprefix("cwru_")
    return np.ascontiguousarray(mat[f"{pid}_DE_time"]).ravel()


@lru_cache(maxsize=4)
def _jnu_channel(source_file: str) -> np.ndarray:
    return np.loadtxt(REPO_ROOT / source_file)


@lru_cache(maxsize=8)
def _hit_session(source_file: str) -> np.ndarray:
    return np.load(REPO_ROOT / source_file, mmap_mode="r")


@lru_cache(maxsize=32)
def _mafaulda_channel(source_file: str) -> np.ndarray:
    return pd.read_csv(REPO_ROOT / source_file, header=None,
                       usecols=[2], dtype=np.float64).to_numpy().ravel()


def read_window(row: pd.Series | dict) -> np.ndarray:
    """Return the raw signal for one frozen window-manifest row."""
    ds = row["dataset"]
    s, e = int(row["start_sample"]), int(row["end_sample"])
    if ds == "CWRU":
        x = _cwru_channel(row["source_file"], row["recording_id"])
    elif ds == "JNU":
        x = _jnu_channel(row["source_file"])
    elif ds == "HIT":
        arr = _hit_session(row["source_file"])
        k = int(str(row["recording_id"]).rsplit("rec", 1)[1])
        base = P.HIT_FRAGMENTS_PER_STREAM * k
        frag = P.HIT_FRAGMENT_SAMPLES
        f0, f1 = s // frag, (e - 1) // frag
        seg = np.asarray(arr[base + f0: base + f1 + 1,
                             P.HIT_CH3_ROW_INDEX, :])
        out = seg.reshape(-1)[s - f0 * frag: e - f0 * frag]
        if out.size != e - s:
            raise AssertionError(f"HIT window {row['window_id']}: "
                                 f"bad slice size {out.size}")
        return out
    elif ds == "MAFAULDA":
        x = _mafaulda_channel(row["source_file"])
    else:
        raise AssertionError(f"unknown dataset {ds}")
    if e > x.size:
        raise AssertionError(
            f"{row['window_id']}: end {e} beyond source ({x.size})")
    return x[s:e]


def hit_reference_stream(source_file: str, recording_id: str) -> np.ndarray:
    """Reference (non-lazy) reconstruction of one full HIT logical stream
    by explicit ordered concatenation — used by tests to prove the lazy
    fragment slicing is value-identical."""
    arr = _hit_session(source_file)
    k = int(recording_id.rsplit("rec", 1)[1])
    base = P.HIT_FRAGMENTS_PER_STREAM * k
    frags = [np.asarray(arr[base + i, P.HIT_CH3_ROW_INDEX, :])
             for i in range(P.HIT_FRAGMENTS_PER_STREAM)]
    return np.concatenate(frags)
