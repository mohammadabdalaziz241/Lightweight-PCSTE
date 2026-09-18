"""Read-only integrity primitives: hashing and signal-level checks."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    """Hash the numeric content of an array (container-independent)."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def signal_checks(x: np.ndarray, expected_min_len: int | None = None) -> dict:
    """Non-destructive checks on a 1-D (or flattened channel) signal."""
    x = np.asarray(x, dtype=np.float64).ravel()
    out = {
        "n": int(x.size),
        "n_nan": int(np.isnan(x).sum()),
        "n_inf": int(np.isinf(x).sum()),
        "is_constant": bool(x.size and np.nanmax(x) == np.nanmin(x)),
        "min": float(np.nanmin(x)) if x.size else None,
        "max": float(np.nanmax(x)) if x.size else None,
        "mean": float(np.nanmean(x)) if x.size else None,
        "std": float(np.nanstd(x)) if x.size else None,
    }
    out["too_short"] = (expected_min_len is not None
                        and x.size < expected_min_len)
    out["ok"] = (out["n_nan"] == 0 and out["n_inf"] == 0
                 and not out["is_constant"] and not out["too_short"]
                 and x.size > 0)
    return out


def boundary_jump_probe(x: np.ndarray, boundaries: list[int],
                        window: int = 50_000) -> list[dict]:
    """Probe for concatenation discontinuities at given sample indices.

    Compares the absolute first-difference at each boundary against the
    99.9th percentile of first-differences in the surrounding window. A
    boundary-difference far above that percentile is evidence that two
    independently recorded segments were concatenated there. Read-only,
    diagnostic only.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    results = []
    for b in boundaries:
        if not (0 < b < x.size):
            continue
        lo = max(0, b - window // 2)
        hi = min(x.size, b + window // 2)
        local = np.abs(np.diff(x[lo:hi]))
        ref = float(np.percentile(local, 99.9)) if local.size else float("nan")
        jump = float(abs(x[b] - x[b - 1]))
        results.append({
            "boundary": int(b),
            "abs_jump": jump,
            "local_p999_absdiff": ref,
            "jump_ratio": (jump / ref) if ref else float("inf"),
        })
    return results
