"""Nested label-fraction manifests — deterministic, label-blind beyond
class stratification, group-even, and identical for S0 and S1.

For every (fold, seed, dataset, class): windows are ranked by
SHA256(fold|seed|dataset|class|group|window_id); the selection ORDER is
a round-robin over the class's groups (groups sorted by group_id), each
group yielding its own hash-ranked windows in turn. The nested property
holds automatically: fraction f selects the first ceil(f * N_class)
positions of that fixed order (minimum one window per class by ceil).
No signal values, energies, predictions or validation results are used.
"""
from __future__ import annotations

import hashlib
import math

import pandas as pd

from ..part3b_windows import PART3B_DIR
from .heads import CLASS_ORDERS, window_class

FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.00)
SEEDS = (42, 1337, 2026)
FOLDS = (1, 2, 3)


def _rank_key(fold: int, seed: int, ds: str, cls: str, group: str,
              window_id: str) -> str:
    return hashlib.sha256(
        f"{fold}|{seed}|{ds}|{cls}|{group}|{window_id}".encode()
    ).hexdigest()


def build_subset_table(fold: int, seed: int) -> pd.DataFrame:
    """One row per TRAIN window with its class-selection position and
    per-fraction membership flags."""
    man = pd.read_csv(PART3B_DIR / f"window_manifest_fold_{fold}.csv")
    tr = man[man["split"] == "train"].copy()
    rows = []
    for ds in CLASS_ORDERS:
        sub = tr[tr["dataset"] == ds]
        for cls in CLASS_ORDERS[ds]:
            cw = sub[sub.apply(lambda r: window_class(ds, r) == cls,
                               axis=1)]
            if cw.empty:
                raise AssertionError(f"class {cls} missing in fold "
                                     f"{fold} {ds} TRAIN")
            # per-group hash-ranked queues
            queues = {}
            for gid, grp in cw.groupby("group_id"):
                ranked = sorted(
                    grp["window_id"],
                    key=lambda w: _rank_key(fold, seed, ds, cls,
                                            str(gid), w))
                queues[str(gid)] = list(ranked)
            order = []
            gids = sorted(queues)
            while any(queues[g] for g in gids):     # round-robin
                for g in gids:
                    if queues[g]:
                        order.append((queues[g].pop(0), g))
            n_cls = len(order)
            cuts = {f: math.ceil(f * n_cls) for f in FRACTIONS}
            for pos, (wid, gid) in enumerate(order):
                row = {"fold": fold, "seed": seed, "dataset": ds,
                       "window_id": wid, "class": cls, "group_id": gid,
                       "class_rank": pos, "n_class": n_cls}
                for f in FRACTIONS:
                    row[f"frac_{int(f * 100)}"] = pos < cuts[f]
                rows.append(row)
    df = pd.DataFrame(rows).sort_values(
        ["dataset", "class", "class_rank"]).reset_index(drop=True)
    return df


def realised_fractions(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (ds, cls), grp in df.groupby(["dataset", "class"]):
        n = len(grp)
        for f in FRACTIONS:
            k = int(grp[f"frac_{int(f * 100)}"].sum())
            out.append({"dataset": ds, "class": cls, "n_class": n,
                        "fraction": f, "selected": k,
                        "realised_pct": round(100 * k / n, 2),
                        "requested_pct": 100 * f})
    return pd.DataFrame(out)
