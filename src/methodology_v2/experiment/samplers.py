"""Frozen samplers.

SSL:        Dataset -> Group -> Window       (LABEL-FREE, balanced 16x4)
Supervised: Dataset -> Class -> Group -> Window (over the selected
            labelled subset only, balanced 16x4)

Both use replacement, deterministic seeded RNG, effective batch 64.
The SSL sampler is constructed from a manifest view containing ONLY
(dataset, group_id, window_id) — label columns are structurally
inaccessible (test-enforced).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")
PER_DATASET = 16          # 16 x 4 datasets = effective batch 64
EFFECTIVE_BATCH = 64


class SSLSampler:
    """Label-free hierarchical sampler over TRAIN windows."""

    ALLOWED_COLUMNS = {"dataset", "group_id", "window_id"}

    def __init__(self, train_manifest: pd.DataFrame, seed: int):
        extra = set(train_manifest.columns) - self.ALLOWED_COLUMNS
        if extra:
            raise AssertionError(
                f"SSL sampler must not see label-bearing columns: {extra}")
        self._groups: dict[str, dict[str, list[str]]] = {}
        for ds in DATASETS:
            sub = train_manifest[train_manifest["dataset"] == ds]
            self._groups[ds] = {
                str(g): list(grp["window_id"])
                for g, grp in sub.groupby("group_id")}
        self._rng = np.random.default_rng(seed)

    def next_batch(self) -> list[tuple[str, str]]:
        """[(dataset, window_id) x 64], 16 per dataset."""
        batch = []
        for ds in DATASETS:                      # fixed dataset order
            gids = sorted(self._groups[ds])
            for _ in range(PER_DATASET):
                g = gids[self._rng.integers(len(gids))]
                wins = self._groups[ds][g]
                batch.append((ds, wins[self._rng.integers(len(wins))]))
        return batch


class SupervisedSampler:
    """Hierarchical labelled sampler: dataset -> class (cycled) ->
    group (uniform) -> window (uniform), over SELECTED windows only.
    Class balance holds over steps via deterministic class cycling."""

    def __init__(self, subset: pd.DataFrame, fraction: float, seed: int):
        col = f"frac_{int(fraction * 100)}"
        sel = subset[subset[col]]
        self._tree: dict[str, dict[str, dict[str, list[str]]]] = {}
        for ds in DATASETS:
            dsub = sel[sel["dataset"] == ds]
            self._tree[ds] = {}
            for cls, cgrp in dsub.groupby("class"):
                self._tree[ds][str(cls)] = {
                    str(g): list(grp["window_id"])
                    for g, grp in cgrp.groupby("group_id")}
        self._cls_cycle = {ds: 0 for ds in DATASETS}
        self._rng = np.random.default_rng(seed)

    def next_batch(self) -> list[tuple[str, str, str]]:
        """[(dataset, class, window_id) x 64], 16 per dataset,
        classes cycled deterministically within each dataset."""
        batch = []
        for ds in DATASETS:
            classes = sorted(self._tree[ds])
            for _ in range(PER_DATASET):
                cls = classes[self._cls_cycle[ds] % len(classes)]
                self._cls_cycle[ds] += 1
                groups = self._tree[ds][cls]
                gids = sorted(groups)
                g = gids[self._rng.integers(len(gids))]
                wins = groups[g]
                batch.append((ds, cls,
                              wins[self._rng.integers(len(wins))]))
        return batch
