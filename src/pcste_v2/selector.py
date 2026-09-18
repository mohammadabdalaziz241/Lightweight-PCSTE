"""Versioned SSL checkpoint selectors. v1 = the sealed dissertation rule (mean of per-dataset masked MSE);
v2 = mean per-dataset NMSE (each dataset's MSE divided by its masked-target energy on the fixed validation masks),
recommended in analysis/pcste_v2_diagnostics/SSL_CHECKPOINT_SELECTOR_ANALYSIS.csv."""
from __future__ import annotations

import numpy as np

SELECTORS = {"mean_mse.v1": "sealed dissertation rule: mean over datasets of masked-cell MSE (minimise)",
             "mean_nmse.v2": "mean over datasets of MSE / masked-target energy E[x^2] (minimise)"}
DEFAULT_SELECTOR = "mean_nmse.v2"


def selector_value(name: str, per_dataset_mse: dict[str, float], per_dataset_target_energy: dict[str, float] | None) -> float:
    ds = sorted(per_dataset_mse)
    if name == "mean_mse.v1":
        return float(np.mean([per_dataset_mse[d] for d in ds]))
    if name == "mean_nmse.v2":
        assert per_dataset_target_energy is not None, "mean_nmse.v2 needs the per-dataset masked-target energy"
        return float(np.mean([per_dataset_mse[d] / max(per_dataset_target_energy[d], 1e-12) for d in ds]))
    raise KeyError(name)
