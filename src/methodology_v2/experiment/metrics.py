"""Frozen metric implementations (model-independent, numpy only).

Class order comes from the frozen taxonomy (heads.CLASS_ORDERS); absent
classes are NEVER dropped — they appear with zero counts and contribute
zero-division-safe scores (precision/recall/F1 = 0.0 where undefined,
explicitly). Verified against scikit-learn in tests where available.
"""
from __future__ import annotations

import numpy as np

from .heads import CLASS_ORDERS


def confusion_matrix(y_true: list[str], y_pred: list[str],
                     classes: tuple[str, ...]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[idx[str(t)], idx[str(p)]] += 1
    return cm


def classification_report(y_true, y_pred, dataset: str) -> dict:
    classes = CLASS_ORDERS[dataset]
    cm = confusion_matrix(y_true, y_pred, classes)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    n = cm.sum()
    return {
        "dataset": dataset, "classes": list(classes),
        "accuracy": float(tp.sum() / n) if n else 0.0,
        "macro_f1": float(f1.mean()),           # absent classes count as 0
        "per_class_precision": [round(float(v), 6) for v in precision],
        "per_class_recall": [round(float(v), 6) for v in recall],
        "per_class_f1": [round(float(v), 6) for v in f1],
        "support": [int(v) for v in support],
        "confusion_matrix": cm.tolist(),
        "zero_division_policy": "undefined precision/recall/F1 := 0.0",
    }


def macro_domain_f1(per_dataset_reports: dict[str, dict]) -> float:
    """PRIMARY metric: mean of the four dataset Macro-F1 scores —
    every dataset contributes exactly 25%."""
    expected = set(CLASS_ORDERS)
    if set(per_dataset_reports) != expected:
        raise AssertionError(
            f"MacroDomainF1 requires all datasets {expected}")
    return float(np.mean([per_dataset_reports[ds]["macro_f1"]
                          for ds in sorted(expected)]))


def macro_domain_recon_mse(per_dataset_mse: dict[str, float]) -> float:
    """SSL checkpoint criterion: mean over the four per-dataset
    reconstruction MSEs (each dataset's MSE = mean of its windows'
    masked-valid-cell MSEs). Never pooled over cells or windows."""
    expected = set(CLASS_ORDERS)
    if set(per_dataset_mse) != expected:
        raise AssertionError("MacroDomainReconMSE requires all datasets")
    return float(np.mean([per_dataset_mse[ds]
                          for ds in sorted(expected)]))
