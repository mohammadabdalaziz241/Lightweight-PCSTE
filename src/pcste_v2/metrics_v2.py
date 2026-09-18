"""Per-specimen / per-configuration / severity-stratified metrics, macro AUC, and the evaluation output writer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.methodology_v2.experiment.heads import CLASS_ORDERS
from src.methodology_v2.experiment.metrics import classification_report, macro_domain_f1


def macro_ovr_auc(y_true: np.ndarray, probs: np.ndarray, classes: list[str]) -> float:
    from sklearn.metrics import roc_auc_score
    aucs = []
    for j, c in enumerate(classes):
        y = (y_true == c).astype(int)
        if y.min() == y.max():
            continue
        aucs.append(roc_auc_score(y, probs[:, j]))
    return float(np.mean(aucs)) if aucs else float("nan")


def group_recall_table(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Recall and dominant prediction per group (specimen / configuration / severity)."""
    rows = []
    for (ds, cls, g), q in df.groupby(["dataset", "y_true", key]):
        rows.append(dict(dataset=ds, group_key=key, group=g, y_true=cls, n=len(q), recall=float((q.y_pred == q.y_true).mean()),
                         main_pred=q.y_pred.mode().iloc[0], mean_pmax=float(q.p_max.mean())))
    return pd.DataFrame(rows)


def evaluate_split(pred_df: pd.DataFrame) -> dict:
    """pred_df columns: dataset, window_id, y_true, y_pred, p_max, prob__<class>..., physical_specimen, fault_severity, rpm, load."""
    reports, extras = {}, {}
    for ds in CLASS_ORDERS:
        q = pred_df[pred_df.dataset == ds]
        if q.empty:
            continue
        classes = list(CLASS_ORDERS[ds]); rep = classification_report(list(q.y_true), list(q.y_pred), ds)
        probs = q[[f"prob__{c}" for c in classes]].to_numpy(dtype=float)
        rep["macro_ovr_auc"] = macro_ovr_auc(q.y_true.to_numpy(), probs, classes)
        rep["balanced_accuracy"] = float(np.mean(rep["per_class_recall"]))
        reports[ds] = rep
    out = {"per_dataset_reports": reports, "macro_domain_f1": macro_domain_f1(reports) if set(reports) == set(CLASS_ORDERS) else None,
           "macro3_f1_excl_cwru": float(np.mean([reports[d]["macro_f1"] for d in ("JNU", "HIT", "MAFAULDA") if d in reports])),
           "per_specimen": group_recall_table(pred_df, "physical_specimen").to_dict("records"),
           "per_severity": group_recall_table(pred_df, "fault_severity").to_dict("records")}
    if "CWRU" in reports:
        cw = pred_df[pred_df.dataset == "CWRU"]
        out["cwru_per_specimen_recall"] = {s: float((g.y_pred == g.y_true).mean()) for s, g in cw.groupby("physical_specimen")}
        out["cwru_per_class_f1"] = dict(zip(CLASS_ORDERS["CWRU"], reports["CWRU"]["per_class_f1"]))
        out["cwru_race_recall_min"] = float(min(reports["CWRU"]["per_class_recall"][0], reports["CWRU"]["per_class_recall"][1]))
    if "MAFAULDA" in reports:
        mf = pred_df[pred_df.dataset == "MAFAULDA"]
        out["mafaulda_per_configuration_recall"] = {s: float((g.y_pred == g.y_true).mean()) for s, g in mf.groupby("group_id")}
    return out
