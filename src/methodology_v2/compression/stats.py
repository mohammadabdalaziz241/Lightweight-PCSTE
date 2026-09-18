"""Pre-registered paired statistics for Part 6 (9 fold x seed cells).

Per contrast: all nine deltas, mean, median, SD (ddof=1), fraction
positive, effect size mean/SD, exact sign-flip tests over all 2^9 = 512
sign patterns.

Non-inferiority (one-sided, margin m > 0): H0: mean(delta) <= -m.
  Test statistic on the MARGIN-SHIFTED deltas d' = delta + m; exact
  one-sided sign-flip p = #(mean(flipped d') >= mean(d')) / 512.
  NI passes iff p < alpha.  (Shift first, then flip — flipping the raw
  deltas would test H0: mean = 0, the wrong hypothesis.)
Superiority (two-sided): p = #(|mean(flipped)| >= |mean(observed)|)/512
  — the primary experiment's exact test, reproduced here.
Hierarchical H1: two-sided superiority is reported ONLY if NI passed.
Holm step-down over each family (m = 3 confirmatory, m = 4 secondary).
Push claim: mean delta >= 0.02 AND two-sided p < 0.05, else
"not distinguishable".
Sensitivity: per-dataset deltas; MacroDomainF1 excluding CWRU.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .protocol import (NI_MARGIN_ARCH, NI_MARGIN_PTQ, PUSH_ALPHA,
                       PUSH_MIN_DELTA)

DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")


def _all_sign_patterns(n: int) -> np.ndarray:
    return np.array(list(itertools.product((-1.0, 1.0), repeat=n)))


def sign_flip_two_sided(deltas) -> float:
    d = np.asarray(deltas, dtype=np.float64)
    obs = abs(d.mean())
    means = (_all_sign_patterns(len(d)) * d).mean(axis=1)
    return float((np.abs(means) >= obs - 1e-12).mean())


def sign_flip_one_sided_greater(values) -> float:
    """H1: mean > 0 (used on margin-shifted deltas for NI)."""
    d = np.asarray(values, dtype=np.float64)
    obs = d.mean()
    means = (_all_sign_patterns(len(d)) * d).mean(axis=1)
    return float((means >= obs - 1e-12).mean())


def descriptives(deltas) -> dict:
    d = np.asarray(deltas, dtype=np.float64)
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return {"n": int(len(d)), "deltas": [float(v) for v in d],
            "mean": float(d.mean()), "median": float(np.median(d)),
            "sd": sd, "fraction_positive": float((d > 0).mean()),
            "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum()),
            "n_ties": int((d == 0).sum()),
            "effect_size_mean_over_sd": (float(d.mean() / sd)
                                         if sd and sd > 0 else None)}


def non_inferiority(deltas, margin: float, alpha: float = 0.05) -> dict:
    d = np.asarray(deltas, dtype=np.float64)
    shifted = d + margin
    p = sign_flip_one_sided_greater(shifted)
    return {"margin": margin, "shifted_mean": float(shifted.mean()),
            "p_one_sided": p, "passes": bool(p < alpha and shifted.mean() > 0),
            "h0": f"mean(delta) <= -{margin}",
            "method": "exact one-sided sign-flip on margin-shifted deltas"}


def superiority(deltas) -> dict:
    return {"p_two_sided": sign_flip_two_sided(deltas),
            "method": "exact two-sided sign-flip (512 patterns)"}


def contrast(name: str, a: list[float], b: list[float], kind: str,
             margin: float | None = None) -> dict:
    """delta = a - b per cell. kind: 'ni_then_superiority' | 'two_sided'
    | 'ni' | 'push'."""
    if len(a) != len(b):
        raise ValueError("unequal cell counts")
    d = [float(x) - float(y) for x, y in zip(a, b)]
    out = {"contrast": name, "kind": kind, **descriptives(d)}
    if kind in ("ni", "ni_then_superiority"):
        out["non_inferiority"] = non_inferiority(d, margin)
        out["p_for_holm"] = out["non_inferiority"]["p_one_sided"]
        if kind == "ni_then_superiority":
            if out["non_inferiority"]["passes"]:
                out["superiority"] = superiority(d)
            else:
                out["superiority"] = {"not_tested": "NI failed — hierarchical "
                                                    "rule blocks superiority"}
    elif kind == "two_sided":
        out["superiority"] = superiority(d)
        out["p_for_holm"] = out["superiority"]["p_two_sided"]
    elif kind == "push":
        out["superiority"] = superiority(d)
        out["p_for_holm"] = out["superiority"]["p_two_sided"]
        out["push"] = {"min_mean_delta": PUSH_MIN_DELTA, "alpha": PUSH_ALPHA,
                       "claim": bool(out["mean"] >= PUSH_MIN_DELTA
                                     and out["p_for_holm"] < PUSH_ALPHA),
                       "verdict": ("push" if (out["mean"] >= PUSH_MIN_DELTA
                                              and out["p_for_holm"] < PUSH_ALPHA)
                                   else "not distinguishable")}
    else:
        raise ValueError(kind)
    return out


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict:
    """Holm step-down: sort p ascending; p_adj_(i) = max_{j<=i} min(1,
    (m-j+1) p_(j)); reject while p_adj <= alpha."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    rejected = {k: bool(adj[k] <= alpha) for k, _ in items}
    return {"m": m, "alpha": alpha, "p_adjusted": adj, "rejected": rejected,
            "order": [k for k, _ in items]}


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------
def confirmatory_family(k1, s1, c_small, q8_k1) -> dict:
    """H1 K1 vs S1 (NI 0.02 then superiority); H2 K1 vs C_small
    (two-sided); H3 Q8(K1) vs K1 (NI 0.01). Holm m=3 on the primary p of
    each hypothesis (NI p for H1/H3, two-sided p for H2)."""
    h1 = contrast("H1 K1 vs S1", k1, s1, "ni_then_superiority", NI_MARGIN_ARCH)
    h2 = contrast("H2 K1 vs C_small", k1, c_small, "two_sided")
    h3 = contrast("H3 Q8(K1) vs K1", q8_k1, k1, "ni", NI_MARGIN_PTQ)
    fam = {"H1": h1, "H2": h2, "H3": h3}
    fam["holm"] = holm({k: v["p_for_holm"] for k, v in fam.items()})
    if h1["non_inferiority"]["passes"] and "p_two_sided" in h1.get("superiority", {}):
        fam["H1_superiority_holm_note"] = (
            "superiority p is reported after NI passed (hierarchical); "
            "Holm-adjusted superiority requires ~8-9/9 positive cells")
    return fam


def secondary_family(k0, s0, k1, q8_s1, s1, q8_s0) -> dict:
    c1 = contrast("K0 vs S0", k0, s0, "ni", NI_MARGIN_ARCH)
    c2 = contrast("K1 vs K0", k1, k0, "two_sided")
    c3 = contrast("Q8(S1) vs S1", q8_s1, s1, "ni", NI_MARGIN_PTQ)
    c4 = contrast("Q8(S0) vs S0", q8_s0, s0, "ni", NI_MARGIN_PTQ)
    fam = {"K0_vs_S0": c1, "K1_vs_K0": c2, "Q8S1_vs_S1": c3, "Q8S0_vs_S0": c4}
    fam["holm"] = holm({k: v["p_for_holm"] for k, v in fam.items()})
    return fam


def push_family(b1=None, s1=None, f1=None, s1_l010=None) -> dict:
    out = {}
    if b1 is not None:
        out["B1_vs_S1"] = contrast("B1 vs S1", b1, s1, "push")
    if f1 is not None:
        out["F1_vs_S1_10pct"] = contrast("F1 vs S1-10%", f1, s1_l010, "push")
    return out


# ---------------------------------------------------------------------------
# sensitivity analyses
# ---------------------------------------------------------------------------
def per_dataset_deltas(a_reports: list[dict], b_reports: list[dict]) -> dict:
    """a/b: per cell {dataset: macro_f1}."""
    out = {}
    for ds in DATASETS:
        d = [float(x[ds]) - float(y[ds]) for x, y in zip(a_reports, b_reports)]
        out[ds] = descriptives(d)
    return out


def macro_excluding(per_dataset: dict, exclude: str = "CWRU") -> float:
    return float(np.mean([v for k, v in per_dataset.items() if k != exclude]))


def excluding_cwru_contrast(name: str, a_reports, b_reports, kind="two_sided",
                            margin=None) -> dict:
    a = [macro_excluding(r) for r in a_reports]
    b = [macro_excluding(r) for r in b_reports]
    return contrast(name + " (excl. CWRU)", a, b, kind, margin)
