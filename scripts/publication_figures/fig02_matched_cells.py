#!/usr/bin/env python3
"""Figure 2 — matched fold-seed Full-S1 vs K1 performance.

Panel (a) is a paired (dumbbell) plot of Macro-4 Macro-F1 in each of the nine
matched fold-seed cells. Panel (b) shows the nine paired differences against the
prespecified non-inferiority margin, with the aggregate endpoints alongside.

The non-inferiority margin is drawn in panel (b) only, and is labelled as a
Macro-F1 margin: no margin was defined for Macro-AUC, and the Macro-AUC block is
explicitly marked as descriptive.

All values are read from the sealed TEST artifacts.

Run:  python scripts/publication_figures/fig02_matched_cells.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import pubfig_style as S


def build() -> None:
    S.apply_style()
    cells = S.load_matched_cells()
    agg = S.load_aggregate()

    paired = agg["paired_macro_4_f1"]
    ni = paired["non_inferiority"]
    margin = -ni["margin"]
    f1_full, f1_k1 = agg["full_s1_macro_4_f1"], agg["k1_macro_4_f1"]
    auc_full, auc_k1 = agg["macro_4_auc"]["full_s1"], agg["macro_4_auc"]["k1"]
    d_auc = np.array([c["delta_auc"] for c in cells])

    labels = [f"GF{c['fold']}-{c['seed']}" for c in cells]
    full = np.array([c["full_s1_f1"] for c in cells])
    k1 = np.array([c["k1_f1"] for c in cells])
    delta = np.array([c["delta_f1"] for c in cells])
    y = np.arange(len(cells))[::-1]  # first cell at the top

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(S.TEXT_WIDTH_IN, 3.05),
        gridspec_kw={"width_ratios": [1.20, 1.0], "wspace": 0.42})

    # ------------------------------------------------ (a) paired cell values
    for yi, f, k in zip(y, full, k1):
        axa.plot([f, k], [yi, yi], color="#b5b5b5", linewidth=1.0, zorder=1)
    axa.scatter(full, y, s=26, color=S.FULL_S1, marker="o",
                zorder=3, label="Full-S1", edgecolor="white", linewidth=0.5)
    axa.scatter(k1, y, s=26, color=S.K1, marker="s",
                zorder=3, label="K1", edgecolor="white", linewidth=0.5)

    axa.set_yticks(y)
    axa.set_yticklabels(labels)
    axa.set_xlabel("Macro-4 Macro-F1")
    axa.set_xlim(0.86, 1.0)
    axa.set_ylim(-1.45, len(cells) - 0.35)
    S.tidy_axes(axa, grid_axis="x")
    axa.legend(loc="lower left", handletextpad=0.4, borderaxespad=0.3,
               labelspacing=0.3)
    S.panel_tag(axa, "(a)", dx=-0.30, dy=1.13)
    axa.set_title("Matched fold–seed cells", fontsize=S.FS_LABEL,
                  color=S.INK, pad=5)

    # the single cell where K1 is lower, marked without relying on colour
    worst = int(np.argmin(delta))
    axa.annotate("only cell\nfavouring Full-S1",
                 xy=(k1[worst], y[worst]), xytext=(0.864, y[worst] - 1.05),
                 fontsize=S.FS_SMALL, color=S.MUTED, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=S.MUTED, linewidth=0.5,
                                 shrinkA=1, shrinkB=3))

    # ------------------------------------------- (b) paired differences vs NI
    axb.axvspan(margin, 0, color="#f0f0f0", zorder=0)
    axb.axvline(0, color="#9a9a9a", linewidth=0.7, zorder=1)
    axb.axvline(margin, color=S.INK, linewidth=0.9, linestyle=(0, (3, 2)),
                zorder=2)

    axb.scatter(delta, y, s=22, color=S.K1, marker="s", zorder=3,
                edgecolor="white", linewidth=0.5)
    axb.set_yticks(y)
    axb.set_yticklabels([])
    axb.tick_params(axis="y", length=0)
    axb.set_ylim(-1.45, len(cells) - 0.35)
    axb.set_xlim(-0.028, 0.050)
    axb.set_xlabel("Paired $\\Delta$ Macro-4 Macro-F1  (K1 $-$ Full-S1)")
    S.tidy_axes(axb, grid_axis="x")
    S.panel_tag(axb, "(b)", dx=-0.10, dy=1.13)
    axb.set_title("Paired difference vs. margin", fontsize=S.FS_LABEL,
                  color=S.INK, pad=5)

    axb.text(margin - 0.0016, (len(cells) - 1) / 2.0,
             f"non-inferiority margin {margin:+.2f} (Macro-F1)",
             fontsize=S.FS_SMALL, color=S.INK, ha="center", va="center",
             rotation=90)

    # mean +/- SD of the paired difference
    axb.errorbar(paired["mean"], -0.62, xerr=paired["sd"], fmt="D",
                 color=S.INK, markersize=4, capsize=2.5, linewidth=0.9,
                 zorder=4)
    axb.text(paired["mean"], -0.95,
             f"mean {paired['mean']:+.4f} $\\pm$ {paired['sd']:.4f}",
             fontsize=S.FS_SMALL, color=S.INK, ha="center", va="top")

    # ------------------------------------------------- aggregate summary band
    summary = (
        f"Macro-4 Macro-F1   Full-S1 {f1_full['mean']:.4f} $\\pm$ {f1_full['sd']:.4f}"
        f"   ·   K1 {f1_k1['mean']:.4f} $\\pm$ {f1_k1['sd']:.4f}"
        f"   ·   $\\Delta$ {paired['mean']:+.4f} $\\pm$ {paired['sd']:.4f}"
        f"   ·   win/tie/loss {paired['n_positive']}/{paired['n_ties']}/{paired['n_negative']}"
        f"   ·   exact one-sided $p$ = {ni['p_one_sided']:.9f}\n"
        f"Macro-4 Macro-AUC (descriptive; no margin defined)   "
        f"Full-S1 {auc_full['mean']:.4f} $\\pm$ {auc_full['sd']:.4f}"
        f"   ·   K1 {auc_k1['mean']:.4f} $\\pm$ {auc_k1['sd']:.4f}"
        f"   ·   $\\Delta$ {d_auc.mean():+.4f} $\\pm$ {d_auc.std(ddof=1):.4f}"
    )
    fig.text(0.5, -0.115, summary, ha="center", va="top",
             fontsize=S.FS_SMALL, color=S.INK, linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7f7f7",
                       edgecolor=S.PANEL_EDGE, linewidth=0.5))

    S.save_figure(fig, "fig02_matched_cells")
    print("figure 2 written")


if __name__ == "__main__":
    build()
