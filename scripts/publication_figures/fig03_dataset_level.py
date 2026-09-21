#!/usr/bin/env python3
"""Figure 3 — dataset-level Macro-F1 and Macro-AUC.

Paired dot-and-error-bar panels: mean across the nine matched cells with sample
SD whiskers, for Full-S1 and K1 on each dataset. Panel (a) is Macro-F1 and
panel (b) Macro-AUC.

Dots rather than bars: both endpoints sit far from zero, and a bar chart on a
truncated axis would exaggerate the differences. The paired difference is
printed beside each dataset so the small HIT decrease stays visible and signed
rather than being lost in the marker overlap.

All values are read from the frozen PER_DATASET_SUMMARY.csv.

Run:  python scripts/publication_figures/fig03_dataset_level.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

import pubfig_style as S

#: frozen key -> display name
DATASETS = [("CWRU", "CWRU"), ("JNU", "JNU"), ("HIT", "HIT"),
            ("MAFAULDA", "MaFaulDa")]
OFFSET = 0.16


def panel(ax, data, metric, title, xlim, xlabel):
    y = np.arange(len(DATASETS))[::-1]
    for yi, (key, _) in zip(y, DATASETS):
        m = data[key][metric]
        ax.plot([m["full_s1_mean"], m["k1_mean"]], [yi + OFFSET, yi - OFFSET],
                color="#c4c4c4", linewidth=0.8, zorder=1)
        ax.errorbar(m["full_s1_mean"], yi + OFFSET, xerr=m["full_s1_sd"],
                    fmt="o", color=S.FULL_S1, markersize=4.6, capsize=2.2,
                    linewidth=0.9, zorder=3, markeredgecolor="white",
                    markeredgewidth=0.4)
        ax.errorbar(m["k1_mean"], yi - OFFSET, xerr=m["k1_sd"],
                    fmt="s", color=S.K1, markersize=4.4, capsize=2.2,
                    linewidth=0.9, zorder=3, markeredgecolor="white",
                    markeredgewidth=0.4)
        # signed paired difference in a column outside the plot area, so the
        # HIT decrease stays explicit without colliding with the whiskers
        delta = m["delta_mean"]
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(1.04, yi, f"$\\Delta$ {delta:+.4f}", transform=trans,
                fontsize=S.FS_SMALL,
                color=S.K1 if delta > 0 else S.FULL_S1,
                ha="left", va="center",
                fontweight="bold" if delta < 0 else "normal", clip_on=False)

    ax.set_yticks(y)
    ax.set_yticklabels([name for _, name in DATASETS])
    ax.set_ylim(-0.6, len(DATASETS) - 0.4)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=S.FS_LABEL, color=S.INK, pad=5)
    S.tidy_axes(ax, grid_axis="x")


def build() -> None:
    S.apply_style()
    data = S.load_per_dataset()

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(S.TEXT_WIDTH_IN, 2.45),
        gridspec_kw={"wspace": 0.62})

    panel(axa, data, "macro_f1", "Macro-F1", (0.76, 1.03),
          "Dataset Macro-F1  (mean $\\pm$ SD over 9 cells)")
    panel(axb, data, "macro_auc", "Macro-AUC", (0.935, 1.008),
          "Dataset Macro-AUC  (mean $\\pm$ SD over 9 cells)")

    handles = [
        plt.Line2D([], [], color=S.FULL_S1, marker="o", linestyle="none",
                   markersize=4.6, label="Full-S1"),
        plt.Line2D([], [], color=S.K1, marker="s", linestyle="none",
                   markersize=4.4, label="K1"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.44, 1.10), handletextpad=0.4,
               columnspacing=1.8, frameon=False)

    S.panel_tag(axa, "(a)", dx=-0.22, dy=1.20)
    S.panel_tag(axb, "(b)", dx=-0.16, dy=1.20)

    # NB: the "HIT decreases / truncated axes" note lives in the LaTeX caption
    # (figures/fig03_dataset_level.tex) and is deliberately not repeated here.

    S.save_figure(fig, "fig03_dataset_level")
    print("figure 3 written")


if __name__ == "__main__":
    build()
