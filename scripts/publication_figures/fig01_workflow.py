#!/usr/bin/env python3
"""Figure 1 — study workflow overview.

A schematic of the frozen study pipeline, from the four public datasets through
Full-S1, the same-fold three-seed teacher ensemble, structural reduction and
ensemble distillation to K1, and on to the sealed evaluation and the corrected
efficiency benchmark. The secondary Q8(K1) extension is drawn deliberately as a
subordinate, dashed branch.

The figure contains no measured results: only the architectural constants and
protocol facts that are already stated in the manuscript. Parameter counts are
read from the frozen efficiency artifact rather than typed in.

Run:  python scripts/publication_figures/fig01_workflow.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import pubfig_style as S

# ---------------------------------------------------------------- geometry
FIG_W, FIG_H = S.TEXT_WIDTH_IN, 4.30
MARGIN = 0.10
BAND_Y = {1: 3.72, 2: 2.60, 3: 1.48}
BOX_H = 0.70
Q8_Y, Q8_H = 0.52, 0.50

NEUTRAL_FACE = "#f4f4f4"
NEUTRAL_EDGE = "#bdbdbd"


def tint(hex_color: str, amount: float) -> str:
    """Blend a hue toward white; used for soft box fills."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def box(ax, x, y, w, h, text, *, face=NEUTRAL_FACE, edge=NEUTRAL_EDGE,
        weight="normal", fs=S.FS_SMALL, dashed=False, text_color=None):
    ax.add_patch(FancyBboxPatch(
        (x, y - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.06",
        linewidth=0.7, facecolor=face, edgecolor=edge,
        linestyle=(0, (2.5, 1.8)) if dashed else "solid", zorder=2))
    ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=fs,
            color=text_color or S.INK, fontweight=weight, linespacing=1.45,
            zorder=3)


def arrow(ax, x0, y0, x1, y1, *, dashed=False, color=None, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=8, linewidth=0.8,
        color=color or "#8a8a8a", zorder=1,
        linestyle=(0, (2.5, 1.8)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0, shrinkB=0))


def build() -> None:
    S.apply_style()
    eff = S.load_efficiency()["not_recomputed_referenced_from_frozen_artifacts"]
    p_full = eff["parameters"]["encoder"]["Full-S1"]
    p_k1 = eff["parameters"]["encoder"]["K1"]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    usable = FIG_W - 2 * MARGIN

    # ---------------- band 1: data -> representation -> Full-S1 (left to right)
    gap1, n1 = 0.34, 3
    w1 = (usable - (n1 - 1) * gap1) / n1
    x1 = [MARGIN + i * (w1 + gap1) for i in range(n1)]
    box(ax, x1[0], BAND_Y[1], w1, BOX_H,
        "Four public datasets\nCWRU · JNU · HIT · MaFaulDa")
    box(ax, x1[1], BAND_Y[1], w1, BOX_H,
        "Shared STFT\nspectro-temporal representation")
    box(ax, x1[2], BAND_Y[1], w1, BOX_H,
        f"Full-S1 shared encoder\n{p_full:,} encoder parameters",
        face=tint(S.FULL_S1, 0.86), edge=S.FULL_S1, weight="bold")
    for i in range(n1 - 1):
        arrow(ax, x1[i] + w1, BAND_Y[1], x1[i + 1], BAND_Y[1])

    # turn down at the right edge
    arrow(ax, x1[2] + w1 / 2, BAND_Y[1] - BOX_H / 2,
          x1[2] + w1 / 2, BAND_Y[2] + BOX_H / 2)

    # ---------------- band 2: ensemble -> reduction -> KD -> K1 (right to left)
    gap2, n2 = 0.30, 4
    w2 = (usable - (n2 - 1) * gap2) / n2
    x2 = [MARGIN + i * (w2 + gap2) for i in range(n2)]
    box(ax, x2[3], BAND_Y[2], w2, BOX_H,
        "Same-fold 3-seed\nFull-S1 teacher ensemble")
    box(ax, x2[2], BAND_Y[2], w2, BOX_H,
        "Structural reduction\nforward-only half_4x1")
    box(ax, x2[1], BAND_Y[2], w2, BOX_H,
        "Ensemble KD\nT = 4, α = 0.5, λ$_\\mathrm{rel}$ = 1.0")
    box(ax, x2[0], BAND_Y[2], w2, BOX_H,
        f"K1 student\n{p_k1:,} encoder parameters",
        face=tint(S.K1, 0.86), edge=S.K1, weight="bold")
    for i in (3, 2, 1):  # flow runs right to left along this band
        arrow(ax, x2[i], BAND_Y[2], x2[i - 1] + w2, BAND_Y[2])

    # turn down at the left edge
    arrow(ax, x2[0] + w2 / 2, BAND_Y[2] - BOX_H / 2,
          x2[0] + w2 / 2, BAND_Y[3] + BOX_H / 2)

    # ---------------- band 3: selection -> frozen plan -> TEST -> efficiency
    gap3, n3 = 0.30, 4
    w3 = (usable - (n3 - 1) * gap3) / n3
    x3 = [MARGIN + i * (w3 + gap3) for i in range(n3)]
    box(ax, x3[0], BAND_Y[3], w3, BOX_H,
        "Validation-only\ncheckpoint selection")
    box(ax, x3[1], BAND_Y[3], w3, BOX_H,
        "Frozen evaluation plan\n3 folds × 3 seeds = 9 matched cells")
    box(ax, x3[2], BAND_Y[3], w3, BOX_H,
        "Sealed TEST\nopened only after freeze", weight="bold")
    box(ax, x3[3], BAND_Y[3], w3, BOX_H,
        "Corrected efficiency\nbenchmark")
    for i in range(n3 - 1):
        arrow(ax, x3[i] + w3, BAND_Y[3], x3[i + 1], BAND_Y[3])

    # ---------------- secondary Q8 branch (dashed, visually subordinate)
    q8_w = usable * 0.62
    q8_x = FIG_W - MARGIN - q8_w
    box(ax, q8_x, Q8_Y, q8_w, Q8_H,
        "Secondary extension: post-training Q8(K1)\n"
        "derived from the frozen K1 checkpoints; storage-oriented",
        face=tint(S.Q8, 0.90), edge=S.Q8, dashed=True,
        text_color="#2f6f5e", fs=S.FS_SMALL)
    arrow(ax, x3[3] + w3 / 2, BAND_Y[3] - BOX_H / 2,
          q8_x + q8_w * 0.72, Q8_Y + Q8_H / 2, dashed=True, color=S.Q8)
    ax.text(q8_x - 0.08, Q8_Y, "secondary", ha="right", va="center",
            fontsize=S.FS_SMALL, color=S.MUTED, style="italic")

    S.save_figure(fig, "fig01_workflow_overview")
    print("figure 1 written")


if __name__ == "__main__":
    build()
