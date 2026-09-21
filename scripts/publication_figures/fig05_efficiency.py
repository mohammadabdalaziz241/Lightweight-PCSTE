#!/usr/bin/env python3
"""Figure 5 — computational efficiency of Full-S1 versus K1.

Four panels, each on a single axis with one unit:

(a) static resource cost expressed as a percentage of Full-S1 (parameters,
    serialized FP32 state, Macro-4 FLOPs, GPU peak memory);
(b) absolute batch-1 latency in ms for the four device/scope combinations;
(c) absolute batch-32 throughput in windows/s;
(d) the K1-over-Full-S1 ratio for every runtime measurement, against a
    reference line at 1.0.

Every latency and throughput value comes from the corrected inference-mode
benchmark (`efficiency_latency_correction_v1`). The superseded
`efficiency_v1` absolute timings are never read by this script.

Run:  python scripts/publication_figures/fig05_efficiency.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pubfig_style as S

BAR_H = 0.34


def paired_barh(ax, labels, full_vals, k1_vals, *, fmt, xlabel, title,
                annot_pad=0.02):
    """Grouped horizontal bars: Full-S1 above, K1 below, values labelled."""
    y = np.arange(len(labels))[::-1]
    ax.barh(y + BAR_H / 2, full_vals, height=BAR_H, color=S.FULL_S1,
            edgecolor="white", linewidth=0.4, label="Full-S1", zorder=2)
    ax.barh(y - BAR_H / 2, k1_vals, height=BAR_H, color=S.K1, hatch="///",
            edgecolor="white", linewidth=0.4, label="K1", zorder=2)

    span = max(max(full_vals), max(k1_vals))
    for yi, fv, kv in zip(y, full_vals, k1_vals):
        ax.text(fv + span * annot_pad, yi + BAR_H / 2, fmt.format(fv),
                va="center", ha="left", fontsize=S.FS_SMALL, color=S.INK)
        ax.text(kv + span * annot_pad, yi - BAR_H / 2, fmt.format(kv),
                va="center", ha="left", fontsize=S.FS_SMALL, color=S.INK)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.62, len(labels) - 0.38)
    ax.set_xlim(0, span * 1.34)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=S.FS_LABEL, color=S.INK, pad=5)
    S.tidy_axes(ax, grid_axis="x")


def build() -> None:
    S.apply_style()
    eff = S.load_efficiency()
    frozen = eff["not_recomputed_referenced_from_frozen_artifacts"]
    lat = eff["corrected_latency"]
    thr = eff["corrected_throughput"]

    p_full = frozen["parameters"]["encoder"]["Full-S1"]
    p_k1 = frozen["parameters"]["encoder"]["K1"]
    b_full = frozen["parameters"]["serialized_state_dict_bytes"]["Full-S1"]
    b_k1 = frozen["parameters"]["serialized_state_dict_bytes"]["K1"]
    g_full = frozen["flops"]["macro4_total_gflops"]["Full-S1"]
    g_k1 = frozen["flops"]["macro4_total_gflops"]["K1"]
    m_full = frozen["gpu_peak_memory"]["macro4_peak_bytes"]["Full-S1"]
    m_k1 = frozen["gpu_peak_memory"]["macro4_peak_bytes"]["K1"]

    fig, axes = plt.subplots(2, 2, figsize=(S.TEXT_WIDTH_IN, 4.10),
                             gridspec_kw={"wspace": 0.52, "hspace": 0.62})
    (axa, axb), (axc, axd) = axes

    # ------------------------------------------- (a) relative resource cost
    rows = [
        ("Encoder parameters", p_k1 / p_full, f"{p_full:,} → {p_k1:,}"),
        ("FP32 state-dict size", b_k1 / b_full,
         f"{b_full / S.MIB:.3f} → {b_k1 / S.MIB:.3f} MiB"),
        ("Macro-4 FLOPs", g_k1 / g_full, f"{g_full:.4f} → {g_k1:.4f} GFLOP"),
        ("GPU peak memory", m_k1 / m_full,
         f"{m_full / S.MIB:.3f} → {m_k1 / S.MIB:.3f} MiB"),
    ]
    y = np.arange(len(rows))[::-1]
    ax = axa
    ax.barh(y, [100.0] * len(rows), height=0.46, color="#e2e2e2",
            edgecolor="white", linewidth=0.4, zorder=1)
    ax.barh(y, [100 * r[1] for r in rows], height=0.46, color=S.K1,
            edgecolor="white", linewidth=0.4, zorder=2)
    for yi, (_, ratio, detail) in zip(y, rows):
        ax.text(100 * ratio - 1.5, yi, f"{100 * ratio:.1f}%", va="center",
                ha="right", fontsize=S.FS_SMALL, color="white",
                fontweight="bold", zorder=3)
        ax.text(101.5, yi, f"$-${100 * (1 - ratio):.2f}%", va="center",
                ha="left", fontsize=S.FS_SMALL, color=S.INK)
        ax.text(2.0, yi - 0.40, detail, va="center", ha="left",
                fontsize=S.FS_SMALL - 0.5, color=S.MUTED, zorder=4,
                bbox=dict(boxstyle="square,pad=0.12", facecolor="white",
                          edgecolor="none"))
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.75, len(rows) - 0.3)
    ax.set_xlim(0, 132)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("K1 resource cost  (% of Full-S1)")
    ax.set_title("Static cost", fontsize=S.FS_LABEL, color=S.INK, pad=5)
    S.tidy_axes(ax, grid_axis="x")

    # ------------------------------------------------- (b) batch-1 latency
    scopes = [("cpu|model_only", "CPU, model-only"),
              ("cpu|end_to_end", "CPU, end-to-end"),
              ("cuda|model_only", "GPU, model-only"),
              ("cuda|end_to_end", "GPU, end-to-end")]
    paired_barh(
        axb, [lbl for _, lbl in scopes],
        [lat[k]["full_s1_macro4_median_ms"] for k, _ in scopes],
        [lat[k]["k1_macro4_median_ms"] for k, _ in scopes],
        fmt="{:.2f}", xlabel="Batch-1 latency (ms)", title="Latency")
    axb.legend(loc="lower right", handlelength=1.2, handletextpad=0.4,
               labelspacing=0.25, borderaxespad=0.3)

    # ---------------------------------------------- (c) batch-32 throughput
    paired_barh(
        axc, ["GPU, batch 32", "CPU, batch 32"],
        [thr["cuda"]["full_s1_macro4_win_s"], thr["cpu"]["full_s1_macro4_win_s"]],
        [thr["cuda"]["k1_macro4_win_s"], thr["cpu"]["k1_macro4_win_s"]],
        fmt="{:.2f}", xlabel="Throughput (windows/s)", title="Throughput")

    # ------------------------------------------------ (d) speed-up summary
    ratios = [
        ("CPU, model-only", lat["cpu|model_only"]["speedup_full_over_k1"]),
        ("CPU, end-to-end", lat["cpu|end_to_end"]["speedup_full_over_k1"]),
        ("GPU, model-only", lat["cuda|model_only"]["speedup_full_over_k1"]),
        ("GPU, end-to-end", lat["cuda|end_to_end"]["speedup_full_over_k1"]),
        ("GPU, batch-32 thr.", thr["cuda"]["ratio_k1_over_full"]),
        ("CPU, batch-32 thr.", thr["cpu"]["ratio_k1_over_full"]),
    ]
    y = np.arange(len(ratios))[::-1]
    axd.axvline(1.0, color=S.INK, linewidth=0.8, linestyle=(0, (3, 2)), zorder=3)
    axd.barh(y, [r for _, r in ratios], height=0.5, color=S.K1, hatch="///",
             edgecolor="white", linewidth=0.4, zorder=2)
    for yi, (_, r) in zip(y, ratios):
        axd.text(r + 0.03, yi, f"{r:.3f}$\\times$", va="center", ha="left",
                 fontsize=S.FS_SMALL, color=S.INK)
    axd.set_yticks(y)
    axd.set_yticklabels([lbl for lbl, _ in ratios])
    axd.set_ylim(-0.6, len(ratios) - 0.4)
    axd.set_xlim(0, 2.45)
    axd.set_xlabel("K1 advantage over Full-S1 ($\\times$)")
    axd.set_title("Runtime gain", fontsize=S.FS_LABEL, color=S.INK, pad=5)
    S.tidy_axes(axd, grid_axis="x")
    axd.text(1.0, -0.52, "parity", fontsize=S.FS_SMALL,
             color=S.MUTED, ha="center", va="bottom")

    for ax, tag in zip((axa, axb, axc, axd), ("(a)", "(b)", "(c)", "(d)")):
        S.panel_tag(ax, tag, dx=-0.30, dy=1.22)

    # NB: the corrected-benchmark / pure-PyTorch caveat lives in the LaTeX
    # caption (figures/fig05_efficiency.tex) and is not repeated here.

    S.save_figure(fig, "fig05_efficiency")
    print("figure 5 written")


if __name__ == "__main__":
    build()
