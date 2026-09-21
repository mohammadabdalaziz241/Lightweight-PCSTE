#!/usr/bin/env python3
"""Figure 6 — secondary Q8(K1) deployment extension.

Deliberately subordinate to Figure 5: a shorter figure, a shared legend, and a
caveat strip that keeps the representation boundary visible.

(a) predictive parity of the weight-only simulated Q8 representation with K1
    FP32 on both Macro-4 endpoints;
(b) stored-artifact size for Full-S1, K1 and Q8(K1);
(c) the modest CPU runtime benefit, against a parity line.

Panel (c) uses the within-run K1-vs-Q8 measurements from the Q8 benchmark. The
chained Full-S1 -> Q8 ratio recorded in the frozen artifacts is derived from the
superseded efficiency_v1 timings and is deliberately not plotted.

Run:  python scripts/publication_figures/fig06_q8.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pubfig_style as S


def build() -> None:
    S.apply_style()
    size = S.load_q8_size()
    aggr = S.load_q8_aggregate()
    q8lat = S.load_q8_latency()

    fig, (axa, axb, axc) = plt.subplots(
        1, 3, figsize=(S.TEXT_WIDTH_IN, 2.30),
        gridspec_kw={"wspace": 0.78})

    # ------------------------------------------------ (a) predictive parity
    endpoints = [("macro_4_macro_f1", "Macro-F1"),
                 ("macro_4_macro_auc", "Macro-AUC")]
    y = np.arange(len(endpoints))[::-1]
    for yi, (key, _) in zip(y, endpoints):
        blk = aggr[key]
        axa.errorbar(blk["k1"]["mean"], yi + 0.14, xerr=blk["k1"]["sd"],
                     fmt="s", color=S.K1, markersize=4.4, capsize=2.2,
                     linewidth=0.9, markeredgecolor="white",
                     markeredgewidth=0.4, zorder=3)
        axa.errorbar(blk["q8"]["mean"], yi - 0.14, xerr=blk["q8"]["sd"],
                     fmt="^", color=S.Q8, markersize=4.6, capsize=2.2,
                     linewidth=0.9, markeredgecolor="white",
                     markeredgewidth=0.4, zorder=3)
        d = blk["paired_delta_q8_minus_k1"]["mean"]
        axa.text(0.04, yi - 0.40, f"$\\Delta$ = {d:+.1e}",
                 transform=axa.get_yaxis_transform(),
                 fontsize=S.FS_SMALL, color=S.MUTED, ha="left", va="center")
    axa.set_yticks(y)
    axa.set_yticklabels([lbl for _, lbl in endpoints])
    axa.set_ylim(-0.62, len(endpoints) - 0.38)
    axa.set_xlim(0.925, 1.005)
    axa.set_xticks([0.94, 0.96, 0.98, 1.00])
    axa.set_xlabel("Macro-4 value\n(mean $\\pm$ SD)")
    axa.set_title("Predictive parity", fontsize=S.FS_LABEL, color=S.INK, pad=5)
    S.tidy_axes(axa, grid_axis="x")

    # ------------------------------------------------------- (b) stored size
    names = ["Full-S1\nFP32", "K1\nFP32", "Q8(K1)\nint8"]
    byts = [size["full_s1_fp32_state_dict_bytes"],
            size["k1_fp32_state_dict_bytes"],
            size["q8_compact_int8_state_bytes"]]
    mib = [b / S.MIB for b in byts]
    colors = [S.FULL_S1, S.K1, S.Q8]
    hatches = ["", "///", "xxx"]
    x = np.arange(len(names))
    axb.bar(x, mib, width=0.58, color=colors, edgecolor="white",
            linewidth=0.5, zorder=2)
    for xi, (bar, val) in enumerate(zip(axb.patches, mib)):
        bar.set_hatch(hatches[xi])
        axb.text(xi, val + 0.25, f"{val:.3f}", ha="center", va="bottom",
                 fontsize=S.FS_SMALL, color=S.INK)
    axb.set_xticks(x)
    axb.set_xticklabels(names)
    axb.set_xlim(-0.65, len(names) - 0.35)
    axb.set_ylim(0, max(mib) * 1.42)
    axb.set_ylabel("Stored state (MiB)")
    axb.set_title("Storage", fontsize=S.FS_LABEL, color=S.INK, pad=5)
    S.tidy_axes(axb, grid_axis="y")
    axb.text(2.0, max(mib) * 0.74,
             f"$-${100 * size['reduction_q8_vs_k1_fp32']:.2f}%\nvs K1\n"
             f"$-${100 * size['reduction_q8_vs_full_s1_fp32']:.2f}%\nvs Full-S1",
             fontsize=S.FS_SMALL, color="#2f6f5e", ha="center", va="center",
             linespacing=1.35)

    # ------------------------------------------------ (c) CPU runtime benefit
    im = q8lat["latency"]["inference_mode"]
    thr = q8lat["throughput_cpu_b32"]
    ratios = [
        ("model-only", im["model_only"]["speedup_k1_over_q8"]),
        ("end-to-end", im["end_to_end"]["speedup_k1_over_q8"]),
        ("batch-32 thr.", thr["speedup_q8_over_k1"]),
    ]
    yr = np.arange(len(ratios))[::-1]
    axc.axvline(1.0, color=S.INK, linewidth=0.8, linestyle=(0, (3, 2)), zorder=3)
    axc.barh(yr, [r for _, r in ratios], height=0.46, color=S.Q8, hatch="xxx",
             edgecolor="white", linewidth=0.4, zorder=2)
    for yi, (_, r) in zip(yr, ratios):
        axc.text(r + 0.012, yi, f"{r:.3f}$\\times$", va="center", ha="left",
                 fontsize=S.FS_SMALL, color=S.INK)
    axc.set_yticks(yr)
    axc.set_yticklabels([lbl for lbl, _ in ratios])
    axc.set_ylim(-0.62, len(ratios) - 0.38)
    axc.set_xlim(0.9, 1.20)
    axc.set_xticks([0.9, 1.0, 1.1, 1.2])
    axc.set_xlabel("Q8 gain over K1 ($\\times$)\nCPU only")
    axc.set_title("CPU runtime gain", fontsize=S.FS_LABEL, color=S.INK, pad=5)
    S.tidy_axes(axc, grid_axis="x")

    for ax, tag in zip((axa, axb, axc), ("(a)", "(b)", "(c)")):
        S.panel_tag(ax, tag, dx=-0.34, dy=1.16)

    # ------------------------------------------------------- shared legend
    handles = [
        plt.Line2D([], [], color=S.FULL_S1, marker="o", linestyle="none",
                   markersize=4.6, label="Full-S1"),
        plt.Line2D([], [], color=S.K1, marker="s", linestyle="none",
                   markersize=4.4, label="K1 FP32"),
        plt.Line2D([], [], color=S.Q8, marker="^", linestyle="none",
                   markersize=4.6, label="Q8(K1) simulated"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.13), handletextpad=0.4,
               columnspacing=1.8, frameon=False)

    # ------------------------------------------------------- caveat strip
    frac = 100 * size["fraction_params_int8"]
    caveat = (
        "Secondary extension.  Post-training per-output-channel symmetric int8 "
        f"weights; {frac:.2f}% of parameters stored as int8; CPU torch.ao dynamic\n"
        "backend; no true CUDA int8 path.  The Q8 non-inferiority result applies to "
        "the weight-only simulated representation, not directly to the\n"
        "deployed cpu_dynamic artifact."
    )
    fig.text(0.5, -0.22, caveat, ha="center", va="top", fontsize=S.FS_SMALL,
             color=S.INK, linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#f4faf8",
                       edgecolor=S.Q8, linewidth=0.6))

    S.save_figure(fig, "fig06_q8_extension")
    print("figure 6 written")


if __name__ == "__main__":
    build()
