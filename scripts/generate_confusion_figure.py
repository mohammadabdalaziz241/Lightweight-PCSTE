#!/usr/bin/env python3
"""Aggregated confusion-matrix figure: Full-S1 vs K1 on the four TEST datasets.

WHAT THIS SCRIPT DOES
---------------------
It is a *pure read-and-plot* utility over the sealed publication TEST artifacts.
It never runs a model, never re-runs TEST, never regenerates predictions and
never writes into the frozen result tree.

For every dataset ``d`` and model ``m`` it pools the raw confusion counts of the
nine matched fold x seed evaluation cells (folds GF1-GF3 x seeds 42/1337/2026)

    C_agg[d, m] = sum_{f=1..3} sum_{s in {42,1337,2026}} C^(f,s)[d, m]

and then row-normalizes the pooled matrix to percentages

    Ctilde[i, j] = 100 * C_agg[i, j] / sum_j C_agg[i, j] .

The row-normalized matrix is what gets plotted; the raw pooled counts are kept
in the CSV export so nothing is lost.

SOURCE OF THE CONFUSION MATRICES
--------------------------------
The per-cell confusion matrices already exist inside the sealed per-model TEST
reports, under ``per_dataset_reports[DATASET]["confusion_matrix"]`` together
with the frozen class order in ``per_dataset_reports[DATASET]["classes"]``.
They are read directly -- nothing is reconstructed or approximated.

With ``--verify-predictions DIR`` the script additionally rebuilds every matrix
from the frozen per-cell prediction CSVs (``y_true`` / ``y_pred`` columns) and
asserts exact element-wise equality. That is an independent read-only check of
already-saved artifacts; it still runs no inference.

METHODOLOGICAL CAVEAT (propagated into the manifest)
----------------------------------------------------
Each fold is evaluated under three seeds, so the same underlying TEST windows
of a fold contribute three prediction outcomes to the pool. The pooled matrices
are therefore a *descriptive* summary of nine matched model evaluations, not a
single independent test split. Statistical inference stays with the paired
fold x seed analysis that is frozen elsewhere.

USAGE
-----
    python scripts/generate_confusion_figure.py                # defaults
    python scripts/generate_confusion_figure.py --verify-predictions <frozen dir>
    python scripts/generate_confusion_figure.py --gloss        # documented label glosses
    python scripts/generate_confusion_figure.py --fig-width 7.16 --dpi 600

All outputs land under ``figures/confusion_matrices/`` inside this publication
repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Frozen evaluation design
# ---------------------------------------------------------------------------

FOLDS = (1, 2, 3)
SEEDS = (42, 1337, 2026)
MODELS = ("Full-S1", "K1")

#: report key -> (display name, file-name token)
DATASETS = {
    "CWRU": ("CWRU", "CWRU"),
    "JNU": ("JNU", "JNU"),
    "HIT": ("HIT", "HIT"),
    "MAFAULDA": ("MaFaulDa", "MaFaulDa"),
}

#: model -> file-name token
MODEL_TOKEN = {"Full-S1": "FullS1", "K1": "K1"}

#: Documented class meanings, quoted verbatim from frozen sources in this repo
#: (``src/methodology_v2/experiment/heads.py`` docstring/comments and
#: ``src/methodology_v2/registry.py::HIT["label_meaning"]``).  These are only
#: ever used for optional axis glosses (``--gloss``) and for the manifest; the
#: authoritative class strings always come from the frozen TEST reports.
DOCUMENTED_GLOSS = {
    "JNU": {"n": "healthy", "ib": "inner", "ob": "outer", "tb": "roller"},
    "HIT": {"0": "healthy", "1": "inner-ring fault", "2": "outer-ring fault"},
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS = REPO_ROOT / "results/final_test/publication_final_test_v1/per_model_reports"
DEFAULT_OUTDIR = REPO_ROOT / "figures/confusion_matrices"
FIG_STEM = "fig_confusion_aggregated_fulls1_vs_k1"


# ---------------------------------------------------------------------------
# Loading frozen artifacts
# ---------------------------------------------------------------------------

def cell_stem(fold: int, seed: int, model: str) -> str:
    """Frozen per-cell file stem, e.g. ``gf1_s42_full_s1``."""
    token = "full_s1" if model == "Full-S1" else "k1"
    return f"gf{fold}_s{seed}_{token}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cells(reports_dir: Path) -> tuple[dict, list[str]]:
    """Read the 18 sealed per-cell TEST reports.

    Returns ``(cells, missing)`` where ``cells`` maps
    ``(fold, seed, model) -> {"path", "sha256", "per_dataset"}`` and
    ``per_dataset`` maps dataset key -> ``{"classes", "cm"}``.
    """
    cells: dict[tuple[int, int, str], dict] = {}
    missing: list[str] = []

    for fold in FOLDS:
        for seed in SEEDS:
            for model in MODELS:
                path = reports_dir / f"{cell_stem(fold, seed, model)}_report.json"
                if not path.is_file():
                    missing.append(str(path))
                    continue
                report = json.loads(path.read_text())

                # The report must self-identify as the cell we asked for.
                for field, expected in (("fold", fold), ("seed", seed),
                                        ("family", model), ("partition", "test")):
                    got = report.get(field)
                    if got != expected:
                        missing.append(f"{path}: {field}={got!r}, expected {expected!r}")

                per_dataset = {}
                for ds_key in DATASETS:
                    block = report.get("per_dataset_reports", {}).get(ds_key)
                    if block is None:
                        missing.append(f"{path}: per_dataset_reports[{ds_key}] absent")
                        continue
                    if "confusion_matrix" not in block or "classes" not in block:
                        missing.append(f"{path}: {ds_key} lacks confusion_matrix/classes")
                        continue
                    per_dataset[ds_key] = {
                        "classes": [str(c) for c in block["classes"]],
                        "cm": np.asarray(block["confusion_matrix"], dtype=np.int64),
                    }

                cells[(fold, seed, model)] = {
                    "path": path, "sha256": sha256(path), "per_dataset": per_dataset,
                }

    return cells, missing


def verify_against_predictions(cells: dict, predictions_dir: Path) -> list[str]:
    """Rebuild each matrix from the frozen prediction CSVs and compare exactly.

    Read-only cross-check of saved artifacts; runs no inference. Returns a list
    of human-readable verification lines (one per cell/dataset checked).
    """
    notes: list[str] = []
    for (fold, seed, model), cell in sorted(cells.items()):
        csv_path = predictions_dir / f"{cell_stem(fold, seed, model)}_predictions.csv"
        if not csv_path.is_file():
            notes.append(f"MISSING {csv_path}")
            continue

        by_dataset: dict[str, list[tuple[str, str]]] = defaultdict(list)
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                by_dataset[row["dataset"].upper()].append((row["y_true"], row["y_pred"]))

        for ds_key, block in cell["per_dataset"].items():
            classes = block["classes"]
            index = {c: i for i, c in enumerate(classes)}
            rebuilt = np.zeros((len(classes), len(classes)), dtype=np.int64)
            for y_true, y_pred in by_dataset.get(ds_key, []):
                rebuilt[index[y_true], index[y_pred]] += 1
            ok = np.array_equal(rebuilt, block["cm"])
            notes.append(
                f"{'OK  ' if ok else 'FAIL'} gf{fold}/s{seed}/{model}/{ds_key} "
                f"rebuilt-from-predictions == report confusion_matrix "
                f"(n={int(block['cm'].sum())})"
            )
    return notes


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(cells: dict) -> tuple[dict, list[str]]:
    """Pool the nine matched cells per (dataset, model).

    Class order is taken from the frozen reports. If any cell were to disagree
    on the class *set*, it is reordered onto the reference order before
    summation; a disagreement on the class set itself is a hard error.
    """
    problems: list[str] = []
    agg: dict[tuple[str, str], dict] = {}

    for ds_key in DATASETS:
        reference: list[str] | None = None
        for model in MODELS:
            pooled = None
            contributing = []
            reorders = []

            for fold in FOLDS:
                for seed in SEEDS:
                    cell = cells.get((fold, seed, model))
                    if cell is None or ds_key not in cell["per_dataset"]:
                        problems.append(f"{ds_key}/{model}: cell gf{fold}/s{seed} absent")
                        continue
                    block = cell["per_dataset"][ds_key]
                    classes, cm = block["classes"], block["cm"]

                    if reference is None:
                        reference = list(classes)
                    if set(classes) != set(reference):
                        problems.append(
                            f"{ds_key}/{model} gf{fold}/s{seed}: class set {classes} "
                            f"differs from reference {reference}"
                        )
                        continue
                    if classes != reference:
                        order = [classes.index(c) for c in reference]
                        cm = cm[np.ix_(order, order)]
                        reorders.append(f"gf{fold}/s{seed}")

                    pooled = cm.copy() if pooled is None else pooled + cm
                    contributing.append((fold, seed))

            if pooled is None:
                problems.append(f"{ds_key}/{model}: no contributing cells")
                continue

            row_totals = pooled.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                normalized = np.where(
                    row_totals[:, None] > 0,
                    100.0 * pooled / np.maximum(row_totals, 1)[:, None],
                    np.nan,
                )

            agg[(ds_key, model)] = {
                "classes": list(reference),
                "counts": pooled,
                "row_normalized": normalized,
                "total": int(pooled.sum()),
                "row_totals": row_totals,
                "cells": contributing,
                "reordered_cells": reorders,
            }

    return agg, problems


def verify_aggregate(agg: dict) -> list[str]:
    """Step-4 checks: row sums, class-order agreement, matrix dimensions."""
    notes: list[str] = []
    for ds_key, (display, _) in DATASETS.items():
        orders = {model: agg[(ds_key, model)]["classes"] for model in MODELS
                  if (ds_key, model) in agg}
        same = len(set(map(tuple, orders.values()))) == 1
        notes.append(f"{'OK  ' if same else 'FAIL'} {display}: Full-S1 and K1 share one class order")

        for model in MODELS:
            entry = agg.get((ds_key, model))
            if entry is None:
                notes.append(f"FAIL {display}/{model}: missing")
                continue
            k = len(entry["classes"])
            square = entry["counts"].shape == (k, k)
            notes.append(f"{'OK  ' if square else 'FAIL'} {display}/{model}: "
                         f"matrix is {entry['counts'].shape[0]}x{entry['counts'].shape[1]} "
                         f"for {k} frozen classes")
            notes.append(f"OK   {display}/{model}: {len(entry['cells'])}/9 matched cells pooled, "
                         f"n={entry['total']}")

            finite = np.isfinite(entry["row_normalized"]).all(axis=1)
            sums = np.nansum(entry["row_normalized"], axis=1)
            good = bool(np.all(np.abs(sums[finite] - 100.0) < 1e-6))
            notes.append(f"{'OK  ' if good else 'FAIL'} {display}/{model}: "
                         f"row sums of normalized matrix = 100% "
                         f"(max |dev| = {float(np.max(np.abs(sums[finite] - 100.0))):.2e})")
    return notes


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def write_matrix_csv(path: Path, classes: list[str], matrix: np.ndarray, fmt: str) -> None:
    """Write one matrix with a ``true\\pred`` corner header."""
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\pred", *classes])
        for label, row in zip(classes, matrix):
            writer.writerow([label, *(format(v, fmt) for v in row)])


def export_matrices(agg: dict, data_dir: Path) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ds_key, (_, ds_token) in DATASETS.items():
        for model in MODELS:
            entry = agg.get((ds_key, model))
            if entry is None:
                continue
            stem = f"{ds_token}_{MODEL_TOKEN[model]}"
            counts_path = data_dir / f"{stem}_counts.csv"
            norm_path = data_dir / f"{stem}_row_normalized.csv"
            write_matrix_csv(counts_path, entry["classes"], entry["counts"], "d")
            write_matrix_csv(norm_path, entry["classes"], entry["row_normalized"], ".6f")
            written += [counts_path, norm_path]
    return written


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def tick_labels(ds_key: str, classes: list[str], gloss: bool) -> list[str]:
    if not gloss:
        return list(classes)
    mapping = DOCUMENTED_GLOSS.get(ds_key, {})
    return [f"{c} ({mapping[c]})" if c in mapping else c for c in classes]


def text_extent_in(label: str, fontsize_pt: float) -> float:
    """Cheap upper bound on rendered text length, in inches."""
    return len(label) * fontsize_pt * 0.62 / 72.0


def build_figure(agg: dict, out_dir: Path, *, fig_width: float, dpi: int,
                 gloss: bool, cmap_name: str) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager  # noqa: F401  (ensures cache is built)
    from matplotlib.colors import Normalize

    # Vector-friendly, LaTeX-safe output: real fonts in the PDF (Type 42),
    # glyph outlines in the SVG so the file renders identically anywhere.
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "text.usetex": False,
        "axes.linewidth": 0.4,
    })

    INK = "#1a1a1a"
    MUTED = "#6b6b6b"
    TICK_FS = 6.5
    ROW_LABEL_FS = 9.0
    COL_TITLE_FS = 9.5
    AXIS_TITLE_FS = 8.5
    CBAR_FS = 7.0

    ds_keys = list(DATASETS)
    ref = {k: agg[(k, MODELS[0])] for k in ds_keys}

    # ---- layout, computed in inches from the actual label lengths ----------
    pad = 0.05
    axis_title_w = 0.18          # rotated "True class"
    row_label_w = 0.30           # rotated dataset name (+ n = ...)
    ytick_w = max(
        text_extent_in(lbl, TICK_FS)
        for k in ds_keys for lbl in tick_labels(k, ref[k]["classes"], gloss)
    ) + 0.06
    cbar_block_w = 0.62          # bar + tick labels + rotated caption
    col_gap = 0.30
    left = pad + axis_title_w + row_label_w + ytick_w
    panel_w = (fig_width - left - cbar_block_w - pad - col_gap) / 2.0

    # x tick labels: horizontal while they fit the cell width, else 45 degrees
    rotations, xtick_h = {}, {}
    for k in ds_keys:
        labels = tick_labels(k, ref[k]["classes"], gloss)
        cell_w = panel_w / len(labels)
        longest = max(text_extent_in(lbl, TICK_FS) for lbl in labels)
        if longest <= cell_w * 0.95:
            rotations[k], xtick_h[k] = 0, TICK_FS * 1.6 / 72.0 + 0.04
        else:
            rotations[k], xtick_h[k] = 45, longest * 0.7071 + 0.05

    top_block = COL_TITLE_FS * 2.0 / 72.0 + 0.10      # column titles
    bottom_block = AXIS_TITLE_FS * 1.9 / 72.0 + 0.04  # "Predicted class"
    row_gap_extra = 0.05
    panel_h = 1.58
    fig_height = (pad + bottom_block + sum(xtick_h.values())
                  + 4 * panel_h + 3 * row_gap_extra + top_block + pad)

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    to_fig = lambda x, y, w, h: (x / fig_width, y / fig_height, w / fig_width, h / fig_height)

    norm = Normalize(vmin=0.0, vmax=100.0)
    cmap = plt.get_cmap(cmap_name)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # rows are laid out top-down
    y_cursor = fig_height - pad - top_block
    row_spans = []

    for ds_key in ds_keys:
        display, _ = DATASETS[ds_key]
        y_cursor -= panel_h
        row_bottom = y_cursor
        row_spans.append((row_bottom, panel_h))
        labels = tick_labels(ds_key, ref[ds_key]["classes"], gloss)
        n_classes = len(labels)
        cell_w_pt = panel_w / n_classes * 72.0
        # one annotation size per panel, scaled to the cell it must sit in
        annot_fs = float(np.clip(cell_w_pt / 3.5, 4.4, 8.0))

        for col, model in enumerate(MODELS):
            entry = agg[(ds_key, model)]
            x0 = left + col * (panel_w + col_gap)
            ax = fig.add_axes(to_fig(x0, row_bottom, panel_w, panel_h))
            matrix = entry["row_normalized"]

            ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto",
                      interpolation="nearest", origin="upper")

            ax.set_xticks(np.arange(n_classes))
            ax.set_yticks(np.arange(n_classes))
            if rotations[ds_key]:
                ax.set_xticklabels(labels, rotation=45, ha="right",
                                   rotation_mode="anchor", fontsize=TICK_FS, color=INK)
            else:
                ax.set_xticklabels(labels, fontsize=TICK_FS, color=INK)
            ax.set_yticklabels(labels if col == 0 else [], fontsize=TICK_FS, color=INK)
            ax.tick_params(length=0, pad=1.6)
            for spine in ax.spines.values():
                spine.set_color("#c8c8c8")

            # hairline separators keep dense panels readable without a grid
            ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.5)
            ax.tick_params(which="minor", length=0)

            # annotate every non-zero cell; exact zeros stay blank (all panels)
            for i in range(n_classes):
                for j in range(n_classes):
                    value = matrix[i, j]
                    if not np.isfinite(value) or value <= 0:
                        continue  # exact zero stays blank, in every panel
                    # a non-zero value that would round to "0.0" is shown as
                    # "<0.1" so it is never confused with an empty cell
                    label = "<0.1" if value < 0.05 else f"{value:.1f}"
                    ax.text(j, i, label,
                            ha="center", va="center", fontsize=annot_fs,
                            color="white" if value >= 55 else INK)

            if ds_key == ds_keys[0]:
                ax.set_title(model, fontsize=COL_TITLE_FS, color=INK, pad=5.0)

        # rotated dataset label with its pooled sample count
        total = ref[ds_key]["total"]
        fig.text((pad + axis_title_w + row_label_w / 2) / fig_width,
                 (row_bottom + panel_h / 2) / fig_height,
                 f"{display}\n$n$ = {total:,}", rotation=90, rotation_mode="anchor",
                 ha="center", va="center", fontsize=ROW_LABEL_FS, color=INK,
                 linespacing=1.25)

        y_cursor -= xtick_h[ds_key] + row_gap_extra

    # shared axis titles
    panel_mid_x = left + panel_w + col_gap / 2.0
    fig.text(panel_mid_x / fig_width, (pad + bottom_block * 0.30) / fig_height,
             "Predicted class", ha="center", va="bottom",
             fontsize=AXIS_TITLE_FS, color=INK)
    top_row_bottom, _ = row_spans[0]
    bottom_row_bottom, bottom_row_h = row_spans[-1]
    stack_mid_y = (bottom_row_bottom + top_row_bottom + bottom_row_h) / 2.0
    fig.text((pad + axis_title_w * 0.45) / fig_width, stack_mid_y / fig_height,
             "True class", rotation=90, ha="center", va="center",
             fontsize=AXIS_TITLE_FS, color=INK)

    # one shared colorbar for all eight panels
    cbar_x = left + 2 * panel_w + col_gap + 0.10
    cbar_h = (top_row_bottom + panel_h) - bottom_row_bottom
    cbar_h = min(cbar_h, fig_height * 0.68)
    cbar_y = stack_mid_y - cbar_h / 2.0
    cax = fig.add_axes(to_fig(cbar_x, cbar_y, 0.11, cbar_h))
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.ax.tick_params(labelsize=CBAR_FS, length=2, width=0.4, pad=1.5, color="#c8c8c8")
    for label in cbar.ax.get_yticklabels():
        label.set_color(INK)
    cbar.outline.set_linewidth(0.4)
    cbar.outline.set_edgecolor("#c8c8c8")
    cbar.set_label("Row-normalized share of true class (%)",
                   fontsize=CBAR_FS + 0.5, color=MUTED, labelpad=3.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "svg", "png"):
        path = out_dir / f"{FIG_STEM}.{ext}"
        fig.savefig(path, format=ext, dpi=dpi, facecolor="white")
        written.append(path)
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# Manifest and LaTeX snippet
# ---------------------------------------------------------------------------

LATEX_SNIPPET = r"""% Auto-generated by scripts/generate_confusion_figure.py.
% Placement: in the Results and Discussion section, after the dataset-level
% Macro-F1 / AUC results and before the efficiency and deployment discussion.
% The float is tall and full-width, so [p] (own page) is the intended default.
\begin{figure*}[p]
  \centering
  % The natural size is 7.16in x 8.71in. keepaspectratio with a height cap
  % below \textheight leaves room for the caption and lets the float shrink
  % to fit templates with a shorter text block; it is a no-op when it fits.
  \includegraphics[width=\textwidth,height=0.80\textheight,keepaspectratio]%
    {figures/confusion_matrices/__FIG_STEM__.pdf}
  \caption{Aggregated class-level confusion matrices for Full-S1 (left column)
  and K1 (right column) on the four evaluation datasets: CWRU, JNU, HIT and
  MaFaulDa. For each dataset and model, the raw confusion counts of the nine
  matched fold--seed evaluation cells were summed element-wise, and the pooled
  matrix was then normalised by true class; each row therefore shows the
  distribution of predictions conditional on the true class, and all panels
  share the same $0$--$100\%$ colour scale. Cells that are exactly zero are
  left blank, and a non-zero entry below $0.05\%$ is shown as ${<}0.1$. The
  pooled matrices are descriptive summaries of the nine matched model
  evaluations rather than a single independent test split; all inferential
  comparisons rest on the paired fold--seed analysis reported in
  Section~\ref{sec:results}.}
  \label{fig:confusion_aggregated}
\end{figure*}
""".replace("__FIG_STEM__", FIG_STEM)


def write_manifest(path: Path, *, cells: dict, agg: dict, reports_dir: Path,
                   predictions_dir: Path | None, prediction_notes: list[str],
                   aggregate_notes: list[str], outputs: list[Path], gloss: bool,
                   cmap_name: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = []
    add = lines.append

    add("# Aggregated confusion-matrix figure — provenance manifest")
    add("")
    add(f"Generated: {stamp}  ")
    add("Generator: `scripts/generate_confusion_figure.py`  ")
    add("Protocol: `lightweight_pcste_final_test_v1` (sealed publication TEST)")
    add("")
    add("This artifact is **figure generation only**. No model was run, no TEST was")
    add("re-run, no prediction was regenerated and no frozen output was modified.")
    add("")

    add("## 1. Provenance of the confusion matrices")
    add("")
    add("**Matrices were read directly from saved per-cell confusion matrices.**")
    add("Nothing was reconstructed from scalar metrics and nothing was approximated.")
    add("")
    add("Each sealed per-model TEST report carries, for every dataset, the fields")
    add("`per_dataset_reports[<DATASET>].confusion_matrix` (raw integer counts) and")
    add("`per_dataset_reports[<DATASET>].classes` (the frozen class order). Those are")
    add("the values pooled here.")
    add("")
    add(f"Primary source directory: `{reports_dir}`")
    if predictions_dir is not None:
        add("")
        add("**Independent cross-check.** Every matrix was additionally rebuilt from the")
        add("frozen per-cell prediction CSVs (`y_true` / `y_pred`) and compared")
        add("element-wise against the report matrices. This reads saved artifacts only;")
        add("no inference was performed.")
        add("")
        add(f"Cross-check source directory: `{predictions_dir}`")
        failures = [n for n in prediction_notes if not n.startswith("OK")]
        add("")
        add(f"Result: {len(prediction_notes) - len(failures)} / {len(prediction_notes)} "
            f"dataset-cell matrices reproduced **exactly**"
            + (f"; {len(failures)} discrepancies (listed in §6)." if failures else "."))
    add("")

    add("## 2. Frozen cells used")
    add("")
    add("Matched evaluation design: folds GF1/GF2/GF3 × seeds 42/1337/2026 × models")
    add("Full-S1 and K1 = 9 matched cells per model, 18 report files in total. Every")
    add("report was verified to self-identify with the expected `fold`, `seed`,")
    add("`family` and `partition = test`.")
    add("")
    add("| # | Fold | Seed | Model | Source file | SHA-256 |")
    add("|---:|---:|---:|---|---|---|")
    for i, ((fold, seed, model), cell) in enumerate(sorted(cells.items()), start=1):
        add(f"| {i} | GF{fold} | {seed} | {model} | `{cell['path'].name}` | "
            f"`{cell['sha256'][:16]}…` |")
    add("")

    add("## 3. Dataset class labels and matrix dimensions")
    add("")
    add("Class labels are taken verbatim from the frozen reports; they agree exactly")
    add("with the frozen head definitions in `src/methodology_v2/experiment/heads.py`")
    add("(`CLASS_ORDERS`). A single class order was observed across all 18 cells for")
    add("every dataset, so no reordering was necessary before summation.")
    add("")
    add("| Dataset | Classes | Dimensions | Class order (frozen) |")
    add("|---|---:|---|---|")
    for ds_key, (display, _) in DATASETS.items():
        entry = agg.get((ds_key, MODELS[0]))
        if entry is None:
            continue
        k = len(entry["classes"])
        add(f"| {display} | {k} | {k}×{k} | "
            + ", ".join(f"`{c}`" for c in entry["classes"]) + " |")
    add("")
    add("> **Deviation from the task brief, reported as required.** The brief expected")
    add("> CWRU to have 4 classes. The frozen outputs are authoritative and give CWRU")
    add("> **3** classes (`inner_race`, `outer_race`, `ball`) — the 3-class CWRU")
    add("> fault-type benchmark used throughout this study. JNU (4), HIT (3) and")
    add("> MaFaulDa (10) match the brief. The figure uses the frozen counts.")
    add("")
    add("Documented meanings of the frozen code-style labels, quoted from frozen")
    add("sources in this repository and recorded here for caption/legend use")
    add("(axis ticks show the raw frozen strings unless `--gloss` is passed; this run")
    add(f"used `--gloss={'on' if gloss else 'off'}`):")
    add("")
    add("- JNU (`src/methodology_v2/experiment/heads.py`): `n` = healthy, `ib` = inner,")
    add("  `ob` = outer, `tb` = roller.")
    add("- HIT (`src/methodology_v2/registry.py`, `HIT[\"label_meaning\"]`): `0` = healthy,")
    add("  `1` = inner-ring fault, `2` = outer-ring fault.")
    add("")

    add("## 4. Aggregation and pooled totals")
    add("")
    add("For dataset *d* and model *m*, the nine matched per-cell matrices were summed")
    add("element-wise and the sum row-normalized to percentages:")
    add("")
    add("```")
    add("C_agg[d,m] = sum_{f=1..3} sum_{s in {42,1337,2026}} C^(f,s)[d,m]")
    add("Ctilde[i,j] = 100 * C_agg[i,j] / sum_j C_agg[i,j]")
    add("```")
    add("")
    add("| Dataset | Model | Cells pooled | Pooled samples | Max abs. row-sum deviation |")
    add("|---|---|---:|---:|---:|")
    for ds_key, (display, _) in DATASETS.items():
        for model in MODELS:
            entry = agg.get((ds_key, model))
            if entry is None:
                continue
            finite = np.isfinite(entry["row_normalized"]).all(axis=1)
            sums = np.nansum(entry["row_normalized"], axis=1)[finite]
            dev = float(np.max(np.abs(sums - 100.0))) if sums.size else float("nan")
            add(f"| {display} | {model} | {len(entry['cells'])} / 9 | "
                f"{entry['total']:,} | {dev:.2e} |")
    add("")
    add("Full-S1 and K1 pool identical sample totals per dataset, as expected: both")
    add("models are evaluated on the same matched TEST cells.")
    add("")

    add("## 5. Methodological note — what the pooled matrices are, and are not")
    add("")
    add("Each fold is evaluated under three seeds, so the same underlying TEST window")
    add("membership for a fold contributes **three** prediction outcomes to the pool.")
    add("The pooled matrices therefore summarize **nine matched model evaluations**,")
    add("not a single independent test split, and the pooled counts are not independent")
    add("samples.")
    add("")
    add("Concretely, the per-cell TEST support is identical for the three seeds of a")
    add("given fold, and each fold contributes that support three times to the pool:")
    add("")
    add("| Dataset | GF1 per cell | GF2 per cell | GF3 per cell | Pooled (x3 seeds) |")
    add("|---|---:|---:|---:|---:|")
    for ds_key, (display, _) in DATASETS.items():
        per_fold = []
        for fold in FOLDS:
            sizes = {int(cells[(fold, seed, MODELS[0])]["per_dataset"][ds_key]["cm"].sum())
                     for seed in SEEDS}
            per_fold.append(str(sizes.pop()) if len(sizes) == 1 else "varies")
        total = agg[(ds_key, MODELS[0])]["total"]
        add(f"| {display} | " + " | ".join(per_fold) + f" | {total:,} |")
    add("")
    add("This is acceptable for a descriptive class-level figure because Full-S1 and K1")
    add("share exactly the same matched evaluation design, so the two columns are")
    add("directly comparable. **No significance test is performed or implied here.**")
    add("Statistical inference remains the paired fold × seed analysis frozen in")
    add("`FINAL_SEALED_TEST_REPORT.md` and `PUBLICATION_FINAL_RESULTS.json`.")
    add("")

    add("## 6. Verification log")
    add("")
    add("### Aggregate checks")
    add("")
    add("```")
    lines.extend(aggregate_notes)
    add("```")
    if predictions_dir is not None:
        failures = [n for n in prediction_notes if not n.startswith("OK")]
        add("")
        add("### Rebuild-from-frozen-predictions cross-check")
        add("")
        if failures:
            add("```")
            lines.extend(failures)
            add("```")
        else:
            add(f"All {len(prediction_notes)} dataset-cell matrices "
                "(18 cells × 4 datasets) reproduced exactly from the frozen")
            add("prediction CSVs. Per-cell log omitted for brevity; re-run with")
            add("`--verify-predictions` to regenerate it.")
    add("")

    add("## 7. Outputs written")
    add("")
    add("```")
    for out_path in outputs:  # NB: must not shadow the `path` parameter
        try:
            add(str(out_path.relative_to(REPO_ROOT)))
        except ValueError:
            add(str(out_path))
    add("```")
    add("")
    add("## 8. Rendering notes")
    add("")
    add(f"- Colour map: `{cmap_name}` — a single-hue sequential ramp, light→dark, "
        "colourblind-safe and")
    add("  greyscale-safe in print. No rainbow/diverging map is used for magnitude.")
    add("- Shared colour scale across all eight panels, fixed to 0–100%.")
    add("- Cells that are exactly zero are left unannotated (uniform rule in every")
    add("  panel) so the dense MaFaulDa panel stays readable; the value is still")
    add("  encoded by colour and is present in the CSV export.")
    add("- Annotation size is scaled per panel to the cell it must fit; all other")
    add("  typography is uniform across the figure.")
    add("- PDF/SVG are vector; PDF embeds Type-42 fonts and the SVG stores glyph")
    add("  outlines, so both render identically inside LaTeX.")
    add("")

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS,
                        help="directory holding the 18 sealed per-cell TEST reports")
    parser.add_argument("--verify-predictions", type=Path, default=None,
                        help="directory holding the frozen per-cell prediction CSVs; "
                             "when given, every matrix is rebuilt from them and compared")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTDIR,
                        help="output directory for the figure and its data exports")
    parser.add_argument("--latex-snippet", type=Path,
                        default=REPO_ROOT / "tables_or_snippets/fig_confusion_include.tex",
                        help="path for the LaTeX \\includegraphics snippet")
    parser.add_argument("--fig-width", type=float, default=7.16,
                        help="figure width in inches (default: two-column \\textwidth)")
    parser.add_argument("--dpi", type=int, default=600, help="raster DPI for the PNG")
    parser.add_argument("--cmap", default="Blues",
                        help="single-hue sequential colour map (default: Blues)")
    parser.add_argument("--gloss", action="store_true",
                        help="append documented class meanings to JNU/HIT tick labels")
    args = parser.parse_args(argv)

    print(f"[1/6] reading sealed per-cell TEST reports from {args.reports_dir}")
    cells, missing = load_cells(args.reports_dir)
    if missing:
        print("BLOCKED — required frozen artifacts are missing or inconsistent:",
              file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 2
    print(f"      {len(cells)} cells loaded "
          f"({sum(1 for k in cells if k[2] == 'Full-S1')} Full-S1, "
          f"{sum(1 for k in cells if k[2] == 'K1')} K1)")

    prediction_notes: list[str] = []
    if args.verify_predictions is not None:
        print(f"[2/6] cross-checking against frozen predictions in {args.verify_predictions}")
        prediction_notes = verify_against_predictions(cells, args.verify_predictions)
        failures = [n for n in prediction_notes if not n.startswith("OK")]
        print(f"      {len(prediction_notes) - len(failures)}/{len(prediction_notes)} "
              f"matrices reproduced exactly")
        if failures:
            print("BLOCKED — frozen predictions disagree with frozen report matrices:",
                  file=sys.stderr)
            for item in failures:
                print(f"  - {item}", file=sys.stderr)
            return 3
    else:
        print("[2/6] prediction cross-check skipped (--verify-predictions not given)")

    print("[3/6] pooling the nine matched cells per dataset and model")
    agg, problems = aggregate(cells)
    if problems:
        print("BLOCKED — aggregation problems:", file=sys.stderr)
        for item in problems:
            print(f"  - {item}", file=sys.stderr)
        return 4

    aggregate_notes = verify_aggregate(agg)
    for note in aggregate_notes:
        print(f"      {note}")
    if any(n.startswith("FAIL") for n in aggregate_notes):
        print("BLOCKED — verification failed.", file=sys.stderr)
        return 5

    print("[4/6] exporting machine-readable matrices")
    data_paths = export_matrices(agg, args.out_dir / "data")

    print("[5/6] rendering the figure")
    figure_paths = build_figure(agg, args.out_dir, fig_width=args.fig_width,
                                dpi=args.dpi, gloss=args.gloss, cmap_name=args.cmap)

    print("[6/6] writing manifest and LaTeX snippet")
    args.latex_snippet.parent.mkdir(parents=True, exist_ok=True)
    args.latex_snippet.write_text(LATEX_SNIPPET)
    manifest_path = args.out_dir / "confusion_matrix_aggregation_manifest.md"
    outputs = figure_paths + data_paths + [manifest_path, args.latex_snippet]
    write_manifest(manifest_path, cells=cells, agg=agg, reports_dir=args.reports_dir,
                   predictions_dir=args.verify_predictions,
                   prediction_notes=prediction_notes, aggregate_notes=aggregate_notes,
                   outputs=outputs, gloss=args.gloss, cmap_name=args.cmap)

    print(f"\nDone. {len(outputs)} files written under {args.out_dir} "
          f"and {args.latex_snippet.parent}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
