"""Shared publication style and frozen-artifact loaders for the main-paper figures.

Every figure in ``scripts/publication_figures/`` imports this module so the six
main-paper figures share one visual language: the same typeface, the same
Full-S1 / K1 / Q8(K1) colour-and-marker encoding, the same axis treatment and
the same export pipeline.

Design rules enforced here
--------------------------
* White background, no gradients, no decorative effects.
* Okabe-Ito colourblind-safe hues. Full-S1 blue, K1 vermillion, Q8 bluish green.
* Identity is never carried by colour alone: every series also has a distinct
  marker or hatch, and values are labelled directly wherever they are compared.
* Vector output (PDF + SVG) with a raster PNG preview. PDFs embed Type-42
  fonts and the SVG stores glyph outlines, so both render identically in LaTeX.

Data provenance
---------------
Loaders read the sealed publication artifacts directly. Nothing is re-measured,
re-run or hand-entered; if an artifact is missing the loader raises rather than
falling back to a literal.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_TEST = REPO_ROOT / "results/final_test/publication_final_test_v1"
FROZEN_EFF = REPO_ROOT / "benchmarks/efficiency_latency_correction_v1"
FROZEN_Q8 = REPO_ROOT / "benchmarks/q8_v1/results"

#: all three formats land here
ASSET_DIR = REPO_ROOT / "figures/publication_figures"
#: only the manuscript-facing PDFs are copied here
MANUSCRIPT_FIG_DIR = REPO_ROOT / "manuscript/figures"

# --------------------------------------------------------------------------
# Visual language
# --------------------------------------------------------------------------

#: Okabe-Ito colourblind-safe palette, assigned in fixed order and never cycled.
FULL_S1 = "#0072B2"   # blue
K1 = "#D55E00"        # vermillion
Q8 = "#009E73"        # bluish green

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d7d7d7"
PANEL_EDGE = "#c8c8c8"
SOFT_FILL = "#f2f2f2"

MODEL_STYLE = {
    "Full-S1": {"color": FULL_S1, "marker": "o", "hatch": ""},
    "K1": {"color": K1, "marker": "s", "hatch": "///"},
    "Q8(K1)": {"color": Q8, "marker": "^", "hatch": "xxx"},
}

#: base width in inches == \textwidth of the manuscript (article, 1in margins)
TEXT_WIDTH_IN = 6.5

FS_TITLE = 9.0
FS_LABEL = 8.0
FS_TICK = 7.5
FS_ANNOT = 7.0
FS_SMALL = 6.5


def apply_style() -> None:
    """Install the shared rcParams. Call once at the top of every figure script."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "text.usetex": False,
        "axes.linewidth": 0.5,
        "axes.edgecolor": PANEL_EDGE,
        "axes.labelcolor": INK,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": FS_ANNOT,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "hatch.linewidth": 0.5,
        "lines.linewidth": 1.2,
    })


def tidy_axes(ax, *, grid_axis: str | None = "y", top=False, right=False) -> None:
    """Recessive axis furniture: drop unused spines, put the grid behind data."""
    ax.spines["top"].set_visible(top)
    ax.spines["right"].set_visible(right)
    if grid_axis:
        ax.grid(True, axis=grid_axis, linestyle="-", linewidth=0.5,
                color=GRID, zorder=0)
        ax.set_axisbelow(True)


def panel_tag(ax, text: str, *, dx: float = -0.06, dy: float = 1.10) -> None:
    """Bold (a)/(b)/(c) tag in the panel's upper-left corner."""
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=FS_TITLE,
            fontweight="bold", color=INK, ha="left", va="top")


def save_figure(fig, stem: str, *, to_manuscript: bool = True) -> list[Path]:
    """Write PDF + SVG + PNG, and copy the PDF into the manuscript tree.

    Returns every path written.
    """
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext, dpi in (("pdf", 600), ("svg", 600), ("png", 300)):
        path = ASSET_DIR / f"{stem}.{ext}"
        fig.savefig(path, format=ext, dpi=dpi, facecolor="white",
                    bbox_inches="tight", pad_inches=0.02)
        written.append(path)
    if to_manuscript:
        MANUSCRIPT_FIG_DIR.mkdir(parents=True, exist_ok=True)
        target = MANUSCRIPT_FIG_DIR / f"{stem}.pdf"
        target.write_bytes((ASSET_DIR / f"{stem}.pdf").read_bytes())
        written.append(target)
    plt.close(fig)
    return written


# --------------------------------------------------------------------------
# Frozen-artifact loaders
# --------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required frozen artifact missing: {path}")
    return json.loads(path.read_text())


def load_matched_cells() -> list[dict]:
    """The nine matched fold-seed Macro-4 Macro-F1 cells, in frozen order."""
    path = FROZEN_TEST / "matched_cells.csv"
    if not path.is_file():
        raise FileNotFoundError(f"required frozen artifact missing: {path}")
    rows = []
    lines = path.read_text().strip().split("\n")
    header = lines[0].split(",")
    for line in lines[1:]:
        values = dict(zip(header, line.split(",")))
        rows.append({
            "fold": int(values["fold"]),
            "seed": int(values["seed"]),
            "full_s1_f1": float(values["full_s1_macro_4_f1"]),
            "k1_f1": float(values["k1_macro_4_f1"]),
            "delta_f1": float(values["delta_f1"]),
            "full_s1_auc": float(values["full_s1_macro_4_auc"]),
            "k1_auc": float(values["k1_macro_4_auc"]),
            "delta_auc": float(values["delta_auc"]),
        })
    return rows


def load_aggregate() -> dict:
    """Macro-4 aggregate endpoints and the frozen non-inferiority result."""
    return _read_json(FROZEN_TEST / "aggregate_summary.json")


def load_per_dataset() -> dict:
    """Dataset-level Macro-F1 / Macro-AUC means, SDs and paired deltas."""
    path = FROZEN_TEST / "PER_DATASET_SUMMARY.csv"
    if not path.is_file():
        raise FileNotFoundError(f"required frozen artifact missing: {path}")
    out: dict[str, dict] = {}
    lines = path.read_text().strip().split("\n")
    header = lines[0].split(",")
    for line in lines[1:]:
        v = dict(zip(header, line.split(",")))
        out.setdefault(v["dataset"], {})[v["metric"]] = {
            "full_s1_mean": float(v["full_s1_mean"]),
            "full_s1_sd": float(v["full_s1_sd"]),
            "k1_mean": float(v["k1_mean"]),
            "k1_sd": float(v["k1_sd"]),
            "delta_mean": float(v["delta_mean"]),
            "delta_sd": float(v["delta_sd"]),
        }
    return out


def load_efficiency() -> dict:
    """Authoritative corrected efficiency benchmark."""
    return _read_json(FROZEN_EFF / "CORRECTED_EFFICIENCY_SUMMARY.json")


def load_q8_size() -> dict:
    return _read_json(FROZEN_Q8 / "q8_size_results.json")


def load_q8_aggregate() -> dict:
    return _read_json(FROZEN_Q8 / "q8_aggregate_summary.json")


def load_q8_latency() -> dict:
    return _read_json(FROZEN_Q8 / "q8_aggregates.json")


def load_q8_throughput() -> dict:
    """K1 vs Q8 batch-32 CPU throughput from the frozen Q8 throughput CSV."""
    path = FROZEN_Q8 / "q8_throughput_results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"required frozen artifact missing: {path}")
    lines = path.read_text().strip().split("\n")
    header = [h.strip() for h in lines[0].split(",")]
    rows = [dict(zip(header, [c.strip() for c in ln.split(",")])) for ln in lines[1:]]
    return {"header": header, "rows": rows}


MIB = 1024.0 ** 2
