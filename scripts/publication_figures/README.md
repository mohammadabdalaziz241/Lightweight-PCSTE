# Main-paper figure generation

Scripts for main-paper Figures 1, 2, 3, 5 and 6. Figure 4 (aggregated confusion
matrices) is produced by `scripts/generate_confusion_figure.py`.

All scripts are pure read-and-plot utilities over the sealed publication
artifacts. None runs a model, re-runs TEST, regenerates predictions, or writes
into a frozen result tree.

## Build

```bash
cd scripts/publication_figures
for f in fig01_workflow fig02_matched_cells fig03_dataset_level \
         fig05_efficiency fig06_q8; do python3 $f.py; done
```

Each script writes PDF + SVG + PNG to `figures/publication_figures/` and copies
the PDF into `manuscript/figures/`.

## Files

| Script | Figure | Frozen sources read at build time |
|---|---|---|
| `pubfig_style.py` | shared style + loaders | — |
| `fig01_workflow.py` | 1, study workflow | `CORRECTED_EFFICIENCY_SUMMARY.json` (parameter counts only) |
| `fig02_matched_cells.py` | 2, matched fold–seed | `matched_cells.csv`, `aggregate_summary.json` |
| `fig03_dataset_level.py` | 3, dataset-level | `PER_DATASET_SUMMARY.csv` |
| `fig05_efficiency.py` | 5, efficiency | `CORRECTED_EFFICIENCY_SUMMARY.json` |
| `fig06_q8.py` | 6, Q8 extension | `q8_size_results.json`, `q8_aggregate_summary.json`, `q8_aggregates.json` |

No figure hard-codes a result: every plotted value is read from a frozen
artifact, so a figure cannot drift from the sealed numbers. The superseded
`efficiency_v1` absolute timings, and the chained Full-S1 → Q8 ratio derived
from them, are never read.

## Shared visual language

Defined once in `pubfig_style.py` and used by every figure, including the
confusion-matrix figure's palette conventions:

- white background, DejaVu Sans, Type-42 fonts in PDF, outlined glyphs in SVG;
- Okabe-Ito colourblind-safe hues — Full-S1 blue `#0072B2` (circle),
  K1 vermillion `#D55E00` (square, `///` hatch), Q8(K1) bluish green `#009E73`
  (triangle, `xxx` hatch);
- identity is never colour-alone: every series also carries a distinct marker or
  hatch, and compared values are labelled directly;
- recessive grid and axes, no gradients or decorative effects.
