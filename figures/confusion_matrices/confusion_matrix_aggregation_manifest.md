# Aggregated confusion-matrix figure — provenance manifest

Generated: 2026-09-21 10:50:41 UTC  
Generator: `scripts/generate_confusion_figure.py`  
Protocol: `lightweight_pcste_final_test_v1` (sealed publication TEST)

This artifact is **figure generation only**. No model was run, no TEST was
re-run, no prediction was regenerated and no frozen output was modified.

## 1. Provenance of the confusion matrices

**Matrices were read directly from saved per-cell confusion matrices.**
Nothing was reconstructed from scalar metrics and nothing was approximated.

Each sealed per-model TEST report carries, for every dataset, the fields
`per_dataset_reports[<DATASET>].confusion_matrix` (raw integer counts) and
`per_dataset_reports[<DATASET>].classes` (the frozen class order). Those are
the values pooled here.

Primary source directory: `/user/HS401/ma06314/Lightweight-PCSTE/results/final_test/publication_final_test_v1/per_model_reports`

**Independent cross-check.** Every matrix was additionally rebuilt from the
frozen per-cell prediction CSVs (`y_true` / `y_pred`) and compared
element-wise against the report matrices. This reads saved artifacts only;
no inference was performed.

Cross-check source directory: `/scratch/ma06314/Standard Project/rotating_machinery_fault_diagnosis/results/pcste_v2/lightweight_v1/publication_final_test_v1`

Result: 72 / 72 dataset-cell matrices reproduced **exactly**.

## 2. Frozen cells used

Matched evaluation design: folds GF1/GF2/GF3 × seeds 42/1337/2026 × models
Full-S1 and K1 = 9 matched cells per model, 18 report files in total. Every
report was verified to self-identify with the expected `fold`, `seed`,
`family` and `partition = test`.

| # | Fold | Seed | Model | Source file | SHA-256 |
|---:|---:|---:|---|---|---|
| 1 | GF1 | 42 | Full-S1 | `gf1_s42_full_s1_report.json` | `5b48589b50b00ac9…` |
| 2 | GF1 | 42 | K1 | `gf1_s42_k1_report.json` | `372988317d49c018…` |
| 3 | GF1 | 1337 | Full-S1 | `gf1_s1337_full_s1_report.json` | `13a670e1c759e7d1…` |
| 4 | GF1 | 1337 | K1 | `gf1_s1337_k1_report.json` | `d7bc1537475de53e…` |
| 5 | GF1 | 2026 | Full-S1 | `gf1_s2026_full_s1_report.json` | `0b53f5abf792a210…` |
| 6 | GF1 | 2026 | K1 | `gf1_s2026_k1_report.json` | `a166282e5f5f3fae…` |
| 7 | GF2 | 42 | Full-S1 | `gf2_s42_full_s1_report.json` | `7a27bec2aced2808…` |
| 8 | GF2 | 42 | K1 | `gf2_s42_k1_report.json` | `08d03359aa7024db…` |
| 9 | GF2 | 1337 | Full-S1 | `gf2_s1337_full_s1_report.json` | `8dd13d004d49695c…` |
| 10 | GF2 | 1337 | K1 | `gf2_s1337_k1_report.json` | `39cbdbd217438559…` |
| 11 | GF2 | 2026 | Full-S1 | `gf2_s2026_full_s1_report.json` | `edc3c9adc1a6e042…` |
| 12 | GF2 | 2026 | K1 | `gf2_s2026_k1_report.json` | `7fbbb2e421d5d6d7…` |
| 13 | GF3 | 42 | Full-S1 | `gf3_s42_full_s1_report.json` | `576a34a5c25819ac…` |
| 14 | GF3 | 42 | K1 | `gf3_s42_k1_report.json` | `15e857aa2144e764…` |
| 15 | GF3 | 1337 | Full-S1 | `gf3_s1337_full_s1_report.json` | `a957c41e9f09c952…` |
| 16 | GF3 | 1337 | K1 | `gf3_s1337_k1_report.json` | `3939f44ebcff5067…` |
| 17 | GF3 | 2026 | Full-S1 | `gf3_s2026_full_s1_report.json` | `c8c27477af68bd3d…` |
| 18 | GF3 | 2026 | K1 | `gf3_s2026_k1_report.json` | `f21c5e5d21c3282a…` |

## 3. Dataset class labels and matrix dimensions

Class labels are taken verbatim from the frozen reports; they agree exactly
with the frozen head definitions in `src/methodology_v2/experiment/heads.py`
(`CLASS_ORDERS`). A single class order was observed across all 18 cells for
every dataset, so no reordering was necessary before summation.

| Dataset | Classes | Dimensions | Class order (frozen) |
|---|---:|---|---|
| CWRU | 3 | 3×3 | `inner_race`, `outer_race`, `ball` |
| JNU | 4 | 4×4 | `n`, `ib`, `ob`, `tb` |
| HIT | 3 | 3×3 | `0`, `1`, `2` |
| MaFaulDa | 10 | 10×10 | `normal`, `imbalance`, `horizontal-misalignment`, `vertical-misalignment`, `underhang/ball_fault`, `underhang/cage_fault`, `underhang/outer_race`, `overhang/ball_fault`, `overhang/cage_fault`, `overhang/outer_race` |

> **Deviation from the task brief, reported as required.** The brief expected
> CWRU to have 4 classes. The frozen outputs are authoritative and give CWRU
> **3** classes (`inner_race`, `outer_race`, `ball`) — the 3-class CWRU
> fault-type benchmark used throughout this study. JNU (4), HIT (3) and
> MaFaulDa (10) match the brief. The figure uses the frozen counts.

Documented meanings of the frozen code-style labels, quoted from frozen
sources in this repository and recorded here for caption/legend use
(axis ticks show the raw frozen strings unless `--gloss` is passed; this run
used `--gloss=off`):

- JNU (`src/methodology_v2/experiment/heads.py`): `n` = healthy, `ib` = inner,
  `ob` = outer, `tb` = roller.
- HIT (`src/methodology_v2/registry.py`, `HIT["label_meaning"]`): `0` = healthy,
  `1` = inner-ring fault, `2` = outer-ring fault.

## 4. Aggregation and pooled totals

For dataset *d* and model *m*, the nine matched per-cell matrices were summed
element-wise and the sum row-normalized to percentages:

```
C_agg[d,m] = sum_{f=1..3} sum_{s in {42,1337,2026}} C^(f,s)[d,m]
Ctilde[i,j] = 100 * C_agg[i,j] / sum_j C_agg[i,j]
```

| Dataset | Model | Cells pooled | Pooled samples | Max abs. row-sum deviation |
|---|---|---:|---:|---:|
| CWRU | Full-S1 | 9 / 9 | 2,340 | 1.42e-14 |
| CWRU | K1 | 9 / 9 | 2,340 | 1.42e-14 |
| JNU | Full-S1 | 9 / 9 | 216 | 0.00e+00 |
| JNU | K1 | 9 / 9 | 216 | 0.00e+00 |
| HIT | Full-S1 | 9 / 9 | 2,520 | 0.00e+00 |
| HIT | K1 | 9 / 9 | 2,520 | 0.00e+00 |
| MaFaulDa | Full-S1 | 9 / 9 | 18,510 | 1.42e-14 |
| MaFaulDa | K1 | 9 / 9 | 18,510 | 1.42e-14 |

Full-S1 and K1 pool identical sample totals per dataset, as expected: both
models are evaluated on the same matched TEST cells.

## 5. Methodological note — what the pooled matrices are, and are not

Each fold is evaluated under three seeds, so the same underlying TEST window
membership for a fold contributes **three** prediction outcomes to the pool.
The pooled matrices therefore summarize **nine matched model evaluations**,
not a single independent test split, and the pooled counts are not independent
samples.

Concretely, the per-cell TEST support is identical for the three seeds of a
given fold, and each fold contributes that support three times to the pool:

| Dataset | GF1 per cell | GF2 per cell | GF3 per cell | Pooled (x3 seeds) |
|---|---:|---:|---:|---:|
| CWRU | 260 | 260 | 260 | 2,340 |
| JNU | 24 | 24 | 24 | 216 |
| HIT | 280 | 280 | 280 | 2,520 |
| MaFaulDa | 2115 | 2205 | 1850 | 18,510 |

This is acceptable for a descriptive class-level figure because Full-S1 and K1
share exactly the same matched evaluation design, so the two columns are
directly comparable. **No significance test is performed or implied here.**
Statistical inference remains the paired fold × seed analysis frozen in
`FINAL_SEALED_TEST_REPORT.md` and `PUBLICATION_FINAL_RESULTS.json`.

## 6. Verification log

### Aggregate checks

```
OK   CWRU: Full-S1 and K1 share one class order
OK   CWRU/Full-S1: matrix is 3x3 for 3 frozen classes
OK   CWRU/Full-S1: 9/9 matched cells pooled, n=2340
OK   CWRU/Full-S1: row sums of normalized matrix = 100% (max |dev| = 1.42e-14)
OK   CWRU/K1: matrix is 3x3 for 3 frozen classes
OK   CWRU/K1: 9/9 matched cells pooled, n=2340
OK   CWRU/K1: row sums of normalized matrix = 100% (max |dev| = 1.42e-14)
OK   JNU: Full-S1 and K1 share one class order
OK   JNU/Full-S1: matrix is 4x4 for 4 frozen classes
OK   JNU/Full-S1: 9/9 matched cells pooled, n=216
OK   JNU/Full-S1: row sums of normalized matrix = 100% (max |dev| = 0.00e+00)
OK   JNU/K1: matrix is 4x4 for 4 frozen classes
OK   JNU/K1: 9/9 matched cells pooled, n=216
OK   JNU/K1: row sums of normalized matrix = 100% (max |dev| = 0.00e+00)
OK   HIT: Full-S1 and K1 share one class order
OK   HIT/Full-S1: matrix is 3x3 for 3 frozen classes
OK   HIT/Full-S1: 9/9 matched cells pooled, n=2520
OK   HIT/Full-S1: row sums of normalized matrix = 100% (max |dev| = 0.00e+00)
OK   HIT/K1: matrix is 3x3 for 3 frozen classes
OK   HIT/K1: 9/9 matched cells pooled, n=2520
OK   HIT/K1: row sums of normalized matrix = 100% (max |dev| = 0.00e+00)
OK   MaFaulDa: Full-S1 and K1 share one class order
OK   MaFaulDa/Full-S1: matrix is 10x10 for 10 frozen classes
OK   MaFaulDa/Full-S1: 9/9 matched cells pooled, n=18510
OK   MaFaulDa/Full-S1: row sums of normalized matrix = 100% (max |dev| = 1.42e-14)
OK   MaFaulDa/K1: matrix is 10x10 for 10 frozen classes
OK   MaFaulDa/K1: 9/9 matched cells pooled, n=18510
OK   MaFaulDa/K1: row sums of normalized matrix = 100% (max |dev| = 1.42e-14)
```

### Rebuild-from-frozen-predictions cross-check

All 72 dataset-cell matrices (18 cells × 4 datasets) reproduced exactly from the frozen
prediction CSVs. Per-cell log omitted for brevity; re-run with
`--verify-predictions` to regenerate it.

## 7. Outputs written

```
figures/confusion_matrices/fig_confusion_aggregated_fulls1_vs_k1.pdf
figures/confusion_matrices/fig_confusion_aggregated_fulls1_vs_k1.svg
figures/confusion_matrices/fig_confusion_aggregated_fulls1_vs_k1.png
figures/confusion_matrices/data/CWRU_FullS1_counts.csv
figures/confusion_matrices/data/CWRU_FullS1_row_normalized.csv
figures/confusion_matrices/data/CWRU_K1_counts.csv
figures/confusion_matrices/data/CWRU_K1_row_normalized.csv
figures/confusion_matrices/data/JNU_FullS1_counts.csv
figures/confusion_matrices/data/JNU_FullS1_row_normalized.csv
figures/confusion_matrices/data/JNU_K1_counts.csv
figures/confusion_matrices/data/JNU_K1_row_normalized.csv
figures/confusion_matrices/data/HIT_FullS1_counts.csv
figures/confusion_matrices/data/HIT_FullS1_row_normalized.csv
figures/confusion_matrices/data/HIT_K1_counts.csv
figures/confusion_matrices/data/HIT_K1_row_normalized.csv
figures/confusion_matrices/data/MaFaulDa_FullS1_counts.csv
figures/confusion_matrices/data/MaFaulDa_FullS1_row_normalized.csv
figures/confusion_matrices/data/MaFaulDa_K1_counts.csv
figures/confusion_matrices/data/MaFaulDa_K1_row_normalized.csv
figures/confusion_matrices/confusion_matrix_aggregation_manifest.md
tables_or_snippets/fig_confusion_include.tex
```

## 8. Rendering notes

- Colour map: `Blues` — a single-hue sequential ramp, light→dark, colourblind-safe and
  greyscale-safe in print. No rainbow/diverging map is used for magnitude.
- Shared colour scale across all eight panels, fixed to 0–100%.
- Cells that are exactly zero are left unannotated (uniform rule in every
  panel) so the dense MaFaulDa panel stays readable; the value is still
  encoded by colour and is present in the CSV export.
- Annotation size is scaled per panel to the cell it must fit; all other
  typography is uniform across the figure.
- PDF/SVG are vector; PDF embeds Type-42 fonts and the SVG stores glyph
  outlines, so both render identically inside LaTeX.

