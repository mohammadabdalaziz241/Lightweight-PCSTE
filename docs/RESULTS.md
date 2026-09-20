# Final Sealed TEST Results

Protocol: `lightweight_pcste_final_test_v1`
Status: `PUBLICATION_FINAL_SEALED_TEST_COMPLETE`

All numbers below are reproduced verbatim from the frozen artifacts in `results/final_test/publication_final_test_v1/`. That directory is authoritative; this document is a reader-facing summary of it and must never diverge from it.

Design: nine matched cells (3 global folds x seeds 42, 1337, 2026), eighteen originally frozen checkpoints, paired within (fold, seed). Dispersion is sample SD (`ddof=1`) across the nine cells.

## Matched Macro-4 cells

| Fold | Seed | Full-S1 F1 | K1 F1 | Delta F1 | Full-S1 AUC | K1 AUC | Delta AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 0.932734 | 0.944172 | +0.011438 | 0.995621 | 0.998797 | +0.003176 |
| 1 | 1337 | 0.953079 | 0.950776 | -0.002303 | 0.995830 | 0.994204 | -0.001626 |
| 1 | 2026 | 0.926615 | 0.941293 | +0.014678 | 0.993423 | 0.996064 | +0.002641 |
| 2 | 42 | 0.947566 | 0.977788 | +0.030222 | 0.998251 | 0.999368 | +0.001117 |
| 2 | 1337 | 0.943998 | 0.972982 | +0.028984 | 0.997849 | 0.999368 | +0.001519 |
| 2 | 2026 | 0.879871 | 0.919195 | +0.039324 | 0.971810 | 0.993053 | +0.021242 |
| 3 | 42 | 0.957175 | 0.967335 | +0.010160 | 0.991825 | 0.996152 | +0.004327 |
| 3 | 1337 | 0.926540 | 0.961007 | +0.034467 | 0.988335 | 0.996468 | +0.008132 |
| 3 | 2026 | 0.944214 | 0.963463 | +0.019249 | 0.993188 | 0.994110 | +0.000922 |

Full precision: `matched_cells.csv`.

## Aggregate Macro-4

| Endpoint | Full-S1 | K1 | Paired delta (K1 - Full-S1) | K1 wins/ties/losses |
|---|---:|---:|---:|---:|
| Macro-4 Macro-F1 (primary) | 0.934644 +/- 0.023258 | 0.955334 +/- 0.018393 | +0.020691 +/- 0.013509 | 8 / 0 / 1 |
| Macro-4 Macro-AUC (secondary) | 0.991793 +/- 0.008102 | 0.996398 +/- 0.002367 | +0.004606 +/- 0.006794 | 8 / 0 / 1 |

## Confirmatory result

The pre-registered confirmatory contrast is non-inferiority of K1 relative to Full-S1 on Macro-4 Macro-F1 at the frozen margin `-0.02`, by exact sign-flip enumeration of all 512 sign patterns over the nine paired differences.

| Quantity | Value |
|---|---|
| Frozen non-inferiority margin | `-0.02` |
| Exact one-sided sign-flip p | `0.001953125` |
| Predefined non-inferiority criterion | **SATISFIED** |
| Gated exact two-sided superiority p | `0.0078125` |

Reading these cautiously:

- The confirmatory claim supported by this study is **non-inferiority at the -0.02 margin**, and that criterion was met exactly as predefined.
- The superiority p-value is reported **only** because the non-inferiority gate passed, under the frozen hierarchical rule. Both p-values come from exact sign-flip enumeration over nine cells, where the attainable resolution is a multiple of `1/512`: the one-sided non-inferiority p of `0.001953125` is exactly the minimum attainable value (`1/512`), and the two-sided superiority p of `0.0078125` is `4/512`. At this sample size the p-values are driven largely by sign consistency, and they bound effect magnitude only weakly.
- The design was framed and frozen as a non-inferiority comparison. The observed positive mean delta is a real, consistently signed observation, but nine matched cells give a limited basis for a strong superiority claim, and it should be presented as supportive secondary evidence.
- Not every cell or dataset favours K1: fold 1 / seed 1337 favours Full-S1 on both metrics, and HIT shows a small negative mean delta on both Macro-F1 (-0.002311) and Macro-AUC (-0.000366).
- No confidence interval for the paired difference was pre-registered, so none is claimed here.

## Per-dataset Macro-F1

| Dataset | Full-S1 | K1 | Paired delta | K1 wins/ties/losses |
|---|---:|---:|---:|---:|
| CWRU | 0.854747 +/- 0.083579 | 0.918945 +/- 0.073432 | +0.064198 +/- 0.044152 | 8 / 0 / 1 |
| JNU | 0.971429 +/- 0.042857 | 0.980952 +/- 0.037796 | +0.009524 +/- 0.028571 | 1 / 8 / 0 |
| HIT | 0.991039 +/- 0.017798 | 0.988728 +/- 0.016133 | -0.002311 +/- 0.007427 | 1 / 5 / 3 |
| MaFaulDa | 0.921360 +/- 0.016432 | 0.932712 +/- 0.017321 | +0.011352 +/- 0.019450 | 6 / 0 / 3 |

## Per-dataset Macro-AUC

| Dataset | Full-S1 | K1 | Paired delta | K1 wins/ties/losses |
|---|---:|---:|---:|---:|
| CWRU | 0.974093 +/- 0.033643 | 0.990379 +/- 0.009921 | +0.016286 +/- 0.027983 | 8 / 0 / 1 |
| JNU | 0.999559 +/- 0.001323 | 1.000000 +/- 0.000000 | +0.000441 +/- 0.001323 | 1 / 8 / 0 |
| HIT | 0.999969 +/- 0.000094 | 0.999602 +/- 0.001082 | -0.000366 +/- 0.001095 | 1 / 6 / 2 |
| MaFaulDa | 0.993549 +/- 0.006247 | 0.995611 +/- 0.004732 | +0.002062 +/- 0.003131 | 7 / 0 / 2 |

Dataset-level tests were not part of the confirmatory family; these are descriptive.

## Class-level detail

`results/final_test/publication_final_test_v1/per_model_reports/` contains one JSON per (fold, seed, family) — eighteen files. Each holds the checkpoint path and SHA256, selected epoch, per-dataset confusion matrices, per-class precision/recall/F1 with support, macro one-vs-rest AUC, CWRU per-specimen recall and race-recall minimum, MaFaulDa per-configuration recall, and per-severity breakdowns.

Raw per-window predictions and probabilities (18 CSV files, ~21.6 MiB total) are **not** stored in Git. See `docs/ARTIFACTS.md`.

## Integrity

- Exact TEST membership verified independently for all eighteen prediction files.
- Every checkpoint rehashed after inference; all matched the frozen plan.
- All nine matched cells retained.
- No training, selection, tuning, protocol change, Q8 run, or rerun occurred.

| Artifact | SHA256 |
|---|---|
| `FINAL_SEALED_TEST_REPORT.md` | `11f8405d7ae5783b283c39a0cd5a4dd16685187125a2dc38b8d738ea035e6a8d` |
| `PUBLICATION_FINAL_RESULTS.json` | `b6f14ff707a952ae860812a2bfa2f099fc8d3c39ab579c8d2605057b5699a527` |
| `PER_DATASET_SUMMARY.csv` | `62fb8bfe5603cf2525c2d1500157e5dc4c889539051e1a5b403affb2ff706520` |
| `matched_cells.csv` | `af62d51c0c299b6e4a17de775015c259b6e575af0e3dc1d2c1e59ad9be74d239` |
| `aggregate_summary.json` | `fffc6c28a42c25d35c85c1515be48b5ab7da0d0767450bfb31d9dbc903a1cf28` |
| Evaluation plan (JSON) | `ce22960df67836b0cb49f476b3fe4c2ed29e5cfcb554f8948e7af3f9fc11a55a` |
| Evaluation plan (MD) | `79521eec872ee0c55ceb74972b65e19c6a2f2a96e478a5640650f5a8039d018f` |
| Frozen model table | `34c1a6b3ed2c9f545cb6b368c46e22a2bdf3b03a1a5134f3f73180b484ab5cd9` |
| Publication barrier | `fa8410c03bcad4d3eea31282ac8ee7828c731a35a025aab872354aa8ffdbe71d` |
| Publication TEST driver | `8f2de50a52ce83fefa8dcd1516ec3c348d9f5c1a6b561b38070fe9f94d149f3e` |
| K1 9-cell freeze manifest | `a9e5fb49a1f38a7d6e2649ff8d155cf619cd55572bf7238b06a1fcb120d64a8b` |
| Historical evaluator (unchanged) | `dd3aa58ce5067298504b9e064cdb51e5f29d0f655b1ca3f4fa6af77600d7f517` |

## Computational efficiency

Efficiency was measured separately under `lightweight_pcste_efficiency_v1` and does not touch any predictive result above. Full detail and per-dataset tables: `benchmarks/efficiency_v1/EFFICIENCY_FINAL_REPORT.md`.

| Metric | Full-S1 | K1 | Reduction / speed-up |
|---|---:|---:|---:|
| Encoder parameters | 2,382,033 | 1,375,953 | 42.24% (1.73x) |
| Complete model parameters | 2,385,893 | 1,379,813 | 42.17% |
| FP32 state_dict size | 9.135 MiB | 5.286 MiB | 42.14% |
| Macro-4 FLOPs / 1 s window | 2.6067 GFLOP | 1.4236 GFLOP | 45.38% (1.83x) |
| CPU batch-1 model-only latency | 91.850 ms | 41.498 ms | 54.82% (2.21x) |
| CPU batch-1 end-to-end latency | 93.327 ms | 42.837 ms | 54.10% (2.18x) |
| GPU batch-1 model-only latency | 11.811 ms | 6.368 ms | 46.08% (1.86x) |
| GPU batch-1 end-to-end latency | 13.409 ms | 7.805 ms | 41.79% (1.72x) |
| GPU batch-32 throughput | 382.2 windows/s | 743.5 windows/s | 1.95x |
| GPU peak memory (batch 1) | 29.466 MiB | 24.823 MiB | 15.76% |

Measured on the deterministically fixed GF1 / seed-42 pair on one idle RTX 4000 Ada host, with deterministic VALIDATION inputs. Latency is the equal-domain mean of four per-dataset batch-1 medians. All timings use the pure-PyTorch reference selective scan; fused Mamba kernels are unavailable on this stack.

Pairing this with the sealed predictive result — K1 non-inferior at the frozen -0.02 margin, paired delta +0.020691 +/- 0.013509 — gives the performance-versus-efficiency trade-off the study set out to quantify. **No manuscript claim is drawn here.**

## Not claimed here

Q8 quantization has not been executed, so no int8 size, latency, or accuracy figure exists. No dissertation-era efficiency number is reused anywhere in this repository. Efficiency was measured on one representative cell on one host; it was not measured across all nine cells or on other hardware.
