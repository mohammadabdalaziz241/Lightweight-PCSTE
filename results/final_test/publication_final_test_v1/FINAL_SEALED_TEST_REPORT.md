# Final Sealed Publication TEST Report

Protocol: `lightweight_pcste_final_test_v1`

Status: `PUBLICATION_FINAL_SEALED_TEST_COMPLETE`

All results use the 18 originally frozen checkpoints. All nine matched cells were retained. No training, checkpoint selection, tuning, protocol change, Q8 run, or scientific rerun occurred.

## Matched Macro-4 results

| Fold | Seed | Full-S1 F1 | K1 F1 | Delta F1 | Full-S1 AUC | K1 AUC | Delta AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 0.932734 | 0.944172 | 0.011438 | 0.995621 | 0.998797 | 0.003176 |
| 1 | 1337 | 0.953079 | 0.950776 | -0.002303 | 0.995830 | 0.994204 | -0.001626 |
| 1 | 2026 | 0.926615 | 0.941293 | 0.014678 | 0.993423 | 0.996064 | 0.002641 |
| 2 | 42 | 0.947566 | 0.977788 | 0.030222 | 0.998251 | 0.999368 | 0.001117 |
| 2 | 1337 | 0.943998 | 0.972982 | 0.028984 | 0.997849 | 0.999368 | 0.001519 |
| 2 | 2026 | 0.879871 | 0.919195 | 0.039324 | 0.971810 | 0.993053 | 0.021242 |
| 3 | 42 | 0.957175 | 0.967335 | 0.010160 | 0.991825 | 0.996152 | 0.004327 |
| 3 | 1337 | 0.926540 | 0.961007 | 0.034467 | 0.988335 | 0.996468 | 0.008132 |
| 3 | 2026 | 0.944214 | 0.963463 | 0.019249 | 0.993188 | 0.994110 | 0.000922 |

## Macro-4 Macro-F1

- Full-S1: 0.934644 +/- 0.023258
- K1: 0.955334 +/- 0.018393
- Paired delta: 0.020691 +/- 0.013509
- K1 wins/ties/losses: 8/0/1
- Frozen non-inferiority margin: -0.02; result: SATISFIED; exact one-sided p = 0.001953125
- Hierarchically gated superiority: exact two-sided p = 0.0078125

## Macro-4 Macro-AUC

- Full-S1: 0.991793 +/- 0.008102
- K1: 0.996398 +/- 0.002367
- Paired delta: 0.004606 +/- 0.006794

## Dataset-level Macro-F1

| Dataset | Full-S1 mean +/- SD | K1 mean +/- SD | Paired delta mean +/- SD |
|---|---:|---:|---:|
| CWRU | 0.854747 +/- 0.083579 | 0.918945 +/- 0.073432 | 0.064198 +/- 0.044152 |
| JNU | 0.971429 +/- 0.042857 | 0.980952 +/- 0.037796 | 0.009524 +/- 0.028571 |
| HIT | 0.991039 +/- 0.017798 | 0.988728 +/- 0.016133 | -0.002311 +/- 0.007427 |
| MAFAULDA | 0.921360 +/- 0.016432 | 0.932712 +/- 0.017321 | 0.011352 +/- 0.019450 |

## Dataset-level Macro-AUC

| Dataset | Full-S1 mean +/- SD | K1 mean +/- SD | Paired delta mean +/- SD |
|---|---:|---:|---:|
| CWRU | 0.974093 +/- 0.033643 | 0.990379 +/- 0.009921 | 0.016286 +/- 0.027983 |
| JNU | 0.999559 +/- 0.001323 | 1.000000 +/- 0.000000 | 0.000441 +/- 0.001323 |
| HIT | 0.999969 +/- 0.000094 | 0.999602 +/- 0.001082 | -0.000366 +/- 0.001095 |
| MAFAULDA | 0.993549 +/- 0.006247 | 0.995611 +/- 0.004732 | 0.002062 +/- 0.003131 |

## Integrity and provenance

- Exact TEST membership verified independently for all 18 prediction files.
- Every checkpoint was rehashed after inference and matched the frozen plan.
- Confusion matrices, class metrics, probabilities, and raw predictions are retained in the per-model report/prediction files.
- Evaluation plan SHA256: `ce22960df67836b0cb49f476b3fe4c2ed29e5cfcb554f8948e7af3f9fc11a55a`
- Publication barrier SHA256: `fa8410c03bcad4d3eea31282ac8ee7828c731a35a025aab872354aa8ffdbe71d`
- Publication driver SHA256: `8f2de50a52ce83fefa8dcd1516ec3c348d9f5c1a6b561b38070fe9f94d149f3e`

`PUBLICATION_FINAL_SEALED_TEST_COMPLETE`
