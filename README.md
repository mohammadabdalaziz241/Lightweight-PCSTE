# Lightweight PC-STE for Rotating Machinery Fault Diagnosis

This repository contains the publication-oriented implementation, frozen protocol, and sealed final results for compressing a shared multi-dataset PC-STE fault-diagnosis encoder into the K1 lightweight model through knowledge distillation.

The scientific workflow is:

`Full PC-STE / S1 teacher ensemble -> knowledge-distilled K1 encoder -> deployment and efficiency evaluation`

K1 uses **one shared 1,375,953-parameter encoder** across CWRU, JNU, HIT, and MaFaulDa, with dataset-specific classification heads. The matched design covers three global folds x three random seeds (nine matched cells, eighteen checkpoints).

> This repository is currently private while the associated manuscript and the deployment/efficiency measurement stage are being completed.

## Status

- **K1 training is complete: 9/9 cells** (GF1 3/3, GF2 3/3, GF3 3/3), frozen in `configs/lightweight_k1/K1_9_CELL_FREEZE_MANIFEST.md` (SHA256 `a9e5fb49a1f38a7d6e2649ff8d155cf619cd55572bf7238b06a1fcb120d64a8b`).
- **The final sealed publication TEST is complete** under protocol `lightweight_pcste_final_test_v1`, status `PUBLICATION_FINAL_SEALED_TEST_COMPLETE`.
- Checkpoint selection used **validation only** (maximum validation Macro-Domain F1, strict improvement, earliest-epoch tie retention, no early stopping). No TEST metric entered training or selection.
- **TEST remained sealed** until all eighteen checkpoints *and* the publication evaluation plan were frozen and externally hash-pinned. TEST was then executed exactly once under that frozen plan.
- **Efficiency benchmarking is complete** under `lightweight_pcste_efficiency_v1` (see below). The optional Q8 quantization extension has **not** been run.

## Design

| Property | Value |
|---|---|
| Datasets | CWRU, JNU, HIT, MaFaulDa |
| Folds x seeds | 3 global folds x seeds 42, 1337, 2026 = 9 matched cells |
| Teacher / base | Full-S1 (full PC-STE v2 S1) |
| Student | K1, `half_4x1`, 1,375,953 encoder parameters, shared encoder + per-dataset heads |
| KD temperature | 4 |
| KD alpha | 0.5 |
| Relational weight | 1.0 |
| Teacher aggregation | `mean_prob_at_T` (same-fold three-seed S1 ensemble) |
| KL direction | teacher \|\| student |
| Epochs per cell | 50 |
| Checkpoint selection | validation-only max Macro-Domain F1 |
| Primary endpoint | Macro-4 Macro-F1 (equal weight over the four datasets) |
| Non-inferiority margin | -0.02 (frozen before TEST) |

## Final sealed TEST results

Nine matched cells, paired within (fold, seed). Mean +/- sample SD (`ddof=1`).

### Macro-4 Macro-F1 (primary endpoint)

| Metric | Full-S1 | K1 | Paired delta (K1 - Full-S1) | K1 wins/ties/losses |
|---|---:|---:|---:|---:|
| Macro-4 Macro-F1 | 0.934644 +/- 0.023258 | 0.955334 +/- 0.018393 | +0.020691 +/- 0.013509 | 8 / 0 / 1 |
| Macro-4 Macro-AUC | 0.991793 +/- 0.008102 | 0.996398 +/- 0.002367 | +0.004606 +/- 0.006794 | 8 / 0 / 1 |

### Per-dataset Macro-F1

| Dataset | Full-S1 | K1 | Paired delta |
|---|---:|---:|---:|
| CWRU | 0.854747 +/- 0.083579 | 0.918945 +/- 0.073432 | +0.064198 +/- 0.044152 |
| JNU | 0.971429 +/- 0.042857 | 0.980952 +/- 0.037796 | +0.009524 +/- 0.028571 |
| HIT | 0.991039 +/- 0.017798 | 0.988728 +/- 0.016133 | -0.002311 +/- 0.007427 |
| MaFaulDa | 0.921360 +/- 0.016432 | 0.932712 +/- 0.017321 | +0.011352 +/- 0.019450 |

### Per-dataset Macro-AUC

| Dataset | Full-S1 | K1 | Paired delta |
|---|---:|---:|---:|
| CWRU | 0.974093 +/- 0.033643 | 0.990379 +/- 0.009921 | +0.016286 +/- 0.027983 |
| JNU | 0.999559 +/- 0.001323 | 1.000000 +/- 0.000000 | +0.000441 +/- 0.001323 |
| HIT | 0.999969 +/- 0.000094 | 0.999602 +/- 0.001082 | -0.000366 +/- 0.001095 |
| MaFaulDa | 0.993549 +/- 0.006247 | 0.995611 +/- 0.004732 | +0.002062 +/- 0.003131 |

### Predefined confirmatory result

The single pre-registered confirmatory contrast is non-inferiority of K1 against Full-S1 on Macro-4 Macro-F1 at the frozen margin `-0.02`, tested by exact sign-flip enumeration over the nine paired differences (all 512 sign patterns).

- **Non-inferiority criterion: SATISFIED.** Exact one-sided sign-flip p = `0.001953125`.
- The hierarchically gated superiority test, reported only because the non-inferiority gate passed, gives exact two-sided sign-flip p = `0.0078125`.

These are exact permutation p-values over nine matched cells, not large-sample tests. The point estimate favours K1, but the study was designed and powered as a non-inferiority comparison; the superiority result is a gated secondary observation from a nine-cell paired design and should be read as supportive rather than definitive. One cell (fold 1, seed 1337) favours Full-S1, and HIT shows a small negative mean delta on both metrics.

Authoritative artifacts: `results/final_test/publication_final_test_v1/`.

## Efficiency and deployment

Fresh publication-grade efficiency measurements were taken under `lightweight_pcste_efficiency_v1` on a single idle RTX 4000 Ada host, using the deterministically fixed **GF1 / seed 42** checkpoint pair (rule: lowest fold, then lowest seed) and deterministic VALIDATION inputs. No dissertation-era efficiency number is reused.

| Metric | Full-S1 | K1 | Reduction / speed-up |
|---|---:|---:|---:|
| Encoder parameters | 2,382,033 | 1,375,953 | 42.24% (1.73×) |
| Complete model parameters | 2,385,893 | 1,379,813 | 42.17% |
| FP32 state_dict size | 9.135 MiB | 5.286 MiB | 42.14% |
| Macro-4 FLOPs / 1 s window | 2.6067 GFLOP | 1.4236 GFLOP | 45.38% (1.83×) |
| CPU batch-1 model-only latency | 91.850 ms | 41.498 ms | 54.82% (2.21×) |
| CPU batch-1 end-to-end latency | 93.327 ms | 42.837 ms | 54.10% (2.18×) |
| GPU batch-1 model-only latency | 11.811 ms | 6.368 ms | 46.08% (1.86×) |
| GPU batch-1 end-to-end latency | 13.409 ms | 7.805 ms | 41.79% (1.72×) |
| GPU batch-32 throughput | 382.2 windows/s | 743.5 windows/s | 1.95× |
| GPU peak memory (batch 1) | 29.466 MiB | 24.823 MiB | 15.76% |

Latency figures are batch-1 medians aggregated as the equal-domain mean of the four per-dataset medians (CPU single-thread; 100/200 warm-up and 1000 timed iterations per cell, ABAB-interleaved). Model-only and end-to-end scopes are reported separately and never mixed; disk I/O is excluded from both. FLOPs are the explicit sum of a profiler dense term and an analytic selective-scan term — never a profiler count alone.

**Caveat carried on every timing figure:** all measurements use the pure-PyTorch reference selective scan, as fused Mamba kernels are unavailable on this stack. Absolute latency, and possibly the Full-S1 / K1 ratio, would differ with fused kernels. Results are from one representative cell on one host.

Full detail: [`benchmarks/efficiency_v1/EFFICIENCY_FINAL_REPORT.md`](benchmarks/efficiency_v1/EFFICIENCY_FINAL_REPORT.md).

The optional Q8 quantization arm remains a separate, not-yet-executed decision.

## Layout

| Path | Contents |
|---|---|
| `src/` | Historical scientific implementation, import layout preserved |
| `scripts/train_ssl/`, `scripts/train_s1/` | Exact SSL and Full-S1 executors |
| `scripts/train_k1/02_k1_production.py` | K1 production executor |
| `scripts/evaluate/` | Historical evaluator and the frozen publication TEST driver |
| `configs/final_s1/` | Frozen Full-S1 specification, statistical plan, selected checkpoints |
| `configs/lightweight_k1/` | Frozen K1 protocol, registered cells, freeze manifest |
| `configs/lightweight_k1/final_test_v1/` | Publication TEST pre-registration (plan, model table, barrier, driver) |
| `protocols/` | Frozen dataset and split manifests |
| `results/lightweight_k1/` | Per-cell validation-stage summaries (9 cells, no checkpoints) |
| `results/final_test/publication_final_test_v1/` | Sealed final TEST results and per-model class-level reports |
| `benchmarks/efficiency_v1/` | Frozen efficiency benchmark: plan, driver, results, final report |
| `docs/` | Protocol, reproducibility, results, artifact policy, inventory |
| `FROZEN_ARTIFACT_HASHES.sha256` | SHA256 manifest of every frozen artifact tracked here |

Verify frozen artifacts with `sha256sum -c FROZEN_ARTIFACT_HASHES.sha256`.

See `docs/EXPERIMENT_PROTOCOL.md`, `docs/RESULTS.md`, `docs/REPRODUCIBILITY.md`, and `docs/ARTIFACTS.md` before using or extending this repository.
