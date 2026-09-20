# Efficiency Benchmark — Frozen Full-S1 vs Frozen K1

Benchmark identity: `lightweight_pcste_efficiency_v1`

Status: `PUBLICATION_EFFICIENCY_BENCHMARK_COMPLETE`

Plan: `EFFICIENCY_BENCHMARK_PLAN.json` (SHA256 `f88db6d957a1f6eec4b3dc8199f7606eb7c73be0958a87322fa297124ff6595b`), frozen before any latency measurement.

Representative pair: **GF1 / seed 42**, fixed by the deterministic rule *lowest fold, then lowest seed*, recorded before measurement and never chosen from efficiency outcomes. Both checkpoint hashes were independently recomputed and matched `PUBLICATION_FROZEN_MODEL_TABLE.csv`.

No model was retrained, no TEST data was read, no frozen artifact was modified, and Q8 was not executed.

## Headline

All latency figures are batch-1 medians aggregated as the **equal-domain mean of the four per-dataset medians**. Speed-up is Full-S1 / K1.

| Metric | Full-S1 | K1 | Reduction / speed-up |
|---|---:|---:|---:|
| Encoder parameters | 2,382,033 | 1,375,953 | 42.24% (1.731×) |
| Complete model parameters | 2,385,893 | 1,379,813 | 42.17% |
| FP32 state_dict size | 9.135 MiB | 5.286 MiB | 42.14% |
| Macro-4 FLOPs / window | 2.6067 GFLOP | 1.4236 GFLOP | 45.38% (1.831×) |
| CPU b1 model-only latency | 91.850 ms | 41.498 ms | 54.82% (2.213×) |
| CPU b1 end-to-end latency | 93.327 ms | 42.837 ms | 54.10% (2.179×) |
| GPU b1 model-only latency | 11.811 ms | 6.368 ms | 46.08% (1.855×) |
| GPU b1 end-to-end latency | 13.409 ms | 7.805 ms | 41.79% (1.718×) |
| GPU b32 throughput | 382.2 win/s | 743.5 win/s | 1.945× |
| GPU peak memory (b1) | 29.466 MiB | 24.823 MiB | 15.76% |

GPU memory is included in the headline because both models were measured in independent clean subprocesses and the two repeats agreed **byte-for-byte** on every dataset.

## Environment

Both models were benchmarked on the **same physical host and hardware**, in the same process for latency and throughput, with ABAB interleaving.

| | |
|---|---|
| Host | `otter135.eps.surrey.ac.uk` |
| CPU | Intel(R) Core(TM) i9-14900 — 24 physical / 32 logical cores |
| RAM | 62.5 GiB |
| GPU | NVIDIA RTX 4000 Ada Generation, 580.126.09, 20475 MiB, P8, Enabled, 3105 MHz, 130.00 W |
| CUDA / cuDNN | 13.0 / 92000 |
| Python / PyTorch | 3.12.3 / 2.12.0+cu130 |
| NumPy / SciPy / pandas | 2.4.4 / 1.18.0 / 3.0.3 |
| CPU threads (measurement) | `torch.set_num_threads(1)` |
| Host load at capture | `0.35 0.16 0.13 1/737 154136` |
| GPU state at capture | 0 %, 0 %, processes: (none) |

## Parameters and size

| Component | Full-S1 | K1 | Reduction |
|---|---:|---:|---:|
| Encoder — total params | 2,382,033 | 1,375,953 | 42.24% |
| Encoder — raw FP32 parameter bytes | 9,528,132 | 5,503,812 | 42.24% |
| Encoder — serialized FP32 state_dict | 9,559,777 | 5,524,156 | 42.21% |
| Encoder + all 4 dataset heads — total params | 2,385,893 | 1,379,813 | 42.17% |
| Encoder + all 4 dataset heads — raw FP32 parameter bytes | 9,543,572 | 5,519,252 | 42.17% |
| Encoder + all 4 dataset heads — serialized FP32 state_dict | 9,578,353 | 5,542,468 | 42.14% |

All parameters are trainable in both models (Full-S1 2,385,893 of 2,385,893; K1 1,379,813 of 1,379,813). Optimizer state is not counted. The four dataset heads contribute 3,860 parameters in **both** models — the head stack is identical, so all reduction comes from the encoder.

The K1 encoder count is exactly **1,375,953**, matching its frozen architectural identity.

*Raw FP32 parameter bytes* is `sum(p.numel() × p.element_size())`. *Serialized state_dict* is `torch.save` of a state-dict-only FP32 payload written by an identical procedure for both models to a temporary file, deleted immediately after measurement. Training-checkpoint file sizes are deliberately **not** compared: the two training stacks package checkpoints differently.

## FLOPs per 1-second window

Reported quantity: **FLOPs, not MACs**, as the explicit sum `dense + scan`. The dense term is `FlopCounterMode` (conv/addmm/mm/bmm, 2×MAC convention); it is never reported alone as a total. The scan term is analytic: `n_bands × Σ_layers(directions) × T × d_inner × d_state × C` with `C = 6`, the constant preserved from the historical Part-6 harness.

| Dataset | Full-S1 dense | Full-S1 scan | Full-S1 total | K1 dense | K1 scan | K1 total | Reduction | Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CWRU | 0.9206 | 0.0610 | **0.9816** | 0.5057 | 0.0305 | **0.5362** | 45.38% | 1.831× |
| JNU | 3.5217 | 0.2336 | **3.7553** | 1.9341 | 0.1168 | **2.0509** | 45.39% | 1.831× |
| HIT | 1.8142 | 0.1203 | **1.9345** | 0.9964 | 0.0602 | **1.0565** | 45.39% | 1.831× |
| MaFaulDa | 3.5217 | 0.2336 | **3.7553** | 1.9341 | 0.1168 | **2.0509** | 45.39% | 1.831× |
| **Macro-4 (equal domain)** | 2.4445 | 0.1621 | **2.6067** | 1.3426 | 0.0811 | **1.4236** | 45.38% | 1.831× |

All values in GFLOP per one 1-second window. The Macro-4 row is the unweighted mean over the four datasets — not weighted by sample count.

Grid parameters taken from the live representation: CWRU 9 bands × 23 time patches, JNU and MaFaulDa 33 × 24, HIT 17 × 24; `d_inner` 384, `d_state` 16; summed directions 8 for Full-S1 (bidirectional) and 4 for K1 (forward-only).

**Sensitivity.** A strict per-operation tally of the reference scan gives roughly 8 rather than 6, which would scale the scan term by 4/3 and raise the Macro-4 totals to about 2.6607 and 1.4507 GFLOP. Because the identical constant applies to both models, the reduction and ratio are **unchanged**. The scan term is only 6.2% of the Full-S1 Macro-4 total, so this choice has little influence on the headline.

## CPU batch-1 latency

FP32, `.eval()`, `torch.inference_mode()`, batch 1, single thread (`torch.set_num_threads(1)`). 100 warm-up and 1000 timed iterations per cell, ABAB-interleaved over 4 rounds. Milliseconds per window.

### CPU — Model-only

| Dataset | Model | mean | median | SD | p95 | min | max |
|---|---|---:|---:|---:|---:|---:|---:|
| CWRU | Full-S1 | 39.923 | **39.773** | 1.607 | 40.919 | 38.247 | 67.525 |
| CWRU | K1 | 17.557 | **17.427** | 1.259 | 18.011 | 17.142 | 44.950 |
| JNU | Full-S1 | 133.589 | **132.518** | 4.646 | 141.046 | 128.622 | 163.933 |
| JNU | K1 | 58.172 | **58.010** | 1.000 | 58.844 | 57.597 | 84.686 |
| HIT | Full-S1 | 62.836 | **62.670** | 1.763 | 63.201 | 62.123 | 90.982 |
| HIT | K1 | 32.403 | **32.348** | 0.284 | 32.791 | 31.962 | 35.329 |
| MaFaulDa | Full-S1 | 132.602 | **132.437** | 2.083 | 134.508 | 129.263 | 159.515 |
| MaFaulDa | K1 | 58.292 | **58.205** | 0.348 | 58.814 | 57.762 | 64.161 |

| Dataset | Full-S1 median | K1 median | Speed-up | Latency reduction |
|---|---:|---:|---:|---:|
| CWRU | 39.773 | 17.427 | 2.282× | 56.18% |
| JNU | 132.518 | 58.010 | 2.284× | 56.22% |
| HIT | 62.670 | 32.348 | 1.937× | 48.38% |
| MaFaulDa | 132.437 | 58.205 | 2.275× | 56.05% |
| **Macro-4 (equal domain)** | **91.850** | **41.498** | **2.213×** | **54.82%** |

### CPU — End-to-end

| Dataset | Model | mean | median | SD | p95 | min | max |
|---|---|---:|---:|---:|---:|---:|---:|
| CWRU | Full-S1 | 40.507 | **40.375** | 1.381 | 41.549 | 38.944 | 67.956 |
| CWRU | K1 | 18.204 | **18.054** | 1.269 | 18.742 | 17.760 | 45.555 |
| JNU | Full-S1 | 136.119 | **134.820** | 5.204 | 146.352 | 129.651 | 177.917 |
| JNU | K1 | 60.117 | **60.064** | 0.974 | 60.680 | 59.315 | 87.975 |
| HIT | Full-S1 | 63.957 | **63.816** | 1.696 | 64.318 | 63.305 | 90.928 |
| HIT | K1 | 33.433 | **33.407** | 0.210 | 33.792 | 32.891 | 35.134 |
| MaFaulDa | Full-S1 | 134.422 | **134.296** | 2.171 | 136.548 | 129.363 | 163.179 |
| MaFaulDa | K1 | 59.897 | **59.822** | 0.382 | 60.459 | 59.166 | 65.396 |

| Dataset | Full-S1 median | K1 median | Speed-up | Latency reduction |
|---|---:|---:|---:|---:|
| CWRU | 40.375 | 18.054 | 2.236× | 55.28% |
| JNU | 134.820 | 60.064 | 2.245× | 55.45% |
| HIT | 63.816 | 33.407 | 1.910× | 47.65% |
| MaFaulDa | 134.296 | 59.822 | 2.245× | 55.45% |
| **Macro-4 (equal domain)** | **93.327** | **42.837** | **2.179×** | **54.10%** |

## GPU batch-1 latency

FP32, `.eval()`, `torch.inference_mode()`, batch 1, `torch.cuda.synchronize()` around each timed region. 200 warm-up and 1000 timed iterations per cell, ABAB-interleaved over 4 rounds. Milliseconds per window.

### GPU — Model-only

| Dataset | Model | mean | median | SD | p95 | min | max |
|---|---|---:|---:|---:|---:|---:|---:|
| CWRU | Full-S1 | 11.514 | **11.355** | 1.832 | 11.810 | 11.102 | 42.066 |
| CWRU | K1 | 6.138 | **6.125** | 0.073 | 6.256 | 5.941 | 6.664 |
| JNU | Full-S1 | 12.088 | **11.980** | 1.270 | 12.344 | 11.694 | 40.199 |
| JNU | K1 | 6.519 | **6.445** | 1.239 | 6.590 | 6.269 | 34.457 |
| HIT | Full-S1 | 12.063 | **11.931** | 1.552 | 12.285 | 11.629 | 40.183 |
| HIT | K1 | 6.504 | **6.456** | 0.888 | 6.611 | 6.272 | 34.434 |
| MaFaulDa | Full-S1 | 12.115 | **11.977** | 1.695 | 12.319 | 11.696 | 46.186 |
| MaFaulDa | K1 | 6.497 | **6.447** | 0.890 | 6.608 | 6.260 | 34.499 |

| Dataset | Full-S1 median | K1 median | Speed-up | Latency reduction |
|---|---:|---:|---:|---:|
| CWRU | 11.355 | 6.125 | 1.854× | 46.06% |
| JNU | 11.980 | 6.445 | 1.859× | 46.21% |
| HIT | 11.931 | 6.456 | 1.848× | 45.89% |
| MaFaulDa | 11.977 | 6.447 | 1.858× | 46.17% |
| **Macro-4 (equal domain)** | **11.811** | **6.368** | **1.855×** | **46.08%** |

### GPU — End-to-end

| Dataset | Model | mean | median | SD | p95 | min | max |
|---|---|---:|---:|---:|---:|---:|---:|
| CWRU | Full-S1 | 11.851 | **11.760** | 1.254 | 12.071 | 11.535 | 39.751 |
| CWRU | K1 | 6.674 | **6.593** | 1.239 | 6.783 | 6.460 | 34.307 |
| JNU | Full-S1 | 14.364 | **14.220** | 1.549 | 14.708 | 13.900 | 42.501 |
| JNU | K1 | 8.601 | **8.471** | 1.339 | 8.935 | 8.304 | 39.956 |
| HIT | Full-S1 | 13.628 | **13.492** | 1.794 | 13.776 | 13.157 | 42.249 |
| HIT | K1 | 7.754 | **7.749** | 0.043 | 7.824 | 7.573 | 8.080 |
| MaFaulDa | Full-S1 | 14.259 | **14.165** | 1.294 | 14.477 | 13.819 | 43.098 |
| MaFaulDa | K1 | 8.487 | **8.405** | 1.285 | 8.518 | 8.220 | 37.011 |

| Dataset | Full-S1 median | K1 median | Speed-up | Latency reduction |
|---|---:|---:|---:|---:|
| CWRU | 11.760 | 6.593 | 1.784× | 43.93% |
| JNU | 14.220 | 8.471 | 1.679× | 40.43% |
| HIT | 13.492 | 7.749 | 1.741× | 42.56% |
| MaFaulDa | 14.165 | 8.405 | 1.685× | 40.66% |
| **Macro-4 (equal domain)** | **13.409** | **7.805** | **1.718×** | **41.79%** |

### Cost of the signal-processing front end

End-to-end minus model-only, per dataset (ms), identical STFT + N2b normalisation for both models:

| Dataset | CPU Full-S1 | CPU K1 | GPU Full-S1 | GPU K1 |
|---|---:|---:|---:|---:|
| CWRU | 0.602 | 0.627 | 0.405 | 0.469 |
| JNU | 2.302 | 2.054 | 2.239 | 2.026 |
| HIT | 1.146 | 1.059 | 1.561 | 1.294 |
| MaFaulDa | 1.859 | 1.617 | 2.188 | 1.958 |

The front end is a fixed additive cost shared by both models, so it dilutes the end-to-end speed-up relative to model-only. The effect is largest on GPU, where the model itself is fastest.

## Throughput — GPU batch 32, model-only

Batch 32 was fixed in advance for both models and succeeded for both on every dataset; no per-model batch tuning was performed.

| Dataset | Full-S1 win/s | K1 win/s | Full-S1 ms/window | K1 ms/window | Speed-up |
|---|---:|---:|---:|---:|---:|
| CWRU | 934.7 | 1808.4 | 1.070 | 0.553 | 1.935× |
| JNU | 135.2 | 266.0 | 7.398 | 3.760 | 1.968× |
| HIT | 323.9 | 633.7 | 3.087 | 1.578 | 1.956× |
| MaFaulDa | 135.1 | 266.0 | 7.402 | 3.760 | 1.969× |
| **Macro-4 (equal domain)** | **382.2** | **743.5** | — | — | **1.945×** |

## Memory — GPU batch 1, model-only

Each model measured in its own clean subprocess with `torch.cuda.reset_peak_memory_stats()` and synchronization. Two independent repeats agreed byte-for-byte on every dataset, so this is reported as a valid headline figure rather than exploratory.

| Quantity | Full-S1 | K1 | Reduction |
|---|---:|---:|---:|
| Resident after model load | 9.110 MiB | 5.272 MiB | 42.13% |
| Peak allocated during forward — CWRU | 22.693 MiB | 18.552 MiB | 18.25% |
| Peak allocated during forward — JNU | 34.261 MiB | 29.263 MiB | 14.59% |
| Peak allocated during forward — HIT | 26.649 MiB | 22.214 MiB | 16.64% |
| Peak allocated during forward — MaFaulDa | 34.261 MiB | 29.263 MiB | 14.59% |
| **Peak allocated — Macro-4 mean** | **29.466 MiB** | **24.823 MiB** | **15.76%** |

Peak memory falls by much less than parameters (≈16% vs ≈42%) because activation tensors, which scale with the representation grid rather than with model depth or directionality, dominate the batch-1 footprint.

## Context — already-frozen predictive result

Read from `results/final_test/publication_final_test_v1/aggregate_summary.json`; **not rerun and not re-derived here**.

| Sealed TEST metric | Full-S1 | K1 |
|---|---:|---:|
| Macro-4 Macro-F1 | 0.934644 ± 0.023258 | 0.955334 ± 0.018393 |
| Paired delta (K1 − Full-S1) | — | +0.020691 ± 0.013509 |

The pre-registered non-inferiority criterion at margin −0.02 was satisfied. These values are context for a future performance-versus-efficiency figure. **No manuscript claim is made here**, and nothing in this benchmark alters any frozen predictive result.

## Interpretation and limits

- K1 removes the backward scan direction, halving recurrent work while keeping depth, width and the head stack. Parameters fall 42.24% and Macro-4 FLOPs 45.38%.

- Measured CPU speed-up (2.21×) **exceeds** the FLOP ratio (1.83×), because the reference selective scan is a sequential Python loop whose iteration count halves with direction count — a latency saving that FLOP accounting does not capture.

- GPU speed-up (1.85×) tracks the FLOP ratio more closely; at batch 1 the GPU is launch-bound rather than compute-bound.

- All timings use the **pure-PyTorch reference selective scan**; fused Mamba kernels are unavailable on this stack. Absolute latency would change materially with fused kernels, and the Full-S1 / K1 ratio could change too. This caveat applies to every latency, throughput and memory figure here.

- Results are for **one representative cell (GF1/seed 42)** on **one host**. Efficiency is an architectural property and both models share fold- and seed-independent structure, so these figures are expected to transfer across cells; that was not measured.

- Single-thread CPU is the deployment-facing primary. Multi-thread CPU scaling was not measured.

## Reproduction

```bash
python benchmarks/efficiency_v1/run_benchmark.py --stage env
python benchmarks/efficiency_v1/run_benchmark.py --stage params
python benchmarks/efficiency_v1/run_benchmark.py --stage flops
python benchmarks/efficiency_v1/run_benchmark.py --stage latency --device cpu
python benchmarks/efficiency_v1/run_benchmark.py --stage latency --device cuda
python benchmarks/efficiency_v1/run_benchmark.py --stage throughput --device cuda
python benchmarks/efficiency_v1/run_benchmark.py --stage memory
```

The frozen checkpoints and normalizers are excluded from Git by the artifact policy; `bench_common.py` reads them from the canonical scientific repository (override with `PCSTE_CANONICAL_REPO`). Raw datasets must be reachable at `<repo>/data` — see `docs/DATASETS.md`.

No historical benchmark source was edited: `src/methodology_v2/compression/benchmark.py` remains at SHA256 `641f437b670b03d5bffb3ed11bfa2c8a758266392beec1988b90b8f4cecd211b`.

