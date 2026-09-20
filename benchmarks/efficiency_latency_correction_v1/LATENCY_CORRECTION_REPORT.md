# Corrected Latency & Throughput Benchmark

Protocol identity: `lightweight_pcste_efficiency_latency_correction_v1`

Status: `PUBLICATION_LATENCY_CORRECTION_COMPLETE`

Plan: `LATENCY_CORRECTION_PLAN.json` (SHA256 `a1a327c2529c2169027e88ed2dadb7f2e35569eae2d4b7370da04f9e6e792730`), frozen and hashed **before** any timing.

> **This is the authoritative publication timing result.** `lightweight_pcste_efficiency_v1` remains frozen for provenance, but its absolute latency and throughput timings omitted inference mode. Its parameter, size, FLOP and GPU-memory results remain valid and are reused unchanged.

## 1. The defect, confirmed from source

`EFFICIENCY_BENCHMARK_PLAN.json` states the measurement mode as `torch.inference_mode(), model .eval()`. The `.eval()` half was applied; **the `torch.inference_mode()` half never ran on the timed path**, so every timed forward built an autograd graph it did not need.

Source `benchmarks/efficiency_v1/run_benchmark.py`, SHA256 `2ab812e9629843b58e01f3c26d0d35a802531b8f461bfd1bc3466d4901e0e04e`:

| Where | Line | Status |
|---|---:|---|
| `latency_stage` → `_time_loop` → `bench_common.model_only_fn` / `end_to_end_fn` | 159 / 141 / 229 / 205 | **missing** — no `inference_mode`, no `no_grad` anywhere in the chain |
| `throughput_stage` → `_time_loop` | 203 / 141 | **missing** — inline `fn()` with no grad context |
| `flop_report` | 103 | present and correct |
| `memory_stage` subprocess (`_MEM_ONE`) | 249, 254 | present and correct — wraps warm-up *and* measured forward |
| `param_report` | 43 | not applicable — performs no forward |

**Affected:** CPU batch-1 latency (both scopes), GPU batch-1 latency (both scopes), GPU batch-32 throughput.  
**Unaffected:** parameter counts, serialized model size, FLOPs, GPU peak memory.

This matches the expected interpretation, and was verified against the source rather than assumed.

## 2. How the correction is proven, not merely asserted

This benchmark exists because a stated measurement condition turned out not to hold. So the corrected run does not merely *claim* inference mode — it **verifies it from inside the timed loop**:

- `torch.is_inference_mode_enabled()` asserted at the first, last and every 100th timed sample, plus before every warm-up block.
- A false assertion raises `InferenceModeViolation` and **aborts the run**.
- Models are additionally asserted to be in `.eval()` before timing.

| Stage | Device | Assertions | Violations |
|---|---|---:|---:|
| latency | cpu | 264 | 0 |
| latency | cuda | 264 | 0 |
| throughput | cpu | 24 | 0 |
| throughput | cuda | 32 | 0 |
| **Total** | — | **584** | **0** |

All timed forwards ran under inference mode: **True**.

## 3. Held identical to the original

Same deterministic **GF1 / seed 42** pair (`7d6cea73…` Full-S1, `10770c53…` K1, both re-verified against the frozen `efficiency_v1` plan *and* `PUBLICATION_FROZEN_MODEL_TABLE.csv`); the same deterministic **VALIDATION** windows verified field-by-field; the same scopes; the same ABAB interleaving; the same synchronized host timing; `bench_common.py` imported unmodified. **TEST was never touched.**

## 4. Corrected headline results

Batch-1 medians, equal-domain mean of the four per-dataset medians. Speed-up = Full-S1 / K1.

| Metric | Full-S1 | K1 | Reduction / speed-up |
|---|---:|---:|---:|
| CPU b1 model-only | 54.709 ms | 28.560 ms | 47.80% (1.916×) |
| CPU b1 end-to-end | 55.699 ms | 29.725 ms | 46.63% (1.874×) |
| GPU b1 model-only | 7.711 ms | 4.164 ms | 46.00% (1.852×) |
| GPU b1 end-to-end | 9.577 ms | 5.864 ms | 38.76% (1.633×) |
| GPU b32 throughput | 392.85 win/s | 785.47 win/s | 1.999× |
| CPU b32 throughput | 13.84 win/s | 26.27 win/s | 1.897× |

CPU batch-32 is **new** — `efficiency_v1` measured batch-32 throughput on GPU only. It uses a reduced schedule (5 warm-up / 25 timed) because each CPU batch-32 sample takes seconds and already aggregates 32 windows; that deviation is recorded in the plan rather than presented as equal to the batch-1 schedule.

## 5. Old versus corrected

### Absolute latency (equal-domain mean of per-dataset medians, ms)

| Scope | Old Full-S1 | Corrected Full-S1 | Change | Old K1 | Corrected K1 | Change |
|---|---:|---:|---:|---:|---:|---:|
| CPU model-only | 91.850 | **54.709** | -40.44% | 41.498 | **28.560** | -31.18% |
| CPU end-to-end | 93.327 | **55.699** | -40.32% | 42.837 | **29.725** | -30.61% |
| GPU model-only | 11.811 | **7.711** | -34.71% | 6.368 | **4.164** | -34.62% |
| GPU end-to-end | 13.409 | **9.577** | -28.58% | 7.805 | **5.864** | -24.86% |

Absolute latency fell everywhere, as expected — by roughly 40% on CPU and 30–35% on GPU. The removed cost is autograd graph construction that no inference deployment would pay.

### Ratios — calculated, not assumed

| Scope | Old speed-up | Corrected speed-up | Change |
|---|---:|---:|---:|
| CPU model-only | 2.2134× | **1.9156×** | -0.2978× (-13.45%) |
| CPU end-to-end | 2.1787× | **1.8738×** | -0.3048× (-13.99%) |
| GPU model-only | 1.8547× | **1.8520×** | -0.0026× (-0.14%) |
| GPU end-to-end | 1.7181× | **1.6331×** | -0.0850× (-4.95%) |
| GPU b32 throughput | 1.9453× | **1.9994×** | +0.0541× |

**The ratios were *not* uniformly preserved, and the original assumption that they were is only partly right.**

- **CPU speed-ups were overstated by about 13%**: model-only falls from 2.2134× to **1.9156×**, end-to-end from 2.1787× to **1.8738×**.
- **GPU model-only is essentially unchanged**: 1.8547× → **1.8520×** (-0.14%).
- **GPU end-to-end fell 4.95%**, and **GPU batch-32 throughput rose slightly** (1.9453× → 1.9994×).

Why CPU moved most: autograd bookkeeping is a per-operation overhead, and Full-S1's bidirectional scan issues roughly twice as many small operations as K1's forward-only scan. Removing that overhead therefore takes proportionally more off Full-S1's *apparent* disadvantage on the op-dispatch-bound CPU path than on the GPU, where batch-1 latency is launch-bound for both models.

### Does the qualitative conclusion survive?

**Yes.** `K1 is faster than Full-S1` holds on every device, every scope and every dataset, before and after the correction. What changes is magnitude, not direction: the CPU advantage is ~1.9× rather than ~2.2×. Any published claim of a 2.2× CPU speed-up should be revised to the corrected figure.

## 6. Per-dataset corrected detail

### CPU batch-1, model-only

| Dataset | Model | mean | median | SD | p95 | min | max |
|---|---|---:|---:|---:|---:|---:|---:|
| CWRU | Full-S1 | 19.924 | **19.840** | 0.297 | 20.372 | 19.658 | 23.058 |
| CWRU | K1 | 10.607 | **10.553** | 0.187 | 10.912 | 10.448 | 11.912 |
| JNU | Full-S1 | 82.404 | **82.338** | 0.689 | 83.402 | 80.366 | 87.021 |
| JNU | K1 | 42.044 | **41.885** | 0.451 | 42.800 | 41.685 | 46.252 |
| HIT | Full-S1 | 37.509 | **37.450** | 0.226 | 37.813 | 37.319 | 40.337 |
| HIT | K1 | 19.800 | **19.743** | 0.574 | 20.141 | 19.629 | 37.171 |
| MaFaulDa | Full-S1 | 80.257 | **79.210** | 1.781 | 83.446 | 78.374 | 87.711 |
| MaFaulDa | K1 | 42.143 | **42.058** | 0.297 | 42.762 | 41.814 | 45.459 |

| Dataset | Full-S1 median | K1 median | Speed-up |
|---|---:|---:|---:|
| CWRU | 19.840 | 10.553 | 1.880× |
| JNU | 82.338 | 41.885 | 1.966× |
| HIT | 37.450 | 19.743 | 1.897× |
| MaFaulDa | 79.210 | 42.058 | 1.883× |
| **Macro-4** | **54.709** | **28.560** | **1.916×** |

### CPU batch-1, end-to-end

| Dataset | Model | mean | median | SD | p95 | min | max |
|---|---|---:|---:|---:|---:|---:|---:|
| CWRU | Full-S1 | 20.360 | **20.321** | 0.182 | 20.598 | 20.210 | 22.776 |
| CWRU | K1 | 11.023 | **11.013** | 0.077 | 11.133 | 10.896 | 12.432 |
| JNU | Full-S1 | 80.842 | **80.681** | 0.485 | 81.600 | 79.904 | 87.075 |
| JNU | K1 | 42.408 | **42.322** | 0.281 | 42.939 | 42.193 | 45.059 |
| HIT | Full-S1 | 38.514 | **38.387** | 0.512 | 39.256 | 38.171 | 44.250 |
| HIT | K1 | 20.768 | **20.699** | 0.277 | 21.101 | 20.568 | 23.366 |
| MaFaulDa | Full-S1 | 83.560 | **83.408** | 0.748 | 84.733 | 82.746 | 94.717 |
| MaFaulDa | K1 | 45.029 | **44.867** | 0.732 | 46.042 | 43.939 | 54.271 |

| Dataset | Full-S1 median | K1 median | Speed-up |
|---|---:|---:|---:|
| CWRU | 20.321 | 11.013 | 1.845× |
| JNU | 80.681 | 42.322 | 1.906× |
| HIT | 38.387 | 20.699 | 1.855× |
| MaFaulDa | 83.408 | 44.867 | 1.859× |
| **Macro-4** | **55.699** | **29.725** | **1.874×** |

### GPU batch-1, model-only

| Dataset | Model | mean | median | SD | p95 | min | max |
|---|---|---:|---:|---:|---:|---:|---:|
| CWRU | Full-S1 | 7.493 | **7.468** | 0.172 | 7.618 | 7.390 | 12.285 |
| CWRU | K1 | 4.042 | **4.030** | 0.054 | 4.107 | 3.981 | 4.895 |
| JNU | Full-S1 | 7.805 | **7.786** | 0.071 | 7.909 | 7.721 | 8.432 |
| JNU | K1 | 4.219 | **4.204** | 0.064 | 4.297 | 4.146 | 5.168 |
| HIT | Full-S1 | 7.812 | **7.790** | 0.077 | 7.924 | 7.724 | 8.450 |
| HIT | K1 | 4.221 | **4.207** | 0.056 | 4.310 | 4.161 | 4.857 |
| MaFaulDa | Full-S1 | 7.826 | **7.801** | 0.098 | 7.952 | 7.737 | 9.062 |
| MaFaulDa | K1 | 4.232 | **4.214** | 0.059 | 4.355 | 4.170 | 4.655 |

| Dataset | Full-S1 median | K1 median | Speed-up |
|---|---:|---:|---:|
| CWRU | 7.468 | 4.030 | 1.853× |
| JNU | 7.786 | 4.204 | 1.852× |
| HIT | 7.790 | 4.207 | 1.852× |
| MaFaulDa | 7.801 | 4.214 | 1.851× |
| **Macro-4** | **7.711** | **4.164** | **1.852×** |

### GPU batch-1, end-to-end

| Dataset | Model | mean | median | SD | p95 | min | max |
|---|---|---:|---:|---:|---:|---:|---:|
| CWRU | Full-S1 | 8.073 | **8.051** | 0.082 | 8.217 | 7.973 | 8.716 |
| CWRU | K1 | 4.621 | **4.608** | 0.066 | 4.707 | 4.551 | 5.835 |
| JNU | Full-S1 | 10.559 | **10.594** | 0.168 | 10.674 | 9.839 | 10.891 |
| JNU | K1 | 6.513 | **6.740** | 0.333 | 6.802 | 6.007 | 8.000 |
| HIT | Full-S1 | 9.095 | **9.094** | 0.023 | 9.134 | 9.032 | 9.182 |
| HIT | K1 | 5.376 | **5.373** | 0.030 | 5.411 | 5.331 | 6.061 |
| MaFaulDa | Full-S1 | 10.579 | **10.568** | 0.064 | 10.664 | 10.476 | 11.694 |
| MaFaulDa | K1 | 6.738 | **6.736** | 0.026 | 6.774 | 6.687 | 6.906 |

| Dataset | Full-S1 median | K1 median | Speed-up |
|---|---:|---:|---:|
| CWRU | 8.051 | 4.608 | 1.747× |
| JNU | 10.594 | 6.740 | 1.572× |
| HIT | 9.094 | 5.373 | 1.693× |
| MaFaulDa | 10.568 | 6.736 | 1.569× |
| **Macro-4** | **9.577** | **5.864** | **1.633×** |

### Throughput, batch 32

| Device | Dataset | Full-S1 win/s | K1 win/s | Ratio |
|---|---|---:|---:|---:|
| GPU | CWRU | 950.16 | 1918.56 | 2.019× |
| GPU | JNU | 136.02 | 267.25 | 1.965× |
| GPU | HIT | 349.33 | 688.36 | 1.971× |
| GPU | MaFaulDa | 135.90 | 267.72 | 1.970× |
| **GPU** | **Macro-4** | **392.85** | **785.47** | **1.999×** |
| CPU | CWRU | 30.71 | 58.07 | 1.891× |
| CPU | JNU | 5.59 | 10.78 | 1.926× |
| CPU | HIT | 13.37 | 25.22 | 1.887× |
| CPU | MaFaulDa | 5.70 | 11.00 | 1.931× |
| **CPU** | **Macro-4** | **13.84** | **26.27** | **1.897×** |

## 7. Carried forward unchanged

Read programmatically from the frozen artifacts named below — not re-measured, not hand-entered:

| Quantity | Full-S1 | K1 | Source |
|---|---:|---:|---|
| Encoder parameters | 2,382,033 | 1,375,953 | `efficiency_v1/parameter_size_results.json` |
| Complete model parameters | 2,385,893 | 1,379,813 | same |
| FP32 state_dict | 9.135 MiB | 5.286 MiB | same |
| Macro-4 GFLOP / window | 2.6067 | 1.4236 | `efficiency_v1/flops_results.json` |
| GPU peak memory (b1) | 29.466 MiB | 24.823 MiB | `efficiency_v1/memory_results.json` |

GPU memory was **not** rerun: source inspection confirms `memory_stage` already wrapped both its warm-up and measured forwards in `torch.inference_mode()` (lines 249, 254), so it was never affected. Its reduction is 15.76%.

## 8. Relationship to Q8

**Q8 was not rerun** and remains frozen. Its extension already measured K1 FP32 versus CPU-dynamic Q8 with inference mode active, in one process.

A Full-S1 → Q8 CPU figure must therefore be a **chained ratio**, never a cross-run division: the corrected Full-S1/K1 CPU model-only ratio (1.916×) composed with the Q8 extension's K1/Q8 ratio (1.084×) gives about **2.077×**. That estimate assumes the two ratios compose; both were measured under inference mode on the same host, which makes the assumption more defensible than before this correction, but it remains an assumption and is labelled as such.

## 9. Provenance

`benchmarks/efficiency_v1/` is **unmodified** — all 13 of its artifact hashes verify. This correction adds a new directory and rewrites no history. The old absolute timings remain on record as the measurements that were actually taken; they are simply no longer the publication-facing numbers.

