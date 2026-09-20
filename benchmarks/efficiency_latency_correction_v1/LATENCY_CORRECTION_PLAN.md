# Latency/Throughput Correction Plan

Protocol identity: `lightweight_pcste_efficiency_latency_correction_v1`

Status at freeze: `FROZEN_BEFORE_TIMING`

Publication Git HEAD at freeze: `7c9118e3c272673c34c324d3a2c841a569b20fe9`

Machine-readable plan: `LATENCY_CORRECTION_PLAN.json`

## 1. Why this run exists

**This is not a new scientific experiment.** It corrects exactly one defect in the frozen `lightweight_pcste_efficiency_v1` benchmark and changes nothing else.

`EFFICIENCY_BENCHMARK_PLAN.json` states the measurement mode as `torch.inference_mode(), model .eval()`. The `.eval()` half was applied. **The `torch.inference_mode()` half was never entered on the timed code path**, so every timed forward built an autograd graph it did not need.

Predictive results, parameters, model size, FLOPs, GPU memory and the Q8 extension are unaffected and are **not** recomputed. No file in `benchmarks/efficiency_v1/` or `benchmarks/q8_v1/` is modified.

## 2. The defect, verified from source

Source: `benchmarks/efficiency_v1/run_benchmark.py`, SHA256 `2ab812e9629843b58e01f3c26d0d35a802531b8f461bfd1bc3466d4901e0e04e`.

**Where it was missing:**

| Function | Line | Detail |
|---|---:|---|
| `latency_stage` | 159 | Calls `builders[fam]()` through `_time_loop` (line 141). Neither the closure (`bench_common.model_only_fn` line 229 / `end_to_end_fn` line 205) nor `_time_loop` establishes any grad context. |
| `throughput_stage` | 203 | Inline `fn()` timed via `_time_loop`, likewise with no grad context. |

Neither `torch.inference_mode()` nor `torch.no_grad()` appears anywhere in that chain.

**Where it was present and correct:**

| Function | Line | Detail |
|---|---:|---|
| `flop_report` | 103 | `with torch.inference_mode():` wraps the `FlopCounterMode` forward |
| `memory_stage` subprocess template `_MEM_ONE` | 249, 254 | Both the warm-up forwards and the measured forward are wrapped |

`param_report` (line 43) performs no forward at all — it is pure `numel()` / `element_size()` / `torch.save`.

**Conclusion, from source rather than assumption:**

- **Affected:** CPU batch-1 latency (both scopes), GPU batch-1 latency (both scopes), GPU batch-32 throughput.
- **Unaffected:** parameter counts, serialized model size, FLOPs, GPU peak memory.

This matches the expected interpretation, and it was checked rather than taken on trust.

**Expected direction:** absolute latency should fall. Ratios are *expected* to be roughly preserved, because the omission applied identically to both models within one ABAB-interleaved process — but that is measured here, not assumed.

## 3. Held identical to the original

| | |
|---|---|
| Model pair | **GF1 / seed 42**, the same deterministic cell (lowest fold, then lowest seed) |
| Full-S1 | `7d6cea73ba61b3da23c73be14a8edae048276e88162517ee8d72a7c2a133ba0c` |
| K1 | `10770c5359337cae6e8f82665a3533b87548fa7f57fad42c5d4d6ba2f9952c5b` |
| Verified against | the frozen `efficiency_v1` plan **and** `PUBLICATION_FROZEN_MODEL_TABLE.csv` |
| Inputs | the same deterministic **VALIDATION** windows, verified field-by-field (window id, source file, split, sample range, rate, spec shape, patch grid, normalizer SHA256). **TEST is never touched.** |
| Setup module | `benchmarks/efficiency_v1/bench_common.py` imported **unmodified** (SHA256 `1de4d6081ba76b3f833369d8f21809540210899a2a1b6b8cf0899b06bab98b92`) |
| Scopes | model-only and end-to-end, defined exactly as before; disk I/O excluded from both |
| Timing | `time.perf_counter`, with `torch.cuda.synchronize()` bounding each timed GPU region — the original convention, preserved; CUDA events are not substituted |
| Interleaving | ABAB — rounds alternate Full-S1/K1 within each dataset and scope; the realised order is written to `order_log_*.json` |
| Aggregation | equal-domain mean of the four per-dataset medians; speed-up = Full-S1 / K1 |

## 4. The correction, and how it is proven

The entire warm-up and timed region of both latency and throughput now runs inside `with torch.inference_mode():`.

It is not enough to write that and trust it — this benchmark exists because exactly that assumption failed last time. So:

- `torch.is_inference_mode_enabled()` is asserted **inside the timed loop** at the first sample, the last sample, and every 100th sample, plus once before warm-up.
- A false assertion raises `InferenceModeViolation` and **aborts the run**.
- The assertion tally and violation count are written to `inference_mode_verification_*.json` and published with the results.
- Models are additionally asserted to be in `.eval()` before timing begins.

## 5. Schedule

| Measurement | Batch | Threads | Warm-up | Timed | Same as original? |
|---|---:|---:|---:|---:|---|
| CPU latency | 1 | 1 | 100 | 1000 (4 ABAB rounds) | Yes |
| GPU latency | 1 | — | 200 | 1000 (4 ABAB rounds) | Yes |
| GPU throughput | 32 | — | 50 | 200 | Yes |
| CPU throughput | 32 | 1 | 5 | 25 | **No — see below** |

**CPU batch-32 deviation, stated openly.** `efficiency_v1` measured batch-32 throughput on GPU only. CPU batch-32 is added here because the same infrastructure supports it symmetrically for both models at an identical batch size. But each CPU batch-32 sample takes seconds, so a reduced schedule is used. Each sample already aggregates 32 windows. This reduced schedule is recorded as a deviation rather than presented as equivalent to the batch-1 schedule. It is included only if both models complete cleanly at the identical batch size.

**No valid run is repeated because a speed-up is smaller than expected.** Only an infrastructure-invalid run may be repeated, with the reason recorded.

## 6. Not recomputed

These were unaffected by the defect, remain frozen, and are **referenced from artifacts rather than re-measured or hand-copied**:

| Quantity | Source artifact |
|---|---|
| Parameters, serialized size | `benchmarks/efficiency_v1/parameter_size_results.json` |
| FLOPs | `benchmarks/efficiency_v1/flops_results.json` |
| GPU peak memory | `benchmarks/efficiency_v1/memory_results.json` |
| Predictive results | `results/final_test/publication_final_test_v1/aggregate_summary.json` |
| Q8 extension | `benchmarks/q8_v1/results/q8_aggregate_summary.json` |

## 7. Relationship to Q8

**Q8 is not rerun.** The Q8 extension already measured K1 FP32 versus CPU-dynamic Q8 with inference mode active, and remains frozen. Any Full-S1 → Q8 figure must be a clearly labelled **chained ratio**; direct cross-run division is not permitted unless timing conditions are demonstrably identical.

## 8. Hardware gate

The same physical host as `efficiency_v1` is required: `otter135.eps.surrey.ac.uk`, Intel Core i9-14900, NVIDIA RTX 4000 Ada Generation. If that host or hardware were unavailable the run would stop and report `PUBLICATION_LATENCY_CORRECTION_BLOCKED` rather than silently benchmark elsewhere. Full environment capture is in `environment.json`.

## 9. Prohibitions observed

No retraining · no TEST access or rerun · no modification of frozen predictive results · no modification of `efficiency_v1` or `q8_v1` artifacts · no recomputation of parameters/size/FLOPs · no new quantization experiment · no modification of the dissertation repository · no deletion or rewrite of historical benchmark files.
