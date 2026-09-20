# Efficiency Benchmark Plan — Frozen Full-S1 vs Frozen K1

Benchmark identity: `lightweight_pcste_efficiency_v1`

Status at freeze: `FROZEN_BEFORE_MEASUREMENT`

Publication Git HEAD at freeze: `5bf74db7c6ed8587687746fa53f9221004372236`

Machine-readable plan: `EFFICIENCY_BENCHMARK_PLAN.json`

This plan is written and hashed **before any latency result is collected**. It measures the computational and deployment cost of two already-frozen models. It does not retrain, does not touch TEST, does not run Q8, and does not modify any frozen predictive artifact.

## 1. Model pair — fixed by rule, not by outcome

Efficiency is an architectural property, so the pair is pinned by a deterministic rule recorded before measurement: **lowest fold, then lowest seed** → **GF1 / seed 42**. No other cell was benchmarked and no cell was selected on the basis of an efficiency result.

| Family | Checkpoint | SHA256 |
|---|---|---|
| Full-S1 | `results/pcste_v2/final_v1/pcstev2_final_v1_s1_gf1_s42/best.pt` | `7d6cea73ba61b3da23c73be14a8edae048276e88162517ee8d72a7c2a133ba0c` |
| K1 | `results/pcste_v2/lightweight_v1/pcstev2_lightweight_v1_k1_gf1_s42/best.pt` | `10770c5359337cae6e8f82665a3533b87548fa7f57fad42c5d4d6ba2f9952c5b` |

Both hashes were independently recomputed against `PUBLICATION_FROZEN_MODEL_TABLE.csv` before benchmarking. Both state dictionaries strict-load. The K1 encoder is re-checked against its frozen architectural identity of **1,375,953** parameters.

| | Full-S1 | K1 |
|---|---|---|
| Temporal layer | `BiMambaLayer` | `UniMambaLayer` |
| Directions per layer | 2 | 1 |
| Blocks | 4 | 4 |
| `d_model` / `d_inner` / `d_state` / `dt_rank` / `d_conv` | 192 / 384 / 16 / 12 / 4 | 192 / 384 / 16 / 12 / 4 |
| Heads | `DatasetHeads` | `DatasetHeads` (identical class) |

## 2. Inputs — VALIDATION only, never TEST

The manifest is filtered to `split == "validation"` and the **first window per dataset in frozen manifest order** is taken. Each selected row is asserted to be a validation row. The identical input is used for both models.

| Dataset | Window ID | Raw | STFT (n_fft, hop) | Spec (bins × frames) | Patch grid (bands × T) | Head classes |
|---|---|---|---|---|---|---:|
| CWRU | `n12L:CWRU:cwru_n12_DE_120:DE:validation:0-12000` | 12 000 @ 12 kHz = 1.0 s | (256, 64) | 129 × 184 | 9 × 23 | 3 |
| JNU | `f1:JNU:jnu_ib1000_2:D:acc_vertical:validation:325300-375300` | 50 000 @ 50 kHz = 1.0 s | (1024, 256) | 513 × 192 | 33 × 24 | 4 |
| HIT | `f1:HIT:hit_data1_rec02:ch3:validation:0-25000` | 25 000 @ 25 kHz = 1.0 s | (512, 128) | 257 × 192 | 17 × 24 | 3 |
| MaFaulDa | `v2f1:MAFAULDA:mafaulda_horizontal-misalignment_1.5mm_12.288:col3:validation:0-50000` | 50 000 @ 50 kHz = 1.0 s | (1024, 256) | 513 × 192 | 33 × 24 | 10 |

Every window is exactly one second, so per-window cost is directly per-second-of-signal cost. Normalizer SHA256 values are pinned in the JSON plan.

## 3. Two timing scopes, never mixed

- **Model-only** — prepared resident representation → encoder → dataset head → logits. Isolates architectural efficiency.
- **End-to-end** — resident raw 1 s waveform → `stft_rep` (log1p|STFT|) → N2b normalisation → collate → encoder → head → logits. Deployment-facing, including the common signal-processing front end.

**Disk I/O is excluded from both.** The waveform is read into memory once before timing, and the normalizer `lru_cache` is warmed before timing, so no timed iteration touches disk.

## 4. Measurement schedule

FP32, `.eval()`, `torch.inference_mode()` throughout.

| | CPU | GPU | Throughput |
|---|---|---|---|
| Batch | 1 | 1 | 32 |
| Threads | `torch.set_num_threads(1)` | — | — |
| Warm-up | 100 | 200 | 50 |
| Timed | 1000 | 1000 | 200 |
| Scopes | model-only, end-to-end | model-only, end-to-end | model-only |
| Timer | `time.perf_counter` | `time.perf_counter` + `torch.cuda.synchronize()` per timed region | same as GPU |

**Deviation from the historical schedule, stated explicitly.** The preserved Part-6 harness uses warm-up 5, timed 20, batches (1, 16) on CPU and (16, 64) on GPU, threads (1, 4). That schedule is **weaker** — 50× fewer timed samples — so it is not preserved. The publication schedule above is adopted instead, with batch-1 single-thread as the deployment-facing primary. The historical **synchronized host-timing method** (`perf_counter` around a `cuda.synchronize()`-bounded region) *is* preserved rather than substituting CUDA events.

Batch 32 is fixed identically for both models and is not tuned per model. If it fails for either model, the highest common predetermined batch size is used and the failure recorded.

## 5. Order and bias control

Within each process, for every dataset and scope, measurement rounds **alternate Full-S1 / K1 (ABAB)** so thermal drift and background load affect both models equally. Memory is measured in separate clean subprocesses, one per model, to avoid CUDA allocator contamination.

A completed valid measurement is never repeated. Only an infrastructure-invalid run may be repeated, and the reason must be recorded.

## 6. FLOP convention

The reported quantity is **FLOPs, not MACs**, as the explicit sum of two terms:

**Dense term** — `torch.utils.flop_counter.FlopCounterMode`, covering `aten.convolution`, `aten.addmm`, `aten.mm`, `aten.bmm`, in the standard 2 × MAC convention for matrix products. *This term alone is not the total:* it does not count the selective-scan recurrence, and is never labelled as total FLOPs.

**Scan term** — analytic, preserving the historical Part-6 rule:

```
scan_flops = batch × n_bands × (Σ_layers directions) × T × d_inner × d_state × C,   C = 6
```

`C` is the approximate number of elementary flops per (`d_inner` × `d_state`) state element per sequential step of `h_t = exp(Δ_t A)·h_{t-1} + (Δ_t B_t)·x_t ; y_t = C_t·h_t`. The identical formula and constant apply to both architectures, differing only through `Σ_layers directions` (Full-S1 = 8, K1 = 4).

*Sensitivity, stated honestly:* a strict per-operation tally of the reference scan gives roughly **8** rather than 6, which would scale the scan term by 4/3. The historical constant is preserved for continuity. Because the same constant applies to both models, **the Full-S1 / K1 ratio is unaffected** by this choice; only the absolute scan total shifts.

```
total_flops = dense_term + scan_term
```

Reported per one 1-second window per dataset, plus the equal-domain Macro-4 mean (no weighting by sample count).

### Why the historical FLOP implementation is not used verbatim

`src/methodology_v2/compression/benchmark.py` (`synthetic_batch`, `compute_axis`, `analytic_scan_flops`) is bound to the **dissertation-era** representation. It hardcodes `DATASET_SHAPES["CWRU"] = (513, 184)` and `n_bands = 33 if dataset != "HIT" else 17`. The publication CWRU representation is the **native 12 kHz** grid: `(129, 184)` with **9** frequency bands. Using the historical function unchanged would overstate CWRU scan cost by a factor of 33/9 ≈ 3.67 for **both** models.

The **counting convention is preserved exactly**; only the input parameterisation (`n_bands`, `T`, directions) is taken from the live model and the live representation instead of the hardcoded historical table. **No historical benchmark source file is edited.** `benchmark.py` remains byte-identical at SHA256 `641f437b670b03d5bffb3ed11bfa2c8a758266392beec1988b90b8f4cecd211b`.

## 7. Size convention

- **Raw FP32 parameter bytes** — `sum(p.numel() × p.element_size())` over parameters.
- **Serialized state_dict size** — `torch.save` of a state-dict-only FP32 payload to a temporary file, by an identical procedure for both models, measured in bytes. Temporary files are deleted after measurement and never enter Git.

Training-checkpoint file sizes are **not** compared: checkpoint packaging differs between the two training stacks (Full-S1 stores `model`/`heads`/`cfg`, K1 stores `encoder`/`heads`/`val_reports`), so that comparison would be an artifact of packaging rather than of the models.

**Q8 is not measured.** The historical `size_axis()` calls `apply_q8_simulated()`; it is deliberately not used, because Q8 must not be run.

## 8. Memory convention

GPU, batch 1, model-only. Each model is measured in its **own clean subprocess**. Within it: `torch.cuda.reset_peak_memory_stats()`, synchronize, record allocated bytes after model load, peak allocated during forward, and the increment.

Reported as a headline claim only if the clean-process measurements are self-consistent on repeat; otherwise labelled **exploratory** and kept out of the headline table.

## 9. Audit of the preserved benchmark implementation

| Component | Source | SHA256 | Applicable as-is? |
|---|---|---|---|
| Four-axis harness | `src/methodology_v2/compression/benchmark.py` | `641f437b670b03d5bffb3ed11bfa2c8a758266392beec1988b90b8f4cecd211b` | Partly — see below |
| Part-6 latency runner | `scripts/reproduce/benchmark_part6_latency.py` | `64d2f52902dfcde719e6f1218ebd14252a1f9b515407350257f483be84c26444` | No — dissertation-era registry/spec driven |
| Architecture/scan helpers | `src/methodology_v2/compression/student.py` | pinned in the JSON plan | Yes — `count_params`, `scan_steps_per_forward` are direction-aware and correct |
| Measurement spec | `configs/lightweight_k1/measurement_spec.yaml` | pinned in the JSON plan | Partly — schedule superseded |

Findings:

- **FLOP counting** — `FlopCounterMode` (dense, 2×MAC) plus a *separate* analytic scan estimate. The harness correctly never sums them into a single "total FLOPs" label. Convention reusable; **CWRU parameterisation wrong for the publication representation** (33 vs 9 bands).
- **Latency** — `_time_forward`: fixed warm-up then per-iteration `perf_counter` with optional `cuda.synchronize()`; ABAB interleaving against a baseline in the same session; summary is median + IQR + min. Method sound; schedule (5/20) far weaker than required; no end-to-end scope exists.
- **Model size** — exact `numel` plus measured `torch.save` bytes. Sound, **but `size_axis()` also invokes Q8**, so it cannot be called here.
- **Memory** — `VmHWM` peak RSS of a fresh subprocess (CPU-side); the docstring mentions `cuda.max_memory_allocated` but `memory_axis()` implements only the RSS path, and it rebuilds a model from a `StudentSpec` rather than loading a frozen checkpoint. Not usable for a checkpoint-faithful GPU memory figure.
- **Scan assumption** — `scan_steps_per_forward` counts 2 steps per `BiMambaLayer` and 1 per `UniMambaLayer`, so direction handling is correct for both architectures.
- **Batch/warm-up/preprocessing** — historical harness is model-only on synthetic shape-faithful random tensors; preprocessing is excluded and there is no end-to-end path.

Conclusion: a single fixed counting convention **can** be established for both architectures, so the FLOP component proceeds under the convention in §6. Nothing is silently substituted; every deviation is recorded above.

## 10. Outputs

`environment.json`, `parameter_size_results.json`, `flops_results.json`, `latency_raw.csv`, `latency_summary.csv`, `throughput_results.csv`, `memory_results.json`, `EFFICIENCY_FINAL_REPORT.md`, and `EFFICIENCY_ARTIFACT_HASHES.sha256`.

## 11. Prohibitions observed

No retraining · no TEST access or rerun · no modification of frozen TEST artifacts · no modification of K1 training artifacts · no Q8 execution · no reuse of dissertation-era efficiency numbers as publication results · no modification of the dissertation repository · no edit of historical benchmark source.
