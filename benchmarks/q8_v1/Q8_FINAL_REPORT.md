# Q8 Deployment Extension — Final Report

Protocol identity: `lightweight_pcste_q8_v1`

Status: `PUBLICATION_Q8_EXTENSION_COMPLETE`

Plan: `Q8_EVALUATION_PLAN.json` (SHA256 `5904c3657e5ba68987caca2b204b725e153607883d98116fd25b9ed6b8220be4`) and barrier `Q8_EVALUATION_BARRIER.json` (SHA256 `d3b6898a109e571ff721f18f0d21ddd1649afdf235d315cc20b63f7ba141373d`), both frozen and hashed **before** any Q8 inference.

## Stage separation — read this first

| | |
|---|---|
| **Primary study** | Full-S1 vs K1 under `lightweight_pcste_final_test_v1`: **pre-registered, sealed TEST, confirmatory non-inferiority at margin −0.02.** Completed and frozen before this extension existed. |
| **Primary efficiency benchmark** | `lightweight_pcste_efficiency_v1`. Completed and frozen before this extension existed. |
| **This work** | `lightweight_pcste_q8_v1` — a **secondary, post-primary deployment/compression extension**, frozen after the primary study and evaluated afterwards. |

**Q8 was not part of the original sealed Full-S1-vs-K1 comparison and is not presented as such.** The primary confirmatory claim and its −0.02 margin are untouched by anything here. Q8 is not a trained model: it is derived deterministically from each already-frozen K1 selected checkpoint. Chain: `Full-S1 → K1 FP32 → Q8(K1)`.

All nine K1 `best.pt` checkpoints were independently re-hashed before conversion and matched both the frozen 9-cell manifest and `PUBLICATION_FROZEN_MODEL_TABLE.csv`. K1 inference was **not** rerun; the comparison uses the already-frozen K1 TEST results.

## 1. Q8 implementation — source truth

Source `src/methodology_v2/compression/quantization.py`, SHA256 `c91e0eaa970534a5099ce27f8a44f94d9c6b35442dccf56e8b809d98bbb0578d`, **unmodified**. Behaviour below was verified empirically on the live model, not assumed from the docstring.

| Property | Value |
|---|---|
| Quantization type | Post-training, **weight-only** for the accuracy representation |
| Granularity | **Per-output-channel** (axis = dim 0 of the `(out, in)` weight) |
| Symmetry | **Symmetric**, zero-point fixed at 0 (not stored, not learned) |
| Scale | `scale[c] = max(abs(w[c,:])) / 127`, clamped at 1e-12; `q = round(w/scale).clamp(-127,127)` as int8 |
| Calibration | **none** — and none used |
| Learned statistics | none |
| Quantized | **23 `nn.Linear` modules**: `stem.proj`, `coords.proj`, 12 × `temporal.layers.{0..3}.fwd.{in_proj,x_proj,out_proj}`, `mixer.score.0/2`, `mixer.value/gate/context`, and the 4 dataset heads |
| Retained FP32 | `*.dt_proj` (an `nn.Linear`, explicitly denylisted), `A_log`, `D`, `conv1d`, every LayerNorm, activations, and **`stem.conv`** (a `Conv2d`, outside the `nn.Linear` allowlist by construction). The recurrent `exp(Δ·A)` selective-scan path is not a parameterised module and is untouched by construction. |
| Parameters int8 | 1,322,816 of 1,379,813 (**95.87%**) — weights of the 23 allowlisted Linears |
| Max weight reconstruction error | 1.008e-03 |
| CPU backend | `torch.ao.quantization.quantize_dynamic` with `per_channel_dynamic_qconfig`, engine **fbgemm**; all 23 planned Linears converted |
| GPU backend | **none — no true hardware INT8 CUDA path** (see §6) |

Two registered representations: **`sim`** (int8 weight-only, dequantised to fp32 for compute; identical CPU/GPU numerics) is the *accuracy* representation and is what TEST accuracy is evaluated on, per the frozen designation in `quantization_spec.yaml`. **`cpu_dynamic`** (int8 weights **plus dynamic int8 activations**) is the *CPU deployment* representation. §7 quantifies the gap between them.

## 2. Predictive performance — nine cells, TEST membership unchanged

| Fold | Seed | K1 Macro-4 F1 | Q8 Macro-4 F1 | Δ F1 | K1 Macro-4 AUC | Q8 Macro-4 AUC | Δ AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 0.944172 | 0.944028 | -0.000144 | 0.998797 | 0.998808 | +0.0000105 |
| 1 | 1337 | 0.950776 | 0.950776 | +0.000000 | 0.994204 | 0.994203 | -0.0000008 |
| 1 | 2026 | 0.941293 | 0.941293 | +0.000000 | 0.996064 | 0.996040 | -0.0000242 |
| 2 | 42 | 0.977788 | 0.977788 | +0.000000 | 0.999368 | 0.999373 | +0.0000043 |
| 2 | 1337 | 0.972982 | 0.972982 | +0.000000 | 0.999368 | 0.999366 | -0.0000018 |
| 2 | 2026 | 0.919195 | 0.919195 | +0.000000 | 0.993053 | 0.993025 | -0.0000280 |
| 3 | 42 | 0.967335 | 0.967220 | -0.000115 | 0.996152 | 0.996165 | +0.0000127 |
| 3 | 1337 | 0.961007 | 0.961258 | +0.000251 | 0.996468 | 0.996481 | +0.0000134 |
| 3 | 2026 | 0.963463 | 0.963463 | +0.000000 | 0.994110 | 0.994096 | -0.0000141 |

| Endpoint | K1 FP32 | Q8(K1) | Paired Δ (Q8 − K1) | Q8 wins/ties/losses |
|---|---:|---:|---:|---:|
| Macro-4 Macro-F1 | 0.955334 ± 0.018393 | 0.955334 ± 0.018405 | -0.0000009 ± 0.0001101 | 1 / 6 / 2 |
| Macro-4 Macro-AUC | 0.996398 ± 0.002367 | 0.996395 ± 0.002376 | -0.00000310 ± 0.00001564 | 4 / 0 / 5 |

**Six of the nine cells are bit-identical to K1 on Macro-4 Macro-F1.** The three that differ do so in the fourth-to-fifth decimal place. Q8 retains K1 predictive performance essentially exactly.

### Per dataset

| Dataset | Metric | K1 FP32 | Q8(K1) | Paired Δ | Q8 w/t/l |
|---|---|---:|---:|---:|---:|
| CWRU | Macro-F1 | 0.918945 ± 0.073432 | 0.918945 ± 0.073432 | +0.0000000 ± 0.0000000 | 0/9/0 |
| CWRU | Macro-AUC | 0.990379 ± 0.009921 | 0.990364 ± 0.009950 | -0.0000152 ± 0.0000503 | 2/3/4 |
| JNU | Macro-F1 | 0.980952 ± 0.037796 | 0.980952 ± 0.037796 | +0.0000000 ± 0.0000000 | 0/9/0 |
| JNU | Macro-AUC | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | +0.0000000 ± 0.0000000 | 0/9/0 |
| HIT | Macro-F1 | 0.988728 ± 0.016133 | 0.988728 ± 0.016133 | +0.0000000 ± 0.0000000 | 0/9/0 |
| HIT | Macro-AUC | 0.999602 ± 0.001082 | 0.999602 ± 0.001069 | +0.0000000 ± 0.0000177 | 1/7/1 |
| MaFaulDa | Macro-F1 | 0.932712 ± 0.017321 | 0.932709 ± 0.017416 | -0.0000036 ± 0.0004406 | 1/6/2 |
| MaFaulDa | Macro-AUC | 0.995611 ± 0.004732 | 0.995614 ± 0.004724 | +0.0000028 ± 0.0000128 | 5/0/4 |

CWRU, JNU and HIT Macro-F1 are **unchanged in all nine cells**. Only MaFaulDa moves, by a mean of −3.6e-6.

## 3. Formal non-inferiority — the historical procedure, reused

The historical protocol **does** define this contrast, so it was reused rather than invented. Verified from frozen records before evaluation:

| Item | Value | Source |
|---|---|---|
| Margin | `0.01` | `quantization_spec.yaml` → `ni_margin` (SHA `2b629ae2…`) **and** `protocol.py` → `NI_MARGIN_PTQ` (SHA `5d2ac582…`) |
| Hypothesis | `H3: Q8(K1) vs K1: NI 0.01` | `statistics_spec.yaml` → `confirmatory_family_holm_m3.H3` (SHA `943b8f96…`) |
| Paired unit | fold × seed, 9 cells | `statistics_spec.yaml` → `paired_unit` |
| Procedure | one-sided exact sign-flip on **margin-shifted** deltas, `H0: mean(Δ) ≤ −m`, exact over all 512 patterns | `statistics_spec.yaml` → `non_inferiority`, `sign_flip` |

Exact historical call: `stats.contrast("H3 Q8(K1) vs K1", q8, k1, "ni", 0.01)`.

| Quantity | Value |
|---|---|
| Observed mean Δ (Q8 − K1) | `-0.0000009` |
| Predefined margin | `-0.01` |
| Margin-shifted mean | `0.0099991` |
| Exact one-sided sign-flip p | `0.001953125` |
| Verdict | **SATISFIED** |

Descriptive interpretation, stated carefully:

- The observed mean difference is `-0.0000009` Macro-F1 — three orders of magnitude smaller than the `0.01` margin. Q8 is non-inferior to K1 by a very wide margin on this endpoint.
- The p-value `0.001953125` is `1/512`, the **minimum attainable** under exact sign-flip enumeration at n = 9. It reflects that every margin-shifted delta is positive, not a large effect.
- `kind="ni"` in the historical family runs **non-inferiority only**. No superiority test is defined for Q8 and none was added.
- **Holm was not applied.** The historical Holm m=3 family (H1 K1 vs S1, H2 K1 vs C_small, H3 Q8 vs K1) is not executable — C_small was never evaluated, and the primary study already used a single confirmatory contrast. H3 is therefore run **standalone as a secondary contrast and carries no family-wise error control from that family.**
- The primary architecture margin of −0.02 does **not** apply to Q8; the PTQ margin is −0.01.

## 4. Storage

Representative cell **GF1 / seed 42** — the same deterministic cell as `efficiency_v1`. Actual `torch.save` bytes of state-dict-only payloads, never estimates.

| Artifact | Bytes | MiB | vs K1 FP32 |
|---|---:|---:|---:|
| Full-S1 FP32 state_dict | 9,578,353 | 9.135 | — |
| K1 FP32 state_dict | 5,541,832 | 5.285 | baseline |
| **Q8 compact int8 state** | **1,600,387** | **1.526** | **−71.12% (3.463×)** |
| Q8 theoretical int8 weight bytes | 1,567,700 | 1.495 | lower bound |
| Q8 `torch.ao` packed (cpu_dynamic) | 1,681,909 | 1.604 | includes runtime packing overhead |

**Full-S1 → Q8 overall: 83.29% reduction (5.985×).**

Three quantities are kept separate and never mixed: the **compact int8 state** (int8 tensors + fp32 per-channel scales + untouched fp32 tensors — the honest stored-artifact size), the **theoretical weight bytes** (a lower bound ignoring container overhead), and the **`torch.ao` packed state** (carries packing/observer metadata).

**Q8 does not change the parameter count.** It remains 1,379,813 (encoder + 4 heads); 1,322,816 weight values are stored as int8 and 56,997 parameters stay fp32 (the quantized Linears' biases, plus `dt_proj`, `A_log`, `D`, `conv1d`, all LayerNorms and `stem.conv`), alongside 5,061 fp32 per-channel scale values. Parameter count and storage precision are distinct.

The K1 FP32 baseline here (5,541,832 B) differs by 636 B (0.01%) from the `efficiency_v1` figure (5,542,468 B) because of container-level serialization differences. The Q8 reduction above is computed against the same-process measurement.

## 5. CPU deployment — real packed INT8

Same host as `efficiency_v1` (`otter135`, Intel i9-14900), same deterministic VALIDATION windows, **never TEST**. FP32 vs `torch.ao` dynamic int8 (fbgemm), batch 1, single thread, 100 warm-up, 1000 timed per cell, ABAB-interleaved in one process. Milliseconds per window; equal-domain mean of the four per-dataset medians.

### Primary — with `torch.inference_mode()`

| Dataset | Scope | K1 FP32 median | Q8 int8 median | Speed-up | Reduction |
|---|---|---:|---:|---:|---:|
| CWRU | model-only | 10.513 | 9.748 | 1.078× | 7.27% |
| JNU | model-only | 47.824 | 44.276 | 1.080× | 7.42% |
| HIT | model-only | 19.923 | 18.286 | 1.090× | 8.22% |
| MaFaulDa | model-only | 45.923 | 42.276 | 1.086× | 7.94% |
| **Macro-4** | **model-only** | **31.046** | **28.647** | **1.084×** | **7.73%** |
| CWRU | end-to-end | 10.919 | 10.146 | 1.076× | 7.08% |
| JNU | end-to-end | 48.401 | 44.903 | 1.078× | 7.23% |
| HIT | end-to-end | 20.812 | 19.142 | 1.087× | 8.02% |
| MaFaulDa | end-to-end | 49.242 | 45.674 | 1.078× | 7.24% |
| **Macro-4** | **end-to-end** | **32.343** | **29.966** | **1.079×** | **7.35%** |

Full dispersion (mean / median / SD / p95 / min / max) for every cell is in `results/q8_latency_summary.csv`; all 32,000 timed samples are in `results/q8_latency_raw.csv`.

### A methodological finding that affects the frozen FP32 benchmark

While preparing this run I found that **`efficiency_v1`'s latency and throughput stages never applied `torch.inference_mode()`**, although its plan states `torch.inference_mode(), model .eval()` as the measurement mode. It *is* applied in that benchmark's FLOPs and memory stages, so the omission is confined to latency and throughput.

Measured consequence on this host (K1, JNU, model-only, fresh process): **78.5 ms without `inference_mode` vs 38.6 ms with it** — autograd bookkeeping roughly doubles CPU latency.

What this does and does not invalidate:

- **Ratios in `efficiency_v1` remain valid.** Full-S1 and K1 were measured identically, ABAB-interleaved in one process, so the reported speed-ups are unaffected.
- **Absolute millisecond figures in `efficiency_v1` are inflated** and are not inference-mode numbers.
- `efficiency_v1` results are frozen and were **not** modified here.

Because of this, Q8 CPU latency was measured **both ways** and absolute milliseconds are never compared across the two runs:

| Scope | Mode | K1 FP32 | Q8 int8 | Speed-up |
|---|---|---:|---:|---:|
| model-only | `inference_mode` (primary) | 31.046 | 28.647 | 1.084× |
| model-only | no `inference_mode` (matches efficiency_v1) | 77.349 | 65.900 | 1.174× |
| end-to-end | `inference_mode` (primary) | 32.343 | 29.966 | 1.079× |
| end-to-end | no `inference_mode` (matches efficiency_v1) | 79.910 | 75.613 | 1.057× |

The first Q8 latency run (without `inference_mode`) was superseded as the primary result because the mode does not represent deployment inference — not because of its speed-up. Both runs are retained and hashed; the superseded one is `results/q8_latency_*_no_inference_mode.csv`.

### Speed-up versus Full-S1 — a chained ratio, not a cross-run division

Full-S1 was not re-measured under `inference_mode`, so a Full-S1 → Q8 CPU speed-up is obtained by **chaining ratios**, each from a within-process paired measurement:

| Scope | Full-S1 / K1 (`efficiency_v1`) | K1 / Q8 (`q8_v1`) | Chained Full-S1 / Q8 |
|---|---:|---:|---:|
| model-only | 2.213× | 1.084× | **2.399×** |
| end-to-end | 2.179× | 1.079× | **2.351×** |

This is an estimate. It assumes the two ratios compose, which holds only if the mode change scales both models equally — plausible but unverified.

### Throughput — CPU batch 32

Identical batch size for both models, model-only. `cpu_dynamic` is a CPU-only backend, so throughput is measured on CPU rather than on the GPU used by `efficiency_v1`.

| Dataset | K1 FP32 win/s | Q8 int8 win/s | Speed-up |
|---|---:|---:|---:|
| CWRU | 59.13 | 60.35 | 1.021× |
| JNU | 10.73 | 11.19 | 1.042× |
| HIT | 25.33 | 26.34 | 1.040× |
| MaFaulDa | 11.03 | 11.48 | 1.040× |
| **Macro-4** | **26.56** | **27.34** | **1.029×** |

**The measured CPU gain is modest — about 8% at batch 1 and 3% at batch 32.** This is consistent with the frozen historical statement `latency_claim = "NONE for Q8 (weight-only int8 does not speed the reference scan)"`: the sequential selective scan stays FP32 and dominates runtime, so quantizing the Linear layers cannot buy much. The measurement is reported as a deployment observation; the historical statement is neither quietly dropped nor retro-fitted.

## 6. GPU — no true INT8, and no claim made

Established **empirically before any GPU timing**. Attempting to run the `cpu_dynamic` model on CUDA fails:

```
NotImplementedError: Could not run 'quantized::linear_dynamic' with arguments from the 'CUDA' backend. This could be because the operator doesn't exist for this backend, or was omi
```

| | |
|---|---|
| True hardware INT8 CUDA execution | **No** |
| `sim` representation compute dtype | float32 (weights dequantised before compute) |
| `cpu_dynamic` backend | CPU-only (`fbgemm`) |

**No GPU latency is reported for Q8.** A GPU timing of the `sim` representation would be fp32 arithmetic on quantised-then-dequantised weights — *simulated Q8* — and would measure FP32 speed, not INT8 acceleration. Presenting it as a Q8 GPU deployment speed-up would be wrong, so it is omitted rather than reported with a caveat.

The `sim` representation is still used for its intended purpose: the **accuracy** evaluation in §2, where CPU and GPU numerics agree by construction.

No TensorRT, `torch.compile`, custom kernels, or alternative quantization library was introduced; any of those would be a different implementation, not this one.

## 7. The deployment numerics gap — an honest limitation

TEST accuracy (§2) is evaluated on **`sim`**, the frozen accuracy representation. The **CPU artifact that is actually deployed is `cpu_dynamic`**, which additionally quantizes activations and therefore does not produce identical predictions.

Measured on VALIDATION only (fold 1, up to 200 windows per dataset in frozen manifest order — **TEST was not used for this check**):

| Dataset | Windows | Top-1 agreement (sim vs cpu_dynamic) | Max abs prob difference |
|---|---:|---:|---:|
| CWRU | 200 | 1.0000 | 0.00448 |
| JNU | 24 | 1.0000 | 0.00000 |
| HIT | 200 | 0.9950 | 0.10196 |
| MaFaulDa | 200 | 0.9500 | 0.49397 |
| **Equal-domain mean** | — | **0.9863** | — |

**This matters.** The deployed CPU representation disagrees with the evaluated representation on about 1.4% of validation windows overall, and on 5% for MaFaulDa, with probability differences up to 0.49. So:

- The non-inferiority result in §3 applies to **`sim`**, the weight-only representation.
- It does **not** automatically transfer to the `cpu_dynamic` artifact whose latency is reported in §5.
- No TEST evaluation of `cpu_dynamic` was performed, because the frozen protocol designates `sim` as the accuracy representation and a second TEST touch was not authorised.
- A deployment claim for the CPU int8 artifact would require its own pre-registered TEST evaluation. **That claim is not made here.**

## 8. Memory — descriptive only

| Quantity | Value |
|---|---:|
| K1 FP32 resident parameter+buffer bytes (CPU) | 5,519,316 |
| Q8 `sim` resident bytes (CPU) | 5,519,316 |
| Q8 `cpu_dynamic` packed state bytes | 1,681,205 |

Memory is **not** a headline claim. The `sim` representation holds dequantised fp32 weights, so its runtime footprint equals FP32 by construction; the packed `cpu_dynamic` state mixes weights with `torch.ao` packing metadata. Neither is a clean 'Q8 runtime memory' figure. No GPU memory number is reported: with no true INT8 CUDA path, a GPU measurement would reflect fp32 runtime memory and would not correspond to the stored INT8 artifact.

## 9. Three-stage performance–efficiency table

| Metric | Full-S1 FP32 | K1 FP32 | Q8(K1) |
|---|---:|---:|---:|
| Macro-4 Macro-F1 | 0.934644 ± 0.023258 | 0.955334 ± 0.018393 | 0.955334 ± 0.018405 |
| Macro-4 Macro-AUC | 0.991793 ± 0.008102 | 0.996398 ± 0.002367 | 0.996395 ± 0.002376 |
| Parameters | 2,385,893 | 1,379,813 | 1,379,813 (unchanged) |
| Stored model size | 9.135 MiB | 5.285 MiB | **1.526 MiB** |
| GFLOP / window (Macro-4) | 2.6067 | 1.4236 | 1.4236 (unchanged) |
| CPU b1 model-only latency | 91.850 ms ᵃ | 31.046 ms ᵇ | 28.647 ms ᵇ |
| CPU speed-up vs Full-S1 | 1.00× | 2.213× ᵃ | 2.399× ᶜ |
| CPU b32 throughput | not measured ᵈ | 26.56 win/s | 27.34 win/s |
| GPU b1 latency | 11.811 ms ᵃ | 6.368 ms ᵃ | **not reported** ᵉ |
| GPU execution type | FP32 | FP32 | **no true INT8 path** |
| Peak memory | 29.466 MiB (GPU) ᵃ | 24.823 MiB (GPU) ᵃ | **not reported** ᵉ |

ᵃ from the frozen `efficiency_v1` benchmark, measured **without** `inference_mode` — absolute values not comparable to ᵇ.  
ᵇ this extension, measured **with** `inference_mode`, K1 and Q8 paired in one process.  
ᶜ chained ratio (see §5), not a cross-run division.  
ᵈ `efficiency_v1` measured batch-32 throughput on GPU; Q8's CPU-only backend makes CPU the common ground, so Full-S1 has no comparable CPU figure.  
ᵉ no true INT8 GPU path exists (§6); a simulated number would be misleading.

Every Q8 cell in this table is populated only where the interpretation is scientifically valid; the rest are explicitly marked rather than filled with a plausible-looking number.

## 10. Summary

1. **Performance retention:** Q8 retains K1 accuracy essentially exactly — paired Δ Macro-4 Macro-F1 `-0.0000009`, six of nine cells bit-identical, non-inferiority at the historical −0.01 margin **satisfied** (exact p `0.001953125`).
2. **Size:** 71.12% smaller than K1 FP32 (3.46×), 83.29% smaller than Full-S1 (5.99×).
3. **CPU INT8 latency:** real but modest — 1.084× at batch 1, 1.029× at batch 32, because the FP32 sequential scan dominates.
4. **GPU:** the preserved implementation has **no true INT8 CUDA path**, so no GPU deployment claim is made.
5. **Caveat:** the evaluated accuracy representation and the deployed CPU artifact are not numerically identical (98.62% top-1 agreement on validation); the accuracy claim covers `sim`, not `cpu_dynamic`.

Q8 is therefore a **compression** win, not a compute win, on this stack — and it is a secondary extension, not part of the primary sealed comparison.

