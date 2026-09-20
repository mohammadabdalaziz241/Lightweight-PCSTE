# Model

## Full-S1 (teacher and base arm)

The full PC-STE v2 S1 model, trained on the frozen four-domain recipe from an SSL-pretrained encoder. It serves two roles: the reference arm in the publication comparison, and — as a same-fold three-seed ensemble — the knowledge-distillation teacher.

## K1 (publication lightweight model)

Architecture `half_4x1`: four forward-only temporal Mamba layers with `kept_direction=fwd` and `uni_residual=mean_of_remaining`.

- **1,375,953 verified encoder parameters**, one shared encoder across CWRU, JNU, HIT, and MaFaulDa.
- Dataset-specific classification heads.
- Initialized from the matched Full-S1 checkpoint of the same (fold, seed) cell.

Knowledge distillation uses temperature 4, alpha 0.5, relational weight 1.0, `mean_prob_at_T` teacher aggregation, and KL direction teacher || student. Fifty epochs per cell, no early stopping, validation-only checkpoint selection on Macro-Domain F1 with strict improvement and earliest-epoch tie retention.

All nine cells are frozen in `configs/lightweight_k1/K1_9_CELL_FREEZE_MANIFEST.md`.

## Q8(K1) — secondary deployment extension

An 8-bit post-training quantized variant of K1, derived deterministically from each frozen K1 checkpoint by the historical Part-6 recipe. **It is a secondary extension, frozen and evaluated after the primary sealed Full-S1-vs-K1 study, and is not part of that comparison.**

Per-output-channel symmetric int8 weights (zero-point 0, `scale = max|w_row| / 127`) on 23 allowlisted `nn.Linear` modules; **no calibration**. Retained in FP32: `dt_proj`, `A_log`, `D`, `conv1d`, all LayerNorms, `stem.conv`, and the selective-scan recurrence. The parameter count is unchanged at 1,379,813 — only storage precision changes.

Two registered representations: `sim` (weight-only, fp32 compute) is the accuracy representation used for TEST evaluation; `cpu_dynamic` (`torch.ao`, int8 weights + dynamic int8 activations) is the CPU deployment representation. They are not numerically identical.

Result: accuracy retained essentially exactly (paired ΔMacro-4 Macro-F1 −0.0000009; non-inferiority at the historical −0.01 margin satisfied), storage 71.12% smaller than K1 FP32, CPU batch-1 latency 1.084×, and **no true INT8 GPU path**. See `docs/RESULTS.md` and `benchmarks/q8_v1/Q8_FINAL_REPORT.md`.

## Efficiency

The parameter count above is an architectural property, verified by construction. No FLOP, latency, throughput, or memory figure is claimed as a publication measurement; that benchmarking is the next publication stage and will not reuse dissertation-era numbers.
