# Experiment Protocol

## Scope of this document

This repository has two distinct protocol layers, and they must not be conflated:

| Layer | Identity | Status here |
|---|---|---|
| Historical dissertation study | multi-arm design including S0, S1, K1, Q8 under the dissertation-era evaluation barrier | Source of the frozen splits, normalizers, Full-S1 training protocol and code lineage. **Not** the evaluation protocol used for the publication result. |
| Current publication study | `lightweight_pcste_final_test_v1` | The protocol under which the sealed TEST reported here was executed. Compares exactly Full-S1 and K1. |

Everything presented as a final result in this repository comes from the publication layer.

## Fixed experimental design

- Datasets: CWRU, JNU, HIT, MaFaulDa.
- Global split protocol: `global_v2_final_s0s1_v1`. CWRU uses `pcste_v2_cwru_native12_load_v1` (official native 12 kHz only; TRAIN = 0 hp + 1 hp, VALIDATION = 2 hp, TEST = 3 hp).
- Three global folds x seeds 42, 1337, 2026 = nine matched cells.
- Paired analysis unit: the (fold, seed) cell. Full-S1 and K1 are compared within cells.

## Model arms

**Full-S1** — the full PC-STE v2 S1 model, serving both as the base/reference arm and, as a same-fold three-seed ensemble, as the KD teacher.

**K1** — the lightweight student, architecture `half_4x1`: four forward-only temporal Mamba layers with `kept_direction=fwd` and `uni_residual=mean_of_remaining`. One shared encoder of **1,375,953** parameters across all four datasets, with dataset-specific heads. Initialized from the matched Full-S1 checkpoint.

**S0** — excluded from the publication evaluation.

**Q8** — the optional 8-bit quantization extension. Excluded from the publication evaluation and **not executed**. Its inclusion remains an open scientific decision with a separate frozen comparison margin of `-0.01`.

## K1 training (frozen)

| Parameter | Value |
|---|---|
| Epochs per cell | 50 (no early stopping) |
| KD temperature | 4 |
| KD alpha | 0.5 |
| Relational weight | 1.0 |
| Teacher aggregation | `mean_prob_at_T` over the same-fold three-seed S1 ensemble |
| KL direction | teacher \|\| student |
| Sampler streams | exact frozen streams, SHA-registered per cell |
| Checkpoint selection | maximum validation Macro-Domain F1; strict improvement; exact tie selects the earlier epoch |
| TEST during training/selection | none (`test_used=false` recorded in every cell's `state.json` and `completion.json`) |
| Production executor SHA256 | `1f5d86a1c6b7f2b637ecdd6d61c279a0b86e9edb54bdbe0490d2a4b20fd62308` |

All nine cells completed and were frozen before TEST. The freeze manifest — `configs/lightweight_k1/K1_9_CELL_FREEZE_MANIFEST.md`, SHA256 `a9e5fb49a1f38a7d6e2649ff8d155cf619cd55572bf7238b06a1fcb120d64a8b` — records each cell's best epoch, validation Macro-Domain F1, `best.pt` SHA256, stream SHA, teacher lineage, and TEST-usage flag.

## Publication TEST protocol

Protocol identity: `lightweight_pcste_final_test_v1`. Frozen at `2026-09-20T08:11:47+00:00`, before any TEST access.

### Endpoints

- Primary: **Macro-4 Macro-F1**, defined as the unweighted mean of the four dataset Macro-F1 values (equal domain weights, no window weighting).
- Secondary aggregate: **Macro-4 Macro-AUC**, defined only if all four dataset Macro-AUC values are finite; no domain may be silently dropped.
- Primary matched difference: `K1 - Full-S1` within each (fold, seed) cell.
- Per dataset the plan also fixes accuracy, balanced accuracy, macro precision/recall, per-class precision/recall/F1, and the confusion matrix.

### Statistical procedure (frozen before TEST)

One confirmatory contrast: `contrast(..., "ni_then_superiority", 0.02)`.

1. Non-inferiority, H0: `mean(delta) <= -0.02`. Add `0.02` to every one of the nine deltas, enumerate all 512 sign patterns, and compute the exact one-sided sign-flip p-value. It passes only if `p < 0.05` **and** the observed shifted mean is positive.
2. Only if that gate passes, report the exact two-sided sign-flip superiority result.

Descriptives use sample SD (`ddof=1`) over all nine matched cells plus exact wins/ties/losses. Statistical-plan SHA256: `7d4ba32d9f1df2283c98c012a7268dd7d3664fa52017a84adf9f0b2e9a373629`; implementation `src/methodology_v2/compression/stats.py` SHA256 `efb3d20c777e4ab0eda15491147950c77c816610d1f71ca0072f8dfa36e3cfcb`.

The historical three-hypothesis Holm family is **not** executable under this protocol, because the arms it covered (S0, Q8) are not evaluated here. No replacement hypothesis and no post-TEST test was added.

### Relationship to the historical dissertation barrier

Stated factually, and deliberately not elaborated further:

- The dissertation-era S0/S1 evaluation barrier was **unavailable**.
- It was **not** bypassed, reconstructed, weakened, or modified. The historical evaluator remains unchanged at SHA256 `dd3aa58ce5067298504b9e064cdb51e5f29d0f655b1ca3f4fa6af77600d7f517` (`scripts/evaluate/evaluate_test.py`).
- A **new publication-specific Full-S1-vs-K1 evaluation protocol was pre-registered while TEST remained unopened**, covering the models, split membership, metrics, aggregation, and statistics listed above.
- TEST was then **executed exactly once** under that frozen publication plan.

### Execution barrier

A separate fail-closed driver (`configs/lightweight_k1/final_test_v1/publication_test_driver.py`, SHA256 `8f2de50a52ce83fefa8dcd1516ec3c348d9f5c1a6b561b38070fe9f94d149f3e`, also reachable at `scripts/evaluate/publication_test_driver.py`) enforces the freeze. Preflight requires an externally supplied barrier SHA256, rehashes every source file, registry, manifest, normalizer and checkpoint, validates the fixed nine-cell schema, and strict-loads all eighteen checkpoint state dictionaries **before** TEST membership may be parsed. Execution additionally requires the literal token `AUTHORIZE_LIGHTWEIGHT_PCSTE_FINAL_TEST_V1`. Existing output directories are refused. Any mismatch fails closed. A dry self-test mutates the plan bytes and the non-inferiority margin in memory and requires both alterations to be rejected.

Barrier record: `configs/lightweight_k1/final_test_v1/PUBLICATION_FINAL_EVALUATION_BARRIER.json`, SHA256 `fa8410c03bcad4d3eea31282ac8ee7828c731a35a025aab872354aa8ffdbe71d`.

## Post-TEST integrity

- Exact TEST membership was verified independently for all eighteen prediction files.
- Every checkpoint was rehashed after inference and matched the frozen plan.
- All nine matched cells were retained; none was dropped.
- No training, checkpoint selection, tuning, protocol change, Q8 run, or scientific rerun occurred during or after the sealed TEST.

## Efficiency benchmark protocol

A separate, independently pre-registered protocol, `lightweight_pcste_efficiency_v1`, measures computational cost. It reads no TEST data, trains nothing, and alters no frozen predictive artifact.

- **Model pair fixed by rule before measurement:** lowest fold, then lowest seed -> GF1 / seed 42. Both checkpoint hashes recomputed against `PUBLICATION_FROZEN_MODEL_TABLE.csv`.
- **Inputs:** first VALIDATION window per dataset in frozen manifest order, one second each, identical for both models. Every selected row is asserted to be a validation row.
- **Two scopes, never mixed:** model-only (prepared representation -> logits) and end-to-end (raw waveform -> STFT -> N2b normalisation -> logits). Disk I/O excluded from both.
- **Schedule:** FP32, `.eval()`, `torch.inference_mode()`, batch 1; CPU single-thread with 100 warm-up / 1000 timed, GPU with 200 warm-up / 1000 timed, ABAB-interleaved over four rounds; throughput at a fixed batch 32 for both models.
- **Aggregation:** equal-domain mean of the four per-dataset medians; speed-up = Full-S1 / K1.
- **FLOP convention:** FLOPs, not MACs, as the explicit sum `dense + scan`. Dense is `FlopCounterMode` (2xMAC for matrix products) and is never reported alone as a total; scan is analytic, `n_bands x sum_layers(directions) x T x d_inner x d_state x 6`, with the constant preserved from the historical Part-6 harness and the grid parameters taken from the live representation.
- **Size convention:** raw FP32 parameter bytes and a state_dict-only FP32 serialization written by an identical procedure for both models. Training-checkpoint file sizes are not compared. Q8 is not measured.

The plan was frozen and hashed before any latency result was collected: `benchmarks/efficiency_v1/EFFICIENCY_BENCHMARK_PLAN.json` (SHA256 `f88db6d957a1f6eec4b3dc8199f7606eb7c73be0958a87322fa297124ff6595b`).

The historical Part-6 harness (`src/methodology_v2/compression/benchmark.py`) was audited and left **unedited**. Its counting conventions are preserved, but it could not be called verbatim: its `DATASET_SHAPES`/`n_bands` are bound to the dissertation-era representation (CWRU `(513,184)` with 33 bands, versus the publication native-12 kHz `(129,184)` with 9 bands), its latency schedule is 50x weaker, it has no end-to-end scope, and its `size_axis()` invokes Q8. The audit is recorded in section 9 of `EFFICIENCY_BENCHMARK_PLAN.md`.

## Secondary extension protocol — Q8(K1)

`lightweight_pcste_q8_v1` is a **secondary, post-primary** deployment/compression extension. It was created only after the primary sealed study *and* the efficiency benchmark were complete and frozen; its plan and barrier were hashed before any Q8 inference. It must never be presented as part of the primary sealed comparison.

- **Source models:** all nine frozen K1 `best.pt` checkpoints (`last.pt` forbidden), re-hashed against the 9-cell freeze manifest before conversion. No checkpoint may be chosen on Q8 behaviour.
- **Conversion:** the unmodified historical `src/methodology_v2/compression/quantization.py` (SHA256 `c91e0eaa970534a5099ce27f8a44f94d9c6b35442dccf56e8b809d98bbb0578d`). Per-output-channel symmetric int8, zero-point 0, `scale = max|w_row| / 127`, on 23 allowlisted `nn.Linear` modules. **Calibration = none.** Retained FP32: `dt_proj`, `A_log`, `D`, `conv1d`, all LayerNorms, `stem.conv`, and the selective-scan path.
- **Representations:** `sim` (weight-only, fp32 compute) is the frozen *accuracy* representation and is what TEST accuracy is evaluated on; `cpu_dynamic` (`torch.ao`, int8 weights + dynamic int8 activations, fbgemm) is the *CPU deployment* representation. They are not numerically identical - the gap is quantified on VALIDATION only.
- **Evaluation:** TEST membership unchanged from the primary study and re-asserted per cell. K1 is **not** rerun; the frozen K1 results are the reference. `delta = Q8 - K1` paired within each (fold, seed) cell.
- **Statistics:** the historical definition is reused exactly - margin `0.01` (`quantization_spec.yaml` `ni_margin`, `protocol.py` `NI_MARGIN_PTQ`), paired unit fold x seed, `stats.contrast(..., "ni", 0.01)` = exact one-sided sign-flip on margin-shifted deltas. Non-inferiority only; no superiority test is defined for Q8. **Holm is not applied**: the historical m=3 family is not executable, so H3 runs standalone with no family-wise error control. The primary -0.02 margin does not apply.
- **Deployment measurement:** same host and the same deterministic VALIDATION inputs as `efficiency_v1`; never TEST. No TensorRT, `torch.compile`, custom kernels, or new quantization library.
- **GPU:** established empirically that the preserved implementation has **no true INT8 CUDA path**; no GPU latency or memory figure is reported for Q8.

Plan `benchmarks/q8_v1/Q8_EVALUATION_PLAN.json` (SHA256 `5904c3657e5ba68987caca2b204b725e153607883d98116fd25b9ed6b8220be4`), barrier SHA256 `d3b6898a109e571ff721f18f0d21ddd1649afdf235d315cc20b63f7ba141373d`.

## Nothing further is pending

All three stages - primary sealed study, efficiency benchmark, and the Q8 extension - are complete and frozen. Any future work may not reuse dissertation-era numbers as a publication measurement, and may not alter any frozen artifact listed in `FROZEN_ARTIFACT_HASHES.sha256`, `benchmarks/efficiency_v1/EFFICIENCY_ARTIFACT_HASHES.sha256`, or `benchmarks/q8_v1/Q8_ARTIFACT_HASHES.sha256`.
