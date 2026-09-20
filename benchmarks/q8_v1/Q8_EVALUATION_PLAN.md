# Q8 Deployment Extension — Evaluation Plan

Protocol identity: `lightweight_pcste_q8_v1`

Status at freeze: `FROZEN_BEFORE_Q8_EVALUATION`

Publication Git HEAD at freeze: `6fee10a39aa23cecd2fe1a454e09045af91f5619`

Machine-readable plan: `Q8_EVALUATION_PLAN.json`

## 1. Stage declaration — this is a secondary extension

| | |
|---|---|
| **Primary study** | Full-S1 vs K1, protocol `lightweight_pcste_final_test_v1`: pre-registered, sealed TEST, confirmatory non-inferiority at margin −0.02. **Completed and frozen before this extension existed.** |
| **Primary efficiency benchmark** | `lightweight_pcste_efficiency_v1`. **Completed and frozen before this extension existed.** |
| **This stage** | `lightweight_pcste_q8_v1` — a **secondary, post-primary deployment/compression extension.** |

At the moment this plan was frozen:

- Q8 had **not** been evaluated.
- **No Q8 result had been observed.**
- All nine K1 checkpoints were **already fixed** by the frozen 9-cell manifest.
- TEST membership was **unchanged** from the primary study.
- **No Q8-specific tuning is permitted.**

Q8 must never be presented as part of the original sealed Full-S1-vs-K1 comparison. The primary confirmatory claim, and its −0.02 margin, are unaffected by anything in this extension.

The chain is `Full-S1 → K1 FP32 → Q8(K1)`. **Q8 is not a trained model**: it is derived deterministically from each already-frozen K1 selected checkpoint.

## 2. Audit of the preserved Q8 implementation

Source: `src/methodology_v2/compression/quantization.py`, SHA256 `c91e0eaa970534a5099ce27f8a44f94d9c6b35442dccf56e8b809d98bbb0578d`. **Unmodified.** The source was read and its behaviour verified empirically on the frozen GF1/seed-42 K1 checkpoint before this plan was written; what follows is source truth, not the historical description restated.

| Property | Source truth |
|---|---|
| Quantization type | Post-training, **weight-only** for the accuracy representation |
| Granularity | **Per-output-channel** (axis = dim 0 of the `(out, in)` weight) |
| Symmetry | **Symmetric**, zero-point fixed at 0 |
| Scale | `scale[c] = max(abs(w[c, :])) / 127`, clamped at `1e-12`; `q = round(w / scale).clamp(-127, 127)` as `int8` |
| Zero-point | Not stored and not learned — symmetric, so identically zero |
| Calibration | **None required and none used** |
| Learned statistics | None |

**Quantized (int8), 23 `nn.Linear` modules — verified on the live model:**
`stem.proj`, `coords.proj`, `temporal.layers.{0..3}.fwd.{in_proj, x_proj, out_proj}` (12), `mixer.score.0`, `mixer.score.2`, `mixer.value`, `mixer.gate`, `mixer.context`, and the four dataset heads `heads.{CWRU, JNU, HIT, MAFAULDA}`.

**Retained FP32 — verified on the live model:**
`*.dt_proj` (an `nn.Linear`, explicitly denylisted), `A_log`, `D`, `conv1d`, every LayerNorm (`temporal.layers.*.norm`, `temporal.norm`, `mixer.norm`), the activation modules, and **`stem.conv`** — a `Conv2d`, therefore outside the `nn.Linear` allowlist by construction. The recurrent state / `exp(Δ·A)` selective-scan path is not a parameterised module and is untouched by weight quantization by construction.

Measured on GF1/seed 42: **1,322,816 of 1,379,813 parameters are int8 (95.93%)**; 23 per-channel scale vectors; maximum weight reconstruction error 1.01e-3.

**Two deployment representations** (both part of the registered recipe):

| Representation | What it is | Role |
|---|---|---|
| `sim` | int8 weight-only, **dequantised to fp32 for compute**; identical numerics on CPU and GPU | **The accuracy representation** (frozen designation in `quantization_spec.yaml`) |
| `cpu_dynamic` | `torch.ao` dynamic quantization — per-channel int8 weights **plus dynamic int8 activations**, applied only to allowlisted modules via a qualified-name qconfig dict | **The CPU deployment representation** |

**CPU implementation:** `torch.ao.quantization.quantize_dynamic` with `per_channel_dynamic_qconfig`, engine chosen as fbgemm → x86 → last available. On this host `fbgemm` is selected; all 23 planned Linears convert (19 encoder + 4 heads), confirmed against the plan count.

**GPU implementation: there is none.** `sim` dequantises to fp32, and `torch.ao` dynamic quantization is a CPU backend. **The preserved implementation provides no true hardware INT8 CUDA kernel.** Any GPU figure in this extension is therefore *simulated Q8* and will be labelled as such. It may be used for numerical-equivalence checking; it must not be presented as INT8 GPU acceleration.

**Honest disclosure of a numerics gap.** `cpu_dynamic` additionally quantizes activations, so its outputs differ from `sim`. TEST accuracy is evaluated for `sim` only, per the frozen designation. To bound the deployment gap without a second TEST touch, `sim`-vs-`cpu_dynamic` prediction agreement is quantified on **VALIDATION only**.

**Calibration = none.** The registered accuracy representation is weight-only; the CPU representation computes activation scales at runtime. The static W8A8 variant that *would* need TRAIN calibration is an exploratory, VAL-only, unregistered scaffold and is **not** part of this extension.

## 3. Source models

All nine frozen K1 **selected** checkpoints (`best.pt`; `last.pt` is forbidden), identities taken from `configs/lightweight_k1/K1_9_CELL_FREEZE_MANIFEST.md` and independently recomputed before conversion. All nine matched both that manifest and `PUBLICATION_FROZEN_MODEL_TABLE.csv`. Pinned in `Q8_FROZEN_MODEL_TABLE.csv`.

No checkpoint may be selected, excluded, or reordered on the basis of Q8 behaviour.

## 4. Predictive evaluation

- TEST membership is the `split == "test"` rows of the frozen fold manifests — **unchanged** from the primary study, and re-asserted per cell.
- Datasets: CWRU, JNU, HIT, MaFaulDa. Per dataset: Macro-F1, accuracy, balanced accuracy, macro precision/recall, macro OVR ROC-AUC where defined, per-class precision/recall/F1, confusion matrix.
- Macro-4 = `(CWRU + JNU + HIT + MaFaulDa) / 4`, equal domain weights, no window weighting.
- Endpoints: **Macro-4 Macro-F1** and **Macro-4 Macro-AUC**.
- **K1 is not rerun.** The comparison uses the already-frozen K1 TEST results, pinned by hash in the JSON plan.
- `delta = Q8 − K1`, paired within the same (fold, seed) cell.
- Device `cuda`, matching the primary driver's `--device auto` resolution on this host. `sim` is fp32 compute, so CPU and GPU numerics agree.

## 5. Statistics — the historical procedure, reused exactly

The historical protocol **does** define this contrast, so it is reused rather than invented. Verified from frozen records:

| Item | Value | Source | SHA256 |
|---|---|---|---|
| Margin | `0.01` | `configs/lightweight_k1/quantization_spec.yaml` → `ni_margin` | `2b629ae2ba61a62ea35abd04148759a2009e4943aec9af53f13a57809feb2181` |
| Margin | `0.01` | `src/methodology_v2/compression/protocol.py` → `NI_MARGIN_PTQ` | `5d2ac582b16ec8f49d3d5f155f38c457aedddb53fc14754cb98fae98016664a1` |
| Hypothesis | `H3: Q8(K1) vs K1: NI 0.01` | `configs/lightweight_k1/statistics_spec.yaml` → `confirmatory_family_holm_m3.H3` | `943b8f96489273e0c7441742f2920f487dbce432507618f5631c746639de3419` |
| Paired unit | `fold × seed (9 cells)` | same file → `paired_unit` | same |
| Procedure | one-sided exact sign-flip on **margin-shifted** deltas (`delta + m`), `H0: mean(delta) ≤ −m`, exact over all 512 patterns | same file → `non_inferiority`, `sign_flip` | same |

Exact call, matching the historical `stats.confirmatory_family` signature:

```python
stats.contrast("H3 Q8(K1) vs K1", q8, k1, "ni", 0.01)
```

`kind="ni"` runs **non-inferiority only**. No superiority test is defined for Q8, and none is added.

**Holm is NOT applied.** The historical Holm m=3 family (H1 K1 vs S1, H2 K1 vs C_small, H3 Q8 vs K1) is not executable: C_small was never evaluated, and the primary study already established its single confirmatory contrast without that family. H3 is therefore run **standalone as a secondary contrast and carries no family-wise error control from the historical family.** This is disclosed, not reconstructed.

**The primary −0.02 architecture margin does not apply to Q8.** The PTQ margin is −0.01. α = 0.05.

## 6. Storage size

Representative cell **GF1 / seed 42** — the same deterministic cell used by `efficiency_v1`. Measured as **actual `torch.save` bytes, never an estimate** (the frozen rule in `quantization_spec.yaml`):

- K1 FP32 standardized state_dict
- Q8 compact int8 state (int8 tensors + fp32 per-channel scales + untouched fp32 tensors)
- `torch.ao` `cpu_dynamic` packed state — **reported separately**, because it carries runtime packing overhead
- theoretical int8 weight bytes

Packed-runtime bytes and theoretical int8 weight bytes are **never mixed into one number**. Arbitrary training-checkpoint sizes are not compared.

## 7. Latency

Same physical host and the **same deterministic VALIDATION windows** as `efficiency_v1` — never TEST.

CPU: `torch.ao` dynamic (engine selected by the historical `cpu_dynamic_quantize`), batch 1, single thread, 100 warm-up, 1000 timed, model-only and end-to-end scopes, aggregated as the equal-domain mean of per-dataset medians — matching `efficiency_v1` exactly so the numbers are directly comparable.

GPU: determined empirically before timing; expected to be **simulated only** (see §2). No TensorRT, no `torch.compile`, no custom kernels, no new quantization library — any of those would be a new implementation, not this one.

**Historical latency claim, preserved for context:** `quantization_spec.yaml` states `latency_claim = "NONE for Q8 (weight-only int8 does not speed the reference scan)"`. Measured CPU latency is reported as a deployment observation; it is not retro-fitted to that statement, and the statement is not quietly dropped.

## 8. Fail-closed rule

The Q8 driver re-verifies every pinned input — plan bytes, barrier SHA, all source hashes, all nine checkpoint hashes, the fold manifests, and the frozen K1 reference artifacts — and refuses to run if any differ. Existing output directories are refused.

## 9. Prohibitions observed

No retraining of Full-S1 or K1 · no modification of K1 checkpoints · no rerun of Full-S1 inference · no rerun of K1 inference to reproduce existing metrics · no change to TEST membership · no quantization tuning on TEST · no change to existing Full-S1-vs-K1 statistical claims · no modification of historical frozen TEST artifacts · no reuse of dissertation-era Q8 results · no modification of the dissertation repository · no edit of the historical quantization implementation.
