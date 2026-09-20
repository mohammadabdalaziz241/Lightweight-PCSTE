# Artifact Policy

Git history in this repository contains source code, frozen text manifests, protocol metadata, pre-registration records, tests, and small result summaries **only**. No model weights, no raw data, no caches, no large numeric arrays.

Every frozen artifact that *is* tracked is listed with its SHA256 in `FROZEN_ARTIFACT_HASHES.sha256`. Verify with:

```bash
sha256sum -c FROZEN_ARTIFACT_HASHES.sha256
```

Frozen artifacts are copied verbatim from the canonical scientific repository and are never edited in place. A correction is made by re-freezing upstream and re-importing, never by hand-editing a file here.

## Tracked in Git

| Artifact | Location | Size |
|---|---|---|
| Source implementation | `src/` | ~413 KiB |
| Executors (SSL, S1, K1, evaluation) | `scripts/` | ~344 KiB |
| Frozen split and dataset manifests | `protocols/` | ~23 MiB |
| Frozen Full-S1 and K1 configuration | `configs/` | ~124 KiB (including the pre-registration bundle) |
| Publication TEST pre-registration (plan, model table, barrier, driver) | `configs/lightweight_k1/final_test_v1/` | ~49 KiB |
| K1 per-cell validation summaries (9 cells: `completion.json`, `state.json`, `epoch_metrics.jsonl`) | `results/lightweight_k1/` | ~185 KiB |
| Sealed TEST summaries and statistics | `results/final_test/publication_final_test_v1/` | ~49 KiB |
| Per-model class-level / confusion-matrix reports (18 files) | `results/final_test/publication_final_test_v1/per_model_reports/` | ~535 KiB |
| Efficiency benchmark plan, driver, results and report | `benchmarks/efficiency_v1/` | ~1.7 MiB |
| Q8 extension plan, barrier, driver, results and report | `benchmarks/q8_v1/` | ~2.5 MiB |
| Corrected latency/throughput benchmark | `benchmarks/efficiency_latency_correction_v1/` | ~3.7 MiB |

Per-model reports contain per-dataset confusion matrices, per-class precision/recall/F1 with support, macro one-vs-rest AUC, CWRU per-specimen recall, MaFaulDa per-configuration recall, per-severity breakdowns, and each checkpoint's path and SHA256. They are text JSON and comfortably Git-safe.

## Excluded from Git — size audit

Measured against the canonical scientific repository at final-freeze time. These are candidates for a release asset or Zenodo deposit.

| Artifact | Count | Total size | Why excluded | Archival route |
|---|---:|---:|---|---|
| Raw TEST prediction/probability CSVs | 18 | 22,684,469 B (21.6 MiB) | Large per-window numeric arrays; all derived metrics are already tracked | Zenodo deposit alongside the sealed results |
| K1 selected checkpoints (`best.pt`) | 9 | 49,856,499 B (47.5 MiB) | Binary weights | Release asset or Zenodo — **required** for checkpoint-level reproduction |
| K1 final-epoch checkpoints (`last.pt`) | 9 | 149,711,157 B (142.8 MiB) | Binary weights, training-state audit only | Zenodo, optional |
| Full-S1 selected checkpoints (`best.pt`) | 9 | 86,151,852 B (82.2 MiB) | Binary weights | Release asset or Zenodo — **required** for checkpoint-level reproduction |
| S1 teacher probability caches (`.npz`) | 2 present | 1,369,179,575 B (1.28 GiB) | Very large; regenerable from Full-S1 checkpoints | Not archived; document regeneration instead |
| Frozen normalizer state (`.npz`) | 12 | 160,920 B (157 KiB) | Binary numeric state under the `*.npz` policy | Release asset or a compact archival bundle — **required** for reproduction |
| Raw datasets and archives | — | tens of GiB | Third-party licensing; not redistributable | Original providers only |

The normalizer files are small enough to be Git-safe on size alone, but are excluded to keep a single consistent rule: **no binary numeric state in Git.** Their SHA256 values, fitting rule, window/frame counts and owning fold manifest are fully recorded in `protocols/splits/global_v2_final_s0s1_v1/normalizers/registry_fold_{1,2,3}.csv`, so exclusion costs auditability nothing.

Total excluded footprint: 1,677,744,472 B, approximately **1.56 GiB**, excluding raw datasets.

For comparison, the entire Git-tracked working tree is about **23 MiB**, of which the frozen split manifests under `protocols/` are ~23 MiB.

## Required for the archival deposit

A future release or Zenodo deposit must contain, at minimum, the artifacts marked **required** above: the 9 Full-S1 `best.pt`, the 9 K1 `best.pt`, and the 12 normalizer `.npz`. With those plus this repository, the sealed TEST is reproducible bit-for-bit through the frozen publication driver. The 18 raw prediction CSVs should be deposited too, so that metrics can be re-derived without GPU inference.

## Never added

- `.pt` / `.pth` / `.ckpt` model weights
- Raw datasets and dataset archives
- Teacher caches
- Large `.npz` / `.npy` arrays
- Virtual environments, node logs, build output
- SSH/recovery history and infrastructure archaeology
- Codex/Claude transcripts
- Dissertation PDF/LaTeX sources
- Obsolete experiments and superseded protocols

These are enforced by `.gitignore`. No binary artifact may be added without first verifying its identity, scientific necessity, licence, and storage destination.

## Efficiency benchmark artifacts

`benchmarks/efficiency_v1/` is tracked in full and hashed in `EFFICIENCY_ARTIFACT_HASHES.sha256`. The bulk is `latency_raw.csv` (~1.6 MiB): 32,000 individual timed iterations — 4 datasets x 2 scopes x 2 devices x 2 models x 1000 samples. It is plain text, it is the evidence behind every median and p95 in the report, and it is small enough to keep, so it is kept rather than summarised away.

The benchmark **produces no binary artifact**. Standardized FP32 state_dict serializations are written to temporary files purely to measure their byte size and are deleted immediately; they never enter Git. The benchmark *reads* the two frozen `best.pt` checkpoints and the fold-1 normalizer `.npz` from the canonical scientific repository — those remain excluded here and are part of the archival deposit.

Result files are marked read-only (mode 444). They must never be edited in place; a correction means re-running the benchmark and re-freezing.

## Corrected latency/throughput artifacts

`benchmarks/efficiency_latency_correction_v1/` is tracked in full and hashed in `LATENCY_CORRECTION_ARTIFACT_HASHES.sha256` (21 entries). It is the **authoritative** publication timing result; `benchmarks/efficiency_v1/` stays frozen and unmodified for provenance.

The bulk is raw timing samples. Both the per-device files written directly by each runner invocation (`latency_raw_cpu.csv`, `latency_raw_gpu.csv`) and the merged deliverable (`latency_raw.csv`) are kept: the per-device files are the untouched run outputs, the merged file is the required artifact. That costs ~1.8 MiB of duplication and buys exact provenance for each invocation, which is the better trade for a benchmark whose whole purpose is correcting a measurement-condition error.

Also tracked: `inference_mode_verification_*.json` (one per stage, recording the `torch.is_inference_mode_enabled()` assertion tally — 584 assertions, 0 violations) and `order_log_*.json` (the realised ABAB interleaving order). These exist so the corrected measurement condition is auditable rather than merely claimed.

No binary artifact is produced. Result files are mode 444.

## Q8 extension artifacts

`benchmarks/q8_v1/` is tracked in full and hashed in `Q8_ARTIFACT_HASHES.sha256` (30 entries). The bulk is the two latency raw CSVs (~1.9 MiB combined): 32,000 timed samples each for the primary `inference_mode` run and the secondary run that matches the `efficiency_v1` measurement mode. Both are retained because absolute milliseconds are not comparable across the two benchmarks; keeping only one would make the discrepancy unverifiable.

Also tracked: the nine per-cell Q8 class-level reports (~26 KiB each, ~236 KiB total) carrying confusion matrices, per-class precision/recall/F1, per-specimen and per-configuration recall - the same structure as the primary TEST reports.

**No Q8 binary model artifact enters Git.** Q8 models are derived deterministically from the frozen K1 checkpoints at run time and are never serialized to the repository. `q8_conversion_manifest.json` records, per cell, the module plan, per-channel scale statistics, and a deterministic content digest of the produced int8 state, so any regenerated Q8 artifact can be checked against it byte-for-byte without storing it.

Standardized serializations used for size measurement are written to in-memory buffers only. Result files are mode 444; a correction means re-running and re-freezing, never editing in place.

**Archival plan.** Q8 needs no new deposit entry: with the K1 `best.pt` checkpoints already marked *Required* in the deposit table above, plus this repository's conversion manifest, every Q8 artifact is exactly reproducible.
