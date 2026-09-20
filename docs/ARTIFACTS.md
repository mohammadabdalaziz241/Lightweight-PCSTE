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

## Optional Q8 artifact

The Q8 quantization extension has not been executed for publication. If it is later retained as a scientific result, its artifacts follow the same rule: text records in Git, binary state in the archival deposit.
