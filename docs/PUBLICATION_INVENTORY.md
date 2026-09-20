# Publication Inventory

Status: all four stages are complete and frozen - the **primary study** (K1 nine-cell matrix + sealed publication TEST), the **efficiency benchmark**, the **secondary Q8 deployment extension**, and the **latency/throughput correction** that supersedes the efficiency benchmark's timing rows. The remaining open item is the archival deposit of excluded binary artifacts.

The primary study and the Q8 extension are deliberately kept separate throughout this repository: Q8 was frozen and evaluated only after the primary sealed comparison was complete, and is never presented as part of it.

Classification codes: `CORE` (implementation), `REPRO` (exact executor), `PROTOCOL` (frozen design/provenance), `RESULTS` (frozen outcome), `EXCLUDED_BINARY` (archival deposit), `EXCLUDED_HISTORICAL` (out of scope), `PENDING` (not yet produced).

## Implementation

| Item | Class | Location | Size | In Git | Final |
|---|---|---|---:|---|---|
| `src/pcste_v2/` final-v1 modules | CORE | `src/pcste_v2/` | ~110 KiB | Yes | Yes |
| `src/methodology_v2/encoder/` | CORE | same path | ~38 KiB | Yes | Yes |
| `src/methodology_v2/experiment/` | CORE | same path | ~29 KiB | Yes | Yes |
| `src/methodology_v2/compression/` | CORE | same path | ~150 KiB | Yes | Yes |

## Executors

| Item | Class | Location | Size | In Git | Final |
|---|---|---|---:|---|---|
| SSL executor | REPRO | `scripts/train_ssl/run_ssl.py` | 13 KiB | Yes | Yes |
| Full-S1 executor | REPRO | `scripts/train_s1/run.py` | 23 KiB | Yes | Yes |
| K1 production executor | REPRO | `scripts/train_k1/02_k1_production.py` | 34 KiB | Yes | Yes |
| Historical dissertation evaluator (unchanged) | REPRO | `scripts/evaluate/evaluate_test.py` | 7 KiB | Yes | Yes |
| Publication TEST driver | REPRO | `configs/lightweight_k1/final_test_v1/publication_test_driver.py`, linked at `scripts/evaluate/publication_test_driver.py` | 20 KiB | Yes | Yes |
| Original Part-6 executors and benchmarks | REPRO | `scripts/reproduce/`, `scripts/methodology_v2/` | ~130 KiB | Yes | Yes |

## Frozen protocol and provenance

| Item | Class | Location | Size | In Git | Final |
|---|---|---|---:|---|---|
| CWRU native-12k load protocol manifests | PROTOCOL | `protocols/datasets/cwru_native12_load_v1/` | ~670 KiB | Yes | Yes |
| Global four-dataset fold manifests | PROTOCOL | `protocols/splits/global_v2_final_s0s1_v1/` | ~22 MiB | Yes | Yes |
| Normalizer registries (hashes, rules, counts) | PROTOCOL | `.../normalizers/registry_fold_{1,2,3}.csv` | ~4 KiB | Yes | Yes |
| Full-S1 specification, statistical plan, selected checkpoints | PROTOCOL | `configs/final_s1/` | ~35 KiB | Yes | Yes |
| K1 protocol, registered cells, compatibility/hash records | PROTOCOL | `configs/lightweight_k1/` | ~15 KiB | Yes | Yes |
| **K1 9-cell freeze manifest** | PROTOCOL | `configs/lightweight_k1/K1_9_CELL_FREEZE_MANIFEST.md` | 5 KiB | Yes | **Yes** |
| **Publication evaluation plan (JSON + MD)** | PROTOCOL | `configs/lightweight_k1/final_test_v1/` | 23 KiB | Yes | **Yes** |
| **Publication frozen model table (18 checkpoints)** | PROTOCOL | `configs/lightweight_k1/final_test_v1/` | 3 KiB | Yes | **Yes** |
| **Publication evaluation barrier** | PROTOCOL | `configs/lightweight_k1/final_test_v1/` | 2 KiB | Yes | **Yes** |
| Frozen artifact hash manifest | PROTOCOL | `FROZEN_ARTIFACT_HASHES.sha256` | 3 KiB | Yes | Yes |

## Results

| Item | Class | Location | Size | In Git | Final |
|---|---|---|---:|---|---|
| **K1 validation summaries, all 9 cells** | RESULTS | `results/lightweight_k1/` | ~185 KiB | Yes | **Yes** |
| **Sealed TEST report** | RESULTS | `results/final_test/publication_final_test_v1/FINAL_SEALED_TEST_REPORT.md` | 3 KiB | Yes | **Yes** |
| **Machine-readable final results** | RESULTS | `.../PUBLICATION_FINAL_RESULTS.json` | 36 KiB | Yes | **Yes** |
| **Per-dataset summary** | RESULTS | `.../PER_DATASET_SUMMARY.csv` | 1 KiB | Yes | **Yes** |
| **Matched cells** | RESULTS | `.../matched_cells.csv` | 1 KiB | Yes | **Yes** |
| **Aggregate summary and statistics** | RESULTS | `.../aggregate_summary.json` | 7 KiB | Yes | **Yes** |
| **Per-model class-level / confusion reports (18)** | RESULTS | `.../per_model_reports/` | ~535 KiB | Yes | **Yes** |
| Reader-facing results summary | RESULTS | `docs/RESULTS.md` | 10 KiB | Yes | Yes |

## Efficiency benchmark

| Item | Class | Location | Size | In Git | Final |
|---|---|---|---:|---|---|
| **Efficiency benchmark plan (JSON + MD)** | PROTOCOL | `benchmarks/efficiency_v1/` | 25 KiB | Yes | **Yes** |
| **Benchmark driver and setup module** | REPRO | `benchmarks/efficiency_v1/{run_benchmark,bench_common}.py` | 29 KiB | Yes | **Yes** |
| **Environment record** | RESULTS | `benchmarks/efficiency_v1/environment.json` | 2 KiB | Yes | **Yes** |
| **Parameter / size results** | RESULTS | `benchmarks/efficiency_v1/parameter_size_results.json` | 3 KiB | Yes | **Yes** |
| **FLOP results** | RESULTS | `benchmarks/efficiency_v1/flops_results.json` | 6 KiB | Yes | **Yes** |
| **Latency raw (32,000 timed iterations)** | RESULTS | `benchmarks/efficiency_v1/latency_raw.csv` | 1.6 MiB | Yes | **Yes** |
| **Latency summary** | RESULTS | `benchmarks/efficiency_v1/latency_summary.csv` | 5 KiB | Yes | **Yes** |
| **Throughput results** | RESULTS | `benchmarks/efficiency_v1/throughput_results.csv` | 1 KiB | Yes | **Yes** |
| **Memory results** | RESULTS | `benchmarks/efficiency_v1/memory_results.json` | 4 KiB | Yes | **Yes** |
| **Headline aggregates** | RESULTS | `benchmarks/efficiency_v1/aggregates.json` | 4 KiB | Yes | **Yes** |
| **Efficiency final report** | RESULTS | `benchmarks/efficiency_v1/EFFICIENCY_FINAL_REPORT.md` | 15 KiB | Yes | **Yes** |
| **Efficiency artifact hash manifest** | PROTOCOL | `benchmarks/efficiency_v1/EFFICIENCY_ARTIFACT_HASHES.sha256` | 1 KiB | Yes | **Yes** |

## Q8 secondary extension

| Item | Class | Location | Size | In Git | Final |
|---|---|---|---:|---|---|
| **Q8 evaluation plan (JSON + MD)** | PROTOCOL | `benchmarks/q8_v1/` | 27 KiB | Yes | **Yes** |
| **Q8 evaluation barrier** | PROTOCOL | `benchmarks/q8_v1/Q8_EVALUATION_BARRIER.json` | 2 KiB | Yes | **Yes** |
| **Q8 frozen model table (9 K1 checkpoints)** | PROTOCOL | `benchmarks/q8_v1/Q8_FROZEN_MODEL_TABLE.csv` | 2 KiB | Yes | **Yes** |
| **Q8 fail-closed driver** | REPRO | `benchmarks/q8_v1/q8_driver.py` | 21 KiB | Yes | **Yes** |
| **Q8 deployment bench** | REPRO | `benchmarks/q8_v1/q8_bench.py` | 17 KiB | Yes | **Yes** |
| **Q8 conversion manifest (9 cells)** | RESULTS | `benchmarks/q8_v1/results/q8_conversion_manifest.json` | 56 KiB | Yes | **Yes** |
| **Q8 matched cells / per-dataset / aggregate** | RESULTS | `benchmarks/q8_v1/results/` | 10 KiB | Yes | **Yes** |
| **Q8 per-cell class-level reports (9)** | RESULTS | `benchmarks/q8_v1/results/gf*_q8_report.json` | ~236 KiB | Yes | **Yes** |
| **Q8 size / GPU probe / agreement / memory** | RESULTS | `benchmarks/q8_v1/results/` | 5 KiB | Yes | **Yes** |
| **Q8 latency raw + summary (both modes)** | RESULTS | `benchmarks/q8_v1/results/q8_latency_*` | ~1.9 MiB | Yes | **Yes** |
| **Q8 throughput** | RESULTS | `benchmarks/q8_v1/results/q8_throughput_results.csv` | 1 KiB | Yes | **Yes** |
| **Q8 final report** | RESULTS | `benchmarks/q8_v1/Q8_FINAL_REPORT.md` | 20 KiB | Yes | **Yes** |
| **Q8 artifact hash manifest (30 entries)** | PROTOCOL | `benchmarks/q8_v1/Q8_ARTIFACT_HASHES.sha256` | 3 KiB | Yes | **Yes** |

No Q8 binary model artifact is stored: Q8 is regenerated deterministically from the frozen K1 checkpoints, and the conversion manifest carries a content digest for byte-level verification.

## Corrected latency/throughput benchmark (authoritative timings)

| Item | Class | Location | Size | In Git | Final |
|---|---|---|---:|---|---|
| **Correction plan (JSON + MD)** | PROTOCOL | `benchmarks/efficiency_latency_correction_v1/` | 26 KiB | Yes | **Yes** |
| **Correction runner** | REPRO | `benchmarks/efficiency_latency_correction_v1/run_correction.py` | 11 KiB | Yes | **Yes** |
| **Environment record** | RESULTS | `.../environment.json` | 2 KiB | Yes | **Yes** |
| **Latency raw (32,000 timed samples)** | RESULTS | `.../latency_raw.csv` (+ per-device) | ~3.3 MiB | Yes | **Yes** |
| **Latency summary** | RESULTS | `.../latency_summary.csv` (+ per-device) | 8 KiB | Yes | **Yes** |
| **Throughput (GPU b32 + CPU b32)** | RESULTS | `.../throughput_results.csv` (+ per-device) | 4 KiB | Yes | **Yes** |
| **Inference-mode verification (4 stages)** | RESULTS | `.../inference_mode_verification_*.json` | 2 KiB | Yes | **Yes** |
| **ABAB order logs** | RESULTS | `.../order_log_*.json` | 12 KiB | Yes | **Yes** |
| **Corrected efficiency summary** | RESULTS | `.../CORRECTED_EFFICIENCY_SUMMARY.json` | 7 KiB | Yes | **Yes** |
| **Correction report** | RESULTS | `.../LATENCY_CORRECTION_REPORT.md` | 13 KiB | Yes | **Yes** |
| **Correction hash manifest (21 entries)** | PROTOCOL | `.../LATENCY_CORRECTION_ARTIFACT_HASHES.sha256` | 2 KiB | Yes | **Yes** |

This directory supersedes `efficiency_v1` for **latency and throughput only**. `efficiency_v1` remains frozen and unmodified, and its parameter, size, FLOP and GPU-memory results remain authoritative.

## Excluded binary artifacts (archival deposit)

Sizes audited in `docs/ARTIFACTS.md`. Text-level identity (SHA256, selection rule, provenance) for every item below is tracked in Git.

| Item | Class | Count / size | Deposit priority |
|---|---|---:|---|
| Full-S1 selected checkpoints | EXCLUDED_BINARY | 9 / 82.2 MiB | Required |
| K1 selected checkpoints | EXCLUDED_BINARY | 9 / 47.5 MiB | Required |
| Frozen normalizer NPZ | EXCLUDED_BINARY | 12 / 157 KiB | Required |
| Raw TEST prediction/probability CSVs | EXCLUDED_BINARY | 18 / 21.6 MiB | Recommended |
| K1 `last.pt` final-epoch states | EXCLUDED_BINARY | 9 / 142.8 MiB | Optional |
| S1 teacher probability caches | EXCLUDED_BINARY | 1.28 GiB | Not deposited; regenerable |
| Raw datasets and archives | EXCLUDED_BINARY | tens of GiB | Never; original providers only |

## Excluded historical material

| Item | Class | Reason |
|---|---|---|
| Dissertation PDF/LaTeX sources | EXCLUDED_HISTORICAL | Not part of the publication implementation |
| Superseded protocols and obsolete experiments | EXCLUDED_HISTORICAL | Not the publication design |
| Recovery, node, and SSH logs; agent transcripts | EXCLUDED_HISTORICAL | Infrastructure archaeology, not science |

The dissertation repository is a separate project and is deliberately **not** configured as a Git remote here.

## Pending

| Item | Class | Notes |
|---|---|---|
| Efficiency and Q8 deployment across all nine cells / other hardware | PENDING | Both benchmarks cover the deterministic GF1/seed-42 pair on one host. Efficiency is architectural, so transfer is expected but unmeasured. |
| TEST evaluation of the Q8 `cpu_dynamic` representation | PENDING | TEST accuracy was evaluated on the frozen `sim` accuracy representation. A deployment accuracy claim for the CPU int8 artifact would need its own pre-registered TEST evaluation. |
| Fused Mamba kernel timings | PENDING | All current timings use the pure-PyTorch reference selective scan. |
| Zenodo / release deposit | PENDING | Must contain the artifacts marked Required above, with a DOI referenced from `README.md` and `docs/ARTIFACTS.md`. |
| `PCSTE_DATA_ROOT` support in the dataset registry | PENDING | Currently documented as a provenance annotation only; `src/methodology_v2/registry.py` resolves `<repo root>/data` directly. |
| Licence file | PENDING | To be decided before the repository is made public. |
