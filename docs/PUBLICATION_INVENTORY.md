# Publication Inventory

Status: the K1 nine-cell matrix, the sealed publication TEST, and the efficiency benchmark are all complete. The remaining open items are the optional Q8 decision and the archival deposit of excluded binary artifacts.

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
| Q8 quantization arm | PENDING | Excluded from the publication evaluation; frozen comparison margin `-0.01` if it is ever run. Scientific inclusion undecided. |
| Efficiency across all nine cells / other hardware | PENDING | The benchmark covers the deterministic GF1/seed-42 pair on one host. Efficiency is architectural, so transfer is expected but unmeasured. |
| Fused Mamba kernel timings | PENDING | All current timings use the pure-PyTorch reference selective scan. |
| Zenodo / release deposit | PENDING | Must contain the artifacts marked Required above, with a DOI referenced from `README.md` and `docs/ARTIFACTS.md`. |
| `PCSTE_DATA_ROOT` support in the dataset registry | PENDING | Currently documented as a provenance annotation only; `src/methodology_v2/registry.py` resolves `<repo root>/data` directly. |
| Licence file | PENDING | To be decided before the repository is made public. |
