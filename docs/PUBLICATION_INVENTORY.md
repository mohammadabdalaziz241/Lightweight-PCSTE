# Publication Inventory

| Source path | Classification | Proposed destination | Why needed | Size | Git-safe? | Final now? |
|---|---|---|---|---:|---|---|
| `src/pcste_v2/` selected final-v1 modules | INCLUDE_CORE | `src/pcste_v2/` | Model, STFT/N2b representation, final SSL/S1 protocol | ~80 KB | Yes | Yes |
| `src/methodology_v2/encoder/` | INCLUDE_CORE | same path | PC-STE, coordinates, Mamba/SSM, patching | ~38 KB | Yes | Yes |
| `src/methodology_v2/experiment/` | INCLUDE_CORE | same path | Heads, normalization-facing registries, samplers, metrics, trainers | ~29 KB | Yes | Yes |
| `src/methodology_v2/compression/` | INCLUDE_CORE | same path | Exact student, KD losses, teachers, trainer, quantization, benchmark, statistics | ~150 KB | Yes | Yes |
| `scripts/pcste_v2/run_ssl.py` | INCLUDE_REPRODUCIBILITY | `scripts/train_ssl/` | Exact final SSL executor | 13 KB | Yes | Yes |
| `scripts/pcste_v2/run.py` | INCLUDE_REPRODUCIBILITY | `scripts/train_s1/` | Exact final S1 executor | 23 KB | Yes | Yes |
| `analysis/.../02_k1_production.py` | INCLUDE_REPRODUCIBILITY | `scripts/train_k1/` | Final K1 production adapter/executor | 34 KB | Yes | Yes |
| Original Part6 executors/benchmarks | INCLUDE_REPRODUCIBILITY | `scripts/reproduce/` | Historical K1/efficiency traceability | ~130 KB | Yes | Yes |
| `cwru_native12_load_v1/` text manifests | INCLUDE_PROTOCOL | `protocols/datasets/` | Final repaired native-12k load protocol | ~670 KB | Yes | Yes |
| `global_v2_final_s0s1_v1/` CSV/text | INCLUDE_PROTOCOL | `protocols/splits/` | Exact four-dataset global folds | ~22 MB | Yes | Yes |
| Global normalizer NPZ files | EXCLUDE_LARGE_ARTIFACT | hash/reference only | Required binary normalization state | ~160 KB | No initially | Archive later |
| Final S1 specification and selected registry | INCLUDE_PROTOCOL | `configs/final_s1/` | Frozen teacher baseline and checkpoint identities | ~35 KB | Yes | Yes |
| K1 protocol, registered cells, compatibility/hash records | INCLUDE_PROTOCOL | `configs/lightweight_k1/` | Exact nine-cell KD design and provenance | ~15 KB | Yes | Yes |
| Six completed K1 `completion/state/epoch_metrics` summaries | INCLUDE_RESULTS | `results/lightweight_k1/` | Validation-stage audit without checkpoints | ~130 KB | Yes | Yes |
| Full-S1 checkpoints | EXCLUDE_LARGE_ARTIFACT | hash/reference only | Teacher identity | large | No | Archive later |
| Teacher cache NPZ | EXCLUDE_LARGE_ARTIFACT | external reference | KD input cache (~645 MB) | 645 MB+ | No | Archive later |
| K1 checkpoints and `last.pt` | EXCLUDE_LARGE_ARTIFACT | hash/reference only | Selected/generated binary states | ~133 MB current | No | Archive later |
| Final TEST executor | DOCUMENT_ONLY | `scripts/evaluate/` | Preserve gated evaluation implementation | small | Yes | Script final; results pending |
| Final TEST results | FINALIZE_AFTER_GF3 | `results/final_test/` | Sealed comparison | — | TBD | No |
| GF3 SSL/S1/K1 summaries | FINALIZE_AFTER_GF3 | results/config registries | Complete nine-cell design | — | Yes later | No |
| Q8 code | DOCUMENT_ONLY | compression source/config | Optional deployment extension | small | Yes | Scientific inclusion pending |
| Dissertation documents and obsolete protocols | EXCLUDE_HISTORICAL | none | Not part of publication implementation | large | N/A | Excluded |
| Recovery/node/SSH logs and histories | EXCLUDE_HISTORICAL | none | Infrastructure archaeology, not science | variable | No | Excluded |

