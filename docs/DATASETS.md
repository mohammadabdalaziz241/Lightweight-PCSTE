# Datasets

The final study uses CWRU, JNU, HIT, and MaFaulDa. Raw datasets are **not** distributed in this repository. Obtain each under its original terms from its original provider.

CWRU uses only official native 12 kHz recordings, without 48 kHz inputs or resampling. Its final load partition is TRAIN = 0 hp + 1 hp, VALIDATION = 2 hp, TEST = 3 hp (`cwru_native12_load_v1`). The global split protocol is `global_v2_final_s0s1_v1`, with three folds and a sealed TEST partition.

Frozen text manifests under `protocols/` describe every recording, window, fold assignment, and hash. They are complete and tracked in Git.

## Local layout

`src/methodology_v2/registry.py` resolves dataset paths from a hardcoded `DATA_ROOT = <repo root>/data`; it does not read an environment variable. Expected layout:

```
<repo root>/data/raw/                        # CWRU official native 12 kHz
<repo root>/data/raw_cwru_48k/               # CWRU 48 kHz (registered, unused by the final protocol)
<repo root>/data/raw_jnu/JNU-Bearing-Dataset/
<repo root>/data/raw_hit/HIT-dataset/
<repo root>/data/raw_mafaulda/full/
```

`data/` is git-ignored; symlink it to external storage.

`PCSTE_DATA_ROOT` in `.env.example` is recorded into run provenance by `scripts/train_s1/run.py` but is **not** currently read by any dataset loader. Treat it as an annotation, not a configuration knob. `PCSTE_V2_RESULTS_ROOT` *is* honoured by the SSL and S1 executors and defaults to `<repo root>/results/pcste_v2`.

## Normalization state

Per-fold, per-dataset normalizer `.npz` files are excluded from Git under the binary-artifact policy. Their SHA256 values, fitting rule (`denominator = max(std_train, 0.05)`), window/frame counts, and owning fold manifest are recorded in `protocols/splits/global_v2_final_s0s1_v1/normalizers/registry_fold_{1,2,3}.csv`. They are required for checkpoint-level reproduction and will be part of the archival deposit; see `docs/ARTIFACTS.md`.
