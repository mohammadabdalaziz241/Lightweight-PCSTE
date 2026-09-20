# Reproducibility

This document states what a clean reader can reproduce from this repository alone, and what requires artifacts that are not in Git. Nothing here claims that an excluded binary artifact is stored in this repository.

## 1. Python environment

- `requirements.txt` — minimum dependency floors for running the code.
- `requirements_frozen.txt` — the complete pinned environment (142 packages) as captured from the working environment.

The preserved historical runtime was Python 3.12.3, PyTorch 2.12.0+cu130, CUDA 13.0, NumPy 2.4.4, pandas 3.0.3, SciPy 1.18.0, scikit-learn 1.9.0, einops 0.8.2.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements_frozen.txt     # exact environment
# or: pip install -r requirements.txt       # floors only
```

Training and TEST inference need a CUDA GPU. Hash verification, import checks, and the TEST driver's preflight/self-test paths run on CPU.

## 2. Dataset acquisition and layout

Raw data are **not** distributed here. Obtain CWRU, JNU, HIT, and MaFaulDa from their original providers under their own terms; see `docs/DATASETS.md`.

**Path resolution, stated accurately:** `src/methodology_v2/registry.py` resolves dataset locations from a hardcoded `DATA_ROOT = <repo root>/data`. It does **not** read an environment variable. The expected layout is:

```
<repo root>/data/raw/                                  # CWRU official native 12 kHz recordings
<repo root>/data/raw_cwru_48k/                         # CWRU 48 kHz (registered, unused by the final protocol)
<repo root>/data/raw_jnu/JNU-Bearing-Dataset/
<repo root>/data/raw_hit/HIT-dataset/
<repo root>/data/raw_mafaulda/full/
```

`data/` is git-ignored, so a symlink from `<repo root>/data` to an external storage location is the practical arrangement.

**`PCSTE_DATA_ROOT`** is recorded in `.env.example` and written into the `ENVIRONMENT.json` provenance block by `scripts/train_s1/run.py`, but **no loader currently reads it to resolve dataset paths**. Treat it as a provenance annotation, not a working configuration knob, until the registry is changed to honour it. `PCSTE_V2_RESULTS_ROOT` *is* honoured by the SSL and S1 executors and defaults to `<repo root>/results/pcste_v2`.

## 3. Frozen dataset and split manifests

These are text and are tracked in full.

| Item | Path | SHA256 |
|---|---|---|
| Global split protocol | `protocols/splits/global_v2_final_s0s1_v1/` | registry `4b6884b10d30b9b169502a9561e7eed8a0de24dcd29985d5bbca6f718e4165d0` |
| GF1 fold manifest | `.../global_v2_fold_1.csv` | `456a69474a3763647b7e0ad13f5538e8f630d268347671ad617134b08e830d0a` |
| GF2 fold manifest | `.../global_v2_fold_2.csv` | `74029f81a4b5617b13218812aa26568b0f3a268f8f6b450b5ab0293b17f71f93` |
| GF3 fold manifest | `.../global_v2_fold_3.csv` | `fe622f3bc1146998d00fe070ea045f0790145b55a028effea52de00f94000f0e` |
| CWRU load protocol | `protocols/datasets/cwru_native12_load_v1/` | bundle index `ddf4d32573c016b08da65b4f878ee0511a9bae4b15444626e1d932d24ebf39e4` |

### Legacy paths

The frozen records reference the historical repository layout. Those paths resolve here through tracked symlinks, so a frozen manifest can be checked without rewriting it:

| Legacy path | Resolves to |
|---|---|
| `pcste_v2/protocol/global_v2_final_s0s1_v1` | `protocols/splits/global_v2_final_s0s1_v1` |
| `pcste_v2/protocol/cwru_native12_load_v1` | `protocols/datasets/cwru_native12_load_v1` |
| `analysis/pcste_v2_lightweight_v1` | `configs/lightweight_k1` |
| `analysis/pcste_v2_final_s0_s1_v1` | `configs/final_s1` |
| `analysis/pcste_v2_cwru_native12_migration` | `configs/cwru_native12_migration` |
| `analysis/pcste_v2_lightweight_v1/scripts/02_k1_production.py` | `scripts/train_k1/02_k1_production.py` |
| `scripts/pcste_v2/{run,run_ssl,evaluate_test,fit_normalizers}.py` | `scripts/{train_s1,train_ssl,evaluate,protocol}/` |
| `methodology_v2/part6_compression` | `configs/lightweight_k1` |

Clone with symlink support (a default `git clone` on Linux/macOS is fine).

**Frozen bundle verification.** Of the 69 entries in the frozen CWRU bundle index (`protocols/datasets/cwru_native12_load_v1/FREEZE_BUNDLE_INDEX.txt`), **57 resolve in this repository and match their recorded SHA256 exactly, with zero mismatches**. The 12 that are absent are exactly the excluded normalizer `.npz` files. This is asserted by `tests/test_publication_artifacts.py`.

**Normalizer NPZ files are not in Git** (12 files, ~157 KiB total; excluded by the binary-artifact policy in `docs/ARTIFACTS.md`). Their per-file SHA256, fitting rule, window/frame counts and owning fold manifest are fully recorded in `protocols/splits/global_v2_final_s0s1_v1/normalizers/registry_fold_{1,2,3}.csv`, so they are verifiable but must be regenerated or obtained from the archival release.

## 4. Entrypoints

All entrypoints are run from the repository root.

**SSL pretraining** — `scripts/train_ssl/run_ssl.py`

```bash
python scripts/train_ssl/run_ssl.py --protocol global_v2_final_s0s1_v1 --fold {1,2,3} --seed {42,1337,2026} \
    --stage final_v1 --authorize-launch <freeze digest>
```

**Full-S1 training (teacher / base arm)** — `scripts/train_s1/run.py`

```bash
python scripts/train_s1/run.py --variant <variant> --fold {1,2,3} --seed {42,1337,2026} \
    --protocol global_v2_final_s0s1_v1 --stage final_v1 \
    --init-encoder <selected SSL best_encoder.pt> --authorize-launch <freeze digest>
```

TEST windows are never read by either executor (fail-closed assertion).

**K1 training (lightweight student)** — `scripts/train_k1/02_k1_production.py`

```bash
python scripts/train_k1/02_k1_production.py --mode preflight --fold {1,2,3} --seed {42,1337,2026}
python scripts/train_k1/02_k1_production.py --mode run      --fold {1,2,3} --seed {42,1337,2026} --device cuda --authorize <token>
```

`preflight` performs no training and no teacher-cache inference. `run` performs TRAIN+VAL teacher caching, training and validation only; TEST waveform inference is not implemented in this executor.

**Publication TEST evaluation** — `configs/lightweight_k1/final_test_v1/publication_test_driver.py` (also reachable as `scripts/evaluate/publication_test_driver.py`)

```bash
# hash/schema/checkpoint-structure checks only; never reads TEST
python scripts/evaluate/publication_test_driver.py --preflight \
    --expected-barrier-sha256 fa8410c03bcad4d3eea31282ac8ee7828c731a35a025aab872354aa8ffdbe71d

python scripts/evaluate/publication_test_driver.py --self-test \
    --expected-barrier-sha256 fa8410c03bcad4d3eea31282ac8ee7828c731a35a025aab872354aa8ffdbe71d

# real TEST access; requires checkpoints and the authorization token
python scripts/evaluate/publication_test_driver.py --execute \
    --expected-barrier-sha256 fa8410c03bcad4d3eea31282ac8ee7828c731a35a025aab872354aa8ffdbe71d \
    --authorization AUTHORIZE_LIGHTWEIGHT_PCSTE_FINAL_TEST_V1 \
    --output-dir <new directory> --device auto
```

The sealed TEST has already been executed exactly once under this driver. Re-running `--execute` is not part of this repository's workflow; the driver refuses an existing output directory.

**Stated plainly: the driver's preflight cannot complete inside this repository.** It rehashes and strict-loads all eighteen checkpoints and the twelve normalizer `.npz` files, none of which are in Git. Run against this repository it fails closed at the first missing artifact — which is the barrier behaving correctly, not a defect. Full preflight requires the canonical scientific repository, or this repository plus the archival deposit described in section 8.

`scripts/evaluate/evaluate_test.py` is the **historical dissertation-era evaluator**, retained unchanged at SHA256 `dd3aa58ce5067298504b9e064cdb51e5f29d0f655b1ca3f4fa6af77600d7f517` for traceability. It was not used to produce the publication result.

## 5. Frozen configuration

| Item | Path | SHA256 |
|---|---|---|
| K1 9-cell freeze manifest | `configs/lightweight_k1/K1_9_CELL_FREEZE_MANIFEST.md` | `a9e5fb49a1f38a7d6e2649ff8d155cf619cd55572bf7238b06a1fcb120d64a8b` |
| K1 production protocol | `configs/lightweight_k1/K1_PRODUCTION_PROTOCOL.json` | `f2417ad057709b10b9d928ffc84114b5bfc135d818be58e664fb89e2d9134cb3` |
| K1 registered cells | `configs/lightweight_k1/K1_REGISTERED_CELLS.csv` | `6d7e91b0bbd4ade3befbb97fccf18394526be33e8b7e54ef9c9fbcd0a5cee753` |
| Full-S1 selected checkpoints | `configs/final_s1/FINAL_SELECTED_CHECKPOINTS.csv` | `321bdedddec7613bcfc40f17175f427882f28e45c50a7241b1fd2f66365aa2a3` |
| Statistical plan | `configs/final_s1/FINAL_STATISTICAL_PLAN.yaml` | `7d4ba32d9f1df2283c98c012a7268dd7d3664fa52017a84adf9f0b2e9a373629` |
| Statistics implementation | `src/methodology_v2/compression/stats.py` | `efb3d20c777e4ab0eda15491147950c77c816610d1f71ca0072f8dfa36e3cfcb` |
| Publication evaluation plan | `configs/lightweight_k1/final_test_v1/PUBLICATION_FINAL_EVALUATION_PLAN.json` | `ce22960df67836b0cb49f476b3fe4c2ed29e5cfcb554f8948e7af3f9fc11a55a` |
| Publication barrier | `configs/lightweight_k1/final_test_v1/PUBLICATION_FINAL_EVALUATION_BARRIER.json` | `fa8410c03bcad4d3eea31282ac8ee7828c731a35a025aab872354aa8ffdbe71d` |
| Frozen model table (18 checkpoints) | `configs/lightweight_k1/final_test_v1/PUBLICATION_FROZEN_MODEL_TABLE.csv` | `34c1a6b3ed2c9f545cb6b368c46e22a2bdf3b03a1a5134f3f73180b484ab5cd9` |

The Full-S1 final-v1 implementation derives from source commit `214f4f7454b3e9757e16639ef714284b2582a7b1`. The K1 production executor SHA256 is `1f5d86a1c6b7f2b637ecdd6d61c279a0b86e9edb54bdbe0490d2a4b20fd62308`. Use the hash manifests in `configs/lightweight_k1/ORIGINAL_PART6_*.sha256` to confirm the unmodified historical Part-6 implementation.

## 6. Result locations

| Result | Path |
|---|---|
| K1 validation-stage per-cell records (9 cells) | `results/lightweight_k1/pcstev2_lightweight_v1_k1_gf{1,2,3}_s{42,1337,2026}/` |
| Sealed final TEST (summary + statistics) | `results/final_test/publication_final_test_v1/` |
| Per-model class-level / confusion reports (18) | `results/final_test/publication_final_test_v1/per_model_reports/` |
| Reader-facing results summary | `docs/RESULTS.md` |

## 7. Verifying this repository

```bash
sha256sum -c FROZEN_ARTIFACT_HASHES.sha256     # every frozen artifact tracked here
pytest tests/ -q                               # includes publication-safe artifact checks
```

`tests/test_publication_artifacts.py` verifies the frozen hashes, parses every frozen JSON/CSV artifact, checks the nine-cell and eighteen-checkpoint schema, cross-checks `docs/RESULTS.md` against the frozen result files, and confirms documented paths exist. It touches no TEST data and runs no inference.

## 8. What is required for full checkpoint-level reproduction (not in Git)

Bit-exact reproduction of the sealed TEST numbers requires artifacts that are deliberately excluded from Git and will be published as release assets or a Zenodo deposit. Until that deposit exists, these are **not** obtainable from this repository:

| Needed artifact | Count / size | Purpose |
|---|---|---|
| Full-S1 selected checkpoints (`best.pt`) | 9 files, ~82.2 MiB | Teacher/base arm inference; SHA256 in the frozen model table |
| K1 selected checkpoints (`best.pt`) | 9 files, ~47.5 MiB | Student arm inference; SHA256 in the K1 freeze manifest |
| K1 final-epoch checkpoints (`last.pt`) | 9 files, ~142.8 MiB | Training-state audit only; not needed for TEST |
| Frozen normalizer NPZ | 12 files, ~157 KiB | Required input normalization state; SHA256 in the normalizer registries |
| S1 teacher probability caches | ~1.28 GiB | KD training input; regenerable from Full-S1 checkpoints |
| Raw TEST prediction/probability CSVs | 18 files, ~21.6 MiB | Re-deriving metrics without rerunning inference |

Everything needed to *audit* those artifacts — their SHA256 values, selection rules, epochs, and provenance — is tracked here in text form. What is missing is only the bytes.

## 9. Not yet measured

Deployment and efficiency benchmarking (parameters, FLOPs, latency, throughput, memory) has not been run under the publication protocol, and dissertation-era efficiency numbers are not carried over. The optional Q8 quantization arm has not been executed.
