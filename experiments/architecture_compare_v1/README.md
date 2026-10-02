# Architecture comparison: 4x1 versus 2x2

Status: **prepared, not executed**. This package addresses the TA's question about
removing direction rather than depth. It does not establish a new result.

## Evidence and scope

The original dissertation record selected 4x1 from nine TRAINING-FREE validation
comparisons: median 0.6842917091 versus 0.5574894608, winning 9/9. Those teacher
hashes do not match the publication teachers. Source:
https://github.com/mohammadabdalaziz241/Foundation-Models-for-Industrial-Fault-Diagnosis-/blob/c07107ab4a4c72907509e342ba3eb65dd4424709/methodology_v2/part6_compression/half_student_decision.json

This follow-up evaluates current teachers before recovery and trains a 2x2
student with the existing K1 recovery recipe. `reference_validation.json` is
derived from all nine frozen publication K1 epoch histories at commit
`d3e3bc932238a67f5221cd3baa95ac87f7bf5fa5`. It includes both the selected and final
epoch results; all epoch numbers are zero-based. No existing results are changed.

The protocol was prepared after publication TEST results were known. It is an
exploratory validation follow-up, not a newly blinded TEST study. Validation is
also used for checkpoint selection. Do not interpret validation differences as
independent held-out generalization evidence or as causal proof of KD's value.

The comparison concerns the **specified compression recipes**: 2x2 inherits
layers [0,2]; 4x1 keeps forward branches and uses unit residual scale. It does not
establish an optimum over all depth mappings, directions or residual scales.

## Requirements

Use the original **canonical scientific workspace** on the university machine:

- The original executor at
  `analysis/pcste_v2_lightweight_v1/scripts/02_k1_production.py`, with SHA256
  `1f5d86a1c6b7f2b637ecdd6d61c279a0b86e9edb54bdbe0490d2a4b20fd62308`.
- The original `src/` implementation and its frozen code hash manifest.
- All nine final Full-S1 checkpoints; frozen global manifests; native-12 source
  allowlist/freeze bundle; normalizers; raw datasets at their original paths.
- The original Python/PyTorch environment and adequate RAM for the full fold's
  TRAIN+VAL representations and teacher cache. A GPU is recommended.

The publication GitHub checkout alone does **not** contain those numeric assets.
Do not replace hash expectations or remove checks to make an incomplete checkout
run. Resolve missing prerequisites in the canonical workspace.

Keep this package in the publication checkout. It imports the unchanged executor
from `--canonical-repo`, and imports `src` from that canonical repository.
It does not copy over or edit the original executor/configuration.

## Commands

Activate the existing scientific environment first. Set paths explicitly:

```bash
export PCSTE_CANONICAL_REPO=/absolute/path/to/canonical/scientific/repository
export PCSTE_COMPARE_PACKAGE=/absolute/path/to/Lightweight-PCSTE/experiments/architecture_compare_v1
```

First run the read-only preflight. It hashes source/checkpoints/raw inputs,
verifies the native-12 gate and exact K1 sample stream. It performs no model
inference and no training. Do not continue if it fails.

```bash
python "$PCSTE_COMPARE_PACKAGE/run.py" --mode preflight \
  --canonical-repo "$PCSTE_CANONICAL_REPO" --fold 1 --seed 42 --device cuda
```

Then screen one cell and inspect `screen.json`. This is the required real-data
runtime smoke: three architecture forwards on VALIDATION, no training. The local
guard tests do not substitute for this check.

```bash
python "$PCSTE_COMPARE_PACKAGE/run.py" --mode screen \
  --canonical-repo "$PCSTE_CANONICAL_REPO" --fold 1 --seed 42 --device cuda
```

If successful, screen the remaining eight cells (run sequentially per GPU):

```bash
for fold in 1 2 3; do
  for seed in 42 1337 2026; do
    if [ "$fold" = 1 ] && [ "$seed" = 42 ]; then continue; fi
    python "$PCSTE_COMPARE_PACKAGE/run.py" --mode screen \
      --canonical-repo "$PCSTE_CANONICAL_REPO" --fold "$fold" --seed "$seed" --device cuda || exit 1
  done
done
python "$PCSTE_COMPARE_PACKAGE/summarize.py" \
  --canonical-repo "$PCSTE_CANONICAL_REPO" --phase screen
```

Review the screening output before recovery training. Run all nine 2x2 cells
without changing hyperparameters in response to validation results:

```bash
for fold in 1 2 3; do
  for seed in 42 1337 2026; do
    python "$PCSTE_COMPARE_PACKAGE/run.py" --mode train \
      --canonical-repo "$PCSTE_CANONICAL_REPO" --fold "$fold" --seed "$seed" --device cuda || exit 1
  done
done
python "$PCSTE_COMPARE_PACKAGE/summarize.py" \
  --canonical-repo "$PCSTE_CANONICAL_REPO" --phase train
```

Do not run multiple cache builders for the same fold concurrently. Verified
existing K1 caches are reused read-only; otherwise new caches are built under
this experiment's directory. An exclusive cache-build lock prevents two builders.

## Outputs and failure handling

Everything new is under
`<canonical>/results/pcste_v2/architecture_compare_v1/`:

- `screen_gf*_s*/`: provenance, plan snapshot and Full-S1/2x2/4x1 scores.
- `train_gf*_s*/`: provenance, epoch-zero diagnostic, 50 epoch records, selected
  `best.pt`, and completion summary. Epoch-zero scores cannot win checkpoint selection.
- `summary_screen/` and `summary_train/`: nine paired rows plus descriptive
  aggregate and per-dataset differences. Positive deltas mean **2x2 minus 4x1**.

Every run refuses an existing output directory. No automatic retry or resume is
implemented in v1. An interrupted run is recorded as failed; retain its evidence
and investigate before scheduling a documented clean replacement. No existing
experiment folder or cache may be deleted just to get past a guard.

The full manifest is read as metadata to identify excluded TEST windows. Only
TRAIN windows enter gradients, only VALIDATION enters scoring, and no TEST
waveform, representation or prediction is evaluated by this package.

## Interpretation and remaining work

Compare the nine cells as paired observations, including dataset-level changes.
The summary deliberately has no significance test, automatic winner declaration,
or new TEST evaluation. Shared data and teacher ensembles limit independence.

Similar parameter and recurrent-work budgets do not guarantee equal latency.
A matched 2x2/4x1 runtime benchmark on identical validation inputs is still needed
before claiming a superior performance-cost trade-off. No timing result is
invented or inherited from the full-model-versus-K1 benchmark.

## Verification available here

```bash
cd "$PCSTE_COMPARE_PACKAGE"
python -m unittest -v test_guards
python -m py_compile run.py summarize.py test_guards.py
```

Guard tests cover TRAIN-only optimization, rejected TEST requests, duplicate
window identities, changed executors, missing comparison cells, mixed versions,
and delta orientation. They do **not** execute scientific PyTorch code. PyTorch,
raw datasets, trained checkpoints and the university GPU were unavailable in the
preparation environment. The real preflight/screen commands above remain pending.
