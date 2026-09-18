# Lightweight PC-STE for Rotating Machinery Fault Diagnosis

This repository contains the publication-oriented implementation and frozen protocol for compressing a shared multi-dataset PC-STE fault-diagnosis encoder into the K1 lightweight model through knowledge distillation.

The scientific workflow is:

`Full PC-STE / S1 teacher ensemble -> knowledge-distilled K1 encoder -> deployment and efficiency evaluation`

K1 uses one shared 1,375,953-parameter encoder across CWRU, JNU, HIT, and MaFaulDa, with dataset-specific classification heads. The matched design covers three global folds and three random seeds. Training uses frozen streams and validation-only checkpoint selection; TEST remains sealed until the complete experiment matrix is frozen.

The repository preserves the exact historical scientific implementation where import layout is material. Publication-facing protocol, configuration, provenance, and small result summaries are organized separately for review.

> This repository is currently private while the associated manuscript and final lightweight evaluation are being completed.

## Current status

- Full-S1 teacher reconstruction for two GF3 seeds is in progress outside this repository.
- Six of nine K1 validation cells (GF1 and GF2) are complete.
- GF3 K1 and final sealed TEST evaluation are not yet included.
- No non-inferiority or final TEST claim is made at this stage.

See `docs/EXPERIMENT_PROTOCOL.md`, `docs/REPRODUCIBILITY.md`, and `docs/ARTIFACTS.md` before running experiments.

