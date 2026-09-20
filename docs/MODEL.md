# Model

## Full-S1 (teacher and base arm)

The full PC-STE v2 S1 model, trained on the frozen four-domain recipe from an SSL-pretrained encoder. It serves two roles: the reference arm in the publication comparison, and — as a same-fold three-seed ensemble — the knowledge-distillation teacher.

## K1 (publication lightweight model)

Architecture `half_4x1`: four forward-only temporal Mamba layers with `kept_direction=fwd` and `uni_residual=mean_of_remaining`.

- **1,375,953 verified encoder parameters**, one shared encoder across CWRU, JNU, HIT, and MaFaulDa.
- Dataset-specific classification heads.
- Initialized from the matched Full-S1 checkpoint of the same (fold, seed) cell.

Knowledge distillation uses temperature 4, alpha 0.5, relational weight 1.0, `mean_prob_at_T` teacher aggregation, and KL direction teacher || student. Fifty epochs per cell, no early stopping, validation-only checkpoint selection on Macro-Domain F1 with strict improvement and earliest-epoch tie retention.

All nine cells are frozen in `configs/lightweight_k1/K1_9_CELL_FREEZE_MANIFEST.md`.

## Q8 (optional extension)

An 8-bit quantized variant of K1. **Not executed for publication.** Its inclusion is an open scientific decision with a separately frozen comparison margin of `-0.01`.

## Efficiency

The parameter count above is an architectural property, verified by construction. No FLOP, latency, throughput, or memory figure is claimed as a publication measurement; that benchmarking is the next publication stage and will not reuse dissertation-era numbers.
