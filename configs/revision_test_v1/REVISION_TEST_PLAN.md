# Revision ablation TEST plan — `lightweight_pcste_revision_ablation_test_v1`

Written on 8 October 2026, before any ablation checkpoint has been evaluated on TEST
and while wave 2 of the ablation training is still running. The machine-readable
version of everything below is the `ANALYSES` constant in
`revision_test_driver.py`; the driver's SHA256 is pinned in the barrier at freeze
time, so the analyses cannot change after this point without the barrier failing.

## 1. Status of this evaluation

- The primary study (K1 vs Full-S1, non-inferiority at −0.02) is **complete and
  sealed** (`lightweight_pcste_final_test_v1`). Nothing here changes it.
- This is a **secondary evaluation** requested in supervisor review. Analyses A and
  B execute tests that were written down **before training** in earlier frozen plans;
  analysis C is **post-hoc** and is labelled as such everywhere it is reported.
- Every ablation arm was trained on the registered K1 batch stream, with K1's
  optimiser, schedule, epoch budget and validation-only checkpoint rule; each differs
  from K1 only in the factor named below. The training executor reproduced K1's
  first two epochs exactly (twice, on two machines) before the ablations were trusted.

## 2. Models (nine matched fold × seed cells each)

| Family | What it is | Source |
|---|---|---|
| Full-S1 | full bidirectional PC-STE, SSL-initialised | published, sealed |
| K1 | 4 layers × forward, initialised from S1, 3-teacher KD + relational | published, sealed |
| S0 | full PC-STE trained from scratch (no SSL) | original pre-registered checkpoints; gf3_s2026 is a documented retrain |
| C_small | K1 architecture, random init, hard-label CE | ablation |
| P1 | K1 architecture, S1 init, hard-label CE (no KD) | ablation |
| K1-2x2 | 2 layers × 2 directions, S1 layers 0 and 2, KD | ablation |
| K1-single | as K1, but one same-seed teacher | ablation |
| K0 | as K1, but S0 init and S0 teachers (no SSL anywhere) | ablation |

## 3. Endpoint and test

- **Endpoint:** Macro-4 Macro-F1 on TEST — the equal mean of the four dataset
  Macro-F1 values within a cell (identical to the primary study).
- **Unit:** one paired difference per (fold, seed) cell; nine pairs.
- **Test:** exact paired sign-flip over all 512 sign patterns, using the frozen
  module `src/methodology_v2/compression/stats.py` unchanged.
- **Multiplicity:** Holm within each family, α = 0.05.
- The smallest attainable two-sided p with nine pairs is 2/512 = 0.0039. With Holm
  over three contrasts, a significant result needs essentially 9/9 or 8/9 cells in
  one direction. Non-significant results are reported as such, not as "no effect".

## 4. Analyses

**A — Does SSL pretraining help the full model? (S1 vs S0)**
Executes `FINAL_STATISTICAL_PLAN.yaml` (final_s0_s1.v1, frozen before training)
without change: two-sided tests on Macro-4 F1 and Macro-4 AUC, Holm m = 2, and the
frozen interpretation labels (SSL_BENEFICIAL / SSL_COMPETITIVE / SSL_UNFAVOURABLE /
SSL_INCONCLUSIVE_OR_MIXED). *Deviation:* S0 gf3_s2026 is a retrained replacement;
the same tests on the eight cells with original S0 checkpoints are reported as a
sensitivity analysis.

**B — Is K1 explained by its small architecture alone, and does SSL matter for the
student?** From the original compression protocol (`statistics_spec.yaml`):
K1 vs C_small (two-sided; originally H2), K1 vs K0 (two-sided), K0 vs S0
(non-inferiority, margin 0.02). Holm m = 3. *Deviation:* the original families also
contained Q8 contrasts, which are reported separately; the three executable
contrasts form one family here.

**C — Review ablations (post-hoc).** K1 vs P1 (does distillation add value beyond
the initialisation?), K1 vs K1-2x2 (is removing a direction better than removing
depth?), K1 vs K1-single (is the three-teacher ensemble needed?). Two-sided, Holm
m = 3. For each, an exact TOST at ±0.02 is reported descriptively, so that a
"no difference" reading is supported by an equivalence test rather than by a
non-significant p.

**Descriptive, for every contrast:** all nine deltas, mean, median, SD, wins/ties/
losses, per-dataset Macro-F1 deltas, Macro-3 excluding CWRU, Macro-4 AUC deltas.

## 5. Safeguards in the driver

1. `--freeze` refuses unless all 45 ablation runs are COMPLETE and TEST-free, and
   every one of the 72 checkpoints matches its recorded SHA256 and strict-loads.
2. The barrier pins the plan, the model table, the driver, the statistics, metrics,
   model and representation code, the fold manifests, the CWRU freeze bundle and
   the 18 sealed publication reports.
3. `--execute` first re-evaluates the 18 already-sealed Full-S1/K1 checkpoints
   (no new TEST exposure) and must reproduce their sealed reports exactly
   (identical confusion matrices). If not, it stops **before** any new model is
   evaluated.
4. TEST is executed once; the output directory cannot be overwritten.

## 6. Known limitations, stated in advance

- CWRU's nine cells reuse one fixed 260-window TEST population (three seeds × three
  joint-training contexts), not nine independent CWRU test sets.
- JNU's TEST set has 24 windows per cell; JNU Macro-F1 moves in steps of about 0.09.
- JNU/HIT and most MaFaulDa TEST windows were scored in an earlier dissertation
  stage (already disclosed in `FINAL_TEST_EXPOSURE_SUMMARY.json`); CWRU 3 hp had
  not been TEST-scored before the primary sealed test.
- Validation saturation (CWRU and JNU at or near 1.0) limited checkpoint selection
  for every arm equally.
