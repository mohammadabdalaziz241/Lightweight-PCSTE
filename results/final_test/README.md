# Final TEST results

Contains the sealed publication TEST executed exactly once under protocol `lightweight_pcste_final_test_v1`.

- `publication_final_test_v1/` — authoritative frozen results. Status `PUBLICATION_FINAL_SEALED_TEST_COMPLETE`.
  - `FINAL_SEALED_TEST_REPORT.md` — human-readable sealed report
  - `PUBLICATION_FINAL_RESULTS.json` — machine-readable results
  - `aggregate_summary.json` — Macro-4 aggregates and the frozen statistical outcome
  - `matched_cells.csv` — the nine paired (fold, seed) cells
  - `PER_DATASET_SUMMARY.csv` — per-dataset Macro-F1 / Macro-AUC with wins/ties/losses
  - `per_model_reports/` — 18 per-(fold, seed, family) class-level and confusion-matrix reports
  - `STATUS` — `COMPLETE`

These files are frozen. They are copied verbatim from the canonical scientific repository, their SHA256 values are recorded in `FROZEN_ARTIFACT_HASHES.sha256`, and they must never be edited here.

Raw per-window prediction and probability CSVs (18 files, ~21.6 MiB) are excluded from Git; see `docs/ARTIFACTS.md`.

Read `docs/RESULTS.md` for the reader-facing summary and the cautious statement of the pre-registered non-inferiority outcome.
