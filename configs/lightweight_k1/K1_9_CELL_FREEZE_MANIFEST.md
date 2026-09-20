# K1 9-Cell Validation Freeze Manifest

Freeze verdict: `K1_9_OF_9_FROZEN_READY_FOR_TEST`

This manifest freezes the completed K1 validation-stage matrix. TEST remained sealed throughout training, checkpoint selection, preservation, and this audit. No TEST metric was computed or used.

## Frozen K1 identity

- Architecture: `half_4x1`
- Encoder parameters: `1,375,953`
- Knowledge-distillation temperature: `4`
- Alpha: `0.5`
- Relational weight: `1.0`
- Teacher aggregation: `mean_prob_at_T`
- KL direction: `teacher || student`
- Checkpoint selection: validation-only maximum Macro-Domain F1; strict improvement; exact tie selects the earlier epoch; no early stopping
- Architecture non-inferiority margin: `-0.02`
- Executor SHA256: `1f5d86a1c6b7f2b637ecdd6d61c279a0b86e9edb54bdbe0490d2a4b20fd62308`
- Epochs per cell: `50`
- TEST used: `false` for every cell

## Frozen cells

| Fold | Seed | Best epoch | Val Macro-Domain F1 | best.pt SHA256 | Stream SHA | Executor SHA | TEST used |
|---:|---:|---:|---:|---|---|---|---|
| 1 | 42 | 23 | 0.9934071749256772 | `10770c5359337cae6e8f82665a3533b87548fa7f57fad42c5d4d6ba2f9952c5b` | `699b74096d780bdd5d23128f3d7de7684ac19d33927a1c417f72ad63fe29f91e` | `1f5d86a1c6b7f2b637ecdd6d61c279a0b86e9edb54bdbe0490d2a4b20fd62308` | false |
| 1 | 1337 | 25 | 0.9905436529191768 | `bda95639ad23241c6b047cbec0c825ee53a31bab4bed147484ad615062714725` | `9829ae898a1d0f71b9482d18ce30f4c510359b6e7d293e8de9c5166748b21325` | `1f5d86a1c6b7f2b637ecdd6d61c279a0b86e9edb54bdbe0490d2a4b20fd62308` | false |
| 1 | 2026 | 1 | 0.9886525380030676 | `665fa67f9521a4239c9a727e0633276e5eba56c0ed8dd0513fd77c64c2b21cf5` | `af8164298762c1b54e9ea6d7fa3e08bf5c66418d023539e1a6215ab8d0a3a9b9` | `1f5d86a1c6b7f2b637ecdd6d61c279a0b86e9edb54bdbe0490d2a4b20fd62308` | false |
| 2 | 42 | 11 | 0.9917302455105466 | `51cd8364afcd10a5657760c4b3493d0a104929d5347efb7b279ee94738eba1fe` | `a14c209bba2ccee48d698edaef7324c114becb0af37bfd7a6c238d4acb99a400` | `1f5d86a1c6b7f2b637ecdd6d61c279a0b86e9edb54bdbe0490d2a4b20fd62308` | false |
| 2 | 1337 | 20 | 0.9841301559829307 | `7018d01c48d1b2c11e96f53cee8c4e8d6579f8f95a9f905dfdcbc2e85c23dfca` | `39151bed0031d5cd9e0befdfa894e1825f07301b01d04d6fce736653db59b6eb` | `1f5d86a1c6b7f2b637ecdd6d61c279a0b86e9edb54bdbe0490d2a4b20fd62308` | false |
| 2 | 2026 | 2 | 0.9785258384983887 | `c5e88907837768cfe25ea3a05fdd3eee6386f586e94f48c908d155fec022af25` | `8de0fe7e97aeb2a17dd134041127799a84c5e8d6a380aeaaa216f58dccf6bbca` | `1f5d86a1c6b7f2b637ecdd6d61c279a0b86e9edb54bdbe0490d2a4b20fd62308` | false |
| 3 | 42 | 33 | 0.9870684656766702 | `10a0b2b24facf2093c8f65cb0fb362f6a5fc7e5477454b4c77a1c8b0f56c016e` | `2efc74989f4708b9443ae86f2e94dc3e2ab4f0350f721d73befc95df7cf9b86a` | `1f5d86a1c6b7f2b637ecdd6d61c279a0b86e9edb54bdbe0490d2a4b20fd62308` | false |
| 3 | 1337 | 17 | 0.9714522759323951 | `e74e71f26eabb69272e851ce2e08b9439a4f3d6990bf7b674174c3299ea90e34` | `3a7571030af82605bbb20a16ab356327236ac8b797f330b0ddcc37c382ed26f9` | `1f5d86a1c6b7f2b637ecdd6d61c279a0b86e9edb54bdbe0490d2a4b20fd62308` | false |
| 3 | 2026 | 7 | 0.987262678921944 | `4840188f9ff5e6cdc60d1c820fcfc266c908d8e47a743208583eb6939719bae1` | `6574aa23907ff47a2c441a6770c2c8329c92da989009b113a03292a600b56309` | `1f5d86a1c6b7f2b637ecdd6d61c279a0b86e9edb54bdbe0490d2a4b20fd62308` | false |

Best epoch values are the zero-based indices stored in `state.json` and `completion.json`.

## Teacher provenance

Every cell records `teacher_cache.name=s1`, `teacher_cache.ensemble_rule=mean_prob_at_T`, and the complete three-seed teacher ensemble for its fold. Each cell's `source_s1_sha256` matches its registered fold/seed teacher.

- GF1 ensemble: `7d6cea73ba61b3da23c73be14a8edae048276e88162517ee8d72a7c2a133ba0c`, `a14075f0234c232f98d9acb0b02f6094b36c6990a85f5b44c4602c3a95920799`, `cfb742f74eb4d55e8f7afceac78e176376b206619f07c6119d02d1d704cc94d2`
- GF2 ensemble: `ce6becd8eb0a23a6c4e3c65a2ae8c88f02aff8101a7cb48721c17292170166c2`, `0e01d16be6a01b971aadcbf84df2960eb9e55435a6b3d617cf91367a27dd8a20`, `847b7190d095f8734eb4c2aecc5bce34cf05a4d7e587a9dc6f7e32a9e1ee92eb`
- GF3 ensemble: `7ed58dacc4f7cdc35b4f313396a03b30dfe23d59e20cab1477d56335d8c13b89`, `9b0b6e7f60e06755382f48a40537d2c9860398b17e9bc68809876a6533052cb4`, `57caf78847b988f0a46fea21260043422681c6b28af435b82a8e4a2b8d3eb515`

## Freeze audit

- Matrix completeness: GF1 `3/3`, GF2 `3/3`, GF3 `3/3`, total `9/9`.
- All nine `state.json` and `completion.json` files exist and parse as valid JSON.
- All nine states report `COMPLETE`, the correct fold and seed, and matching selection metadata in state/completion.
- All nine `epoch_metrics.jsonl` files parse as exactly 50 JSON records.
- Every registered stream SHA and teacher lineage matches `K1_REGISTERED_CELLS.csv`.
- Every selected `best.pt` exists; its SHA256 was recomputed and matches both metadata records.
- The preserved production executor recomputes to the frozen executor SHA256.
- TEST remained sealed and `test_used=false` in both state and completion for all nine cells.
