# Publication Final Evaluation Plan — Frozen Full-S1 vs Frozen K1

Protocol identity: `lightweight_pcste_final_test_v1`

Freeze timestamp: `2026-09-20T08:11:47+00:00`

Machine-readable plan SHA256: `ce22960df67836b0cb49f476b3fe4c2ed29e5cfcb554f8948e7af3f9fc11a55a`

Status at freeze: TEST remained completely sealed. No TEST dataframe was loaded; no TEST sample, representation, label through the evaluation loader, prediction, inference, or metric was accessed or produced.

## Scope and provenance

This is a new publication-specific evaluation of exactly `Full-S1` and `K1` over the already-fixed 3 folds x 3 seeds. `S0` and `Q8` are excluded. The dissertation-era S0/S1 barrier is unavailable; it was not reconstructed, bypassed, weakened, or modified. The historical evaluator remains unchanged at SHA256 `dd3aa58ce5067298504b9e064cdb51e5f29d0f655b1ca3f4fa6af77600d7f517`.

## Frozen models

| Fold | Seed | Full-S1 SHA256 | K1 SHA256 |
|---:|---:|---|---|
| 1 | 42 | `7d6cea73ba61b3da23c73be14a8edae048276e88162517ee8d72a7c2a133ba0c` | `10770c5359337cae6e8f82665a3533b87548fa7f57fad42c5d4d6ba2f9952c5b` |
| 1 | 1337 | `a14075f0234c232f98d9acb0b02f6094b36c6990a85f5b44c4602c3a95920799` | `bda95639ad23241c6b047cbec0c825ee53a31bab4bed147484ad615062714725` |
| 1 | 2026 | `cfb742f74eb4d55e8f7afceac78e176376b206619f07c6119d02d1d704cc94d2` | `665fa67f9521a4239c9a727e0633276e5eba56c0ed8dd0513fd77c64c2b21cf5` |
| 2 | 42 | `ce6becd8eb0a23a6c4e3c65a2ae8c88f02aff8101a7cb48721c17292170166c2` | `51cd8364afcd10a5657760c4b3493d0a104929d5347efb7b279ee94738eba1fe` |
| 2 | 1337 | `0e01d16be6a01b971aadcbf84df2960eb9e55435a6b3d617cf91367a27dd8a20` | `7018d01c48d1b2c11e96f53cee8c4e8d6579f8f95a9f905dfdcbc2e85c23dfca` |
| 2 | 2026 | `847b7190d095f8734eb4c2aecc5bce34cf05a4d7e587a9dc6f7e32a9e1ee92eb` | `c5e88907837768cfe25ea3a05fdd3eee6386f586e94f48c908d155fec022af25` |
| 3 | 42 | `7ed58dacc4f7cdc35b4f313396a03b30dfe23d59e20cab1477d56335d8c13b89` | `10a0b2b24facf2093c8f65cb0fb362f6a5fc7e5477454b4c77a1c8b0f56c016e` |
| 3 | 1337 | `9b0b6e7f60e06755382f48a40537d2c9860398b17e9bc68809876a6533052cb4` | `e74e71f26eabb69272e851ce2e08b9439a4f3d6990bf7b674174c3299ea90e34` |
| 3 | 2026 | `57caf78847b988f0a46fea21260043422681c6b28af435b82a8e4a2b8d3eb515` | `4840188f9ff5e6cdc60d1c820fcfc266c908d8e47a743208583eb6939719bae1` |

The exact checkpoint paths are frozen in `PUBLICATION_FROZEN_MODEL_TABLE.csv`. All 18 hashes were independently recomputed, and every checkpoint strict-loaded on CPU into its frozen architecture without reading TEST. Full-S1 selection is pinned by `FINAL_SELECTED_CHECKPOINTS.csv` (SHA256 `321bdedddec7613bcfc40f17175f427882f28e45c50a7241b1fd2f66365aa2a3`). K1 selection is pinned by `K1_9_CELL_FREEZE_MANIFEST.md` (SHA256 `a9e5fb49a1f38a7d6e2649ff8d155cf619cd55572bf7238b06a1fcb120d64a8b`).

## Frozen TEST membership

Global protocol: `global_v2_final_s0s1_v1`.

| Fold | Manifest SHA256 |
|---:|---|
| GF1 | `456a69474a3763647b7e0ad13f5538e8f630d268347671ad617134b08e830d0a` |
| GF2 | `74029f81a4b5617b13218812aa26568b0f3a268f8f6b450b5ab0293b17f71f93` |
| GF3 | `fe622f3bc1146998d00fe070ea045f0790145b55a028effea52de00f94000f0e` |

The global hash registry SHA256 is `4b6884b10d30b9b169502a9561e7eed8a0de24dcd29985d5bbca6f718e4165d0`. CWRU uses `pcste_v2_cwru_native12_load_v1`; its complete freeze bundle index digest is `ddf4d32573c016b08da65b4f878ee0511a9bae4b15444626e1d932d24ebf39e4`. Freeze validation inspects bytes/hashes only and does not parse TEST membership through an evaluation loader.

## Frozen metrics

For each of CWRU, JNU, HIT, and MaFaulDa: Macro-F1, accuracy, balanced accuracy, macro precision, macro recall, macro one-vs-rest ROC-AUC where defined, per-class precision/recall/F1, and confusion matrix.

Macro-4 is `(CWRU + JNU + HIT + MaFaulDa) / 4`, with equal domain weights and no window weighting. Macro-4 AUC is defined only if all four dataset Macro-AUC values are finite; no domain may be silently dropped.

- Primary endpoint: `Macro-4 Macro-F1`
- Secondary aggregate endpoint: `Macro-4 Macro-AUC`
- Primary matched difference: `K1 - Full-S1`

## Frozen statistical procedure

The statistical-plan SHA256 is `7d4ba32d9f1df2283c98c012a7268dd7d3664fa52017a84adf9f0b2e9a373629`; `src/methodology_v2/compression/stats.py` SHA256 is `efb3d20c777e4ab0eda15491147950c77c816610d1f71ca0072f8dfa36e3cfcb`.

The nine paired Macro-4 Macro-F1 deltas use the existing `contrast(..., "ni_then_superiority", 0.02)` implementation. Non-inferiority tests H0 `mean(delta) <= -0.02`: add `0.02` to every delta, enumerate all 512 sign patterns, and compute the exact one-sided sign-flip p-value. It passes only if `p < 0.05` and the observed shifted mean is positive. The exact two-sided sign-flip superiority result is reported only after this gate passes.

This protocol has one confirmatory contrast. The historical three-hypothesis Holm family is not executable because its excluded arms are not evaluated; no replacement hypothesis or post-TEST test will be added. Descriptives use sample SD (`ddof=1`) across all nine matched cells and exact wins/ties/losses.

## Strict publication barrier

The separate driver `publication_test_driver.py` has SHA256 `8f2de50a52ce83fefa8dcd1516ec3c348d9f5c1a6b561b38070fe9f94d149f3e`. Preflight requires an externally supplied frozen barrier SHA, rehashes every source/registry/manifest/checkpoint, validates the fixed 9-cell schema, and strict-loads all checkpoint state dictionaries before TEST membership can be parsed.

TEST execution additionally requires the exact token `AUTHORIZE_LIGHTWEIGHT_PCSTE_FINAL_TEST_V1`. Existing output directories are refused. Any mismatch fails closed. The dry self-test changes plan bytes and the non-inferiority margin in memory and requires both alterations to be rejected.
