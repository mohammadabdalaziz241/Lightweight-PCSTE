"""methodology_v2 Part 6 — PC-STE lightweight study (compression).

Isolated Part-6 namespace: Student-D surgery, same-fold teacher
ensembles + TRAIN/VAL teacher caching, KD losses, frozen-recipe trainer
(no epoch loop here), optional chunked selective-scan backend + parity
harness, Q8 post-training quantization, Stage-2 training-free
sensitivity tools, deterministic run-registry/seal machinery, TEST
policy (single sealed session + touch ledger), four-axis benchmark
harness and the pre-registered paired statistics.

Nothing in this package modifies the frozen primary artifacts; primary
checkpoints are opened read-only and hash-verified. Epoch loops live in
scripts/methodology_v2/part6_compression.py (guard-tested absent here).
"""
from .protocol import (PART6_DIR, PART6_RESULTS, PART6_VERSION,  # noqa: F401
                       KD_ALPHA, KD_TEMPERATURE, NI_MARGIN_ARCH,
                       NI_MARGIN_PTQ, PUSH_MIN_DELTA, FOLDS, SEEDS)
