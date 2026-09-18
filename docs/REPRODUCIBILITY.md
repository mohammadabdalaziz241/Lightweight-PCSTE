# Reproducibility

The exact historical code layout is retained under `src/` and the original executors under `scripts/`. Frozen protocol manifests are under `protocols/`, while publication-facing configurations and immutable provenance records are under `configs/`.

The Full-S1 final-v1 implementation derives from source commit `214f4f7454b3e9757e16639ef714284b2582a7b1`. The K1 adapter production executor SHA256 is `1f5d86a1c6b7f2b637ecdd6d61c279a0b86e9edb54bdbe0490d2a4b20fd62308`.

Use the hash manifests in `configs/lightweight_k1/` to verify the unmodified historical Part6 implementation. Generated checkpoints, teacher caches, normalizer binaries, raw data, and local logs are intentionally absent from Git.

The preserved historical runtime used Python 3.12.3, PyTorch 2.12.0+cu130, CUDA 13.0, NumPy 2.4.4, pandas 3.0.3, SciPy 1.18.0, scikit-learn 1.9.0, and einops 0.8.2.

