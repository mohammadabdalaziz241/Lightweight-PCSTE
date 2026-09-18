# Experiment Protocol

The frozen design contains three global folds and seeds 42, 1337, and 2026. K1 uses 50 epochs, exact frozen sampler streams, and maximum validation Macro-Domain F1 checkpoint selection with strict improvement and earliest-epoch tie retention. TEST is not used during training or selection.

The architecture non-inferiority margin is -0.02. The optional Q8 comparison margin is -0.01. These thresholds are frozen and must not be changed after observing outcomes.

Current publication state: GF1 and GF2 K1 are complete (6/9 cells); GF3 is pending reconstructed teacher lineage. No provisional validation result is presented as a TEST result.

