"""Part-5D experiment infrastructure — methodology_v2.

Frozen S0/S1 experimental machinery: dataset-specific linear heads,
nested label-fraction manifests, label-free SSL sampler, hierarchical
supervised sampler, metrics, trainer classes and the immutable run
registry. This package PREPARES the frozen experiment; it never launches
the real 60/50-epoch matrix (bounded smoke steps only, results discarded
and labelled NOT_AN_EXPERIMENT).
"""
from .heads import CLASS_ORDERS, DatasetHeads, window_class  # noqa: F401
from .metrics import (classification_report,  # noqa: F401
                      macro_domain_f1)
