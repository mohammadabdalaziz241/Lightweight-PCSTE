"""methodology_v2 — Part 1: dataset selection, audit, and common task definition.

This package is an isolated namespace for the redesigned dissertation
methodology (self-supervised vs supervised comparison over CWRU, JNU, HIT
and MaFaulDa). It is strictly an *audit* package:

- it only ever opens raw data files read-only;
- it never creates windows, splits, spectrograms, or any training artefact;
- it must never import torch or any module from the legacy pipelines
  (src/trainer.py, src/preprocessing.py, ...), so that previous experiments
  remain byte-reproducible and no training code can be invoked by accident.

Deliverables are written to methodology_v2/part1_audit/ at the repository
root. Nothing under data/ is ever modified.
"""

FORBIDDEN_IMPORTS = ("torch", "src.trainer", "src.preprocessing",
                     "src.joint_ssl", "src.carrier_ssl", "src.datasets",
                     "src.data_loader")
