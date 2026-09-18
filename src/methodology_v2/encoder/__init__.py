"""PC-STE encoder package — methodology_v2 Part 5B.

Model implementation lives in this subpackage (outside the audit-purity
scans that protect src/methodology_v2/*.py). Contains ONLY the encoder
and its verification utilities: no SSL objectives, no masking policy, no
reconstruction modules, no supervised heads, no optimisation loops.
"""
from .pcste import PCSTE, PCSTEConfig  # noqa: F401
from .collate import collate_representations  # noqa: F401
