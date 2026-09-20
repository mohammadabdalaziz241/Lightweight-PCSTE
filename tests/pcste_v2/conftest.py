"""Collection guard for tests that depend on superseded protocol bundles.

`test_native12.py` is the historical implementation test for
`pcste_v2_cwru_native12_specimen_v1`. That protocol was superseded by the
final `cwru_native12_load_v1` used throughout the publication study, and its
manifest bundle (together with the superseded `global_v2_PB_MAFv2_v1` splits,
~24 MB) is deliberately not imported into this repository — see
`docs/ARTIFACTS.md`. The test is retained verbatim for traceability and is
skipped here rather than left as a hard collection error.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

collect_ignore = []
if not (REPO / "pcste_v2/protocol/cwru_native12_specimen_v1").is_dir():
    collect_ignore.append("test_native12.py")
