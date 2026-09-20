"""Skip guards for Part-6 tests that require historical dissertation bundles.

`test_part6_compression.py` is retained verbatim and most of it exercises the
PC-STE / K1 implementation directly. A minority of its cases read artifacts of
the *original* dissertation-era Part-3/Part-6 study:

  * `methodology_v2/part3_windows/window_manifest_fold_*.csv` (~24 MB of
    superseded window manifests), and
  * `methodology_v2/part6_compression/part6_run_registry.csv` plus the
    historical assignment/ledger records.

Those bundles are superseded by the frozen publication protocol under
`protocols/` and `configs/lightweight_k1/`, and are excluded by the artifact
policy in `docs/ARTIFACTS.md`. In this repository `methodology_v2/part6_compression`
is a symlink to the publication K1 configuration, not the historical directory.

The affected cases are skipped when those bundles are absent, so the remaining
Part-6 coverage still runs. They pass unchanged in the canonical scientific
repository, where the historical bundles are present.
"""
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_PART3_WINDOWS = REPO / "methodology_v2/part3_windows"
_PART6_REGISTRY = REPO / "methodology_v2/part6_compression/part6_run_registry.csv"

# Cases that read the historical Part-3 window manifests or Part-6 run registry.
_NEEDS_HISTORICAL_BUNDLE = {
    "test_registry_deterministic_template_and_final_requires_checkpoints",
    "test_registry_final_mode_with_real_checkpoints",
    "test_assignment_rule_balance_and_determinism",
    "test_assignment_file_roundtrip_and_tamper",
    "test_test_guards_and_manifest_gating",
    "test_part6_protocol_dir_has_no_checkpoints_and_specs_exist",
}


def pytest_collection_modifyitems(config, items):
    if _PART3_WINDOWS.is_dir() and _PART6_REGISTRY.is_file():
        return
    skip = pytest.mark.skip(
        reason="requires the historical Part-3/Part-6 dissertation bundles, "
               "which are excluded from the publication repository"
    )
    for item in items:
        if item.name.split("[")[0] in _NEEDS_HISTORICAL_BUNDLE:
            item.add_marker(skip)
