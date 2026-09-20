"""Publication-safe checks on the frozen artifacts of `lightweight_pcste_final_test_v1`.

These tests never read TEST data, never load a checkpoint, and never run
inference. They verify byte identity of frozen artifacts, structural validity
of the frozen records, consistency between the reader-facing documentation and
the authoritative result files, and that documented paths exist.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

FOLDS = (1, 2, 3)
SEEDS = (42, 1337, 2026)
DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")

FINAL_TEST = REPO / "results/final_test/publication_final_test_v1"
PREREG = REPO / "configs/lightweight_k1/final_test_v1"

# Authoritative hashes, as recorded in the canonical scientific repository.
AUTHORITATIVE = {
    "configs/lightweight_k1/K1_9_CELL_FREEZE_MANIFEST.md":
        "a9e5fb49a1f38a7d6e2649ff8d155cf619cd55572bf7238b06a1fcb120d64a8b",
    "configs/lightweight_k1/final_test_v1/PUBLICATION_FINAL_EVALUATION_PLAN.json":
        "ce22960df67836b0cb49f476b3fe4c2ed29e5cfcb554f8948e7af3f9fc11a55a",
    "configs/lightweight_k1/final_test_v1/PUBLICATION_FINAL_EVALUATION_PLAN.md":
        "79521eec872ee0c55ceb74972b65e19c6a2f2a96e478a5640650f5a8039d018f",
    "configs/lightweight_k1/final_test_v1/PUBLICATION_FROZEN_MODEL_TABLE.csv":
        "34c1a6b3ed2c9f545cb6b368c46e22a2bdf3b03a1a5134f3f73180b484ab5cd9",
    "configs/lightweight_k1/final_test_v1/PUBLICATION_FINAL_EVALUATION_BARRIER.json":
        "fa8410c03bcad4d3eea31282ac8ee7828c731a35a025aab872354aa8ffdbe71d",
    "configs/lightweight_k1/final_test_v1/publication_test_driver.py":
        "8f2de50a52ce83fefa8dcd1516ec3c348d9f5c1a6b561b38070fe9f94d149f3e",
    "configs/final_s1/FINAL_SELECTED_CHECKPOINTS.csv":
        "321bdedddec7613bcfc40f17175f427882f28e45c50a7241b1fd2f66365aa2a3",
    "configs/final_s1/FINAL_STATISTICAL_PLAN.yaml":
        "7d4ba32d9f1df2283c98c012a7268dd7d3664fa52017a84adf9f0b2e9a373629",
    "src/methodology_v2/compression/stats.py":
        "efb3d20c777e4ab0eda15491147950c77c816610d1f71ca0072f8dfa36e3cfcb",
    "scripts/evaluate/evaluate_test.py":
        "dd3aa58ce5067298504b9e064cdb51e5f29d0f655b1ca3f4fa6af77600d7f517",
    "scripts/train_k1/02_k1_production.py":
        "1f5d86a1c6b7f2b637ecdd6d61c279a0b86e9edb54bdbe0490d2a4b20fd62308",
    "protocols/splits/global_v2_final_s0s1_v1/global_v2_hashes.json":
        "4b6884b10d30b9b169502a9561e7eed8a0de24dcd29985d5bbca6f718e4165d0",
    "protocols/splits/global_v2_final_s0s1_v1/global_v2_fold_1.csv":
        "456a69474a3763647b7e0ad13f5538e8f630d268347671ad617134b08e830d0a",
    "protocols/splits/global_v2_final_s0s1_v1/global_v2_fold_2.csv":
        "74029f81a4b5617b13218812aa26568b0f3a268f8f6b450b5ab0293b17f71f93",
    "protocols/splits/global_v2_final_s0s1_v1/global_v2_fold_3.csv":
        "fe622f3bc1146998d00fe070ea045f0790145b55a028effea52de00f94000f0e",
    "protocols/datasets/cwru_native12_load_v1/FREEZE_BUNDLE_INDEX.txt":
        "ddf4d32573c016b08da65b4f878ee0511a9bae4b15444626e1d932d24ebf39e4",
    "results/final_test/publication_final_test_v1/FINAL_SEALED_TEST_REPORT.md":
        "11f8405d7ae5783b283c39a0cd5a4dd16685187125a2dc38b8d738ea035e6a8d",
    "results/final_test/publication_final_test_v1/PUBLICATION_FINAL_RESULTS.json":
        "b6f14ff707a952ae860812a2bfa2f099fc8d3c39ab579c8d2605057b5699a527",
    "results/final_test/publication_final_test_v1/PER_DATASET_SUMMARY.csv":
        "62fb8bfe5603cf2525c2d1500157e5dc4c889539051e1a5b403affb2ff706520",
    "results/final_test/publication_final_test_v1/matched_cells.csv":
        "af62d51c0c299b6e4a17de775015c259b6e575af0e3dc1d2c1e59ad9be74d239",
    "results/final_test/publication_final_test_v1/aggregate_summary.json":
        "fffc6c28a42c25d35c85c1515be48b5ab7da0d0767450bfb31d9dbc903a1cf28",
}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Frozen-file hash verification
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rel,expected", sorted(AUTHORITATIVE.items()))
def test_frozen_artifact_matches_authoritative_hash(rel, expected):
    path = REPO / rel
    assert path.is_file(), f"missing frozen artifact: {rel}"
    assert sha256_of(path) == expected, f"frozen artifact modified: {rel}"


def test_hash_manifest_is_complete_and_correct():
    """FROZEN_ARTIFACT_HASHES.sha256 must cover every frozen artifact and be accurate."""
    manifest = REPO / "FROZEN_ARTIFACT_HASHES.sha256"
    assert manifest.is_file()

    recorded = {}
    for line in manifest.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        recorded[rel.strip()] = digest

    for rel, digest in recorded.items():
        path = REPO / rel
        assert path.is_file(), f"manifest references a missing file: {rel}"
        assert sha256_of(path) == digest, f"manifest hash mismatch: {rel}"

    missing = set(AUTHORITATIVE) - set(recorded)
    assert not missing, f"frozen artifacts absent from the hash manifest: {sorted(missing)}"


# --------------------------------------------------------------------------
# Structural validity of the frozen records
# --------------------------------------------------------------------------

def test_all_frozen_json_artifacts_parse():
    paths = sorted(PREREG.glob("*.json")) + sorted(FINAL_TEST.glob("*.json")) \
        + sorted((FINAL_TEST / "per_model_reports").glob("*.json"))
    assert paths
    for path in paths:
        json.loads(path.read_text())


def test_barrier_pins_the_frozen_plan():
    barrier = json.loads((PREREG / "PUBLICATION_FINAL_EVALUATION_BARRIER.json").read_text())
    assert barrier["protocol_identity"] == "lightweight_pcste_final_test_v1"
    assert barrier["status"] == "FROZEN_BEFORE_TEST"
    assert barrier["matched_cells"] == 9
    assert barrier["checkpoints"] == 18
    assert barrier["model_families"] == ["Full-S1", "K1"]
    assert barrier["excluded_families"] == ["S0", "Q8"]
    # The barrier must record that TEST was untouched at freeze time.
    assert barrier["test_loader_invoked_before_freeze"] is False
    assert barrier["test_inference_before_freeze"] is False
    assert barrier["test_metrics_before_freeze"] is False

    pairs = [
        ("evaluation_plan_sha256", "PUBLICATION_FINAL_EVALUATION_PLAN.json"),
        ("human_plan_sha256", "PUBLICATION_FINAL_EVALUATION_PLAN.md"),
        ("model_table_sha256", "PUBLICATION_FROZEN_MODEL_TABLE.csv"),
        ("publication_driver_sha256", "publication_test_driver.py"),
    ]
    for key, name in pairs:
        assert barrier[key] == sha256_of(PREREG / name), f"barrier no longer pins {name}"

    assert barrier["k1_freeze_manifest_sha256"] == sha256_of(
        REPO / "configs/lightweight_k1/K1_9_CELL_FREEZE_MANIFEST.md")
    assert barrier["full_s1_registry_sha256"] == sha256_of(
        REPO / "configs/final_s1/FINAL_SELECTED_CHECKPOINTS.csv")
    assert barrier["statistical_plan_sha256"] == sha256_of(
        REPO / "configs/final_s1/FINAL_STATISTICAL_PLAN.yaml")

    splits = REPO / "protocols/splits/global_v2_final_s0s1_v1"
    for fold in FOLDS:
        assert barrier["global_fold_manifest_sha256"][f"GF{fold}"] == \
            sha256_of(splits / f"global_v2_fold_{fold}.csv")
    assert barrier["cwru_freeze_digest_sha256"] == sha256_of(
        REPO / "protocols/datasets/cwru_native12_load_v1/FREEZE_BUNDLE_INDEX.txt")


def test_frozen_model_table_has_eighteen_checkpoints_over_nine_cells():
    with (PREREG / "PUBLICATION_FROZEN_MODEL_TABLE.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 9
    cells = {(int(r["fold"]), int(r["seed"])) for r in rows}
    assert cells == {(f, s) for f in FOLDS for s in SEEDS}

    hashes = set()
    for row in rows:
        for key in ("full_s1_sha256", "k1_sha256"):
            digest = row[key]
            assert re.fullmatch(r"[0-9a-f]{64}", digest), f"bad digest in {key}"
            hashes.add(digest)
    assert len(hashes) == 18, "the 18 frozen checkpoints must be distinct"


def test_k1_freeze_manifest_declares_nine_complete_cells():
    text = (REPO / "configs/lightweight_k1/K1_9_CELL_FREEZE_MANIFEST.md").read_text()
    assert "K1_9_OF_9_FROZEN_READY_FOR_TEST" in text
    assert "GF1 `3/3`, GF2 `3/3`, GF3 `3/3`, total `9/9`" in text
    assert "`half_4x1`" in text
    assert "1,375,953" in text
    assert "TEST used: `false` for every cell" in text


def test_k1_validation_cells_are_all_present_and_complete():
    """All nine validation-stage cells must be recorded, with TEST unused."""
    root = REPO / "results/lightweight_k1"
    for fold in FOLDS:
        for seed in SEEDS:
            cell = root / f"pcstev2_lightweight_v1_k1_gf{fold}_s{seed}"
            assert cell.is_dir(), f"missing K1 cell: {cell.name}"

            state = json.loads((cell / "state.json").read_text())
            completion = json.loads((cell / "completion.json").read_text())
            blob = json.dumps({"state": state, "completion": completion})
            assert "COMPLETE" in blob, f"{cell.name} is not marked complete"
            assert '"test_used": true' not in blob, f"{cell.name} reports TEST usage"

            lines = [ln for ln in (cell / "epoch_metrics.jsonl").read_text().splitlines() if ln.strip()]
            assert len(lines) == 50, f"{cell.name}: expected 50 epoch records, got {len(lines)}"
            for line in lines:
                json.loads(line)


def test_sealed_test_status_and_report():
    assert (FINAL_TEST / "STATUS").read_text().strip() == "COMPLETE"
    report = (FINAL_TEST / "FINAL_SEALED_TEST_REPORT.md").read_text()
    assert "PUBLICATION_FINAL_SEALED_TEST_COMPLETE" in report
    assert "lightweight_pcste_final_test_v1" in report


def test_matched_cells_cover_the_nine_frozen_cells():
    with (FINAL_TEST / "matched_cells.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 9
    assert {(int(r["fold"]), int(r["seed"])) for r in rows} == {(f, s) for f in FOLDS for s in SEEDS}
    for row in rows:
        delta = float(row["k1_macro_4_f1"]) - float(row["full_s1_macro_4_f1"])
        assert abs(delta - float(row["delta_f1"])) < 1e-9, "delta_f1 is not the paired difference"


def test_per_model_reports_cover_every_cell_and_family():
    reports = sorted((FINAL_TEST / "per_model_reports").glob("*_report.json"))
    assert len(reports) == 18
    seen = set()
    for path in reports:
        data = json.loads(path.read_text())
        seen.add((data["fold"], data["seed"], data["family"]))
        assert re.fullmatch(r"[0-9a-f]{64}", data["checkpoint_sha256"])
        assert set(data["per_dataset_reports"]) == set(DATASETS)
        for name, block in data["per_dataset_reports"].items():
            n_classes = len(block["classes"])
            matrix = block["confusion_matrix"]
            assert len(matrix) == n_classes, f"{path.name}/{name}: confusion matrix is not square"
            assert all(len(row) == n_classes for row in matrix)
            for key in ("per_class_f1", "per_class_precision", "per_class_recall", "support"):
                assert len(block[key]) == n_classes, f"{path.name}/{name}: {key} length"
    assert seen == {(f, s, fam) for f in FOLDS for s in SEEDS for fam in ("K1", "Full-S1")}


# --------------------------------------------------------------------------
# Documentation consistency
# --------------------------------------------------------------------------

def test_results_doc_agrees_with_the_frozen_result_files():
    """docs/RESULTS.md must not drift from the authoritative artifacts."""
    doc = (REPO / "docs/RESULTS.md").read_text()
    readme = (REPO / "README.md").read_text()
    agg = json.loads((FINAL_TEST / "aggregate_summary.json").read_text())
    flat = json.dumps(agg)

    # Values quoted in both documents, verified against the frozen JSON.
    for value in ("0.001953125", "0.0078125"):
        assert value in flat, f"{value} is not present in aggregate_summary.json"
        assert value in doc and value in readme

    for value in ("0.934644", "0.955334", "0.020691", "0.991793", "0.996398", "0.004606"):
        assert value in doc and value in readme, f"{value} missing from the documentation"

    report = (FINAL_TEST / "FINAL_SEALED_TEST_REPORT.md").read_text()
    for value in ("0.934644", "0.955334", "0.020691", "0.001953125", "0.0078125"):
        assert value in report, f"{value} is not in the frozen sealed report"

    # The non-inferiority margin must be quoted, never silently altered.
    assert "-0.02" in doc and "-0.02" in readme
    assert "SATISFIED" in doc and "SATISFIED" in readme

    # No efficiency claim may be made before benchmarking.
    for banned in ("FLOPs reduction", "speedup", "x faster"):
        assert banned not in readme, f"unbenchmarked efficiency claim in README: {banned}"


def test_documented_paths_exist():
    expected = [
        "README.md",
        "FROZEN_ARTIFACT_HASHES.sha256",
        "requirements.txt",
        "requirements_frozen.txt",
        "docs/ARTIFACTS.md",
        "docs/DATASETS.md",
        "docs/EXPERIMENT_PROTOCOL.md",
        "docs/MODEL.md",
        "docs/PUBLICATION_INVENTORY.md",
        "docs/REPRODUCIBILITY.md",
        "docs/RESULTS.md",
        "scripts/train_ssl/run_ssl.py",
        "scripts/train_s1/run.py",
        "scripts/train_k1/02_k1_production.py",
        "scripts/evaluate/evaluate_test.py",
        "scripts/evaluate/publication_test_driver.py",
        "configs/final_s1/FINAL_SELECTED_CHECKPOINTS.csv",
        "configs/lightweight_k1/K1_9_CELL_FREEZE_MANIFEST.md",
        "configs/lightweight_k1/K1_REGISTERED_CELLS.csv",
        "protocols/splits/global_v2_final_s0s1_v1/normalizers/registry_fold_1.csv",
        "protocols/datasets/cwru_native12_load_v1/FREEZE_DIGEST.json",
        "results/final_test/publication_final_test_v1/FINAL_SEALED_TEST_REPORT.md",
    ]
    missing = [rel for rel in expected if not (REPO / rel).exists()]
    assert not missing, f"documented paths do not exist: {missing}"


def test_legacy_symlinked_paths_resolve():
    """Frozen records reference legacy relative paths; they must still resolve."""
    for rel in [
        "analysis/pcste_v2_lightweight_v1/K1_9_CELL_FREEZE_MANIFEST.md",
        "analysis/pcste_v2_lightweight_v1/final_test_v1/PUBLICATION_FINAL_EVALUATION_PLAN.json",
        "analysis/pcste_v2_final_s0_s1_v1/FINAL_SELECTED_CHECKPOINTS.csv",
        "pcste_v2/protocol/global_v2_final_s0s1_v1/global_v2_fold_1.csv",
        "pcste_v2/protocol/cwru_native12_load_v1/FREEZE_DIGEST.json",
    ]:
        assert (REPO / rel).is_file(), f"legacy path does not resolve: {rel}"


def test_no_binary_scientific_artifacts_are_tracked():
    """The artifact policy forbids weights, arrays, and raw data in the tree."""
    offenders = []
    for pattern in ("*.pt", "*.pth", "*.ckpt", "*.npz", "*.npy"):
        for path in REPO.rglob(pattern):
            if ".git" in path.parts:
                continue
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, f"binary scientific artifacts present: {offenders}"


def test_publication_driver_resolves_the_repository_root():
    """The driver derives REPO from its own location; the layout must satisfy it."""
    driver = PREREG / "publication_test_driver.py"
    assert driver.resolve().parents[3] == REPO


def test_cwru_freeze_bundle_index_resolves_and_matches():
    """Every text entry of the frozen CWRU bundle index must resolve and match.

    Only the 12 normalizer `.npz` files may be absent: they are excluded binary
    state under `docs/ARTIFACTS.md`, and their hashes remain recorded both here
    and in the normalizer registries.
    """
    index = REPO / "protocols/datasets/cwru_native12_load_v1/FREEZE_BUNDLE_INDEX.txt"
    matched, absent, mismatched = 0, [], []
    for line in index.read_text().splitlines():
        if not line.strip():
            continue
        rel, expected = line.rsplit(None, 1)
        path = REPO / rel.strip()
        if not path.is_file():
            absent.append(rel.strip())
        elif sha256_of(path) == expected:
            matched += 1
        else:
            mismatched.append(rel.strip())

    assert not mismatched, f"frozen bundle entries no longer match: {mismatched}"
    assert all(rel.endswith(".npz") for rel in absent), \
        f"non-binary bundle entries are missing: {[r for r in absent if not r.endswith('.npz')]}"
    assert len(absent) == 12, f"expected exactly the 12 excluded normalizers, got {len(absent)}"
    assert matched == 57


def test_k1_production_executor_matches_the_frozen_manifest():
    """The executor SHA recorded in the K1 freeze manifest must be the one shipped."""
    manifest = (REPO / "configs/lightweight_k1/K1_9_CELL_FREEZE_MANIFEST.md").read_text()
    digest = sha256_of(REPO / "scripts/train_k1/02_k1_production.py")
    assert f"Executor SHA256: `{digest}`" in manifest
    # and it must be reachable at the legacy path the frozen records reference
    legacy = REPO / "analysis/pcste_v2_lightweight_v1/scripts/02_k1_production.py"
    assert legacy.is_file() and sha256_of(legacy) == digest


def test_legacy_executor_paths_resolve():
    """Frozen records reference the historical `scripts/pcste_v2/` layout."""
    for rel, real in [
        ("scripts/pcste_v2/run.py", "scripts/train_s1/run.py"),
        ("scripts/pcste_v2/run_ssl.py", "scripts/train_ssl/run_ssl.py"),
        ("scripts/pcste_v2/evaluate_test.py", "scripts/evaluate/evaluate_test.py"),
        ("scripts/pcste_v2/fit_normalizers.py", "scripts/protocol/fit_normalizers.py"),
    ]:
        assert (REPO / rel).is_file(), f"legacy executor path does not resolve: {rel}"
        assert sha256_of(REPO / rel) == sha256_of(REPO / real)
