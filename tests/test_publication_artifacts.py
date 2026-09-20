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


# --------------------------------------------------------------------------
# efficiency benchmark (efficiency_v1)
# --------------------------------------------------------------------------

BENCH = REPO / "benchmarks/efficiency_v1"


def test_efficiency_artifact_hashes_are_correct():
    manifest = BENCH / "EFFICIENCY_ARTIFACT_HASHES.sha256"
    assert manifest.is_file()
    n = 0
    for line in manifest.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        path = BENCH / rel.strip()
        assert path.is_file(), f"missing efficiency artifact: {rel}"
        assert sha256_of(path) == digest, f"efficiency artifact modified: {rel}"
        n += 1
    assert n >= 13, f"expected the full efficiency artifact set, got {n}"


def test_efficiency_benchmark_never_touched_test_or_q8():
    plan = json.loads((BENCH / "EFFICIENCY_BENCHMARK_PLAN.json").read_text())
    assert plan["inputs"]["test_access"] is False
    assert plan["inputs"]["rule"].startswith("first VALIDATION window")
    for meta in plan["inputs"]["per_dataset"].values():
        assert meta["split"] == "validation", "a non-validation input was pinned"
    assert "no Q8 execution" in plan["prohibitions_observed"]
    assert "no TEST access" in plan["prohibitions_observed"]
    assert plan["size_convention"]["q8"].startswith("NOT measured")

    # The raw latency samples must cover only the four datasets, never a TEST split.
    with (BENCH / "latency_raw.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 32000, f"expected 32000 timed samples, got {len(rows)}"
    assert {r["dataset"] for r in rows} == set(DATASETS)
    assert {r["family"] for r in rows} == {"Full-S1", "K1"}
    assert {r["scope"] for r in rows} == {"model_only", "end_to_end"}
    assert {r["device"] for r in rows} == {"cpu", "cuda"}


def test_efficiency_pair_is_the_deterministic_cell():
    """The benchmarked pair must be GF1/seed42 with the frozen table's hashes."""
    plan = json.loads((BENCH / "EFFICIENCY_BENCHMARK_PLAN.json").read_text())
    sel = plan["model_pair_selection"]
    assert sel["rule"] == "lowest fold, then lowest seed"
    assert sel["fold"] == 1 and sel["seed"] == 42

    with (PREREG / "PUBLICATION_FROZEN_MODEL_TABLE.csv").open() as fh:
        row = next(r for r in csv.DictReader(fh) if r["fold"] == "1" and r["seed"] == "42")
    assert sel["checkpoints"]["Full-S1"]["sha256"] == row["full_s1_sha256"]
    assert sel["checkpoints"]["K1"]["sha256"] == row["k1_sha256"]


def test_efficiency_results_are_internally_consistent():
    ps = json.loads((BENCH / "parameter_size_results.json").read_text())
    fl = json.loads((BENCH / "flops_results.json").read_text())
    ag = json.loads((BENCH / "aggregates.json").read_text())

    # K1 encoder must still be the frozen architectural identity.
    assert ps["per_model"]["K1"]["encoder_only"]["total_params"] == 1_375_953
    enc = ps["comparison"]["encoder_only"]
    assert abs(enc["k1_reduction_fraction"] - (1 - enc["k1"] / enc["full_s1"])) < 1e-12

    # FLOPs must be the explicit sum of the two terms, never the profiler alone.
    for fam in ("Full-S1", "K1"):
        for ds in DATASETS:
            d = fl["per_model"][fam]["per_dataset"][ds]
            assert abs(d["total_flops"] - (d["dense_flops"] + d["scan_flops"])) < 1.0
            assert d["scan_flops"] > 0, "scan term must be counted"
    # Direction handling: Full-S1 bidirectional (8), K1 forward-only (4).
    assert fl["per_model"]["Full-S1"]["per_dataset"]["JNU"]["directions_summed_over_layers"] == 8
    assert fl["per_model"]["K1"]["per_dataset"]["JNU"]["directions_summed_over_layers"] == 4
    # CWRU must use the publication grid (9 bands), not the historical 33.
    assert fl["per_model"]["K1"]["per_dataset"]["CWRU"]["n_bands"] == 9

    # Headline latency = equal-domain mean of the four per-dataset medians.
    for key, h in ag["latency_headline"].items():
        for fam, field in (("Full-S1", "full_s1_macro4_median_ms"), ("K1", "k1_macro4_median_ms")):
            want = sum(h["per_dataset_median_ms"][ds][fam] for ds in DATASETS) / 4
            assert abs(h[field] - want) < 1e-9, f"{key}/{fam} is not the equal-domain mean"
        assert h["speedup_full_over_k1"] > 1.0


def test_historical_benchmark_source_was_not_edited():
    assert sha256_of(REPO / "src/methodology_v2/compression/benchmark.py") == \
        "641f437b670b03d5bffb3ed11bfa2c8a758266392beec1988b90b8f4cecd211b"


def test_efficiency_report_agrees_with_result_files():
    doc = (BENCH / "EFFICIENCY_FINAL_REPORT.md").read_text()
    readme = (REPO / "README.md").read_text()
    ps = json.loads((BENCH / "parameter_size_results.json").read_text())
    ag = json.loads((BENCH / "aggregates.json").read_text())

    assert f"{ps['comparison']['encoder_only']['full_s1']:,}" in doc
    assert "1,375,953" in doc and "1,375,953" in readme
    for key in ("cpu|model_only", "cuda|model_only"):
        med = ag["latency_headline"][key]["full_s1_macro4_median_ms"]
        assert f"{med:.3f}" in doc, f"{key} headline median missing from the report"
        assert f"{med:.3f}" in readme, f"{key} headline median missing from the README"
    # The reference-scan caveat must be carried, not dropped.
    assert "reference selective scan" in doc and "reference selective scan" in readme


# --------------------------------------------------------------------------
# Q8 secondary extension (q8_v1)
# --------------------------------------------------------------------------

Q8 = REPO / "benchmarks/q8_v1"


def test_q8_artifact_hashes_are_correct():
    manifest = Q8 / "Q8_ARTIFACT_HASHES.sha256"
    assert manifest.is_file()
    n = 0
    for line in manifest.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        path = Q8 / rel.strip()
        assert path.is_file(), f"missing Q8 artifact: {rel}"
        assert sha256_of(path) == digest, f"Q8 artifact modified: {rel}"
        n += 1
    assert n >= 30, f"expected the full Q8 artifact set, got {n}"


def test_q8_is_declared_a_secondary_post_primary_extension():
    plan = json.loads((Q8 / "Q8_EVALUATION_PLAN.json").read_text())
    sd = plan["stage_declaration"]
    assert sd["primary_study_status"] == "PUBLICATION_FINAL_SEALED_TEST_COMPLETE"
    assert sd["q8_evaluated_when_this_plan_was_frozen"] is False
    assert sd["q8_result_observed_when_this_plan_was_frozen"] is False
    assert sd["k1_checkpoints_already_fixed"] is True
    assert sd["test_membership_unchanged"] is True
    assert sd["q8_specific_tuning_permitted"] is False
    assert sd["retroactive_inclusion_in_primary_claim_permitted"] is False

    barrier = json.loads((Q8 / "Q8_EVALUATION_BARRIER.json").read_text())
    assert barrier["status"] == "FROZEN_BEFORE_Q8_EVALUATION"
    for flag in ("q8_evaluated_before_freeze", "q8_result_observed_before_freeze",
                 "test_membership_changed", "k1_inference_rerun",
                 "full_s1_inference_rerun", "q8_tuned_on_test"):
        assert barrier[flag] is False, f"barrier declaration not sealed: {flag}"
    # the barrier must still pin the plan it was frozen against
    assert barrier["evaluation_plan_sha256"] == sha256_of(Q8 / "Q8_EVALUATION_PLAN.json")
    assert barrier["model_table_sha256"] == sha256_of(Q8 / "Q8_FROZEN_MODEL_TABLE.csv")


def test_q8_used_the_historical_margin_and_procedure():
    """The Q8 margin must come from frozen records, not from the driver's opinion."""
    qspec = json.loads((REPO / "configs/lightweight_k1/quantization_spec.yaml").read_text())
    sspec = json.loads((REPO / "configs/lightweight_k1/statistics_spec.yaml").read_text())
    assert qspec["ni_margin"] == 0.01
    assert "NI 0.01" in sspec["confirmatory_family_holm_m3"]["H3"]
    assert "Q8(K1) vs K1" in sspec["confirmatory_family_holm_m3"]["H3"]
    assert sspec["paired_unit"] == "fold x seed (9 cells)"
    src = (REPO / "src/methodology_v2/compression/protocol.py").read_text()
    assert "NI_MARGIN_PTQ = 0.01" in src

    agg = json.loads((Q8 / "results/q8_aggregate_summary.json").read_text())
    ni = agg["non_inferiority"]
    assert ni["margin"] == 0.01
    assert ni["defined_by_historical_protocol"] is True
    assert ni["holm_family_applied"] is False, "Holm must not be reconstructed"
    assert ni["primary_margin_does_not_apply"] == -0.02
    res = ni["result"]
    assert res["kind"] == "ni"
    assert "superiority" not in res, "no superiority test is defined for Q8"
    assert res["non_inferiority"]["h0"] == "mean(delta) <= -0.01"
    assert res["non_inferiority"]["passes"] is True


def test_q8_did_not_rerun_or_alter_the_primary_study():
    """Q8 must reuse frozen K1 results and leave every primary artifact intact."""
    plan = json.loads((Q8 / "Q8_EVALUATION_PLAN.json").read_text())
    for rel, want in plan["evaluation"]["k1_reference_artifacts"].items():
        assert sha256_of(FINAL_TEST / rel) == want, f"frozen K1 artifact changed: {rel}"
    for key, rec in plan["evaluation"]["test_membership"]["global_fold_manifests"].items():
        assert sha256_of(REPO / rec["path"]) == rec["sha256"], f"TEST membership changed: {key}"
    assert plan["evaluation"]["k1_reference"].startswith("the ALREADY-FROZEN")

    # every pinned source, including the quantization implementation, unmodified
    for rel, want in plan["source_hashes"].items():
        assert sha256_of(REPO / rel) == want, f"pinned source changed: {rel}"

    # Q8 must be built from selected checkpoints only
    with (Q8 / "Q8_FROZEN_MODEL_TABLE.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 9
    assert {(int(r["fold"]), int(r["seed"])) for r in rows} == {(f, s) for f in FOLDS for s in SEEDS}
    for r in rows:
        assert r["k1_checkpoint"].endswith("best.pt"), "last.pt is forbidden"
    # and they must be the same checkpoints the primary study sealed
    with (PREREG / "PUBLICATION_FROZEN_MODEL_TABLE.csv").open() as fh:
        prim = {(int(r["fold"]), int(r["seed"])): r["k1_sha256"] for r in csv.DictReader(fh)}
    for r in rows:
        assert prim[(int(r["fold"]), int(r["seed"]))] == r["k1_sha256"]


def test_q8_calibration_was_none_and_test_was_never_used_for_tuning():
    conv = json.loads((Q8 / "results/q8_conversion_manifest.json").read_text())
    assert conv["calibration"] == "none"
    assert len(conv["cells"]) == 9
    for c in conv["cells"]:
        assert c["calibration"] == "none"
        assert c["n_scale_tensors"] > 0
        # the denylist must never be quantized
        for name in c["int8_modules"]:
            assert "dt_proj" not in name and "conv1d" not in name and not name.endswith("norm")
        assert any("dt_proj" in n for n in c["fp32_modules"])
        assert any("conv1d" in n for n in c["fp32_modules"])
    agree = json.loads((Q8 / "results/q8_sim_vs_cpu_dynamic_agreement.json").read_text())
    assert "VALIDATION only" in agree["scope"]


def test_q8_makes_no_int8_gpu_claim():
    probe = json.loads((Q8 / "results/q8_gpu_probe.json").read_text())
    assert probe["true_int8_cuda"] is False
    report = (Q8 / "Q8_FINAL_REPORT.md").read_text()
    readme = (REPO / "README.md").read_text()
    for doc, label in ((report, "Q8 report"), (readme, "README")):
        assert "no true INT8" in doc.lower() or "no true int8" in doc.lower(), label
    # no GPU latency number may be presented for Q8
    assert "not reported" in report
    mem = json.loads((Q8 / "results/q8_memory_results.json").read_text())
    assert mem["headline_eligible"] is False


def test_q8_report_agrees_with_result_files():
    agg = json.loads((Q8 / "results/q8_aggregate_summary.json").read_text())
    sz = json.loads((Q8 / "results/q8_size_results.json").read_text())
    report = (Q8 / "Q8_FINAL_REPORT.md").read_text()
    readme = (REPO / "README.md").read_text()

    assert f"{agg['macro_4_macro_f1']['q8']['mean']:.6f}" in report
    assert "0.001953125" in report and "0.001953125" in readme
    assert f"{sz['n_params_total_true']:,}" in report
    # the parameter count must be stated as unchanged
    assert "does not change the parameter count" in report
    assert sz["n_params_int8"] + sz["n_params_fp32"] == sz["n_params_total_true"]
    # nine matched cells, delta consistent
    with (Q8 / "results/q8_matched_cells.csv").open() as fh:
        cells = list(csv.DictReader(fh))
    assert len(cells) == 9
    for c in cells:
        d = float(c["q8_macro_4_f1"]) - float(c["k1_macro_4_f1"])
        assert abs(d - float(c["delta_f1"])) < 1e-12


def test_documentation_separates_primary_study_from_q8_extension():
    readme = (REPO / "README.md").read_text()
    results = (REPO / "docs/RESULTS.md").read_text()
    protocol = (REPO / "docs/EXPERIMENT_PROTOCOL.md").read_text()
    for doc, label in ((readme, "README"), (results, "docs/RESULTS.md"), (protocol, "docs/EXPERIMENT_PROTOCOL.md")):
        low = doc.lower()
        assert "secondary" in low, f"{label} does not mark Q8 as secondary"
        assert "primary" in low, f"{label} does not name the primary study"
    # the primary margin must never be attached to Q8
    assert "-0.01" in results or "\u22120.01" in results
    assert "does not apply to Q8" in results or "does not apply" in results
