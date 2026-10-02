"""Lightweight safety/aggregation tests. Scientific GPU smoke is still required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import run
import summarize


class InputGuards(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"window_id": ["a", "b", "c"], "split": ["train", "validation", "test"]})

    def test_test_request_is_rejected(self):
        with self.assertRaises(ValueError):
            run.checked_ids(self.frame, "test")

    def test_duplicate_and_unknown_ids_are_rejected(self):
        duplicate = pd.concat([self.frame, self.frame.iloc[:1]])
        with self.assertRaises(ValueError):
            run.checked_ids(duplicate, "validation")
        unknown = self.frame.copy()
        unknown.loc[0, "split"] = "unknown"
        with self.assertRaises(ValueError):
            run.checked_ids(unknown, "validation")

    def test_train_stream_rejects_validation_and_test(self):
        for invalid in ("b", "c", "missing"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                run.assert_training_stream([[('JNU', 'healthy', invalid)] * 64], ["a"])
        run.assert_training_stream([[('JNU', 'healthy', 'a')] * 64], ["a"])

    def test_no_empty_or_wrong_batch(self):
        with self.assertRaises(ValueError):
            run.checked_ids(self.frame[self.frame.split != "validation"], "validation")
        with self.assertRaises(ValueError):
            run.assert_training_stream([[('JNU', 'healthy', 'a')]], ["a"])

    def test_wrong_executor_refused_before_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = root / "analysis/pcste_v2_lightweight_v1/scripts/02_k1_production.py"
            executor.parent.mkdir(parents=True)
            executor.write_text("raise AssertionError('must not execute')")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                run.load_base(root, {"original_executor_sha256": "0" * 64})


class AggregationGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.reference = json.loads((run.HERE / "reference_validation.json").read_text())
        for cell in self.reference["cells"]:
            f, s = cell["fold"], cell["seed"]
            p = self.root / f"screen_gf{f}_s{s}"
            p.mkdir()
            run.write_json(p / "provenance.json", {"fold": f, "seed": s,
                "plan_sha256": run.sha(run.HERE / "plan.json"),
                "runner_sha256": run.sha(run.HERE / "run.py"), "new_test_inference": False})
            ds = {d: 0.7 for d in ("CWRU", "JNU", "HIT", "MAFAULDA")}
            run.write_json(p / "screen.json", {"status": "COMPLETE", "split": "validation",
                "fold": f, "seed": s, "scores": {
                    a: {"macro_domain_f1": v, "per_dataset_macro_f1": {d: v for d in ds}}
                    for a, v in (("full", .9), ("2x2", .7), ("4x1", .8))}})

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_matrix_has_correct_delta_orientation(self):
        rows, summary = summarize.collect(self.root, "screen")
        self.assertEqual(len(rows), 9)
        self.assertAlmostEqual(summary["paired_delta"]["mean"], -.1)
        self.assertEqual(summary["paired_delta"]["negative"], 9)

    def test_missing_cell_cannot_be_silently_dropped(self):
        (self.root / "screen_gf3_s2026/screen.json").unlink()
        with self.assertRaises(FileNotFoundError):
            summarize.collect(self.root, "screen")

    def test_mixed_code_or_test_usage_rejected(self):
        p = self.root / "screen_gf1_s42/provenance.json"
        original = json.loads(p.read_text())
        for key, value in (("runner_sha256", "changed"), ("new_test_inference", True)):
            x = dict(original, **{key: value})
            run.write_json(p, x)
            with self.assertRaises(ValueError):
                summarize.collect(self.root, "screen")
            run.write_json(p, original)


if __name__ == "__main__":
    unittest.main()
