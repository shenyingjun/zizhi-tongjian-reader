import unittest

from p10_independent_dev_compare import _compare, select_recipe


class IndependentDevCompareTest(unittest.TestCase):
    def test_selects_f1_then_precision_then_recall(self):
        metrics = {
            "round6": {"f1": 0.8, "precision": 0.9, "recall": 0.7},
            "round7": {"f1": 0.9, "precision": 0.8, "recall": 0.8},
            "round8": {"f1": 0.9, "precision": 0.9, "recall": 0.7},
            "round9": {"f1": 0.9, "precision": 0.9, "recall": 0.8},
        }
        self.assertEqual("round9", select_recipe(metrics))

    def test_exact_tie_prefers_earlier_recipe(self):
        metric = {"f1": 0.9, "precision": 0.9, "recall": 0.9}
        self.assertEqual(
            "round6",
            select_recipe({name: metric for name in (
                "round6", "round7", "round8", "round9"
            )}),
        )

    def test_compares_exact_geometry_sets(self):
        surface = "甲"
        reference = [{
            "para_id": 1, "start": 0, "end": 1, "surface": surface
        }]
        before = [{
            "id": "example",
            "reference_spans": reference,
            "prediction_spans": [],
        }]
        after = [{
            "id": "example",
            "reference_spans": reference,
            "prediction_spans": reference,
        }]
        totals, changes = _compare(before, after)
        self.assertEqual(1, totals["raw_additions"])
        self.assertEqual(1, totals["reference_recoveries"])
        self.assertEqual(1, totals["changed_jies"])
        self.assertEqual(1, len(changes))


if __name__ == "__main__":
    unittest.main()
