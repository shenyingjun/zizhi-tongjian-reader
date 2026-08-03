import unittest

from p7_failure_attribution import _effect, _stability


class Round7FailureAttributionTest(unittest.TestCase):
    def test_classifies_change_effects(self):
        self.assertEqual(
            "regression",
            _effect({"action": "removed", "reference_status": "true_positive"}),
        )
        self.assertEqual(
            "removed_false_positive",
            _effect({"action": "removed", "reference_status": "false_positive"}),
        )

    def test_groups_exact_changes_by_seed(self):
        row = {
            "id": "juan-027-jie-0001",
            "action": "removed",
            "para_id": 1,
            "start": 2,
            "end": 3,
            "surface": "甲",
            "reference_status": "true_positive",
        }
        result = _stability({1: [row], 2: [row], 3: []})
        self.assertEqual(1, result["counts"]["two_seeds"])
        self.assertEqual(1, result["harmful_counts"]["two_seeds"])


if __name__ == "__main__":
    unittest.main()
