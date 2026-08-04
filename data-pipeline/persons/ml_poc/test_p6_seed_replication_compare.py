import unittest

from p6_seed_replication_compare import (
    _change_stability,
    _summarize_metrics,
)


def _pair(old_f1, new_f1, changes):
    metrics = {
        "round4": {"exact": {"f1": old_f1}},
        "round6": {"exact": {"f1": new_f1}},
    }
    value = {
        "metrics": metrics,
        "f1_delta": new_f1 - old_f1,
        "surfaced_changes": changes,
    }
    return {"dev": value, "evaluation": value}


class SeedReplicationCompareTest(unittest.TestCase):
    def test_summarizes_mixed_paired_directions(self):
        pairs = {
            1: _pair(0.9, 0.8, []),
            2: _pair(0.8, 0.9, []),
            3: _pair(0.7, 0.8, []),
        }
        summary = _summarize_metrics(pairs)
        delta = summary["evaluation"]["paired_f1_delta"]
        self.assertEqual(2, delta["positive_seeds"])
        self.assertEqual(1, delta["negative_seeds"])
        self.assertAlmostEqual(1 / 30, delta["mean"])

    def test_groups_exact_changes_by_seed_count(self):
        shared = {
            "id": "juan-001-jie-0001",
            "action": "removed",
            "surface": "甲",
            "reference_status": "true_positive",
            "para_id": 1,
            "start": 2,
            "end": 3,
        }
        unique = {**shared, "surface": "乙", "start": 4, "end": 5}
        pairs = {
            1: _pair(0.9, 0.8, [shared, unique]),
            2: _pair(0.9, 0.8, [shared]),
            3: _pair(0.9, 0.8, [shared]),
        }
        stability = _change_stability(pairs, "evaluation")
        self.assertEqual(1, stability["counts"]["all_three_seeds"])
        self.assertEqual(1, stability["counts"]["one_seed"])
        self.assertEqual([1, 2, 3], stability["all_three_seeds"][0]["seeds"])


if __name__ == "__main__":
    unittest.main()
