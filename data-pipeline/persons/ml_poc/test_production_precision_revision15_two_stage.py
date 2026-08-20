from __future__ import annotations

import unittest

import numpy as np

from production_precision_revision14_two_stage import (
    _class_balanced_weights,
)
import production_precision_revision15_two_stage as ts


class ClassBalancedWeightsTest(unittest.TestCase):
    def test_weights_balance_training_class_mass(self):
        labels = np.asarray([1, 1, 1, 0, 0], dtype=np.float32)
        indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
        weights, inventory = _class_balanced_weights(labels, indices)

        self.assertEqual(inventory["positive_rows"], 3)
        self.assertEqual(inventory["negative_rows"], 1)
        self.assertAlmostEqual(float(weights[indices][labels[indices] == 1].sum()), 2.0)
        self.assertAlmostEqual(float(weights[indices][labels[indices] == 0].sum()), 2.0)
        self.assertEqual(float(weights[4]), 1.0)

    def test_missing_fold_class_fails_closed(self):
        labels = np.asarray([1, 1], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "both classes"):
            _class_balanced_weights(labels, np.asarray([0, 1], dtype=np.int64))


class GreedyResolutionTest(unittest.TestCase):
    def test_bridge_does_not_suppress_disjoint_exact_spans(self):
        rows = [
            {
                "id": "j",
                "juan": 1,
                "jie_index": 1,
                "para_id": 1,
                "start": 0,
                "end": 2,
                "surface": "ab",
                "class": "exact_reference",
                "existence_label": 1,
            },
            {
                "id": "j",
                "juan": 1,
                "jie_index": 1,
                "para_id": 1,
                "start": 1,
                "end": 4,
                "surface": "bcd",
                "class": "boundary_alternative",
                "existence_label": 1,
            },
            {
                "id": "j",
                "juan": 1,
                "jie_index": 1,
                "para_id": 1,
                "start": 3,
                "end": 5,
                "surface": "de",
                "class": "exact_reference",
                "existence_label": 1,
            },
        ]
        table = ts.end_to_end_metrics(
            rows,
            np.asarray([0.99, 0.99, 0.99], dtype=np.float32),
            np.asarray([3.0, 1.0, 2.0], dtype=np.float32),
            [row["class"] for row in rows],
            np.asarray([0, 0, 0], dtype=np.int64),
            np.asarray([1, 0, 1], dtype=np.int8),
            {("j", 1, 0, 2), ("j", 1, 3, 5)},
            {("j", 1, 1, 4)},
            greedy_resolution=True,
        )
        row = next(item for item in table if item["threshold"] == 0.50)

        self.assertEqual(row["e2e_selected_count"], 2)
        self.assertEqual(row["e2e_exact_recall"], 1.0)
        self.assertEqual(row["e2e_boundary_component_accuracy"], 1.0)

    def test_higher_ranked_bridge_blocks_both_exact_spans(self):
        candidates = [
            {
                "id": "j", "juan": 1, "jie_index": 1, "para_id": 1,
                "start": 0, "end": 2,
            },
            {
                "id": "j", "juan": 1, "jie_index": 1, "para_id": 1,
                "start": 1, "end": 4,
            },
            {
                "id": "j", "juan": 1, "jie_index": 1, "para_id": 1,
                "start": 3, "end": 5,
            },
        ]
        selected = ts.select_greedy_nonoverlap(
            candidates,
            np.asarray([2.0, 3.0, 1.0], dtype=np.float32),
            [0, 1, 2],
        )
        self.assertEqual(selected, [1])

    def test_component_accuracy_only_requires_admitted_exact_spans(self):
        rows = [
            {
                "id": "j",
                "juan": 1,
                "jie_index": 1,
                "para_id": 1,
                "start": 0,
                "end": 2,
                "surface": "ab",
                "class": "exact_reference",
                "existence_label": 1,
            },
            {
                "id": "j",
                "juan": 1,
                "jie_index": 1,
                "para_id": 1,
                "start": 1,
                "end": 3,
                "surface": "bc",
                "class": "boundary_alternative",
                "existence_label": 1,
            },
        ]
        table = ts.end_to_end_metrics(
            rows,
            np.asarray([0.1, 0.1], dtype=np.float32),
            np.asarray([2.0, 1.0], dtype=np.float32),
            [row["class"] for row in rows],
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([1, 0], dtype=np.int8),
            {("j", 1, 0, 2)},
            {("j", 1, 1, 3)},
            greedy_resolution=True,
        )
        row = next(item for item in table if item["threshold"] == 0.50)

        self.assertEqual(row["e2e_boundary_component_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
