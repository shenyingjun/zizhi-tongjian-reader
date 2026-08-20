from __future__ import annotations

import unittest

import numpy as np

from production_precision_revision14_two_stage import (
    _stage1_training_indices,
    _three_stratum_weights,
)


class Revision16InventoryTest(unittest.TestCase):
    def test_training_inventory_excludes_synthetic_boundaries(self):
        classes = [
            "exact_reference",
            "boundary_alternative",
            "boundary_alternative",
            "semantic_negative",
            "easy_negative",
            "reconciled_nonoverlap",
        ]
        real_labels = np.asarray([1, 1, -1, 0, -1, -1], dtype=np.int8)

        selected = _stage1_training_indices(
            np.arange(len(classes), dtype=np.int64),
            classes,
            real_labels,
            real_only=True,
            structural_negatives=True,
        )

        self.assertEqual(selected.tolist(), [0, 1, 3, 4, 5])


class ThreeStratumWeightsTest(unittest.TestCase):
    def test_weights_assign_frozen_stratum_mass(self):
        labels = np.asarray([1, 1, 0, 0, 0, 0], dtype=np.float32)
        classes = [
            "exact_reference",
            "boundary_alternative",
            "semantic_negative",
            "semantic_negative",
            "easy_negative",
            "reconciled_nonoverlap",
        ]
        indices = np.arange(len(labels), dtype=np.int64)

        weights, inventory = _three_stratum_weights(
            labels, classes, indices
        )

        self.assertAlmostEqual(float(weights[:2].sum()), 0.5)
        self.assertAlmostEqual(float(weights[2:4].sum()), 0.25)
        self.assertAlmostEqual(float(weights[4:].sum()), 0.25)
        self.assertEqual(inventory["positive_rows"], 2)
        self.assertEqual(inventory["semantic_negative_rows"], 2)
        self.assertEqual(inventory["structural_negative_rows"], 2)

    def test_missing_stratum_fails_closed(self):
        labels = np.asarray([1, 0], dtype=np.float32)
        classes = ["exact_reference", "easy_negative"]

        with self.assertRaisesRegex(ValueError, "all three strata"):
            _three_stratum_weights(
                labels,
                classes,
                np.arange(len(labels), dtype=np.int64),
            )

    def test_ambiguous_stratum_ownership_fails_closed(self):
        labels = np.asarray([1, 0, 0, 0], dtype=np.float32)
        classes = [
            "exact_reference",
            "semantic_negative",
            "easy_negative",
            "other_negative",
        ]

        with self.assertRaisesRegex(ValueError, "stratum ownership"):
            _three_stratum_weights(
                labels,
                classes,
                np.arange(len(labels), dtype=np.int64),
            )


if __name__ == "__main__":
    unittest.main()
