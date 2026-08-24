from __future__ import annotations

import unittest

import numpy as np

from production_precision_revision21_specialists import (
    _specialist_training_indices,
    _specialist_weights,
)


class Revision21SpecialistTest(unittest.TestCase):
    def test_specialist_weights_split_mass_equally(self) -> None:
        labels = np.asarray([1, 1, 1, 0, 0], dtype=np.float32)
        indices = np.arange(5, dtype=np.int64)

        weights, counts = _specialist_weights(labels, indices)

        self.assertAlmostEqual(float(weights[:3].sum()), 0.5)
        self.assertAlmostEqual(float(weights[3:].sum()), 0.5)
        self.assertEqual(counts["positive_rows"], 3)
        self.assertEqual(counts["negative_rows"], 2)

    def test_specialists_keep_negative_families_separate(self) -> None:
        base_train = np.asarray([0, 1, 2, 3], dtype=np.int64)
        base_positive = np.asarray([0, 4], dtype=np.int64)
        semantic_negative = np.asarray([1], dtype=np.int64)
        structural_negative = np.asarray([2, 3], dtype=np.int64)
        augmentation_positive = np.asarray([5], dtype=np.int64)
        augmentation_semantic = np.asarray([6], dtype=np.int64)

        semantic = _specialist_training_indices(
            base_train,
            base_positive,
            semantic_negative,
            augmentation_positive,
            augmentation_semantic,
        )
        structural = _specialist_training_indices(
            base_train,
            base_positive,
            structural_negative,
            augmentation_positive,
            np.asarray([], dtype=np.int64),
        )

        self.assertEqual(semantic.tolist(), [0, 1, 5, 6])
        self.assertEqual(structural.tolist(), [0, 2, 3, 5])


if __name__ == "__main__":
    unittest.main()
