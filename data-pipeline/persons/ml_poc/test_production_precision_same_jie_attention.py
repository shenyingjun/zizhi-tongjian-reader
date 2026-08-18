from __future__ import annotations

import unittest

import numpy as np

from production_precision_same_jie_attention import (
    DISTANCE_BUCKETS,
    STRATUM_MASSES,
    _distance_bucket,
    _folds,
    _stratum_weights,
)


class SameJieAttentionContractTest(unittest.TestCase):
    def test_round_robin_juan_folds_are_four_each(self):
        assignments = _folds(list(range(1, 29)))

        self.assertEqual(set(range(7)), set(assignments.values()))
        self.assertEqual(
            [4] * 7,
            [list(assignments.values()).count(fold) for fold in range(7)],
        )

    def test_distance_buckets_are_same_jie_and_edge_based(self):
        self.assertEqual(
            DISTANCE_BUCKETS.index("inside"),
            _distance_bucket(5, 5, 7, 3, 3),
        )
        self.assertEqual(
            DISTANCE_BUCKETS.index("1"),
            _distance_bucket(4, 5, 7, 3, 3),
        )
        self.assertEqual(
            DISTANCE_BUCKETS.index("2-4"),
            _distance_bucket(9, 5, 7, 3, 3),
        )
        self.assertEqual(
            DISTANCE_BUCKETS.index("other_paragraph"),
            _distance_bucket(5, 5, 7, 4, 3),
        )

    def test_fold_local_weights_preserve_masses(self):
        strata = [
            *(["real_positive"] * 5),
            *(["real_negative"] * 2),
            *(["mined_negative"] * 7),
        ]

        weights, _ = _stratum_weights(strata)

        for name, mass in STRATUM_MASSES.items():
            indices = np.asarray([
                index for index, value in enumerate(strata) if value == name
            ])
            self.assertAlmostEqual(
                mass, float(weights[indices].sum()) / len(strata), places=7
            )


if __name__ == "__main__":
    unittest.main()
