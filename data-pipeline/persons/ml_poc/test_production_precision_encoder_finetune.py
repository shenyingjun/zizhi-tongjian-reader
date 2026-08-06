from __future__ import annotations

import unittest

import numpy as np

from production_precision_encoder_finetune import (
    FOLDS,
    LABEL_BY_NAME,
    STRATUM_MASSES,
    _classify_real_row,
    _expand_bounds,
    _stratum_weights,
    _table,
)


class EncoderFinetuneContractTest(unittest.TestCase):
    def test_real_rows_are_reclassified_from_exact_geometry(self):
        exact = {
            "label": 1,
            "para_id": 3,
            "start": 5,
            "end": 7,
            "overlapping_references": [
                {"para_id": 3, "start": 5, "end": 7}
            ],
        }
        boundary = {
            **exact,
            "end": 8,
        }
        negative = {
            **exact,
            "label": 0,
            "overlapping_references": [],
        }

        self.assertEqual(
            ("exact_person", LABEL_BY_NAME["exact_person"]),
            _classify_real_row(exact),
        )
        self.assertEqual(
            (
                "boundary_alternative",
                LABEL_BY_NAME["boundary_alternative"],
            ),
            _classify_real_row(boundary),
        )
        self.assertEqual(
            ("real_not_person", LABEL_BY_NAME["not_person"]),
            _classify_real_row(negative),
        )

    def test_expansion_uses_remaining_side_after_edge(self):
        text = "abcdefgh"

        bounds = _expand_bounds(
            text,
            0,
            1,
            lambda start, end: end - start <= 5,
        )

        self.assertEqual((0, 5), bounds)

    def test_fold_local_weights_preserve_masses(self):
        strata = [
            *(["exact_person"] * 5),
            *(["boundary_alternative"] * 2),
            *(["real_not_person"] * 3),
            *(["mined_not_person"] * 7),
        ]

        weights, _ = _stratum_weights(strata)

        for name, mass in STRATUM_MASSES.items():
            indices = np.asarray([
                index for index, value in enumerate(strata)
                if value == name
            ])
            self.assertAlmostEqual(
                mass, float(weights[indices].sum()) / len(strata), places=7
            )

    def test_table_does_not_count_boundary_alternative_as_recall(self):
        labels = np.asarray(
            [
                LABEL_BY_NAME["exact_person"],
                LABEL_BY_NAME["boundary_alternative"],
                LABEL_BY_NAME["not_person"],
                LABEL_BY_NAME["not_person"],
            ]
            * FOLDS,
            dtype=np.int64,
        )
        strata = [
            "exact_person",
            "boundary_alternative",
            "real_not_person",
            "mined_not_person",
        ] * FOLDS
        fold_ids = np.repeat(np.arange(FOLDS), 4)
        scores = np.asarray([0.99, 0.99, 0.01, 0.01] * FOLDS)

        threshold = next(
            row for row in _table(scores, labels, strata, fold_ids)
            if row["threshold"] == 0.50
        )

        self.assertEqual(7, threshold["true_positive"])
        self.assertEqual(14, threshold["prediction_rows"])
        self.assertEqual(1.0, threshold["recall"])
        self.assertEqual(0.5, threshold["precision"])
        self.assertEqual(0.0, threshold["boundary_alternative_rejection"])


if __name__ == "__main__":
    unittest.main()
