from __future__ import annotations

import unittest

import numpy as np

from production_precision_revision14_two_stage import (
    _fold_local_augmentation_indices,
    _prepare_training_augmentation,
    _stage1_training_indices,
    _three_stratum_weights,
)


def _row(
    start: int,
    end: int,
    *,
    juan: int = 1,
    jie_id: str = "jie-1",
    row_class: str,
    label: int,
) -> dict:
    return {
        "id": jie_id,
        "juan": juan,
        "jie_index": 1,
        "para_id": 1,
        "start": start,
        "end": end,
        "surface": "甲乙丙丁戊己"[start:end],
        "class": row_class,
        "existence_label": label,
    }


class Revision19AugmentationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base_rows = [
            _row(0, 2, row_class="exact_reference", label=1),
            _row(2, 3, row_class="easy_negative", label=0),
            _row(5, 6, row_class="easy_negative", label=0),
        ]
        self.inventory = {
            "rows": self.base_rows,
            "binary_labels": np.asarray([1, 0, 0], dtype=np.float32),
            "classes": [row["class"] for row in self.base_rows],
            "real_labels": np.asarray([1, -1, -1], dtype=np.int8),
            "exact_by_geometry": {
                ("jie-1", 1, 0, 2): self.base_rows[0],
            },
        }
        self.examples = {
            "jie-1": {
                "id": "jie-1",
                "juan": 1,
                "jie_index": 1,
                "text": "甲乙丙丁戊己",
                "segments": [{
                    "para_id": 1,
                    "assembled_start": 0,
                    "assembled_end": 6,
                }],
            }
        }
        self.outside_example = {
            "id": "jie-2",
            "juan": 2,
            "jie_index": 1,
            "text": "甲乙丙丁戊己",
            "segments": [{
                "para_id": 1,
                "assembled_start": 0,
                "assembled_end": 6,
            }],
        }

    def test_deduplicates_rows_and_excludes_same_juan_by_fold(self) -> None:
        exact = [{**self.base_rows[0]}]
        semantic = [
            {**self.base_rows[1]},
            _row(
                3,
                4,
                juan=2,
                jie_id="jie-2",
                row_class="semantic_negative",
                label=0,
            ),
        ]
        rank_pairs = [{
            "positive": {**self.base_rows[0]},
            "negative": _row(
                1,
                2,
                row_class="reviewed_boundary_alternative",
                label=1,
            ),
        }]

        result = _prepare_training_augmentation(
            self.inventory,
            self.examples,
            [self.outside_example],
            exact,
            semantic,
            rank_pairs,
            boundary_stage1_positive=True,
        )

        self.assertEqual(len(result["rows"]), 5)
        self.assertEqual(result["stage1_indices"].tolist(), [0, 1, 3, 4])
        self.assertEqual(result["weight_classes"][1], "semantic_negative")
        self.assertEqual(len(result["pair_indices"]), 1)
        fold_zero = _fold_local_augmentation_indices(
            result["stage1_indices"],
            result["rows"],
            {1: 0},
            0,
        )
        self.assertEqual(fold_zero.tolist(), [3])
        fold_one = _fold_local_augmentation_indices(
            result["stage1_indices"],
            result["rows"],
            {1: 0},
            1,
        )
        self.assertEqual(fold_one.tolist(), [0, 1, 3, 4])

    def test_augmented_training_preserves_three_stratum_mass(self) -> None:
        result = _prepare_training_augmentation(
            self.inventory,
            self.examples,
            [self.outside_example],
            [{**self.base_rows[0]}],
            [
                {**self.base_rows[1]},
                _row(
                    3,
                    4,
                    juan=2,
                    jie_id="jie-2",
                    row_class="semantic_negative",
                    label=0,
                ),
            ],
            [],
        )
        base_indices = _stage1_training_indices(
            np.arange(len(self.base_rows), dtype=np.int64),
            self.inventory["classes"],
            self.inventory["real_labels"],
            real_only=True,
            structural_negatives=True,
        )
        training_indices = np.asarray(sorted(set(
            base_indices.tolist() + result["stage1_indices"].tolist()
        )))

        weights, counts = _three_stratum_weights(
            result["binary_labels"],
            result["weight_classes"],
            training_indices,
        )

        positive = [
            index for index in training_indices
            if result["binary_labels"][index] == 1
        ]
        semantic = [
            index for index in training_indices
            if result["weight_classes"][index] == "semantic_negative"
        ]
        structural = [
            index for index in training_indices
            if result["weight_classes"][index] == "easy_negative"
        ]
        self.assertAlmostEqual(float(weights[positive].sum()), 0.5)
        self.assertAlmostEqual(float(weights[semantic].sum()), 0.25)
        self.assertAlmostEqual(float(weights[structural].sum()), 0.25)
        self.assertEqual(counts["positive_rows"], 1)
        self.assertEqual(len(self.inventory["rows"]), 3)

    def test_overlap_positive_collision_with_negative_blocks(self) -> None:
        positive = _row(
            1,
            3,
            row_class="reviewed_exact_reference",
            label=1,
        )

        with self.assertRaisesRegex(ValueError, "conflicts with base label"):
            _prepare_training_augmentation(
                self.inventory,
                self.examples,
                [],
                [positive],
                [],
                [{
                    "positive": positive,
                    "negative": {**self.base_rows[1]},
                }],
                boundary_stage1_positive=True,
            )


if __name__ == "__main__":
    unittest.main()
