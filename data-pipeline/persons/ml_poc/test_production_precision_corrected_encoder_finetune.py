from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import production_precision_corrected_encoder_finetune as corrected


def _examples() -> list[dict]:
    return [
        {
            "id": f"juan-{juan:03d}-jie-0001",
            "juan": juan,
            "jie_index": 1,
            "text": "abcd",
            "segments": [{
                "para_id": juan,
                "assembled_start": 0,
                "assembled_end": 4,
            }],
        }
        for juan in range(1, 29)
    ]


def _row(juan: int, start: int, end: int) -> dict:
    text = "abcd"
    return {
        "id": f"juan-{juan:03d}-jie-0001",
        "juan": juan,
        "jie_index": 1,
        "fold": (juan - 1) % 7,
        "para_id": juan,
        "start": start,
        "end": end,
        "surface": text[start:end],
    }


def _synthetic_inputs() -> tuple:
    references = [_row(juan, 0, 2) for juan in (1, 2)]
    boundary = [_row(juan, 0, 1) for juan in (1, 2)]
    semantic = [{**_row(juan, 2, 3), "label": 0} for juan in (1, 2)]
    exact_real = [{**row, "label": 1} for row in references]
    boundary_real = [{**row, "label": 1} for row in boundary]
    rank_pairs = [
        {"positive": references[0], "negative": boundary[0]},
        {"positive": references[0], "negative": boundary[0]},
        {"positive": references[1], "negative": boundary[1]},
    ]
    easy = [_row(juan, 3, 4) for juan in (1, 2)]
    mined = [boundary[0]]
    return (
        _examples(),
        references,
        [*exact_real, *boundary_real, *semantic],
        rank_pairs,
        easy,
        mined,
    )


def _count_patch():
    return patch.multiple(
        corrected,
        EXPECTED_REFERENCES=2,
        EXPECTED_EXISTENCE=6,
        EXPECTED_RANK_PAIRS=3,
        EXPECTED_RANK_BOUNDARY=2,
        EXPECTED_BOUNDARY=2,
        EXPECTED_SEMANTIC=2,
        EXPECTED_EASY=2,
        EXPECTED_MINED_BOUNDARIES=1,
        EXPECTED_INVENTORY=8,
    )


class CorrectedEncoderInventoryTest(unittest.TestCase):
    def test_assembles_unique_disjoint_inventory_and_real_mapping(self):
        with _count_patch():
            inventory = corrected._assemble_inventory(*_synthetic_inputs())

        self.assertEqual(8, len(inventory["rows"]))
        self.assertEqual(
            {
                "exact_person": 2,
                "boundary_alternative": 2,
                "real_not_person": 2,
                "mined_not_person": 2,
            },
            {
                name: inventory["strata"].count(name)
                for name in corrected.STRATUM_MASSES
            },
        )
        self.assertEqual(6, int((inventory["real_labels"] >= 0).sum()))
        self.assertEqual(2, int((inventory["real_labels"] == 1).sum()))

    def test_rejects_class_geometry_collision(self):
        values = list(_synthetic_inputs())
        values[4][0] = {**values[2][-2], "label": 0}
        with _count_patch(), self.assertRaisesRegex(
            ValueError, "class/stratum geometry collision"
        ):
            corrected._assemble_inventory(*values)

    def test_rejects_unreachable_rank_positive(self):
        values = list(_synthetic_inputs())
        values[3][0] = {
            "positive": _row(1, 2, 4),
            "negative": values[3][0]["negative"],
        }
        with _count_patch(), self.assertRaisesRegex(
            ValueError, "rank positive is not a reference"
        ):
            corrected._assemble_inventory(*values)

    def test_rejects_unreachable_reference(self):
        values = list(_synthetic_inputs())
        values[1][1] = _row(2, 1, 3)
        values[3][2] = dict(values[3][0])
        with (
            _count_patch(),
            patch.object(corrected, "EXPECTED_RANK_BOUNDARY", 1),
            patch.object(corrected, "EXPECTED_BOUNDARY", 3),
            patch.object(corrected, "EXPECTED_INVENTORY", 9),
            self.assertRaisesRegex(ValueError, "reference is unreachable"),
        ):
            corrected._assemble_inventory(*values)

    def test_rejects_count_difference(self):
        values = list(_synthetic_inputs())
        values[1] = values[1][:-1]
        with _count_patch(), self.assertRaisesRegex(
            ValueError, "reference count differs"
        ):
            corrected._assemble_inventory(*values)

    def test_historical_source_fold_does_not_override_seven_fold_map(self):
        values = list(_synthetic_inputs())
        values[2][0]["fold"] = 99
        with _count_patch():
            inventory = corrected._assemble_inventory(*values)
        self.assertEqual(0, int(inventory["fold_ids"][0]))


class CorrectedEncoderMetricTest(unittest.TestCase):
    def _metric_arrays(self):
        labels = np.asarray(
            [
                corrected.LABEL_BY_NAME["exact_person"],
                corrected.LABEL_BY_NAME["boundary_alternative"],
                corrected.LABEL_BY_NAME["not_person"],
                corrected.LABEL_BY_NAME["not_person"],
            ]
            * corrected.FOLDS,
            dtype=np.int64,
        )
        strata = [
            "exact_person",
            "boundary_alternative",
            "real_not_person",
            "mined_not_person",
        ] * corrected.FOLDS
        folds = np.repeat(np.arange(corrected.FOLDS), 4)
        real_labels = np.asarray([1, 0, 0, -1] * corrected.FOLDS)
        return labels, strata, folds, real_labels

    def test_perfect_small_data_is_eligible_without_wilson_gate(self):
        labels, strata, folds, real_labels = self._metric_arrays()
        scores = np.asarray(
            [0.99, 0.01, 0.01, 0.01] * corrected.FOLDS,
            dtype=np.float32,
        )

        row = next(
            value
            for value in corrected._threshold_table(
                scores, labels, strata, folds, real_labels
            )
            if value["threshold"] == 0.50
        )

        self.assertTrue(row["eligible"])
        self.assertEqual(1.0, row["exact_recall"])
        self.assertEqual(1.0, row["real_candidate_precision"])
        self.assertLess(
            row["all_row_wilson_precision_lower_one_sided_95"], 0.99
        )

    def test_boundary_failure_blocks_eligibility(self):
        labels, strata, folds, real_labels = self._metric_arrays()
        scores = np.asarray(
            [0.99, 0.01, 0.01, 0.01] * corrected.FOLDS,
            dtype=np.float32,
        )
        scores[1] = np.float32(0.50)

        row = next(
            value
            for value in corrected._threshold_table(
                scores, labels, strata, folds, real_labels
            )
            if value["threshold"] == 0.50
        )

        self.assertFalse(row["eligible"])
        self.assertEqual(6 / 7, row["boundary_alternative_rejection"])
        self.assertEqual(7 / 8, row["real_candidate_precision"])

    def test_fold_absence_is_explicitly_null(self):
        labels, strata, folds, real_labels = self._metric_arrays()
        scores = np.asarray(
            [0.99, 0.01, 0.01, 0.01] * corrected.FOLDS,
            dtype=np.float32,
        )
        strata[6] = "mined_not_person"

        row = next(
            value
            for value in corrected._threshold_table(
                scores, labels, strata, folds, real_labels
            )
            if value["threshold"] == 0.50
        )

        self.assertIsNone(row["fold_metrics"][1]["semantic_rejection"])
        self.assertEqual(0, row["fold_metrics"][1]["semantic_rows"])


if __name__ == "__main__":
    unittest.main()
