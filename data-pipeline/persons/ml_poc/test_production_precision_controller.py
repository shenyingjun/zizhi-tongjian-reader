from __future__ import annotations

import math
import unittest

from production_precision_infer import _span_confidence
from production_precision_reference import _labels_from_annotations
from production_precision_select import _metric, _wilson_lower
from production_verifier_lattice import _intrinsic_vetoes
from production_span_verifier import _resolve_group


class PrecisionControllerTest(unittest.TestCase):
    def test_freezes_review_annotations_as_bio(self):
        row = {
            "id": "juan-001-jie-0001",
            "text": "曹操\n帝",
            "segments": [
                {"para_id": 4, "assembled_start": 0, "assembled_end": 2},
                {"para_id": 5, "assembled_start": 3, "assembled_end": 4},
            ],
        }
        task = {"jies": [{"text": row["text"], "segments": row["segments"]}]}

        labels = _labels_from_annotations(
            row,
            task,
            [
                {"para_id": 4, "start": 0, "end": 2, "surface": "曹操"},
                {"para_id": 5, "start": 0, "end": 1, "surface": "帝"},
            ],
        )

        self.assertEqual(labels, ["B-PER", "I-PER", "O", "B-PER"])

    def test_span_confidence_is_geometric_mean(self):
        example = {
            "id": "juan-001-jie-0001",
            "segments": [
                {"para_id": 4, "assembled_start": 0, "assembled_end": 2},
            ],
        }

        confidence = _span_confidence(
            example,
            (4, 0, 2, "曹操"),
            [0.81, 0.49],
        )

        self.assertAlmostEqual(confidence, math.sqrt(0.81 * 0.49))

    def test_precision_metric_uses_one_sided_wilson_bound(self):
        reference = set(range(400))
        prediction = set(range(399))

        metric = _metric(reference, prediction)

        self.assertEqual(metric["true_positive"], 399)
        self.assertEqual(metric["precision"], 1.0)
        self.assertLess(metric["wilson_precision_lower_one_sided_95"], 1.0)
        self.assertAlmostEqual(
            metric["wilson_precision_lower_one_sided_95"],
            _wilson_lower(399, 399),
        )

    def test_intrinsic_veto_does_not_reject_numeral_shaped_name_character(self):
        self.assertEqual(_intrinsic_vetoes("万安"), [])
        self.assertEqual(
            _intrinsic_vetoes("张三，"),
            ["numeric_punctuation_or_symbol"],
        )

    def test_overlap_resolution_penalizes_weak_fragments(self):
        def candidate(start, end, score):
            return {
                "id": "juan-001-jie-0001",
                "para_id": 4,
                "start": start,
                "end": end,
                "surface": "人物名"[start:end],
                "score": score,
                "support_count": 3,
                "seed_confidences": {"a": 0.9, "b": 0.9, "c": 0.9},
            }

        selected = _resolve_group(
            [
                candidate(0, 3, 0.99),
                candidate(0, 1, 0.51),
                candidate(1, 3, 0.51),
            ],
            0.50,
        )

        self.assertEqual([(row["start"], row["end"]) for row in selected], [(0, 3)])

    def test_overlap_resolution_can_keep_two_strong_adjacent_spans(self):
        def candidate(start, end, score):
            return {
                "id": "juan-001-jie-0001",
                "para_id": 4,
                "start": start,
                "end": end,
                "surface": "甲乙丙丁"[start:end],
                "score": score,
                "support_count": 3,
                "seed_confidences": {"a": 0.9, "b": 0.9, "c": 0.9},
            }

        selected = _resolve_group(
            [
                candidate(0, 4, 0.51),
                candidate(0, 2, 0.99),
                candidate(2, 4, 0.99),
            ],
            0.50,
        )

        self.assertEqual(
            [(row["start"], row["end"]) for row in selected],
            [(0, 2), (2, 4)],
        )


if __name__ == "__main__":
    unittest.main()
