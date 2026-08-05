from __future__ import annotations

import math
import unittest

from production_precision_infer import _span_confidence
from production_precision_reference import _labels_from_annotations
from production_precision_select import _metric, _wilson_lower


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


if __name__ == "__main__":
    unittest.main()
