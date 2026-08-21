import unittest

from production_precision_revision17_candidates import merge_predictions


def _example():
    return {
        "id": "juan-001-jie-0001",
        "juan": 1,
        "jie_index": 1,
        "text": "甲乙丙",
        "segments": [{
            "para_id": 7,
            "assembled_start": 0,
            "assembled_end": 3,
        }],
        "labels": [0, 0, 0],
        "target_mask": [True, True, True],
    }


def _prediction(spans):
    return [{
        "id": "juan-001-jie-0001",
        "juan": 1,
        "jie_index": 1,
        "reference_spans": [],
        "prediction_spans": spans,
    }]


class Revision17CandidatesTest(unittest.TestCase):
    def test_merges_support_and_maximum_confidence(self):
        shared = {
            "para_id": 7,
            "start": 0,
            "end": 2,
            "surface": "甲乙",
        }
        rows = merge_predictions(
            [_example()],
            {
                20260727: _prediction([{**shared, "confidence": 0.7}]),
                20260728: _prediction([{**shared, "confidence": 0.9}]),
                20260729: _prediction([{
                    "para_id": 7,
                    "start": 2,
                    "end": 3,
                    "surface": "丙",
                    "confidence": 0.8,
                }]),
            },
        )

        self.assertEqual(2, len(rows))
        self.assertEqual(2, rows[0]["generator_support"])
        self.assertEqual(0.9, rows[0]["maximum_generator_confidence"])
        self.assertEqual(1, rows[1]["generator_support"])

    def test_rejects_non_source_exact_surface(self):
        bad = {
            "para_id": 7,
            "start": 0,
            "end": 2,
            "surface": "甲丙",
            "confidence": 0.7,
        }
        with self.assertRaisesRegex(ValueError, "candidate source differs"):
            merge_predictions(
                [_example()],
                {
                    20260727: _prediction([bad]),
                    20260728: _prediction([]),
                    20260729: _prediction([]),
                },
            )

    def test_rejects_missing_seed_coverage(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            merge_predictions(
                [_example()],
                {
                    20260727: _prediction([]),
                    20260728: _prediction([]),
                },
            )


if __name__ == "__main__":
    unittest.main()
