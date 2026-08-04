import unittest

from p3_compact_adjudicate import (
    consensus_annotations,
    disagreement_rows,
)


class P3CompactAdjudicateTest(unittest.TestCase):
    def test_initial_annotations_keep_only_consensus_geometry(self):
        annotations = [
            {"para_id": 1, "start": 1, "end": 3, "surface": "曹操"},
            {"para_id": 1, "start": 5, "end": 7, "surface": "刘备"},
        ]
        candidates = [
            {"para_id": 1, "start": 1, "end": 3, "surface": "曹操"},
            {"para_id": 1, "start": 1, "end": 4, "surface": "曹操至"},
        ]

        result = consensus_annotations(annotations, candidates)

        self.assertEqual([annotations[1]], result)

    def test_emits_both_sides_of_exact_geometry_disagreement(self):
        prediction = {
            "reference_spans": [
                {
                    "para_id": 1, "start": 1, "end": 4,
                    "surface": "赵武灵",
                },
                {
                    "para_id": 1, "start": 8, "end": 10,
                    "surface": "曹操",
                },
            ],
            "prediction_spans": [
                {
                    "para_id": 1, "start": 2, "end": 4,
                    "surface": "武灵",
                },
                {
                    "para_id": 1, "start": 8, "end": 10,
                    "surface": "曹操",
                },
            ],
        }

        rows = disagreement_rows(prediction)

        self.assertEqual(2, len(rows))
        self.assertEqual(
            {(1, 1, 4), (1, 2, 4)},
            {(row["para_id"], row["start"], row["end"]) for row in rows},
        )


if __name__ == "__main__":
    unittest.main()
