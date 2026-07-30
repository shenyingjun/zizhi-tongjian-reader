import unittest

from p3_compact_adjudication_finalize import (
    _geometry,
    _expected_candidates,
    _normalized_annotations,
)


class P3CompactAdjudicationFinalizeTest(unittest.TestCase):
    def test_geometry_includes_surface_for_reference_comparison(self):
        rows = [{
            "para_id": 2,
            "start": 1,
            "end": 3,
            "surface": "曹操",
        }]

        self.assertEqual({(2, 1, 3, "曹操")}, _geometry(rows))

    def test_normalized_annotations_reject_duplicates(self):
        rows = [{
            "para_id": 2,
            "start": 1,
            "end": 3,
            "surface": "曹操",
        }]

        with self.assertRaisesRegex(ValueError, "duplicate"):
            _normalized_annotations(rows + rows)

    def test_expected_candidates_reconstructs_hidden_id(self):
        example_id = "juan-001-jie-0001"
        candidates = _expected_candidates(
            1,
            {example_id: {"juan": 1}},
            {example_id: {
                "reference_spans": [{
                    "para_id": 2, "start": 1, "end": 3,
                    "surface": "曹操",
                }],
                "prediction_spans": [],
            }},
        )

        self.assertEqual(1, len(candidates))
        self.assertEqual(["source_hidden"], candidates[0]["channels"])
        self.assertTrue(candidates[0]["id"].startswith("adjudication:"))


if __name__ == "__main__":
    unittest.main()
