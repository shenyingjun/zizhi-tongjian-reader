import unittest

from p6_data_impact import (
    _bio_spans,
    _occurrence_status,
    _profile,
    _synthesize_findings,
)


class DataImpactTest(unittest.TestCase):
    def test_extracts_bio_spans_and_boundary_profiles(self):
        row = {
            "id": "juan-001-jie-0001",
            "text": "甲之乙氏丙公",
            "labels": [
                "B-PER", "O", "B-PER", "I-PER", "B-PER", "O",
            ],
        }
        self.assertEqual(
            [(0, 1, "甲"), (2, 4, "乙氏"), (4, 5, "丙")],
            _bio_spans(row),
        )
        profile = _profile([row])
        self.assertEqual(2, profile["single_character_spans"])
        self.assertEqual(
            {"之": 1, "公": 1},
            profile["single_character_gold_followed_by"],
        )
        self.assertEqual({"氏": 1}, profile["gold_ending_in"])

    def test_classifies_term_against_gold_geometry(self):
        spans = [(1, 3, "甲乙")]
        self.assertEqual("exact_gold", _occurrence_status(1, 3, spans))
        self.assertEqual(
            "inside_larger_gold", _occurrence_status(1, 2, spans)
        )
        self.assertEqual("overlaps_gold", _occurrence_status(2, 4, spans))
        self.assertEqual("untagged", _occurrence_status(4, 5, spans))

    def test_synthesizes_directional_model_shift(self):
        rows = [{
            "id": "juan-001-jie-0001",
            "text": "甲之",
            "labels": ["B-PER", "O"],
        }]
        profiles = {
            "round4_existing": _profile(rows),
            "round5_added": _profile(rows),
        }
        comparisons = {"evaluation": {
            "metrics": {
                "round4": {
                    "prediction_spans": 10,
                    "exact": {
                        "true_positive": 9,
                        "precision": 0.9,
                        "recall": 0.9,
                        "f1": 0.9,
                    },
                },
                "round6": {
                    "prediction_spans": 8,
                    "exact": {
                        "true_positive": 8,
                        "precision": 1.0,
                        "recall": 0.8,
                        "f1": 8 / 9,
                    },
                },
            },
            "attribution": {
                "new_true_positives": 0,
                "lost_true_positives": 1,
                "added_false_positives": 0,
                "removed_false_positives": 1,
            },
        }}
        findings = _synthesize_findings(profiles, comparisons)
        shift = findings["incremental_model_shift"]
        self.assertEqual(-2, shift["prediction_count_delta"])
        self.assertEqual(-1, shift["true_positive_delta"])
        self.assertGreater(shift["precision_delta"], 0)
        self.assertLess(shift["recall_delta"], 0)


if __name__ == "__main__":
    unittest.main()
