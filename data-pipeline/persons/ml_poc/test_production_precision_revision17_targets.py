import unittest

from production_precision_revision17_target_freeze import normalize_target_raw
from production_precision_revision17_targets import build_target_inventory


class Revision17TargetsTest(unittest.TestCase):
    def test_join_counts_semantic_juan_diversity(self):
        decisions = [
            {
                "task_id": "a",
                "candidate_id": "ca",
                "label": "not_person",
            },
            {
                "task_id": "b",
                "candidate_id": "cb",
                "label": "wrong_boundary",
            },
        ]
        selection = [
            {"task_id": "a", "candidate_id": "ca", "juan": 1},
            {"task_id": "b", "candidate_id": "cb", "juan": 2},
        ]
        joined, counts = build_target_inventory(decisions, selection)

        self.assertEqual(2, len(joined))
        self.assertEqual(1, counts["not_person"])
        self.assertEqual(1, counts["not_person_juans"])
        self.assertEqual(1, counts["wrong_boundary"])

    def test_validates_single_extended_exact_target(self):
        task = {
            "target_task_id": "target",
            "candidate_id": "candidate",
            "jie": {
                "text": "甲乙丙",
                "segments": [{
                    "para_id": 7,
                    "assembled_start": 0,
                    "assembled_end": 3,
                }],
            },
            "wrong_candidate": {
                "candidate_id": "candidate",
                "para_id": 7,
                "start": 1,
                "end": 2,
                "surface": "乙",
            },
        }
        result = normalize_target_raw({
            "target_task_id": "target",
            "candidate_id": "candidate",
            "uncertain": False,
            "targets": [{
                "para_id": 7,
                "start": 0,
                "end": 2,
                "surface": "甲乙",
            }],
            "rationale": "The full name includes the preceding character.",
            "reviewer": "copilot-teacher",
            "model": "gpt-5.6-sol",
        }, task)

        self.assertFalse(result["uncertain"])
        self.assertEqual("甲乙", result["targets"][0]["surface"])

    def test_rejects_target_equal_to_wrong_candidate(self):
        task = {
            "target_task_id": "target",
            "candidate_id": "candidate",
            "jie": {
                "text": "甲乙",
                "segments": [{
                    "para_id": 7,
                    "assembled_start": 0,
                    "assembled_end": 2,
                }],
            },
            "wrong_candidate": {
                "candidate_id": "candidate",
                "para_id": 7,
                "start": 0,
                "end": 1,
                "surface": "甲",
            },
        }
        with self.assertRaisesRegex(ValueError, "validity differs"):
            normalize_target_raw({
                "target_task_id": "target",
                "candidate_id": "candidate",
                "uncertain": False,
                "targets": [{
                    "para_id": 7,
                    "start": 0,
                    "end": 1,
                    "surface": "甲",
                }],
                "rationale": "Same geometry.",
                "reviewer": "copilot-teacher",
                "model": "gpt-5.6-sol",
            }, task)

    def test_retains_certain_empty_target_as_contradiction(self):
        task = {
            "target_task_id": "target",
            "candidate_id": "candidate",
            "jie": {
                "text": "甲乙",
                "segments": [{
                    "para_id": 7,
                    "assembled_start": 0,
                    "assembled_end": 2,
                }],
            },
            "wrong_candidate": {
                "candidate_id": "candidate",
                "para_id": 7,
                "start": 0,
                "end": 1,
                "surface": "甲",
            },
        }
        result = normalize_target_raw({
            "target_task_id": "target",
            "candidate_id": "candidate",
            "uncertain": False,
            "targets": [],
            "rationale": "No overlapping person is present.",
            "reviewer": "copilot-teacher",
            "model": "gpt-5.6-sol",
        }, task)

        self.assertTrue(result["contradiction"])
        self.assertEqual([], result["targets"])

    def test_canonicalizes_redundant_surface_without_changing_geometry(self):
        task = {
            "target_task_id": "target",
            "candidate_id": "candidate",
            "jie": {
                "text": "甲乙丙",
                "segments": [{
                    "para_id": 7,
                    "assembled_start": 0,
                    "assembled_end": 3,
                }],
            },
            "wrong_candidate": {
                "candidate_id": "candidate",
                "para_id": 7,
                "start": 1,
                "end": 2,
                "surface": "乙",
            },
        }
        result = normalize_target_raw({
            "target_task_id": "target",
            "candidate_id": "candidate",
            "uncertain": False,
            "targets": [{
                "para_id": 7,
                "start": 0,
                "end": 2,
                "surface": "甲",
            }],
            "rationale": "The exact person occupies the first two characters.",
            "reviewer": "copilot-teacher",
            "model": "gpt-5.6-sol",
        }, task)

        self.assertEqual("甲乙", result["targets"][0]["surface"])
        self.assertEqual("甲", result["targets"][0]["reported_surface"])
        self.assertTrue(result["targets"][0]["surface_corrected"])


if __name__ == "__main__":
    unittest.main()
