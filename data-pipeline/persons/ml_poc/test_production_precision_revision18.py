import unittest

from production_precision_revision17_tasks import (
    TASKS_STATUS as REVIEW_TASKS_STATUS,
)
from production_precision_revision18_adjudication import TASKS_STATUS
from production_precision_revision18_freeze import normalize_raw


def _task():
    return {
        "adjudication_task_id": "task",
        "candidate_id": "candidate",
        "allowed_labels": [
            "exact_person",
            "wrong_boundary",
            "not_person",
            "uncertain",
        ],
        "jie": {
            "text": "甲乙",
            "segments": [{
                "para_id": 1,
                "assembled_start": 0,
                "assembled_end": 2,
            }],
        },
        "candidate": {
            "candidate_id": "candidate",
            "para_id": 1,
            "start": 1,
            "end": 2,
            "surface": "乙",
        },
    }


class Revision18Test(unittest.TestCase):
    def test_source_and_output_task_statuses_are_distinct(self):
        self.assertNotEqual(REVIEW_TASKS_STATUS, TASKS_STATUS)

    def test_normalizes_wrong_boundary_target(self):
        result = normalize_raw({
            "adjudication_task_id": "task",
            "candidate_id": "candidate",
            "label": "wrong_boundary",
            "targets": [{
                "para_id": 1,
                "start": 0,
                "end": 2,
                "surface": "甲乙",
            }],
            "rationale": "The full person span includes both characters.",
            "reviewer": "copilot-teacher",
            "model": "gpt-5.6-sol",
        }, _task())

        self.assertEqual("wrong_boundary", result["label"])
        self.assertFalse(result["contradiction"])
        self.assertEqual("甲乙", result["targets"][0]["surface"])

    def test_rejects_targets_for_not_person(self):
        with self.assertRaisesRegex(ValueError, "decision differs"):
            normalize_raw({
                "adjudication_task_id": "task",
                "candidate_id": "candidate",
                "label": "not_person",
                "targets": [{
                    "para_id": 1,
                    "start": 0,
                    "end": 2,
                    "surface": "甲乙",
                }],
                "rationale": "Not a person.",
                "reviewer": "copilot-teacher",
                "model": "gpt-5.6-sol",
            }, _task())


if __name__ == "__main__":
    unittest.main()
