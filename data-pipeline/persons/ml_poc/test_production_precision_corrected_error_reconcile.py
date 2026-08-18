from __future__ import annotations

import unittest

from production_precision_corrected_error_reconcile import _validate_target_output


class CorrectedErrorReconcileContractTest(unittest.TestCase):
    def setUp(self):
        self.task = {
            "task_id": "task",
            "source_candidate": {
                "candidate_id": "candidate",
                "para_id": 1,
                "start": 1,
                "end": 2,
                "surface": "乙",
            },
            "jie": {
                "text": "甲乙丙",
                "segments": [{
                    "para_id": 1,
                    "assembled_start": 0,
                    "assembled_end": 3,
                }],
            },
        }

    def _raw(self, target):
        return {
            "task_id": "task",
            "candidate_id": "candidate",
            "decision": "targets",
            "targets": [target],
            "rationale": "source reason",
            "reviewer": "copilot-teacher",
            "model": "gpt-5.6-sol",
        }

    def test_distinguishes_unchanged_from_corrected_target(self):
        unchanged = _validate_target_output(
            self.task,
            self._raw({"para_id": 1, "start": 1, "end": 2, "surface": "乙"}),
        )
        corrected = _validate_target_output(
            self.task,
            self._raw({"para_id": 1, "start": 0, "end": 2, "surface": "甲乙"}),
        )

        self.assertEqual("unchanged", unchanged[0])
        self.assertEqual("targets", corrected[0])

    def test_rejects_non_source_exact_target(self):
        with self.assertRaisesRegex(ValueError, "geometry differs"):
            _validate_target_output(
                self.task,
                self._raw({
                    "para_id": 1,
                    "start": 0,
                    "end": 2,
                    "surface": "错误",
                }),
            )


if __name__ == "__main__":
    unittest.main()
