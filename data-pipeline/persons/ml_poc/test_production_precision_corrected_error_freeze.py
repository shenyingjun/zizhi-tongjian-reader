from __future__ import annotations

import unittest

from production_precision_corrected_error_freeze import _normalize_raw


class CorrectedErrorFreezeContractTest(unittest.TestCase):
    def test_normalizes_frozen_rationale_alias_without_changing_judgment(self):
        normalized = _normalize_raw({
            "task_id": "a",
            "candidate_id": "b",
            "label": "wrong_boundary",
            "brief source rationale": "  source reason  ",
            "reviewer": "copilot-teacher",
            "model": "gpt-5.6-sol",
        })

        self.assertEqual("wrong_boundary", normalized["label"])
        self.assertEqual("source reason", normalized["rationale"])

    def test_rejects_extra_raw_fields(self):
        with self.assertRaisesRegex(ValueError, "raw fields differ"):
            _normalize_raw({
                "task_id": "a",
                "candidate_id": "b",
                "label": "exact_person",
                "rationale": "reason",
                "reviewer": "copilot-teacher",
                "model": "gpt-5.6-sol",
                "score": 0.9,
            })


if __name__ == "__main__":
    unittest.main()
