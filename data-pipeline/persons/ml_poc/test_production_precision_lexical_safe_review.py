from __future__ import annotations

import math
import unittest

from production_precision_lexical_safe_review import (
    AUDIT_SIZE,
    _candidate_vetoes,
    _overlaps,
    _text_sha256,
)


class SafeReviewContractTest(unittest.TestCase):
    def test_audit_size_is_minimal_for_one_percent_upper_bound(self):
        self.assertEqual(AUDIT_SIZE, 299)
        self.assertLess(1 - 0.05 ** (1 / AUDIT_SIZE), 0.01)
        self.assertGreaterEqual(1 - 0.05 ** (1 / (AUDIT_SIZE - 1)), 0.01)
        self.assertEqual(
            AUDIT_SIZE, math.ceil(math.log(0.05) / math.log(0.99))
        )

    def test_half_open_overlap(self):
        self.assertTrue(_overlaps((2, 4), (3, 5)))
        self.assertFalse(_overlaps((2, 4), (4, 5)))

    def test_translation_is_a_positive_veto(self):
        paragraph = "甲曹操乙"
        task = {"jie_index": 7}
        candidate = {
            "para_id": 3,
            "start": 1,
            "end": 2,
            "surface": "曹",
        }
        evidence = {"paragraphs": {"3": {
            "text_sha256": _text_sha256(paragraph),
            "jie_index": 7,
            "identities": [{"candidates": [{
                "start": 1,
                "end": 3,
                "surface": "曹操",
                "mapping_status": "mapped_exact_unique_jie",
            }]}],
        }}}
        self.assertEqual(
            ["approved_translation_evidence_overlap"],
            _candidate_vetoes(
                task, candidate, {3: paragraph}, evidence
            ),
        )

    def test_flagged_evidence_cannot_veto(self):
        paragraph = "甲曹操乙"
        task = {"jie_index": 7}
        candidate = {
            "para_id": 3,
            "start": 1,
            "end": 3,
            "surface": "曹操",
        }
        evidence = {"paragraphs": {"3": {
            "text_sha256": _text_sha256(paragraph),
            "jie_index": 7,
            "identities": [{"candidates": [{
                "start": 1,
                "end": 3,
                "surface": "曹操",
                "mapping_status": "flagged_multi_jie_identity",
            }]}],
        }}}
        self.assertEqual(
            [],
            _candidate_vetoes(
                task, candidate, {3: paragraph}, evidence
            ),
        )


if __name__ == "__main__":
    unittest.main()
