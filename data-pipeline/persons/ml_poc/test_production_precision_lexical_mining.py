from __future__ import annotations

import unittest

from production_precision_lexical_mining import (
    _candidate_id,
    _enumerate_candidates,
    _overlaps,
)


class LexicalCandidateTest(unittest.TestCase):
    def _example(self) -> dict:
        return {
            "id": "juan-001-jie-0001",
            "juan": 1,
            "jie_index": 1,
            "text": "甲乙，丙丁",
            "segments": [{
                "para_id": 4,
                "assembled_start": 0,
                "assembled_end": 5,
            }],
        }

    def test_overlap_is_half_open_and_paragraph_local(self):
        references = {(4, 1, 3)}
        self.assertTrue(_overlaps((4, 0, 2), references))
        self.assertFalse(_overlaps((4, 3, 4), references))
        self.assertFalse(_overlaps((5, 1, 2), references))

    def test_enumeration_excludes_reference_overlap_veto_and_existing(self):
        rows = _enumerate_candidates(
            self._example(),
            {(4, 0, 2)},
            {(4, 3, 4)},
        )
        geometries = {
            (row["para_id"], row["start"], row["end"]) for row in rows
        }
        self.assertNotIn((4, 0, 1), geometries)
        self.assertNotIn((4, 2, 3), geometries)  # punctuation hard veto
        self.assertNotIn((4, 3, 4), geometries)  # existing OOF geometry
        self.assertIn((4, 4, 5), geometries)

    def test_candidate_id_uses_full_canonical_key(self):
        first = _candidate_id((1, 2, 3, 4, 5))
        second = _candidate_id((2, 2, 3, 4, 5))
        self.assertEqual(len(first), 20)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
