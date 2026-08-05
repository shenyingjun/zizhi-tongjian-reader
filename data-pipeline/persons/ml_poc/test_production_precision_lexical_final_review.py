from __future__ import annotations

import unittest

from production_precision_lexical_final_review import (
    TEACHER_MODELS,
    _schema_sha256,
)


class FinalReviewContractTest(unittest.TestCase):
    def test_teacher_families_are_pinned(self):
        self.assertEqual(TEACHER_MODELS["A"], "claude-sonnet-5")
        self.assertEqual(TEACHER_MODELS["C"], "claude-sonnet-5")
        self.assertEqual(TEACHER_MODELS["D"], "gpt-5.6-sol")
        self.assertEqual(len(_schema_sha256()), 64)


if __name__ == "__main__":
    unittest.main()
