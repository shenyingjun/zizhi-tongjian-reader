from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from production_precision_hard_label_server import LABELS


class HardLabelServerContractTest(unittest.TestCase):
    def test_labels_are_the_blind_audit_labels(self):
        self.assertEqual(
            {"exact_person", "wrong_boundary", "not_person", "uncertain"},
            LABELS,
        )

    def test_state_directory_can_be_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review = root / "review"
            state = root / "state"
            self.assertNotEqual(review.resolve(), state.resolve())


if __name__ == "__main__":
    unittest.main()
