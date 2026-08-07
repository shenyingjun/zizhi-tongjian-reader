from __future__ import annotations

import unittest

from production_precision_hard_label_freeze import _confusion


class HardLabelFreezeContractTest(unittest.TestCase):
    def test_confusion_separates_original_strata(self):
        rows = [
            {
                "original_label": "real_not_person",
                "audited_label": "exact_person",
            },
            {
                "original_label": "boundary_alternative",
                "audited_label": "wrong_boundary",
            },
        ]

        confusion = _confusion(rows)

        self.assertEqual(1, confusion["real_not_person"]["exact_person"])
        self.assertEqual(
            1, confusion["boundary_alternative"]["wrong_boundary"]
        )
        self.assertEqual(0, confusion["real_not_person"]["not_person"])


if __name__ == "__main__":
    unittest.main()
