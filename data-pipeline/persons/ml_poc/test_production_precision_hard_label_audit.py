from __future__ import annotations

import unittest

from production_precision_hard_label_audit import _candidate_id, _task_id


class HardLabelAuditContractTest(unittest.TestCase):
    def test_candidate_id_binds_grouped_manifest_and_geometry(self):
        row = {
            "juan": 2,
            "jie_index": 3,
            "para_id": 5,
            "start": 7,
            "end": 9,
        }

        first = _candidate_id("a" * 64, row)
        second = _candidate_id("b" * 64, row)

        self.assertEqual(20, len(first))
        self.assertNotEqual(first, second)

    def test_task_id_is_stable_per_numbered_jie(self):
        self.assertEqual(_task_id(2, 3), _task_id(2, 3))
        self.assertNotEqual(_task_id(2, 3), _task_id(2, 4))


if __name__ == "__main__":
    unittest.main()
