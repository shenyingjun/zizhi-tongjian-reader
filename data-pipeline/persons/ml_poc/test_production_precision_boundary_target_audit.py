from __future__ import annotations

import unittest

from production_precision_boundary_target_audit import _task_id


class BoundaryTargetAuditContractTest(unittest.TestCase):
    def test_task_id_binds_candidate(self):
        self.assertEqual(20, len(_task_id("a" * 20)))
        self.assertNotEqual(_task_id("a" * 20), _task_id("b" * 20))


if __name__ == "__main__":
    unittest.main()
