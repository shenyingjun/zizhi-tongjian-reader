from __future__ import annotations

import unittest

from production_precision_corrected_error_target_audit import _opaque_id


class CorrectedErrorTargetAuditContractTest(unittest.TestCase):
    def test_opaque_ids_bind_salt_domain_and_source_ids(self):
        salt = bytes(range(32))
        first = _opaque_id(salt, "task", "a" * 24, "b" * 24)

        self.assertEqual(24, len(first))
        self.assertNotEqual(first, _opaque_id(salt, "candidate", "a" * 24, "b" * 24))
        self.assertNotEqual(
            first, _opaque_id(bytes(reversed(salt)), "task", "a" * 24, "b" * 24)
        )


if __name__ == "__main__":
    unittest.main()
