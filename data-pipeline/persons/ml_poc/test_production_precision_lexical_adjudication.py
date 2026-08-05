from __future__ import annotations

import unittest

from production_precision_lexical_adjudication import _audit_digest


class AuditSelectionTest(unittest.TestCase):
    def test_digest_uses_canonical_geometry(self):
        task = {"juan": 12, "jie_index": 34}
        candidate = {"para_id": 5, "start": 6, "end": 7}
        first = _audit_digest(task, candidate)
        self.assertEqual(first, _audit_digest(task, candidate))
        self.assertNotEqual(
            first,
            _audit_digest(
                {"juan": 13, "jie_index": 34},
                candidate,
            ),
        )


if __name__ == "__main__":
    unittest.main()
