import json
import tempfile
import unittest
from pathlib import Path

import benchmark_reference as BR


class BenchmarkReferenceTest(unittest.TestCase):
    def test_committed_exclusions_are_current_and_unique(self):
        exclusions = BR.load_exclusions()
        summary = BR.exclusion_summary(exclusions)

        self.assertGreater(summary["count"], 0)
        self.assertEqual(
            summary["count"],
            sum(len(rows) for rows in exclusions.values()),
        )

    def test_duplicate_geometry_is_rejected(self):
        row = {
            "juan": 1,
            "pid": 1,
            "start": 0,
            "end": 1,
            "surface": "二",
            "reason": "test",
            "review": "test",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicates.jsonl"
            path.write_text(
                json.dumps(row, ensure_ascii=False) + "\n"
                + json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate geometry"):
                BR.load_exclusions(path, validate=False)


if __name__ == "__main__":
    unittest.main()
