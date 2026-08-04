import tempfile
import unittest
from pathlib import Path

from p14_compact_challenge_evaluate import (
    _aggregate,
    evaluate_compact_challenge,
)


class CompactChallengeEvaluateTest(unittest.TestCase):
    def test_aggregate_uses_micro_counts(self):
        rows = [
            {"counts": {"system": (2, 2, 2)}},
            {"counts": {"system": (8, 4, 4)}},
        ]
        metric = _aggregate(rows, "system")
        self.assertEqual(10, metric["reference_spans"])
        self.assertEqual(6, metric["prediction_spans"])
        self.assertEqual(6, metric["true_positive"])
        self.assertAlmostEqual(0.75, metric["f1"])

    def test_refuses_existing_output_before_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                evaluate_compact_challenge(
                    root / "reference",
                    root / "manifest",
                    root / "rules",
                    root / "artifacts",
                    output,
                )


if __name__ == "__main__":
    unittest.main()
