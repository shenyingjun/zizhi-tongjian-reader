import tempfile
import unittest
from pathlib import Path

from p12_adoption_statistics import (
    bootstrap_intervals,
    compute_adoption_statistics,
)


class AdoptionStatisticsTest(unittest.TestCase):
    def test_bootstrap_is_deterministic(self):
        rows = [
            {"model": (2, 2, 2), "rules": (2, 3, 2)},
            {"model": (1, 2, 1), "rules": (1, 1, 0)},
        ]
        self.assertEqual(
            bootstrap_intervals(rows, seed=7, replicates=100),
            bootstrap_intervals(rows, seed=7, replicates=100),
        )

    def test_refuses_existing_output_before_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                compute_adoption_statistics(
                    root / "reference",
                    root / "comparison",
                    root / "rules",
                    output,
                )


if __name__ == "__main__":
    unittest.main()
