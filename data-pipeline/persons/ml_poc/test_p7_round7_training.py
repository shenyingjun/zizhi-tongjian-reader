import tempfile
import unittest
from pathlib import Path

from p7_round7_training import prepare_round7_training


class Round7TrainingDatasetTest(unittest.TestCase):
    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_round7_training(
                    root / "base",
                    root / "freeze",
                    output,
                )


if __name__ == "__main__":
    unittest.main()
