import tempfile
import unittest
from pathlib import Path

from p7_training_finalize import finalize_training_labels


class Round7TrainingFinalizeTest(unittest.TestCase):
    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                finalize_training_labels(
                    root / "review",
                    root / "state",
                    output,
                )


if __name__ == "__main__":
    unittest.main()
