import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from p4_round4_training import prepare_round4_training


class Round4TrainingTest(unittest.TestCase):
    def test_rejects_existing_output_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            with (
                patch(
                    "p4_round4_training._git_commit_clean",
                    side_effect=AssertionError("must not run"),
                ),
                self.assertRaises(FileExistsError),
            ):
                prepare_round4_training(root / "base", root / "new", output)


if __name__ == "__main__":
    unittest.main()
