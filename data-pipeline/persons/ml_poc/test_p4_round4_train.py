import tempfile
import unittest
from pathlib import Path

from p4_round4_train import run_round4_training


class Round4TrainTest(unittest.TestCase):
    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                run_round4_training(root / "dataset", output)


if __name__ == "__main__":
    unittest.main()
