import tempfile
import unittest
from pathlib import Path

from p6_round6_train import run_round6_training


class Round6TrainTest(unittest.TestCase):
    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                run_round6_training(root / "dataset", output)


if __name__ == "__main__":
    unittest.main()
