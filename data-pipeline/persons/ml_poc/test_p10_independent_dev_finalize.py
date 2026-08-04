import tempfile
import unittest
from pathlib import Path

from p10_independent_dev_finalize import finalize_independent_dev


class IndependentDevFinalizeTest(unittest.TestCase):
    def test_refuses_existing_output_before_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                finalize_independent_dev(
                    root / "review", root / "state", output
                )


if __name__ == "__main__":
    unittest.main()
