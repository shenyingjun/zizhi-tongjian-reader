import tempfile
import unittest
from pathlib import Path

from p10_independent_dev_tasks import prepare_independent_dev_tasks


class IndependentDevTasksTest(unittest.TestCase):
    def test_refuses_existing_output_before_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_independent_dev_tasks(
                    output, root / "historical", root / "dataset"
                )


if __name__ == "__main__":
    unittest.main()
