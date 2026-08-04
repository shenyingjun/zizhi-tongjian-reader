import tempfile
import unittest
from pathlib import Path

from p11_promotion_tasks import prepare_promotion_tasks


class PromotionTasksTest(unittest.TestCase):
    def test_refuses_existing_output_before_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_promotion_tasks(
                    output,
                    root / "historical",
                    root / "dataset",
                    root / "dev",
                    root / "selection",
                )


if __name__ == "__main__":
    unittest.main()
