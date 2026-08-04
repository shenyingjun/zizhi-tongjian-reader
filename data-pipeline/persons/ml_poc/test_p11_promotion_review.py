import tempfile
import unittest
from pathlib import Path

from p11_promotion_review import prepare_promotion_review


class PromotionReviewTest(unittest.TestCase):
    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_promotion_review(
                    root / "tasks", root / "teachers", output
                )


if __name__ == "__main__":
    unittest.main()
