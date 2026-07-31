import unittest

from p7_training_review import _validate_source


class Round7TrainingReviewTest(unittest.TestCase):
    def test_missing_source_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            _validate_source(__import__("pathlib").Path("missing-round7-source"))


if __name__ == "__main__":
    unittest.main()
