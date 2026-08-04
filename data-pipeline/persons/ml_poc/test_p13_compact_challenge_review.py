import tempfile
import unittest
from pathlib import Path

from p13_compact_challenge_review import (
    _teacher_name,
    prepare_compact_challenge_review,
)


class CompactChallengeReviewTest(unittest.TestCase):
    def test_teacher_name_preserves_jie_identity(self):
        self.assertEqual(
            "assisted_juan_218_jie_030.json",
            _teacher_name("blind_juan_218_jie_030.json"),
        )

    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_compact_challenge_review(
                    root / "tasks", root / "teachers", output
                )


if __name__ == "__main__":
    unittest.main()
