import tempfile
import unittest
from pathlib import Path

from p13_compact_challenge_tasks import (
    prepare_compact_challenge_tasks,
    select_challenges,
)


class CompactChallengeTasksTest(unittest.TestCase):
    def test_selection_has_disjoint_strata(self):
        rows = [
            {
                "juan": index + 1,
                "jie_index": 1,
                "text": ("太后" if index < 45 else "可汗") * (index + 1),
                "characters": 100 + index,
            }
            for index in range(90)
        ]
        selected = select_challenges(rows, 7)
        identities = {
            (row["juan"], row["jie_index"]) for row in selected
        }
        self.assertEqual(8, len(selected))
        self.assertEqual(8, len(identities))

    def test_refuses_existing_output_before_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_compact_challenge_tasks(
                    output,
                    root / "tasks",
                    root / "reference",
                    root / "adoption",
                )


if __name__ == "__main__":
    unittest.main()
