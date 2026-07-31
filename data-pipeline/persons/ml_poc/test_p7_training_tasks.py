import unittest

from p7_training_tasks import _eligible_jies, select_unique_juan_rows


class Round7TrainingTasksTest(unittest.TestCase):
    def test_selects_reproducible_unique_juans(self):
        rows = [
            {"juan": juan, "jie_index": jie}
            for juan in range(1, 61)
            for jie in range(2)
        ]
        first = select_unique_juan_rows(rows, seed=27, count=40)
        second = select_unique_juan_rows(rows, seed=27, count=40)
        self.assertEqual(first, second)
        self.assertEqual(40, len({row["juan"] for row in first}))

    def test_excludes_whole_juan_before_jie_selection(self):
        sources = {
            1: {"paragraphs": [{"id": 1, "main": "①" + "甲" * 30}]},
            2: {"paragraphs": [{"id": 2, "main": "①" + "乙" * 30}]},
        }
        rows = _eligible_jies(sources, {1})
        self.assertEqual({2}, {row["juan"] for row in rows})


if __name__ == "__main__":
    unittest.main()
