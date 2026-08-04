import unittest

from p4_next_assisted import select_unique_juan_rows


class NextAssistedTest(unittest.TestCase):
    def test_selects_reproducible_unique_juans(self):
        rows = [
            {"juan": juan, "jie_index": jie}
            for juan in range(1, 81)
            for jie in range(3)
        ]
        first = select_unique_juan_rows(rows, seed=27)
        second = select_unique_juan_rows(rows, seed=27)
        self.assertEqual(first, second)
        self.assertEqual(60, len(first))
        self.assertEqual(60, len({row["juan"] for row in first}))

    def test_rejects_insufficient_unique_juans(self):
        with self.assertRaisesRegex(ValueError, "only 2"):
            select_unique_juan_rows([
                {"juan": 1}, {"juan": 1}, {"juan": 2},
            ], seed=1, count=3)

    def test_juan_selection_is_not_weighted_by_jie_count(self):
        rows = (
            [{"juan": 1, "jie_index": jie} for jie in range(100)]
            + [{"juan": 2, "jie_index": 0}]
            + [{"juan": 3, "jie_index": 0}]
        )
        counts = {1: 0, 2: 0, 3: 0}
        for seed in range(300):
            selected = select_unique_juan_rows(rows, seed=seed, count=1)
            counts[selected[0]["juan"]] += 1
        self.assertLess(max(counts.values()) - min(counts.values()), 40)


if __name__ == "__main__":
    unittest.main()
