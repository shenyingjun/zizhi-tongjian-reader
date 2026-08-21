import random
import unittest

from production_precision_revision17_plan import select_formal_reserve


class Revision17PlanTest(unittest.TestCase):
    def _frame(self):
        rows = []
        for index in range(240):
            if index < 4:
                prefix = "可汗"
            elif index < 70:
                prefix = "太后"
            elif index < 140:
                prefix = "字"
            else:
                prefix = "甲"
            text = prefix + "乙" * (25 + index % 11)
            rows.append({
                "juan": index // 8 + 1,
                "jie_index": index % 8 + 1,
                "jie_number": index % 8 + 1,
                "text": text,
                "segments": [],
                "characters": len(text),
            })
        return rows

    def test_reserve_is_deterministic_disjoint_and_stratified(self):
        frame = self._frame()
        counts = {
            "foreign_title": 4,
            "role_appellation": 5,
            "boundary_anaphora": 6,
            "uniform_random": 10,
        }
        first = select_formal_reserve(frame, seed=17, counts=counts)
        second = select_formal_reserve(frame, seed=17, counts=counts)

        self.assertEqual(first, second)
        self.assertEqual(sum(counts.values()), len(first))
        self.assertEqual(len(first), len({
            (row["juan"], row["jie_index"]) for row in first
        }))
        self.assertEqual(
            counts,
            {
                name: sum(row["stratum"] == name for row in first)
                for name in counts
            },
        )
        self.assertTrue(all(
            "可汗" in row["text"]
            for row in first
            if row["stratum"] == "foreign_title"
        ))

    def test_foreign_reserve_must_exhaust_exact_cohort(self):
        with self.assertRaisesRegex(ValueError, "exactly exhaust"):
            select_formal_reserve(
                self._frame(),
                seed=17,
                counts={
                    "foreign_title": 3,
                    "role_appellation": 5,
                    "boundary_anaphora": 6,
                    "uniform_random": 10,
                },
            )

    def test_challenge_draw_samples_geometry_sorted_cohort(self):
        frame = [{
            "juan": index + 1,
            "jie_index": 1,
            "jie_number": 1,
            "text": (
                "可汗" + "乙" * 30
                if index == 0
                else "太后" + "乙" * (20 + index)
            ),
            "segments": [],
            "characters": 32 if index == 0 else 22 + index,
        } for index in range(12)]
        seed = 17
        selected = select_formal_reserve(
            frame,
            seed=seed,
            counts={
                "foreign_title": 1,
                "role_appellation": 3,
                "boundary_anaphora": 0,
                "uniform_random": 0,
            },
        )

        actual = {
            (row["juan"], row["jie_index"])
            for row in selected
            if row["stratum"] == "role_appellation"
        }
        expected = set(random.Random(seed + 1).sample(
            [(index + 1, 1) for index in range(1, 12)],
            3,
        ))
        self.assertEqual(expected, actual)

    def test_challenge_reserve_cannot_reuse_foreign_jies(self):
        frame = [
            {
                "juan": index + 1,
                "jie_index": 1,
                "jie_number": 1,
                "text": ("可汗太后字" if index < 4 else "甲" * 30),
                "segments": [],
                "characters": 30,
            }
            for index in range(40)
        ]
        with self.assertRaisesRegex(ValueError, "role_appellation"):
            select_formal_reserve(
                frame,
                seed=17,
                counts={
                    "foreign_title": 4,
                    "role_appellation": 1,
                    "boundary_anaphora": 0,
                    "uniform_random": 1,
                },
            )


if __name__ == "__main__":
    unittest.main()
