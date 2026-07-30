import unittest

from p3_active_round import STRATA, select_active_rows


class P3ActiveRoundTest(unittest.TestCase):
    def test_selects_sixty_unique_juans_in_declared_strata(self):
        rows = [
            {
                "juan": index,
                "jie_index": jie_index,
                "characters": 50,
                "model_rule_symmetric_difference": index % 17 + 1,
                "uncertainty": index / 200,
                "role_count": index % 13 + 1,
                "foreign_count": index % 11 + 1,
            }
            for index in range(1, 201)
            for jie_index in (1, 2)
        ]

        selected = select_active_rows(rows)

        self.assertEqual(60, len(selected))
        self.assertEqual(60, len({row["juan"] for row in selected}))
        for stratum, count in STRATA:
            self.assertEqual(
                count,
                sum(
                    row["selection_stratum"] == stratum
                    for row in selected
                ),
            )


if __name__ == "__main__":
    unittest.main()
