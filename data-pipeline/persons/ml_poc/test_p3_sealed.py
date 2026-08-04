import unittest

from p3_sealed import EXCLUDED_JUANS, select_sealed


class P3SealedTest(unittest.TestCase):
    def test_selects_three_random_and_two_distinct_challenges(self):
        sources = {
            juan: {
                "paragraphs": [{
                    "main": (
                        "甲" * juan
                        + "太后" * (juan % 11)
                        + "单于" * (juan % 13)
                    ),
                }],
            }
            for juan in range(1, 80)
        }

        selected = select_sealed(sources, seed=7)

        self.assertEqual(5, len(selected))
        self.assertEqual(3, sum(
            row["role"] == "probability_random" for row in selected
        ))
        self.assertEqual(5, len({row["juan"] for row in selected}))
        self.assertFalse(
            EXCLUDED_JUANS & {row["juan"] for row in selected}
        )
        self.assertEqual(
            {"role_appellation_challenge", "foreign_title_challenge"},
            {
                row["role"] for row in selected
                if row["role"] != "probability_random"
            },
        )


if __name__ == "__main__":
    unittest.main()
