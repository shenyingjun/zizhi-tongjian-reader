import unittest

from p2_assisted import select_expansion


class P2AssistedSelectionTest(unittest.TestCase):
    def test_selects_one_blind_and_four_assisted_without_pilot_juans(self):
        sources = {
            juan: {
                "paragraphs": [{
                    "main": (
                        "帝" * juan
                        + "太子" * (juan % 5)
                        + "单于" * (juan % 7)
                    ),
                }],
            }
            for juan in range(1, 60)
        }

        selected = select_expansion(sources, seed=7)

        self.assertEqual(5, len(selected))
        self.assertEqual(1, sum(mode == "blind_anchor"
                                for mode, _, _ in selected))
        self.assertEqual(4, sum(mode == "assisted"
                                for mode, _, _ in selected))
        self.assertEqual(5, len({juan for _, _, juan in selected}))
        self.assertFalse({13, 27, 52} & {
            juan for _, _, juan in selected
        })


if __name__ == "__main__":
    unittest.main()
