import unittest

from p7_three_seed_select import passes_stability_gate


class Round7ThreeSeedSelectTest(unittest.TestCase):
    def test_requires_both_mean_and_worst_seed_improvement(self):
        baseline = {"mean_f1": 0.96, "min_f1": 0.95}
        self.assertTrue(
            passes_stability_gate(
                baseline, {"mean_f1": 0.97, "min_f1": 0.96}
            )
        )
        self.assertFalse(
            passes_stability_gate(
                baseline, {"mean_f1": 0.97, "min_f1": 0.94}
            )
        )
        self.assertFalse(
            passes_stability_gate(
                baseline, {"mean_f1": 0.95, "min_f1": 0.96}
            )
        )


if __name__ == "__main__":
    unittest.main()
