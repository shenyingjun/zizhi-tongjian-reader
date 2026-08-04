import unittest

from p7_three_seed_select import passes_stability_gate


class Round9StabilitySelectTest(unittest.TestCase):
    def test_fails_when_mean_and_worst_seed_regress(self):
        self.assertFalse(
            passes_stability_gate(
                {"mean_f1": 0.9689453306880186, "min_f1": 0.9607843137254902},
                {"mean_f1": 0.9672389009897024, "min_f1": 0.9556650246305418},
            )
        )


if __name__ == "__main__":
    unittest.main()
