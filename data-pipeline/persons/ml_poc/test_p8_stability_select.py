import unittest

from p7_three_seed_select import passes_stability_gate


class Round8StabilitySelectTest(unittest.TestCase):
    def test_tiny_mean_regression_still_fails_strict_gate(self):
        self.assertFalse(
            passes_stability_gate(
                {"mean_f1": 0.9689453306880186, "min_f1": 0.9607843137254902},
                {"mean_f1": 0.9689438983942078, "min_f1": 0.9658536585365853},
            )
        )


if __name__ == "__main__":
    unittest.main()
