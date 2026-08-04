import unittest

from production_select import _aggregate, _metric, _vote


class ProductionSelectTest(unittest.TestCase):
    def test_exact_geometry_vote(self):
        first = {(1, 0, 2, "甲乙"), (1, 3, 5, "丙丁")}
        second = {(1, 0, 2, "甲乙"), (1, 3, 4, "丙")}
        third = {(1, 0, 2, "甲乙"), (1, 3, 5, "丙丁")}

        self.assertEqual(first, _vote([first, second, third], 2))
        self.assertEqual({(1, 0, 2, "甲乙")}, _vote(
            [first, second, third], 3
        ))

    def test_aggregate_uses_micro_counts(self):
        rows = [
            _metric({1, 2}, {1, 3}),
            _metric({1}, {1}),
        ]

        self.assertEqual({
            "reference_spans": 3,
            "prediction_spans": 3,
            "true_positive": 2,
            "precision": 2 / 3,
            "recall": 2 / 3,
            "f1": 2 / 3,
        }, _aggregate(rows))


if __name__ == "__main__":
    unittest.main()
