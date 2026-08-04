import unittest

from p6_seed_ensemble import vote_predictions


def _row(identity, prediction):
    return {
        "id": identity,
        "reference_spans": [
            {"para_id": 1, "start": 0, "end": 1, "surface": "甲"}
        ],
        "prediction_spans": prediction,
    }


class SeedEnsembleTest(unittest.TestCase):
    def test_votes_on_exact_geometry(self):
        shared = {"para_id": 1, "start": 0, "end": 1, "surface": "甲"}
        unstable = {"para_id": 1, "start": 2, "end": 3, "surface": "乙"}
        predictions = {
            20260727: [_row("example", [shared, unstable])],
            20260728: [_row("example", [shared])],
            20260729: [_row("example", [shared])],
        }
        majority = vote_predictions(predictions, 2)
        unanimous = vote_predictions(predictions, 3)
        self.assertEqual(majority[0]["prediction_spans"], [shared])
        self.assertEqual(unanimous[0]["prediction_spans"], [shared])

    def test_rejects_missing_seed(self):
        with self.assertRaisesRegex(ValueError, "seed inventory"):
            vote_predictions({20260727: []}, 2)


if __name__ == "__main__":
    unittest.main()
