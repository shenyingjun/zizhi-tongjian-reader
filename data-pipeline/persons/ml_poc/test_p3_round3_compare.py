import unittest

from p3_round3_compare import compare_predictions


def span(start: int, end: int, surface: str) -> dict:
    return {
        "para_id": 1,
        "start": start,
        "end": end,
        "surface": surface,
    }


class Round3CompareTest(unittest.TestCase):
    def test_attributes_true_and_false_prediction_changes(self):
        reference = [span(0, 2, "曹操"), span(3, 5, "刘备")]
        old = [{
            "id": "example",
            "reference_spans": reference,
            "prediction_spans": [
                span(0, 1, "曹"),
                span(3, 5, "刘备"),
                span(6, 8, "孙权"),
            ],
        }]
        new = [{
            "id": "example",
            "reference_spans": reference,
            "prediction_spans": [
                span(0, 2, "曹操"),
                span(3, 5, "刘备"),
                span(8, 10, "周瑜"),
            ],
        }]

        result = compare_predictions(old, new)

        self.assertEqual({
            "new_true_positives": 1,
            "lost_true_positives": 0,
            "added_false_positives": 1,
            "removed_false_positives": 2,
        }, result["attribution"])
        self.assertEqual(
            1, result["prediction_geometry"]["geometry_replacements"]
        )
        self.assertEqual(
            2, result["reference_length_recall"]["2"]["round3_hits"]
        )


if __name__ == "__main__":
    unittest.main()
