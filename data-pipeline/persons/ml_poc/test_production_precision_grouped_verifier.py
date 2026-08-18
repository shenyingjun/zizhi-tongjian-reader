from __future__ import annotations

import math
import unittest

from production_precision_grouped_data import _overlaps
from production_precision_grouped_verifier import (
    _confidence_key,
    _resolve,
    _resolve_group,
)


def _candidate(
    start: int,
    end: int,
    existence: float,
    rank: float,
    surface: str,
    *,
    support: int = 3,
) -> dict:
    return {
        "id": "juan-001-jie-0001",
        "para_id": 4,
        "start": start,
        "end": end,
        "surface": surface,
        "existence_score": existence,
        "rank_logit": rank,
        "support_count": support,
        "seed_confidences": {"a": 0.9, "b": 0.8, "c": 0.7},
        "intrinsic_hard_vetoes": [],
    }


class OccurrenceLabelTest(unittest.TestCase):
    def test_half_open_overlap(self):
        self.assertTrue(_overlaps((1, 0, 2), (1, 1, 3)))
        self.assertFalse(_overlaps((1, 0, 2), (1, 2, 3)))
        self.assertFalse(_overlaps((1, 0, 2), (2, 0, 2)))


class OrdinalResolverTest(unittest.TestCase):
    def test_exact_rank_wins_over_fragments(self):
        selected = _resolve_group([
            _candidate(0, 3, 0.8, 2.0, "人物名"),
            _candidate(0, 1, 0.9, 1.0, "人"),
            _candidate(1, 3, 0.9, 0.5, "物名"),
        ], 0.50)
        self.assertEqual(
            [(row["start"], row["end"]) for row in selected], [(0, 3)]
        )

    def test_two_adjacent_exact_spans_beat_lower_rank_merge(self):
        selected = _resolve_group([
            _candidate(0, 2, 0.9, 3.0, "甲乙"),
            _candidate(2, 4, 0.9, 2.0, "丙丁"),
            _candidate(0, 4, 0.9, 1.0, "甲乙丙丁"),
        ], 0.50)
        self.assertEqual(
            [(row["start"], row["end"]) for row in selected],
            [(0, 2), (2, 4)],
        )

    def test_threshold_veto_and_isolated_candidate(self):
        vetoed = _candidate(0, 2, 0.99, 4.0, "甲乙")
        vetoed["intrinsic_hard_vetoes"] = ["numeric_punctuation_or_symbol"]
        weak = _candidate(2, 4, 0.49, 3.0, "丙丁")
        isolated = _candidate(4, 6, 0.50, -10.0, "戊己")
        self.assertEqual(
            _resolve([vetoed, weak, isolated], 0.50),
            {("juan-001-jie-0001", 4, 4, 6, "戊己")},
        )

    def test_nonfinite_rank_rejected(self):
        with self.assertRaises(ValueError):
            _resolve_group([
                _candidate(0, 2, 0.9, math.nan, "甲乙")
            ], 0.50)

    def test_ties_use_support_then_quantized_confidence(self):
        lower_support = _candidate(0, 2, 0.9, 1.0, "甲乙", support=2)
        higher_support = _candidate(0, 3, 0.9, 1.0, "甲乙丙", support=3)
        selected = _resolve_group([lower_support, higher_support], 0.50)
        self.assertEqual(selected[0]["surface"], "甲乙丙")
        self.assertEqual(_confidence_key(lower_support), 2_400_000)


class FeatureContractTest(unittest.TestCase):
    def test_grouped_features_exclude_generator_metadata(self):
        from production_precision_verifier import NUMERIC_SIZE

        self.assertEqual(NUMERIC_SIZE, 1 + 6 + 6 + 2)


if __name__ == "__main__":
    unittest.main()
