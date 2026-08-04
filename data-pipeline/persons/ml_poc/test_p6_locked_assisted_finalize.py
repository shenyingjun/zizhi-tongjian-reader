import unittest

from core import Span
from p6_locked_assisted_finalize import _metrics, _roles


class LockedAssistedFinalizeTest(unittest.TestCase):
    def test_roles_require_declared_strata_counts(self):
        rows = []
        index = 1
        for role, count in (
            ("probability_random", 12),
            ("role_appellation_challenge", 4),
            ("foreign_title_challenge", 4),
        ):
            for _ in range(count):
                rows.append({
                    "juan": index,
                    "jie_index": index,
                    "role": role,
                })
                index += 1
        roles = _roles({"private_selected_jies": rows})
        self.assertEqual(len(roles), 20)
        self.assertEqual(
            list(roles.values()).count("probability_random"), 12
        )

    def test_metrics_aggregate_exact_counts_before_scores(self):
        first = Span(1, 0, 1, "甲")
        second = Span(2, 0, 1, "乙")
        extra = Span(2, 1, 2, "丙")
        metrics = _metrics([
            ([first], [first]),
            ([second], [second, extra]),
        ])
        self.assertEqual(metrics["reference_spans"], 2)
        self.assertEqual(metrics["prediction_spans"], 3)
        self.assertEqual(metrics["true_positive"], 2)
        self.assertEqual(metrics["precision"], 2 / 3)
        self.assertEqual(metrics["recall"], 1.0)
