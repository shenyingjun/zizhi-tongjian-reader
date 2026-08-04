from __future__ import annotations

import unittest

from core import Span
from p1_audit import audit_split


class P1AuditTest(unittest.TestCase):
    def test_scopes_rules_and_classifies_exact_geometry(self):
        examples = [{
            "id": "juan-052-jie-0001",
            "segments": [{
                "para_id": 2,
                "assembled_start": 0,
                "assembled_end": 10,
            }],
        }]
        reference = [
            Span(2, 0, 2, "甲乙"),
            Span(2, 3, 5, "丙丁"),
            Span(2, 6, 8, "戊己"),
        ]
        model = [
            Span(2, 0, 2, "甲乙"),
            Span(2, 3, 4, "丙"),
            Span(2, 6, 8, "戊己"),
            Span(2, 8, 9, "庚"),
        ]
        predictions = [{
            "id": examples[0]["id"],
            "reference_spans": [row.__dict__ for row in reference],
            "prediction_spans": [row.__dict__ for row in model],
        }]
        rules = [
            {**reference[0].__dict__, "field": "main"},
            {**reference[1].__dict__, "field": "main"},
            {
                "para_id": 99,
                "start": 0,
                "end": 1,
                "surface": "外",
                "field": "main",
            },
        ]

        report = audit_split(
            examples,
            predictions,
            rules,
            {2: "甲乙。丙丁。戊己庚。"},
        )

        self.assertEqual(report["gate"]["rule_omissions"], 1)
        self.assertEqual(report["gate"]["recovered_rule_omissions"], 1)
        self.assertEqual(report["gate"]["rule_true_positive_regressions"], 1)
        self.assertEqual(report["delta"]["geometry_replacements"], 1)
        self.assertEqual(report["delta"]["pure_false_positives"], 1)
        self.assertEqual(report["delta"]["pure_misses"], 0)

    def test_rejects_predictions_from_another_split(self):
        with self.assertRaisesRegex(ValueError, "prediction IDs"):
            audit_split(
                [{"id": "expected", "segments": []}],
                [{
                    "id": "unexpected",
                    "reference_spans": [],
                    "prediction_spans": [],
                }],
                [],
                {},
            )

    def test_split_span_is_one_replacement_and_one_pure_addition(self):
        examples = [{
            "id": "split",
            "segments": [{
                "para_id": 1,
                "assembled_start": 0,
                "assembled_end": 3,
            }],
        }]
        predictions = [{
            "id": "split",
            "reference_spans": [{
                "para_id": 1,
                "start": 0,
                "end": 3,
                "surface": "甲乙丙",
            }],
            "prediction_spans": [
                {
                    "para_id": 1,
                    "start": 0,
                    "end": 1,
                    "surface": "甲",
                },
                {
                    "para_id": 1,
                    "start": 1,
                    "end": 3,
                    "surface": "乙丙",
                },
            ],
        }]

        report = audit_split(
            examples,
            predictions,
            [],
            {1: "甲乙丙"},
        )

        self.assertEqual(report["delta"]["geometry_replacements"], 1)
        self.assertEqual(report["delta"]["pure_false_positives"], 1)
        self.assertEqual(report["delta"]["pure_misses"], 0)


if __name__ == "__main__":
    unittest.main()
