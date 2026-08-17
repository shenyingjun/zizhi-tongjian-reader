from __future__ import annotations

import unittest

import numpy as np

from production_precision_corrected_error_audit import (
    LABELS,
    TASK_STATUS,
    _opaque_id,
    _select_error_rows,
    _validate_task,
)


def _row(
    *,
    geometry: tuple[str, int, int, int],
    score: float,
    class_label: str,
    stratum: str,
    inventory_source: str,
) -> dict:
    identity, para_id, start, end = geometry
    return {
        "id": identity,
        "juan": 2,
        "jie_index": 3,
        "para_id": para_id,
        "start": start,
        "end": end,
        "surface": "甲",
        "class_label": class_label,
        "stratum": stratum,
        "inventory_source": inventory_source,
        "fold": 0,
        "oof_exact_probability": score,
    }


class CorrectedErrorAuditContractTest(unittest.TestCase):
    def test_selector_uses_float32_threshold_semantics(self):
        reference = ("juan-002-jie-0003", 1, 2, 3)
        semantic = ("juan-002-jie-0003", 1, 4, 5)
        rounds_to_half = float(
            np.nextafter(np.float64(0.5), np.float64(0.0))
        )
        rows = [
            _row(
                geometry=reference,
                score=rounds_to_half,
                class_label="exact_person",
                stratum="exact_person",
                inventory_source="corrected_reference",
            ),
            _row(
                geometry=semantic,
                score=rounds_to_half,
                class_label="not_person",
                stratum="real_not_person",
                inventory_source="corrected_semantic",
            ),
        ]

        selected = _select_error_rows(
            rows,
            {reference},
            {semantic},
            {0: {2}},
            enforce_production_counts=False,
        )

        self.assertEqual(["semantic_false_positive"], [
            row["selection_side"] for row in selected
        ])
        self.assertEqual(0.5, selected[0]["oof_exact_probability_float32"])

    def test_selector_rejects_non_heldout_juan(self):
        geometry = ("juan-002-jie-0003", 1, 2, 3)
        with self.assertRaisesRegex(ValueError, "not held out"):
            _select_error_rows(
                [
                    _row(
                        geometry=geometry,
                        score=0.1,
                        class_label="exact_person",
                        stratum="exact_person",
                        inventory_source="corrected_reference",
                    )
                ],
                {geometry},
                set(),
                {0: {7}},
                enforce_production_counts=False,
            )

    def test_opaque_ids_bind_salt_domain_and_geometry(self):
        row = {
            "id": "juan-002-jie-0003",
            "para_id": 1,
            "start": 2,
            "end": 3,
        }
        salt = bytes(range(32))

        candidate = _opaque_id(salt, "revision-14-candidate", row)

        self.assertEqual(24, len(candidate))
        self.assertNotEqual(
            candidate, _opaque_id(salt, "revision-14-task", row)
        )
        self.assertNotEqual(
            candidate, _opaque_id(bytes(reversed(salt)), "revision-14-candidate", row)
        )

    def test_public_task_schema_has_one_candidate_and_no_selection_fields(self):
        task = {
            "schema_version": 1,
            "status": TASK_STATUS,
            "phase": "revision-14-blind-corrected-oof-error-audit",
            "task_id": "a" * 24,
            "review_scope": "current-numbered-jie-only",
            "protocol": {
                "decision": "judge",
                "evidence": "this jie",
                "independence": "one first judgment",
            },
            "jie": {
                "text": "甲乙",
                "segments": [{
                    "para_id": 1,
                    "assembled_start": 0,
                    "assembled_end": 2,
                }],
            },
            "candidate": {
                "candidate_id": "b" * 24,
                "para_id": 1,
                "start": 0,
                "end": 1,
                "surface": "甲",
            },
            "allowed_labels": list(LABELS),
        }

        _validate_task(task)
        task["selection_side"] = "missed_exact"
        with self.assertRaisesRegex(ValueError, "task fields differ"):
            _validate_task(task)


if __name__ == "__main__":
    unittest.main()
