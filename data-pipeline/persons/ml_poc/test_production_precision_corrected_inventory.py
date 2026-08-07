from __future__ import annotations

import unittest

from production_precision_corrected_inventory import _build_corrected_inventory


def _example() -> dict:
    return {
        "id": "juan-001-jie-0001",
        "juan": 1,
        "jie_index": 1,
        "text": "甲乙丙丁戊",
        "segments": [{"para_id": 7, "assembled_start": 0, "assembled_end": 5}],
    }


def _row(start: int, end: int, surface: str, label: int, references: list) -> dict:
    return {
        "id": "juan-001-jie-0001",
        "juan": 1,
        "jie_index": 1,
        "fold": 0,
        "para_id": 7,
        "start": start,
        "end": end,
        "surface": surface,
        "label": label,
        "overlapping_references": references,
        "policy_membership": [],
    }


def _audit(
    candidate_id: str, start: int, end: int, surface: str, label: str
) -> dict:
    return {
        "candidate_id": candidate_id,
        "juan": 1,
        "jie_index": 1,
        "para_id": 7,
        "start": start,
        "end": end,
        "surface": surface,
        "audited_label": label,
    }


class CorrectedInventoryTest(unittest.TestCase):
    def test_replaces_reference_and_relabels_real_rows(self):
        old = {"para_id": 7, "start": 3, "end": 4, "surface": "丁"}
        wrong = _row(1, 3, "乙丙", 0, [])
        exact = _row(0, 1, "甲", 1, [{
            "para_id": 7, "start": 0, "end": 1, "surface": "甲"
        }])
        target = {
            "candidate_id": "wrong",
            "decision": "targets",
            "wrong_candidate": {
                "para_id": 7, "start": 1, "end": 3, "surface": "乙丙"
            },
            "targets": [{
                "para_id": 7, "start": 1, "end": 4, "surface": "乙丙丁"
            }],
        }
        result = _build_corrected_inventory(
            [_example()],
            [exact, wrong, _row(3, 4, "丁", 1, [old])],
            [],
            [_audit("wrong", 1, 3, "乙丙", "wrong_boundary")],
            [target],
            [],
            [],
        )
        self.assertEqual(
            [(0, 1), (1, 4)],
            [(row["start"], row["end"]) for row in result["references"]],
        )
        corrected_wrong = [
            row for row in result["existence"] if row["surface"] == "乙丙"
        ][0]
        self.assertEqual(1, corrected_wrong["label"])
        self.assertEqual(
            "乙丙丁", corrected_wrong["overlapping_references"][0]["surface"]
        )
        self.assertEqual(1, result["counts"]["added_rank_pairs"])
        self.assertEqual(1, result["counts"]["removed_references"])

    def test_semantic_negative_cannot_remove_reference(self):
        reference = {"para_id": 7, "start": 1, "end": 3, "surface": "乙丙"}
        row = _row(2, 4, "丙丁", 1, [reference])
        with self.assertRaisesRegex(ValueError, "semantic negative"):
            _build_corrected_inventory(
                [_example()],
                [row],
                [],
                [_audit("negative", 2, 4, "丙丁", "not_person")],
                [],
                [],
                [],
            )

    def test_conflicting_corrected_targets_stop(self):
        left = _row(0, 2, "甲乙", 0, [])
        right = _row(2, 4, "丙丁", 0, [])
        audits = [
            _audit("left", 0, 2, "甲乙", "exact_person"),
            _audit("right", 2, 4, "丙丁", "wrong_boundary"),
        ]
        targets = [{
            "candidate_id": "right",
            "decision": "targets",
            "wrong_candidate": {
                "para_id": 7, "start": 2, "end": 4, "surface": "丙丁"
            },
            "targets": [{
                "para_id": 7, "start": 1, "end": 3, "surface": "乙丙"
            }],
        }]
        with self.assertRaisesRegex(ValueError, "conflict component coverage"):
            _build_corrected_inventory(
                [_example()], [left, right], [], audits, targets, [], []
            )

    def test_easy_negative_cannot_overlap_corrected_reference(self):
        exact = _row(0, 2, "甲乙", 0, [])
        easy = _row(1, 3, "乙丙", 0, [])
        with self.assertRaisesRegex(ValueError, "easy negative"):
            _build_corrected_inventory(
                [_example()],
                [exact],
                [],
                [_audit("exact", 0, 2, "甲乙", "exact_person")],
                [],
                [],
                [easy],
            )

    def test_conflict_overlay_mechanically_revises_all_members(self):
        left = _row(0, 2, "甲乙", 0, [])
        right = _row(1, 3, "乙丙", 0, [])
        audits = [
            _audit("left", 0, 2, "甲乙", "exact_person"),
            _audit("right", 1, 3, "乙丙", "exact_person"),
        ]
        conflict = {
            "task_id": "component",
            "juan": 1,
            "jie_index": 1,
            "component_candidate_ids": ["left", "right"],
            "included_candidates": audits,
            "decision": "targets",
            "targets": [{
                "para_id": 7,
                "start": 0,
                "end": 2,
                "surface": "甲乙",
            }],
        }
        result = _build_corrected_inventory(
            [_example()], [right, left], [], list(reversed(audits)), [],
            [conflict], []
        )
        forward = _build_corrected_inventory(
            [_example()], [left, right], [], audits, [], [conflict], []
        )
        self.assertEqual(result, forward)
        labels = {
            row["candidate_id"]: row["audited_label"]
            for row in result["corrections"]
        }
        self.assertEqual(
            {"left": "exact_person", "right": "wrong_boundary"}, labels
        )
        self.assertEqual(
            [(0, 2)],
            [(row["start"], row["end"]) for row in result["references"]],
        )
        self.assertEqual(
            {"exact_person": 1, "wrong_boundary": 1},
            result["counts"]["revised_audit_labels"],
        )

    def test_conflict_target_external_overlap_stops(self):
        rows = [
            _row(0, 2, "甲乙", 0, []),
            _row(1, 3, "乙丙", 0, []),
            _row(3, 5, "丁戊", 0, []),
        ]
        audits = [
            _audit("left", 0, 2, "甲乙", "exact_person"),
            _audit("middle", 1, 3, "乙丙", "exact_person"),
            _audit("external", 3, 5, "丁戊", "not_person"),
        ]
        conflict = {
            "task_id": "component",
            "juan": 1,
            "jie_index": 1,
            "component_candidate_ids": ["left", "middle"],
            "included_candidates": audits[:2],
            "decision": "targets",
            "targets": [{
                "para_id": 7,
                "start": 1,
                "end": 4,
                "surface": "乙丙丁",
            }],
        }
        with self.assertRaisesRegex(ValueError, "external overlap"):
            _build_corrected_inventory(
                [_example()], rows, [], audits, [], [conflict], []
            )

    def test_uncertain_conflict_stops_downstream(self):
        left = _row(0, 2, "甲乙", 0, [])
        right = _row(1, 3, "乙丙", 0, [])
        audits = [
            _audit("left", 0, 2, "甲乙", "exact_person"),
            _audit("right", 1, 3, "乙丙", "exact_person"),
        ]
        conflict = {
            "task_id": "component",
            "juan": 1,
            "jie_index": 1,
            "component_candidate_ids": ["left", "right"],
            "included_candidates": audits,
            "decision": "uncertain",
            "targets": [],
        }
        with self.assertRaisesRegex(ValueError, "uncertain"):
            _build_corrected_inventory(
                [_example()], [left, right], [], audits, [], [conflict], []
            )

    def test_semantic_negative_cannot_overlap_later_addition(self):
        negative = _row(1, 2, "乙", 0, [])
        exact = _row(0, 3, "甲乙丙", 0, [])
        with self.assertRaisesRegex(ValueError, "final references"):
            _build_corrected_inventory(
                [_example()],
                [negative, exact],
                [],
                [
                    _audit("negative", 1, 2, "乙", "not_person"),
                    _audit("exact", 0, 3, "甲乙丙", "exact_person"),
                ],
                [],
                [],
                [],
            )


if __name__ == "__main__":
    unittest.main()
