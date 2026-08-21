from __future__ import annotations

import unittest
from unittest import mock

from production_precision_revision19_conflicts import conflict_components
from production_precision_revision19_freeze import normalize_raw
from production_precision_revision19_overlay import reconcile_overlay


def _decision(candidate_id: str, start: int, end: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate": {
            "id": "juan-001-jie-0001",
            "juan": 1,
            "jie_index": 1,
            "para_id": 7,
            "start": start,
            "end": end,
            "surface": "x" * (end - start),
        },
    }


class Revision19ConflictTests(unittest.TestCase):
    def test_components_close_over_owners_candidates_and_overlap(self) -> None:
        decisions = [
            _decision("wide-owner", 1, 5),
            _decision("short-owner", 3, 5),
            _decision("semantic", 0, 2),
            _decision("overlap-chain", 4, 7),
            _decision("unrelated", 9, 10),
        ]
        exact_owners = [
            {
                "geometry": ["juan-001-jie-0001", 7, 1, 5],
                "candidate_ids": ["wide-owner"],
            },
            {
                "geometry": ["juan-001-jie-0001", 7, 3, 5],
                "candidate_ids": ["short-owner"],
            },
        ]
        conflicts = [
            {
                "type": "overlapping_exact_additions",
                "left": ["juan-001-jie-0001", 7, 1, 5],
                "right": ["juan-001-jie-0001", 7, 3, 5],
            },
            {
                "type": "semantic_overlaps_exact_addition",
                "semantic": ["juan-001-jie-0001", 7, 0, 2],
                "exact": ["juan-001-jie-0001", 7, 1, 5],
            },
        ]

        components = conflict_components(decisions, exact_owners, conflicts)

        self.assertEqual(len(components), 1)
        self.assertEqual(
            components[0]["candidate_ids"],
            ["overlap-chain", "semantic", "short-owner", "wide-owner"],
        )
        self.assertNotIn(
            ("juan-001-jie-0001", 7, 9, 10),
            components[0]["geometries"],
        )

    def test_disconnected_conflicts_remain_separate(self) -> None:
        decisions = [
            _decision("left", 1, 3),
            _decision("right", 6, 8),
        ]
        exact_owners = [
            {
                "geometry": ["juan-001-jie-0001", 7, 1, 3],
                "candidate_ids": ["left"],
            },
            {
                "geometry": ["juan-001-jie-0001", 7, 6, 8],
                "candidate_ids": ["right"],
            },
        ]
        conflicts = [
            {
                "type": "semantic_overlaps_exact_addition",
                "semantic": ["juan-001-jie-0001", 7, 0, 2],
                "exact": ["juan-001-jie-0001", 7, 1, 3],
            },
            {
                "type": "semantic_overlaps_exact_addition",
                "semantic": ["juan-001-jie-0001", 7, 7, 9],
                "exact": ["juan-001-jie-0001", 7, 6, 8],
            },
        ]

        components = conflict_components(decisions, exact_owners, conflicts)

        self.assertEqual(len(components), 2)

    def test_wrong_boundary_owner_keeps_overlapping_candidate_geometry(self) -> None:
        decisions = [
            _decision("wrong-owner", 1, 4),
            _decision("semantic", 0, 2),
        ]
        exact_owners = [{
            "geometry": ["juan-001-jie-0001", 7, 2, 4],
            "candidate_ids": ["wrong-owner"],
        }]
        conflicts = [{
            "type": "semantic_overlaps_exact_addition",
            "semantic": ["juan-001-jie-0001", 7, 0, 2],
            "exact": ["juan-001-jie-0001", 7, 2, 4],
        }]

        components = conflict_components(decisions, exact_owners, conflicts)

        self.assertEqual(len(components), 1)
        self.assertEqual(
            components[0]["candidate_ids"],
            ["semantic", "wrong-owner"],
        )
        self.assertIn(
            ("juan-001-jie-0001", 7, 1, 4),
            components[0]["geometries"],
        )

    def test_freeze_canonicalizes_surface_and_requires_shown_overlap(self) -> None:
        task = {
            "conflict_task_id": "task",
            "jie": {
                "text": "甲乙丙丁",
                "segments": [{
                    "para_id": 7,
                    "assembled_start": 0,
                    "assembled_end": 4,
                }],
            },
            "shown_geometries": [{
                "para_id": 7,
                "start": 1,
                "end": 3,
                "surface": "乙丙",
            }],
        }
        raw = {
            "conflict_task_id": "task",
            "uncertain": False,
            "exact_people": [{
                "para_id": 7,
                "start": 2,
                "end": 4,
                "surface": "wrong",
            }],
            "rationale": "The overlapping full name is exact.",
            "reviewer": "copilot-teacher",
            "model": "gpt-5.6-sol",
        }

        normalized = normalize_raw(raw, task)

        self.assertEqual(normalized["exact_people"][0]["surface"], "丙丁")
        self.assertTrue(
            normalized["exact_people"][0]["surface_corrected"]
        )

        raw["exact_people"][0].update(start=3, end=4)
        with self.assertRaisesRegex(ValueError, "coverage"):
            normalize_raw(raw, task)

    def test_reconciliation_replaces_only_component_geometry(self) -> None:
        decisions = [
            {
                **_decision("wide", 1, 4),
                "label": "exact_person",
                "decision_source": "old",
                "targets": [],
            },
            {
                **_decision("short", 2, 4),
                "label": "not_person",
                "decision_source": "old",
                "targets": [],
            },
            {
                **_decision("untouched", 6, 8),
                "label": "exact_person",
                "decision_source": "old",
                "targets": [],
            },
        ]
        exact = [
            {
                **decisions[0]["candidate"],
                "class": "mined_exact_reference",
                "existence_label": 1,
            },
            {
                **decisions[2]["candidate"],
                "class": "mined_exact_reference",
                "existence_label": 1,
            },
        ]
        owners = [
            {
                "geometry": list((
                    "juan-001-jie-0001", 7, 1, 4
                )),
                "candidate_ids": ["wide"],
            },
            {
                "geometry": list((
                    "juan-001-jie-0001", 7, 6, 8
                )),
                "candidate_ids": ["untouched"],
            },
        ]
        components = [{
            "conflict_task_id": "task",
            "candidate_ids": ["wide", "short"],
            "geometries": [
                ["juan-001-jie-0001", 7, 1, 4],
                ["juan-001-jie-0001", 7, 2, 4],
            ],
        }]
        adjudications = [{
            "conflict_task_id": "task",
            "uncertain": False,
            "exact_people": [{
                "para_id": 7,
                "start": 2,
                "end": 4,
                "surface": "xx",
            }],
        }]
        examples = [{
            "id": "juan-001-jie-0001",
            "juan": 1,
            "jie_index": 1,
        }]

        with (
            mock.patch(
                "production_precision_revision19_overlay.EXPECTED_DECISIONS", 3
            ),
            mock.patch(
                "production_precision_revision19_overlay.EXPECTED_COMPONENTS", 1
            ),
        ):
            result = reconcile_overlay(
                decisions,
                exact,
                owners,
                examples,
                components,
                adjudications,
            )

        by_id = {
            row["candidate_id"]: row for row in result["final_decisions"]
        }
        self.assertEqual(by_id["wide"]["label"], "wrong_boundary")
        self.assertEqual(by_id["short"]["label"], "exact_person")
        self.assertIs(by_id["untouched"], decisions[2])
        self.assertEqual(len(result["raw_additions"]), 1)
        self.assertEqual(len(result["raw_removals"]), 1)
        self.assertEqual(result["conflicts"], [])


if __name__ == "__main__":
    unittest.main()
