from __future__ import annotations

import unittest

from production_precision_revision14_inventory import _apply_overlay


class Revision14InventoryContractTest(unittest.TestCase):
    def test_overlay_is_one_hop_against_immutable_prior_references(self):
        old = {
            ("jie", 1, 0, 2): {
                "id": "jie",
                "juan": 1,
                "jie_index": 1,
                "para_id": 1,
                "start": 0,
                "end": 2,
                "surface": "甲乙",
                "source": "prior",
            },
        }
        selection = {
            "task": {
                "task_id": "task",
                "id": "jie",
                "juan": 1,
                "jie_index": 1,
                "para_id": 1,
                "start": 1,
                "end": 2,
                "surface": "乙",
            },
        }
        decisions = [{
            "task_id": "task",
            "candidate_id": "candidate",
            "initial_label": "exact_person",
            "final_label": "exact_person",
            "resolution": "unchanged",
        }]

        references, corrections = _apply_overlay(
            old, decisions, selection, {}
        )

        self.assertEqual({("jie", 1, 1, 2)}, set(references))
        self.assertEqual(1, len(corrections[0]["removed_references"]))

    def test_not_person_removes_overlapping_prior_reference(self):
        old = {
            ("jie", 1, 0, 1): {
                "id": "jie",
                "juan": 1,
                "jie_index": 1,
                "para_id": 1,
                "start": 0,
                "end": 1,
                "surface": "甲",
                "source": "prior",
            },
        }
        selection = {
            "task": {
                "task_id": "task",
                "id": "jie",
                "juan": 1,
                "jie_index": 1,
                "para_id": 1,
                "start": 0,
                "end": 1,
                "surface": "甲",
            },
        }
        decisions = [{
            "task_id": "task",
            "candidate_id": "candidate",
            "initial_label": "exact_person",
            "final_label": "not_person",
            "resolution": "unchanged",
        }]

        references, _ = _apply_overlay(old, decisions, selection, {})

        self.assertEqual({}, references)


if __name__ == "__main__":
    unittest.main()
