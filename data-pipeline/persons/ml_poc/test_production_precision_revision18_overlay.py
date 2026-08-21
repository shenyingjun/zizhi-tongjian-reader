import unittest

from production_precision_revision18_overlay import construct_overlay


class Revision18OverlayTest(unittest.TestCase):
    def test_detects_semantic_overlap_and_exact_conflict(self):
        selection = [
            {
                "candidate_id": "a",
                "id": "jie",
                "juan": 1,
                "jie_index": 1,
                "para_id": 1,
                "start": 0,
                "end": 2,
                "surface": "甲乙",
            },
            {
                "candidate_id": "b",
                "id": "jie",
                "juan": 1,
                "jie_index": 1,
                "para_id": 1,
                "start": 1,
                "end": 3,
                "surface": "乙丙",
            },
            {
                "candidate_id": "c",
                "id": "jie",
                "juan": 1,
                "jie_index": 1,
                "para_id": 1,
                "start": 0,
                "end": 1,
                "surface": "甲",
            },
        ]
        first = [
            {"candidate_id": "a", "label": "exact_person"},
            {"candidate_id": "b", "label": "exact_person"},
            {"candidate_id": "c", "label": "not_person"},
        ]
        import production_precision_revision18_overlay as module

        old_decisions = module.EXPECTED_DECISIONS
        old_adjudications = module.EXPECTED_ADJUDICATIONS
        module.EXPECTED_DECISIONS = 3
        module.EXPECTED_ADJUDICATIONS = 0
        try:
            result = construct_overlay(selection, first, [], [])
        finally:
            module.EXPECTED_DECISIONS = old_decisions
            module.EXPECTED_ADJUDICATIONS = old_adjudications

        self.assertEqual(2, len(result["conflicts"]))
        self.assertEqual(
            {
                "overlapping_exact_additions",
                "semantic_overlaps_exact_addition",
            },
            {row["type"] for row in result["conflicts"]},
        )

    def test_rejects_empty_wrong_boundary_adjudication(self):
        selection = [{
            "candidate_id": "a",
            "id": "jie",
            "juan": 1,
            "jie_index": 1,
            "para_id": 1,
            "start": 0,
            "end": 1,
            "surface": "甲",
        }]
        first = [{"candidate_id": "a", "label": "wrong_boundary"}]
        adjudications = [{
            "candidate_id": "a",
            "label": "wrong_boundary",
            "targets": [],
        }]
        import production_precision_revision18_overlay as module

        old_decisions = module.EXPECTED_DECISIONS
        old_adjudications = module.EXPECTED_ADJUDICATIONS
        module.EXPECTED_DECISIONS = 1
        module.EXPECTED_ADJUDICATIONS = 1
        try:
            with self.assertRaisesRegex(ValueError, "contradictory"):
                construct_overlay(selection, first, [], adjudications)
        finally:
            module.EXPECTED_DECISIONS = old_decisions
            module.EXPECTED_ADJUDICATIONS = old_adjudications


if __name__ == "__main__":
    unittest.main()
