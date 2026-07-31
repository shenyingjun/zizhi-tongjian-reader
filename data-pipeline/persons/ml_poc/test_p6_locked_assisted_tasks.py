import unittest

from p6_locked_assisted_tasks import (
    _eligible_jies,
    _manifest_juans,
    evaluation_claim_metadata,
)


class LockedAssistedTasksTest(unittest.TestCase):
    def test_copilot_assistance_does_not_determine_eligibility(self):
        metadata = evaluation_claim_metadata()
        self.assertFalse(metadata["copilot_assistance_is_disqualifying"])
        self.assertFalse(metadata["eligible_for_promotion"])
        self.assertEqual(
            metadata["promotion_eligibility_reason"],
            "predeclared_low_power_diagnostic_without_promotion_gate",
        )

    def test_manifest_juans_collects_all_bound_surfaces(self):
        manifest = {
            "splits": {"train": {"juans": [1, 2]}},
            "selected": [{"juan": 3}],
            "selected_jies": [{"juan": 4}],
            "private_selected_jies": [{"juan": 5}],
        }
        self.assertEqual({1, 2, 3, 4, 5}, _manifest_juans(manifest))

    def test_eligible_jies_excludes_whole_juan(self):
        sources = {
            1: {"paragraphs": [{"id": 1, "main": "①" + "甲" * 30}]},
            2: {"paragraphs": [{"id": 2, "main": "①" + "乙" * 30}]},
        }
        rows = _eligible_jies(sources, {1})
        self.assertEqual({2}, {row["juan"] for row in rows})


if __name__ == "__main__":
    unittest.main()
