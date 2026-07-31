import unittest

from p6_locked_assisted_tasks import _eligible_jies, _manifest_juans


class LockedAssistedTasksTest(unittest.TestCase):
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
