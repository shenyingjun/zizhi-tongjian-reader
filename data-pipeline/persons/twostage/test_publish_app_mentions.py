from __future__ import annotations

import json
import unittest
from pathlib import Path

import publish_app_mentions as P


class PublishAppMentionsTest(unittest.TestCase):
    def test_replaces_only_rule_proven_contained_geometry(self):
        paragraphs = [{"id": 11, "main": "立辉王祚为皇太子。昭宣帝即位"}]
        mentions = [
            {
                "pid": 11,
                "source": "main",
                "start": 2,
                "end": 4,
                "surface": "王祚",
                "person_id": "person:1",
            },
            {
                "pid": 11,
                "source": "main",
                "start": 9,
                "end": 11,
                "surface": "昭宣",
                "person_id": "person:1",
            },
        ]
        occurrences = [
            {
                "para_id": 11,
                "field": "main",
                "start": 1,
                "end": 4,
                "surface": "辉王祚",
                "rule": "jue_name",
            },
            {
                "para_id": 11,
                "field": "main",
                "start": 9,
                "end": 12,
                "surface": "昭宣帝",
                "rule": "posthumous_emperor_title",
            },
        ]

        result = P.replace_contained_mentions(occurrences, mentions, paragraphs)

        self.assertEqual(
            [(1, 4, "辉王祚"), (9, 12, "昭宣帝")],
            [(row["start"], row["end"], row["surface"]) for row in result],
        )
        self.assertTrue(all(row["person_id"] == "person:1" for row in result))

    def test_consumer_facing_juan265_has_full_boundaries(self):
        repo = Path(__file__).resolve().parents[3]
        path = (
            repo / "web" / "public" / "text" / "persons-v2" / "mentions"
            / "juan_265.json"
        )
        mentions = json.loads(path.read_text(encoding="utf-8"))["mentions"]
        geometries = {
            (row["pid"], row["start"], row["end"], row["surface"])
            for row in mentions
        }
        self.assertIn((11, 21, 24, "辉王祚"), geometries)
        self.assertIn((11, 66, 69, "昭宣帝"), geometries)
        self.assertNotIn((11, 22, 24, "王祚"), geometries)
        self.assertNotIn((11, 66, 68, "昭宣"), geometries)


if __name__ == "__main__":
    unittest.main()
