"""Focused tests for office-prefixed relations in appositive genealogy glosses."""
from __future__ import annotations

import unittest

import rules as R


class GlossOfficeRelationTest(unittest.TestCase):
    def test_office_prefixed_person_is_minted_across_controlled_relations(self):
        for relation in (
            "\u5b50", "\u5973", "\u5144", "\u5f1f", "\u59ea", "\u4f84",
            "\u5b59", "\u7236", "\u6bcd", "\u53d4", "\u4f2f", "\u4ece\u5f1f",
            "\u65cf\u59ea", "\u66fe\u5b59",
        ):
            with self.subTest(relation=relation):
                text = (
                    "\u8d77\u5c45\u90ce\u82cf\u6977\uff0c\u793c\u90e8\u5c1a\u4e66"
                    f"\u5faa\u4e4b{relation}\u4e5f"
                )
                self.assertEqual(
                    R.detect_gloss(text),
                    [
                        (10, 11, "\u5faa", "gloss_rel"),
                        (3, 5, "\u82cf\u6977", "gloss_subj"),
                    ],
                )

    def test_unbound_office_name_fragment_is_not_minted(self):
        text = "\u793c\u90e8\u5c1a\u4e66\u5faa\u4e4b\u5b50\u4e5f"

        self.assertEqual(R.detect_gloss(text), [])

    def test_unknown_relation_is_not_minted(self):
        text = "\u8d77\u5c45\u90ce\u82cf\u6977\uff0c\u793c\u90e8\u5c1a\u4e66\u5faa\u4e4b\u53cb\u4e5f"

        self.assertEqual(R.detect_gloss(text), [])

    def test_non_office_prefix_is_not_minted(self):
        text = "\u8d77\u5c45\u90ce\u82cf\u6977\uff0c\u793c\u90e8\u6587\u58eb\u5faa\u4e4b\u5b50\u4e5f"

        self.assertEqual(R.detect_gloss(text), [])

    def test_bare_relation_vocabulary_is_unchanged(self):
        self.assertEqual(R.detect_gloss("\u5faa\u4e4b\u5973\u4e5f"), [])

    def test_compound_surname_is_not_reinterpreted_as_an_office(self):
        text = "\u53f8\u9a6c\u61ff\uff0c\u793c\u90e8\u5c1a\u4e66\u5faa\u4e4b\u5b50\u4e5f"

        self.assertEqual(R.detect_gloss(text), [])


if __name__ == "__main__":
    unittest.main()
