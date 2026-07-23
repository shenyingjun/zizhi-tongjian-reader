"""Focused tests for the kinship + office-nested given-name rule.

These call ``rule_genealogy_given`` directly. Personhood comes only from the local
genealogy structure and POS-given morphology in the same jie; an office/title sitting
between the kinship term and the name is skipped, never tagged.
"""
from __future__ import annotations

import unittest

import rules as R


EMPTY_CORPUS = R.Corpus(set(), {}, set())


def make_ctx(text, gset, gspans=(), tokens=(), consumed_prefix=0):
    ctx = R.Ctx(
        text,
        set(gset),
        EMPTY_CORPUS,
        juan=1,
        sec=1,
        para_id=0,
        ce=None,
        gspans=tuple(gspans),
        tokens=tuple(tokens),
    )
    for offset in range(consumed_prefix):
        ctx.consumed[offset] = True
    return ctx


class GenealogyOfficeTest(unittest.TestCase):
    def test_office_between_kin_term_and_pos_given_name(self):
        # 以其子[牙内诸军使]渥为 -> tag 渥 (office is skipped)
        text = "\u4ee5\u5176\u5b50\u7259\u5185\u8bf8\u519b\u4f7f\u6e25\u4e3a"
        ctx = make_ctx(text, gset={8}, gspans=[(8, 9)])
        result = R.rule_genealogy_given(ctx, 1)
        self.assertEqual(result, (8, 9, "\u6e25", "gloss_kin"))

    def test_office_with_place_prefix_and_multichar_given(self):
        # 为其姪[天雄节度使]继勋求 -> tag 继勋; 天雄 is a place token inside the office,
        # and 求 is an accepted right boundary.
        text = (
            "\u4e3a\u5176\u59ea\u5929\u96c4\u8282\u5ea6\u4f7f\u7ee7\u52cb\u6c42"
        )
        place = R.pos_giv.PosToken(
            "\u5929\u96c4", 3, 5, "PROPN", "PROPN|NameType=Geo", "B-LOC"
        )
        ctx = make_ctx(text, gset={8, 9}, gspans=[(8, 10)], tokens=[place])
        result = R.rule_genealogy_given(ctx, 1)
        self.assertEqual(result, (8, 10, "\u7ee7\u52cb", "gloss_kin"))

    def test_bare_nephew_term_directly_before_name(self):
        # 为姪邺娶 -> tag 邺 (姪 is a newly recognised kinship term)
        text = "\u4e3a\u59ea\u90ba\u5a36"
        ctx = make_ctx(text, gset={2}, gspans=[(2, 3)])
        result = R.rule_genealogy_given(ctx, 1)
        self.assertEqual(result, (2, 3, "\u90ba", "gloss_kin"))

    def test_office_without_trailing_name_is_rejected(self):
        # 其子[牙内诸军使]卒 -> no POS-given name after the office, so nothing is minted
        text = "\u5176\u5b50\u7259\u5185\u8bf8\u519b\u4f7f\u5352"
        ctx = make_ctx(text, gset=set())
        self.assertIsNone(R.rule_genealogy_given(ctx, 0))

    def test_gap_that_is_not_an_office_is_not_skipped(self):
        # 其子甚好渥为 -> 好 is not an office suffix, so 渥 must NOT be minted
        text = "\u5176\u5b50\u751a\u597d\u6e25\u4e3a"
        ctx = make_ctx(text, gset={4}, gspans=[(4, 5)])
        self.assertIsNone(R.rule_genealogy_given(ctx, 0))

    def test_office_span_may_not_swallow_an_interior_name(self):
        # 其子牙渥使温为 -> a POS-given (渥) sits inside the would-be office, so the
        # office is not skipped and 温 is not minted.
        text = "\u5176\u5b50\u7259\u6e25\u4f7f\u6e29\u4e3a"
        ctx = make_ctx(text, gset={3, 5}, gspans=[(3, 4), (5, 6)])
        self.assertIsNone(R.rule_genealogy_given(ctx, 0))

    def test_direct_pos_given_after_kin_term_still_works(self):
        # 其子仁果， -> unchanged direct behaviour (regression guard)
        text = "\u5176\u5b50\u4ec1\u679c\uff0c"
        ctx = make_ctx(text, gset={2, 3}, gspans=[(2, 4)])
        result = R.rule_genealogy_given(ctx, 0)
        self.assertEqual(result, (2, 4, "\u4ec1\u679c", "gloss_kin"))


if __name__ == "__main__":
    unittest.main()
