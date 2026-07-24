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
        text = "杨行密以其子牙内诸军使渥为宣州观察使"
        start = text.index("渥")
        ctx = make_ctx(text, gset={start}, gspans=[(start, start + 1)])
        result = R.rule_genealogy_given(ctx, text.index("其子"))
        self.assertEqual(result, (start, start + 1, "渥", "gloss_kin"))

    def test_office_with_place_prefix_and_multichar_given(self):
        text = "茂贞遣判官赵锽如西川，为其姪天雄节度使继勋求婚"
        kin_start = text.index("其姪")
        place_start = text.index("天雄")
        name_start = text.index("继勋")
        place = R.pos_giv.PosToken(
            "天雄",
            place_start,
            place_start + 2,
            "PROPN",
            "PROPN|NameType=Geo",
            "B-LOC",
        )
        ctx = make_ctx(
            text,
            gset={name_start, name_start + 1},
            gspans=[(name_start, name_start + 2)],
            tokens=[place],
        )
        result = R.rule_genealogy_given(ctx, kin_start)
        self.assertEqual(
            result,
            (name_start, name_start + 2, "继勋", "gloss_kin"),
        )

    def test_direct_sibling_construction_tags_name_only(self):
        text = "初，马殷弟賨，性沈勇"
        start = text.index("賨")
        ctx = make_ctx(text, gset={start}, gspans=[(start, start + 1)])
        result = R.rule_genealogy_given(ctx, text.index("弟"))
        self.assertEqual(result, (start, start + 1, "賨", "gloss_kin"))

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

    def test_complete_bio_continuation_is_not_truncated(self):
        text = "弟拔弥俄突，"
        tokens = tuple(
            R.pos_giv.PosToken(
                char,
                offset,
                offset + 1,
                "PROPN",
                ("B-" if offset == 1 else "I-") + "PROPN|NameType=Giv",
                "B" if offset == 1 else "I",
                0.99,
            )
            for offset, char in enumerate(text[1:5], 1)
        )
        ctx = make_ctx(
            text,
            gset={1, 2, 3, 4},
            gspans=[(1, 5)],
            tokens=tokens,
        )
        self.assertEqual(
            R.rule_genealogy_given(ctx, 0),
            (1, 5, "拔弥俄突", "gloss_kin"),
        )

    def test_punctuation_mistagged_as_new_bio_does_not_continue_name(self):
        text = "弟渥，为"
        tokens = (
            R.pos_giv.PosToken(
                "渥", 1, 2, "PROPN", "B-PROPN|NameType=Giv", "B", 0.99
            ),
            R.pos_giv.PosToken(
                "，", 2, 3, "PROPN", "B-PROPN|NameType=Giv", "B", 0.51
            ),
        )
        ctx = make_ctx(
            text,
            gset={1, 2},
            gspans=[(1, 2), (2, 3)],
            tokens=tokens,
        )
        self.assertEqual(
            R.rule_genealogy_given(ctx, 0),
            (1, 2, "渥", "gloss_kin"),
        )

    def test_consumed_name_is_a_hard_veto(self):
        text = "弟賨，"
        ctx = make_ctx(text, gset={1}, gspans=[(1, 2)])
        ctx.consumed[1] = True
        self.assertIsNone(R.rule_genealogy_given(ctx, 0))

    def test_compound_surname_is_not_reinterpreted_as_kinship(self):
        text = "从长孙俭求"
        tokens = (
            R.pos_giv.PosToken(
                "长", 1, 2, "PROPN", "B-PROPN|NameType=Sur", "B", 0.98
            ),
            R.pos_giv.PosToken(
                "孙", 2, 3, "PROPN", "I-PROPN|NameType=Sur", "I", 0.99
            ),
            R.pos_giv.PosToken(
                "俭", 3, 4, "PROPN", "PROPN|NameType=Giv", None, 0.99
            ),
        )
        ctx = make_ctx(text, gset={3}, gspans=[(3, 4)], tokens=tokens)
        self.assertIsNone(R.rule_genealogy_given(ctx, 1))

    def test_genealogy_anchor_propagates_only_within_numbered_jie(self):
        anchor_text = "①其子渥为"
        anchor_token = R.pos_giv.PosToken(
            "渥", 3, 4, "PROPN", "PROPN|NameType=Giv", None, 0.99
        )
        anchor_evidence = R.pos_giv.GivOffsets(
            offsets={3},
            spans=[(3, 4)],
            tokens=(anchor_token,),
        )
        repeated_token = R.pos_giv.PosToken(
            "渥", 0, 1, "PROPN", "PROPN|NameType=Giv", None, 0.99
        )
        repeated_evidence = R.pos_giv.GivOffsets(
            offsets={0},
            spans=[(0, 1)],
            tokens=(repeated_token,),
        )
        corpus = R.Corpus(set(), {}, set())

        same_jie = R.detect_juan(
            1,
            [
                {"id": 0, "main": anchor_text, "ce_year": 1},
                {"id": 1, "main": "渥，", "ce_year": 1},
            ],
            {0: anchor_evidence, 1: repeated_evidence},
            corpus,
            enabled=R.PRESET_RECALL,
        )
        self.assertTrue(
            any(
                card["para_id"] == 1
                and card["start"] == 0
                and card["surface"] == "渥"
                and card["rule"] == "jie_anaphora"
                for card in same_jie
            )
        )

        next_jie = R.detect_juan(
            1,
            [
                {"id": 0, "main": anchor_text, "ce_year": 1},
                {"id": 1, "main": "②渥，", "ce_year": 1},
            ],
            {
                0: anchor_evidence,
                1: R.pos_giv.GivOffsets(
                    offsets={1},
                    spans=[(1, 2)],
                    tokens=(repeated_token.shifted(1),),
                ),
            },
            corpus,
            enabled=R.PRESET_RECALL,
        )
        self.assertFalse(
            any(
                card["para_id"] == 1
                and card["start"] == 1
                and card["surface"] == "渥"
                for card in next_jie
            )
        )


if __name__ == "__main__":
    unittest.main()
