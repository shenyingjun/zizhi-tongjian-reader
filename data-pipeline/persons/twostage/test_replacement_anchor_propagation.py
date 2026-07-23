from __future__ import annotations

import unittest

import rules as R


PosToken = R.pos_giv.PosToken


def _ctx(text, spans, tokens, ner):
    return R.Ctx(
        text,
        {offset for start, end in spans for offset in range(start, end)},
        R.Corpus(set(ner), {}, set()),
        1,
        1,
        0,
        904,
        gspans=spans,
        tokens=tokens,
    )


class ReplacementAnchorPropagationTest(unittest.TestCase):
    def test_genealogy_replacement_remains_exact_title_anchor(self):
        text = "其弟且鞮侯为单于。且鞮侯单于初立"
        tokens = (
            PosToken("且", 2, 3, "PROPN", "B-PROPN|NameType=Giv", "B", 0.8),
            PosToken("鞮", 3, 4, "PROPN", "I-PROPN|NameType=Giv", "I", 0.8),
            PosToken("侯", 4, 5, "PROPN", "I-PROPN|NameType=Giv", "I", 0.8),
            PosToken("且", 9, 10, "PROPN", "B-PROPN|NameType=Sur", "B", 0.4),
            PosToken("鞮", 10, 11, "PROPN", "I-PROPN|Case=Loc|NameType=Geo", "I", 0.4),
            PosToken("侯", 11, 12, "NOUN", "NOUN", None, 0.9),
        )
        ctx = _ctx(text, [(2, 5)], tokens, {"且鞮侯"})
        ctx.consumed[2:5] = [True] * 3
        cards = [{
            "start": 2,
            "end": 5,
            "surface": "且鞮侯",
            "chunk_type": "gloss_kin",
            "rule": "genealogy_given",
        }]
        self.assertEqual(
            [(9, 12, "且鞮侯", "local_exact_title")],
            R.detect_local_exact_title(ctx, cards),
        )

    def test_only_validated_genealogy_card_can_authorize_title(self):
        text = "其弟且鞮侯为单于。且鞮侯单于初立"
        tokens = (
            PosToken("且", 2, 3, "PROPN", "B-PROPN|NameType=Giv", "B", 0.8),
            PosToken("鞮", 3, 4, "PROPN", "I-PROPN|NameType=Giv", "I", 0.8),
            PosToken("侯", 4, 5, "PROPN", "I-PROPN|NameType=Giv", "I", 0.8),
            PosToken("且", 9, 10, "PROPN", "B-PROPN|NameType=Sur", "B", 0.4),
            PosToken("鞮", 10, 11, "PROPN", "I-PROPN|Case=Loc|NameType=Geo", "I", 0.4),
            PosToken("侯", 11, 12, "NOUN", "NOUN", None, 0.9),
        )
        for rule in ("gloss_geneal", "unrelated_rule"):
            ctx = _ctx(text, [(2, 5)], tokens, {"且鞮侯"})
            ctx.consumed[2:5] = [True] * 3
            cards = [{
                "start": 2,
                "end": 5,
                "surface": "且鞮侯",
                "chunk_type": "gloss_kin",
                "rule": rule,
            }]
            with self.subTest(rule=rule):
                self.assertEqual([], R.detect_local_exact_title(ctx, cards))

    def test_validated_jue_replacement_seeds_clean_given_handle(self):
        for title, handle in (
            ("句龙王", "吾斯"),
            ("高句骊王", "位宫"),
            ("郇王", "素节"),
            ("雍王", "守礼"),
        ):
            first = title + handle
            text = first + "反；" + handle + "等复"
            given_start = len(title)
            target_start = len(first) + 2
            tokens = (
                PosToken(
                    handle[0],
                    given_start,
                    given_start + 1,
                    "PROPN",
                    "B-PROPN|NameType=Giv",
                    "B",
                    0.99,
                ),
                PosToken(
                    handle[1],
                    given_start + 1,
                    given_start + 2,
                    "PROPN",
                    "I-PROPN|NameType=Giv",
                    "I",
                    0.99,
                ),
                PosToken(
                    handle[0],
                    target_start,
                    target_start + 1,
                    "VERB",
                    "VERB",
                    None,
                    0.9,
                ),
                PosToken(
                    handle[1],
                    target_start + 1,
                    target_start + 2,
                    "NOUN",
                    "NOUN",
                    None,
                    0.9,
                ),
            )
            ctx = _ctx(text, [(given_start, given_start + 2)], tokens, {handle})
            ctx.consumed[:len(first)] = [True] * len(first)
            cards = [{
                "start": 0,
                "end": len(first),
                "surface": first,
                "chunk_type": "title_name",
                "rule": "jue_name",
            }]
            with self.subTest(first=first):
                self.assertEqual(
                    [(target_start, target_start + 2, handle, "local_exact_surface")],
                    R.detect_local_exact_surface(ctx, cards),
                )

    def test_other_title_name_cards_do_not_seed_handles(self):
        text = "将军守礼反；守礼等复"
        tokens = (
            PosToken("守", 2, 3, "PROPN", "B-PROPN|NameType=Giv", "B", 0.99),
            PosToken("礼", 3, 4, "PROPN", "I-PROPN|NameType=Giv", "I", 0.99),
            PosToken("守", 6, 7, "VERB", "VERB", None, 0.9),
            PosToken("礼", 7, 8, "NOUN", "NOUN", None, 0.9),
        )
        ctx = _ctx(text, [(2, 4)], tokens, {"守礼"})
        ctx.consumed[:4] = [True] * 4
        cards = [{
            "start": 0,
            "end": 4,
            "surface": "将军守礼",
            "chunk_type": "title_name",
            "rule": "office_name",
        }]
        self.assertEqual([], R.detect_local_exact_surface(ctx, cards))


if __name__ == "__main__":
    unittest.main()
