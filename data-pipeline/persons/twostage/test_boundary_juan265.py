from __future__ import annotations

import unittest

import rules as R

PosToken = R.pos_giv.PosToken
GivOffsets = R.pos_giv.GivOffsets


def _rows(text, offsets, spans, tokens, ce=904):
    """Run the Agent-1 tagger over a single paragraph and return occurrences."""
    corpus = R.Corpus(set(), {}, set())
    paras = [{"id": 0, "main": text, "ce_year": ce}]
    giv = {0: GivOffsets(offsets=offsets, spans=spans, tokens=tokens)}
    return R.detect_juan(1, paras, giv, corpus, enabled=R.PRESET_RECALL)


def _surfaces(rows):
    return {row["surface"] for row in rows}


class Juan265BoundaryTest(unittest.TestCase):
    # ── Issue A: 封号(POS-Geo char) + 王 + given → full 辉王祚 span ──────────────
    def test_fief_char_title_given_full_span(self):
        text = "立辉王祚为"  # 立0 辉1 王2 祚3 为4
        tokens = (
            PosToken("立", 0, 1, "VERB", "VERB", None, 0.99),
            PosToken("辉", 1, 2, "PROPN", "PROPN|Case=Loc|NameType=Geo", None, 0.66),
            PosToken("王", 2, 3, "NOUN", "NOUN", None, 0.91),
            PosToken("祚", 3, 4, "PROPN", "PROPN|NameType=Giv", None, 0.99),
            PosToken("为", 4, 5, "AUX", "AUX|VerbType=Cop", None, 0.99),
        )
        rows = _rows(text, offsets={3}, spans=[(3, 4)], tokens=tokens)
        row = next((r for r in rows if r["surface"] == "辉王祚"), None)
        self.assertIsNotNone(row, f"expected 辉王祚, got {_surfaces(rows)}")
        self.assertEqual((1, 4), (row["start"], row["end"]))
        self.assertNotIn("王祚", _surfaces(rows))

    def test_fief_char_title_without_given_not_tagged(self):
        # 辉王 title only (next char is not a given position) must not tag.
        text = "立辉王城"  # 城 is not a given position
        tokens = (
            PosToken("立", 0, 1, "VERB", "VERB", None, 0.99),
            PosToken("辉", 1, 2, "PROPN", "PROPN|Case=Loc|NameType=Geo", None, 0.66),
            PosToken("王", 2, 3, "NOUN", "NOUN", None, 0.91),
            PosToken("城", 3, 4, "NOUN", "NOUN", None, 0.95),
        )
        rows = _rows(text, offsets=set(), spans=[], tokens=tokens)
        self.assertNotIn("辉王城", _surfaces(rows))
        self.assertNotIn("辉王", _surfaces(rows))

    def test_plain_noun_before_wang_does_not_extend(self):
        # A non-Geo char before 王 must NOT be absorbed as a 封号.
        text = "立大王祚为"  # 大 is an ordinary modifier, not a POS-Geo fief char
        tokens = (
            PosToken("立", 0, 1, "VERB", "VERB", None, 0.99),
            PosToken("大", 1, 2, "ADJ", "ADJ", None, 0.95),
            PosToken("王", 2, 3, "PROPN", "PROPN|NameType=Sur", None, 0.9),
            PosToken("祚", 3, 4, "PROPN", "PROPN|NameType=Giv", None, 0.99),
            PosToken("为", 4, 5, "AUX", "AUX|VerbType=Cop", None, 0.99),
        )
        rows = _rows(text, offsets={3}, spans=[(3, 4)], tokens=tokens)
        self.assertNotIn("大王祚", _surfaces(rows))

    def test_fief_morph_char_rejects_geo_continuation(self):
        # 间 in 河间 continues a Geo name; the single-char 封号 branch must decline it
        # (2-char fief 河间王 is multifief territory), so 河间王；韬 is never split into
        # 间王；韬. A standalone Geo char (辉) after a verb is still accepted.
        text = "封河间王韬"  # 封0 河1 间2 王3 韬4
        tokens = (
            PosToken("封", 0, 1, "VERB", "VERB", None, 0.99),
            PosToken("河", 1, 2, "PROPN", "PROPN|Case=Loc|NameType=Geo", None, 0.7),
            PosToken("间", 2, 3, "PROPN", "PROPN|Case=Loc|NameType=Geo", None, 0.7),
            PosToken("王", 3, 4, "NOUN", "NOUN", None, 0.9),
            PosToken("韬", 4, 5, "PROPN", "PROPN|NameType=Giv", None, 0.99),
        )
        corpus = R.Corpus(set(), {}, set())
        ctx = R.Ctx(text, {4}, corpus, 1, 1, 0, 904, tokens=tokens)
        self.assertFalse(R._fief_morph_char(ctx, 2))  # 间 continues Geo name 河

    def test_jue_name_rejects_punctuation_given_position(self):
        # Defensive: a hard boundary erroneously present in the given offsets must
        # not be glued as a given name (辉王；X must never span ；).
        text = "立辉王；韬为"  # 立0 辉1 王2 ；3 韬4 为5
        tokens = (
            PosToken("立", 0, 1, "VERB", "VERB", None, 0.99),
            PosToken("辉", 1, 2, "PROPN", "PROPN|Case=Loc|NameType=Geo", None, 0.66),
            PosToken("王", 2, 3, "NOUN", "NOUN", None, 0.91),
        )
        corpus = R.Corpus(set(), {}, set())
        ctx = R.Ctx(text, {3, 4}, corpus, 1, 1, 0, 904, tokens=tokens)
        self.assertIsNone(R.rule_jue_name(ctx, 1))

    # ── Issue B: 谥号(2 posthumous epithets) + 帝 → full 昭宣帝 span ─────────────
    def test_posthumous_two_char_epithet_plus_di_full_span(self):
        text = "，昭宣帝即位"  # ，0 昭1 宣2 帝3 即4 位5
        tokens = (
            PosToken("，", 0, 1, "NOUN", "NOUN|Case=Tem", None, 0.35),
            PosToken("昭", 1, 2, "PROPN", "B-PROPN|NameType=Giv", "B", 0.56),
            PosToken("宣", 2, 3, "PROPN", "I-PROPN|NameType=Giv", "I", 0.53),
            PosToken("帝", 3, 4, "NOUN", "NOUN", None, 0.99),
            PosToken("即", 4, 5, "VERB", "VERB", None, 0.99),
            PosToken("位", 5, 6, "NOUN", "NOUN", None, 0.99),
        )
        rows = _rows(text, offsets={1, 2}, spans=[(1, 3)], tokens=tokens)
        row = next((r for r in rows if r["surface"] == "昭宣帝"), None)
        self.assertIsNotNone(row, f"expected 昭宣帝, got {_surfaces(rows)}")
        self.assertEqual((1, 4), (row["start"], row["end"]))
        self.assertNotIn("昭宣", _surfaces(rows))

    def test_epithet_plus_di_without_predicate_not_glued(self):
        # No accession/enthronement support: 帝 must not be glued to 昭宣.
        text = "，昭宣帝之"  # 之 is a particle, no emperor predicate
        tokens = (
            PosToken("，", 0, 1, "NOUN", "NOUN|Case=Tem", None, 0.35),
            PosToken("昭", 1, 2, "PROPN", "B-PROPN|NameType=Giv", "B", 0.56),
            PosToken("宣", 2, 3, "PROPN", "I-PROPN|NameType=Giv", "I", 0.53),
            PosToken("帝", 3, 4, "NOUN", "NOUN", None, 0.99),
            PosToken("之", 4, 5, "PART", "PART", None, 0.99),
        )
        rows = _rows(text, offsets={1, 2}, spans=[(1, 3)], tokens=tokens)
        self.assertNotIn("昭宣帝", _surfaces(rows))

    def test_single_char_epithet_plus_di_not_tagged(self):
        # Reused single-char 谥号+帝 (宣帝) is ambiguous and must not glue.
        text = "，宣帝即位"  # ，0 宣1 帝2 即3 位4
        tokens = (
            PosToken("，", 0, 1, "NOUN", "NOUN|Case=Tem", None, 0.35),
            PosToken("宣", 1, 2, "PROPN", "B-PROPN|NameType=Giv", "B", 0.53),
            PosToken("帝", 2, 3, "NOUN", "NOUN", None, 0.99),
            PosToken("即", 3, 4, "VERB", "VERB", None, 0.99),
            PosToken("位", 4, 5, "NOUN", "NOUN", None, 0.99),
        )
        rows = _rows(text, offsets={1}, spans=[(1, 2)], tokens=tokens)
        self.assertNotIn("宣帝", _surfaces(rows))


if __name__ == "__main__":
    unittest.main()
