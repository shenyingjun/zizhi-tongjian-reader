from __future__ import annotations

import json
import unittest
from pathlib import Path

import rules as R
import translation_evidence as TE

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

    def test_single_char_fief_rejects_left_title_continuations(self):
        cases = (
            ("赵郡王叡", 1),
            ("夫余王夫台", 1),
            ("高句骊王位宫", 2),
            ("焉耆王鸠尸", 1),
            ("右伊秩訾王呼卢", 3),
            ("归义侯势", 1),
            ("拘弥王兴", 1),
            ("离狐王盛", 1),
            ("左贤王信", 1),
            ("左谷蠡王师子", 2),
            ("靺鞨王武艺", 1),
            ("常道鄕公璜", 2),
        )
        for text, start in cases:
            title_at = next(
                index for index in range(start + 1, len(text))
                if text[index] in R.JUE_HEAD
            )
            given_start = title_at + 1
            tokens = tuple(
                PosToken(
                    char,
                    index,
                    index + 1,
                    "PROPN",
                    (
                        "B-PROPN|Case=Loc|NameType=Geo"
                        if index == 0
                        else "I-PROPN|Case=Loc|NameType=Geo"
                    ),
                    "B" if index == 0 else "I",
                    0.8,
                )
                for index, char in enumerate(text[:title_at])
            ) + (
                PosToken(text[title_at], title_at, title_at + 1, "NOUN", "NOUN", None, 0.9),
            ) + tuple(
                PosToken(
                    char,
                    index,
                    index + 1,
                    "PROPN",
                    ("B-" if index == given_start else "I-")
                    + "PROPN|NameType=Giv",
                    "B" if index == given_start else "I",
                    0.99,
                )
                for index, char in enumerate(text[given_start:], given_start)
            )
            corpus = R.Corpus(set(), {}, set())
            ctx = R.Ctx(
                text,
                set(range(given_start, len(text))),
                corpus,
                1,
                1,
                0,
                904,
                gspans=[(given_start, len(text))],
                tokens=tokens,
            )
            with self.subTest(text=text):
                result = R.rule_jue_name(ctx, start)
                self.assertTrue(
                    result is None or result[0] < start,
                    f"must reject or repair the left boundary, got {result}",
                )

    def test_single_char_fief_rejects_non_title_morphology(self):
        cases = (
            ("㉒侯安都", 0, "PROPN|Case=Loc|NameType=Geo", None, 0.9),
            ("大行王恢", 1, "I-PROPN|Case=Loc|NameType=Geo", "I", 0.8),
        )
        for text, start, tag, bio, score in cases:
            title_at = next(i for i in range(start + 1, len(text)) if text[i] in R.JUE_HEAD)
            given_start = title_at + 1
            tokens = (
                PosToken(text[start], start, start + 1, "PROPN", tag, bio, score),
                PosToken(text[title_at], title_at, title_at + 1, "NOUN", "NOUN", None, 0.9),
                PosToken(
                    text[given_start],
                    given_start,
                    given_start + 1,
                    "PROPN",
                    "PROPN|NameType=Giv",
                    None,
                    0.99,
                ),
            )
            corpus = R.Corpus(set(), {}, set())
            ctx = R.Ctx(
                text,
                {given_start},
                corpus,
                1,
                1,
                0,
                904,
                gspans=[(given_start, given_start + 1)],
                tokens=tokens,
            )
            with self.subTest(text=text):
                self.assertIsNone(R.rule_jue_name(ctx, start))

        text = "刺史，代侯渊。"
        tokens = (
            PosToken("代", 3, 4, "PROPN", "PROPN|Case=Loc|NameType=Geo", None, 0.4),
            PosToken("侯", 4, 5, "NOUN", "NOUN", None, 0.9),
            PosToken("渊", 5, 6, "PROPN", "PROPN|NameType=Giv", None, 0.99),
        )
        ctx = R.Ctx(
            text,
            {5},
            R.Corpus(set(), {}, set()),
            1,
            1,
            0,
            904,
            gspans=[(5, 6)],
            tokens=tokens,
        )
        self.assertIsNone(R.rule_jue_name(ctx, 3))

        text = "徐敬成许王子晋"
        tokens = (
            PosToken("成", 2, 3, "PROPN", "PROPN|NameType=Giv", None, 0.99),
            PosToken("许", 3, 4, "PROPN", "PROPN|Case=Loc|NameType=Geo", None, 0.8),
            PosToken("王", 4, 5, "NOUN", "NOUN", None, 0.9),
            PosToken("子", 5, 6, "PROPN", "B-PROPN|NameType=Giv", "B", 0.99),
            PosToken("晋", 6, 7, "PROPN", "I-PROPN|NameType=Giv", "I", 0.99),
        )
        ctx = R.Ctx(
            text,
            {5, 6},
            R.Corpus(set(), {}, set()),
            1,
            1,
            0,
            904,
            gspans=[(5, 7)],
            tokens=tokens,
        )
        self.assertIsNone(R.rule_jue_name(ctx, 3))

    def test_jue_name_uses_complete_given_span(self):
        text = "焉耆王鸠尸卑那"
        tokens = (
            PosToken("耆", 1, 2, "PROPN", "I-PROPN|Case=Loc|NameType=Geo", "I", 0.8),
            PosToken("王", 2, 3, "NOUN", "NOUN", None, 0.9),
        ) + tuple(
            PosToken(
                char,
                i,
                i + 1,
                "PROPN",
                ("B-" if i == 3 else "I-") + "PROPN|NameType=Giv",
                "B" if i == 3 else "I",
                0.99,
            )
            for i, char in enumerate(text[3:], 3)
        )
        ctx = R.Ctx(
            text,
            set(range(3, 7)),
            R.Corpus(set(), {}, set()),
            1,
            1,
            0,
            904,
            gspans=[(3, 7)],
            tokens=tokens,
        )
        self.assertIsNone(R.rule_jue_name(ctx, 1))

    def test_orphaned_geo_continuation_after_verb_is_rejected(self):
        text = "立耆王鸠尸卑那"
        tokens = (
            PosToken("立", 0, 1, "VERB", "VERB", None, 0.99),
            PosToken("耆", 1, 2, "PROPN", "I-PROPN|Case=Loc|NameType=Geo", "I", 0.8),
            PosToken("王", 2, 3, "NOUN", "NOUN", None, 0.9),
        ) + tuple(
            PosToken(
                char,
                index,
                index + 1,
                "PROPN",
                ("B-" if index == 3 else "I-") + "PROPN|NameType=Giv",
                "B" if index == 3 else "I",
                0.99,
            )
            for index, char in enumerate(text[3:], 3)
        )
        corpus = R.Corpus({"鸠尸卑那"}, {}, set())
        ctx = R.Ctx(
            text,
            set(range(3, 7)),
            corpus,
            1,
            1,
            0,
            904,
            gspans=[(3, 7)],
            tokens=tokens,
        )
        self.assertIsNone(R.rule_jue_name(ctx, 1))
        self.assertEqual(
            (3, 7, "鸠尸卑那", "model_ner_name"),
            R.rule_model_ner_name(ctx, 3),
        )

    def test_continued_fief_repairs_complete_foreign_name(self):
        text = "焉耆王鸠尸卑那奔"
        tokens = (
            PosToken("焉", 0, 1, "PROPN", "B-PROPN|NameType=Sur", "B", 0.7),
            PosToken("耆", 1, 2, "PROPN", "I-PROPN|Case=Loc|NameType=Geo", "I", 0.7),
            PosToken("王", 2, 3, "NOUN", "NOUN", None, 0.9),
        ) + tuple(
            PosToken(
                char,
                i,
                i + 1,
                "PROPN",
                ("B-" if i == 3 else "I-") + "PROPN|NameType=Giv",
                "B" if i == 3 else "I",
                0.9,
            )
            for i, char in enumerate(text[3:7], 3)
        ) + (PosToken("奔", 7, 8, "VERB", "VERB", None, 0.99),)
        ctx = R.Ctx(
            text,
            set(range(3, 7)),
            R.Corpus(set(), {}, set()),
            1,
            1,
            0,
            904,
            gspans=[(3, 7)],
            tokens=tokens,
        )
        self.assertEqual(
            (0, 7, "焉耆王鸠尸卑那", "title_name"),
            R.rule_jue_name(ctx, 1),
        )

    def test_unresolved_long_given_does_not_fall_through(self):
        text = "虢王巨苍黄修"
        tokens = (
            PosToken("虢", 0, 1, "PROPN", "PROPN|Case=Loc|NameType=Geo", None, 0.7),
            PosToken("王", 1, 2, "NOUN", "NOUN", None, 0.9),
            PosToken("巨", 2, 3, "PROPN", "B-PROPN|NameType=Giv", "B", 0.8),
            PosToken("苍", 3, 4, "PROPN", "I-PROPN|NameType=Giv", "I", 0.7),
            PosToken("黄", 4, 5, "PROPN", "I-PROPN|NameType=Giv", "I", 0.3),
            PosToken("修", 5, 6, "VERB", "VERB", None, 0.99),
        )
        ctx = R.Ctx(
            text,
            {2, 3, 4},
            R.Corpus({"巨苍", "巨苍黄"}, {}, set()),
            1,
            1,
            0,
            904,
            gspans=[(2, 5)],
            tokens=tokens,
        )
        self.assertIsNone(R.rule_jue_name(ctx, 0))
        self.assertIsNone(R.rule_known_fullname_pos(ctx, 2))
        self.assertIsNone(R.rule_model_ner_name(ctx, 2))

    def test_high_confidence_long_foreign_given_is_complete(self):
        text = "燕王诺曷钵立"
        tokens = (
            PosToken("燕", 0, 1, "PROPN", "PROPN|Case=Loc|NameType=Nat", None, 0.9),
            PosToken("王", 1, 2, "NOUN", "NOUN", None, 0.9),
            PosToken("诺", 2, 3, "PROPN", "B-PROPN|NameType=Giv", "B", 0.99),
            PosToken("曷", 3, 4, "PROPN", "I-PROPN|NameType=Giv", "I", 0.99),
            PosToken("钵", 4, 5, "PROPN", "I-PROPN|NameType=Giv", "I", 0.99),
            PosToken("立", 5, 6, "VERB", "VERB", None, 0.99),
        )
        ctx = R.Ctx(
            text,
            {2, 3, 4},
            R.Corpus(set(), {}, set()),
            1,
            1,
            0,
            904,
            gspans=[(2, 5)],
            tokens=tokens,
        )
        self.assertEqual(
            (0, 5, "燕王诺曷钵", "title_name"),
            R.rule_jue_name(ctx, 0),
        )

    def test_jue_name_extends_model_ner_given_tail(self):
        text = "谯王元名为"
        tokens = (
            PosToken("谯", 0, 1, "PROPN", "PROPN|Case=Loc|NameType=Geo", None, 0.8),
            PosToken("王", 1, 2, "NOUN", "NOUN", None, 0.9),
            PosToken("元", 2, 3, "PROPN", "PROPN|NameType=Giv", None, 0.93),
            PosToken("名", 3, 4, "NOUN", "NOUN", None, 0.43),
            PosToken("为", 4, 5, "AUX", "AUX|VerbType=Cop", None, 0.99),
        )
        ctx = R.Ctx(
            text,
            {2},
            R.Corpus({"元名"}, {}, set()),
            1,
            1,
            0,
            904,
            gspans=[(2, 3)],
            tokens=tokens,
        )
        self.assertEqual((0, 4, "谯王元名", "title_name"), R.rule_jue_name(ctx, 0))

    def test_valid_single_char_fief_targets_remain(self):
        for fief, given in (
            ("辉", "祚"),
            ("沂", "禋"),
            ("祁", "琪"),
            ("岐", "范"),
            ("曹", "皋"),
            ("𣏌", "上"),
        ):
            text = f"立{fief}王{given}为"
            tokens = (
                PosToken("立", 0, 1, "VERB", "VERB", None, 0.99),
                PosToken(fief, 1, 2, "PROPN", "PROPN|Case=Loc|NameType=Geo", None, 0.8),
                PosToken("王", 2, 3, "NOUN", "NOUN", None, 0.9),
                PosToken(given, 3, 4, "PROPN", "PROPN|NameType=Giv", None, 0.99),
                PosToken("为", 4, 5, "AUX", "AUX|VerbType=Cop", None, 0.99),
            )
            ctx = R.Ctx(
                text,
                {3},
                R.Corpus(set(), {}, set()),
                1,
                1,
                0,
                904,
                gspans=[(3, 4)],
                tokens=tokens,
            )
            with self.subTest(text=text):
                self.assertEqual(text[1:4], R.rule_jue_name(ctx, 1)[2])

    def test_lexical_polity_title_after_office_list_remains(self):
        text = "幷州牧、晋公柳，征西"
        tokens = (
            PosToken("晋", 4, 5, "PROPN", "PROPN|Case=Loc|NameType=Nat", None, 0.99),
            PosToken("公", 5, 6, "NOUN", "NOUN", None, 0.96),
            PosToken("柳", 6, 7, "PROPN", "PROPN|NameType=Giv", None, 0.99),
        )
        ctx = R.Ctx(
            text,
            {6},
            R.Corpus(set(), {}, set()),
            1,
            1,
            0,
            904,
            gspans=[(6, 7)],
            tokens=tokens,
        )
        self.assertEqual((4, 7, "晋公柳", "title_name"), R.rule_jue_name(ctx, 4))

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


class Juan265CorpusBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = Path(__file__).resolve().parents[3]
        text_dir = repo / "web" / "public" / "text"
        document = json.loads(
            (text_dir / "juan_265.json").read_text(encoding="utf-8")
        )
        paragraphs = document["paragraphs"]
        evidence = R.pos_giv.giv_for_juan(
            265,
            paragraphs,
            text_dir / "persons" / "pos_giv",
        )
        corpus = R.load_corpus()
        cls.default_rows = R.detect_juan(
            265,
            paragraphs,
            evidence,
            corpus,
            enabled=R.PRESET_RECALL,
            scan_notes=False,
        )
        translated = TE.load_juan(
            Path(__file__).resolve().parent / "translation" / "evidence",
            265,
            paragraphs,
        )
        cls.assisted_rows = R.detect_juan(
            265,
            paragraphs,
            evidence,
            corpus,
            enabled=R.PRESET_RECALL,
            scan_notes=False,
            translation_evidence=translated,
        )

    def test_real_paragraph_uses_full_title_name_boundaries(self):
        expected = {
            (11, 21, 24, "辉王祚", "jue_name"),
            (11, 66, 69, "昭宣帝", "posthumous_emperor_title"),
        }
        rejected = {
            (11, 22, 24, "王祚"),
            (11, 66, 68, "昭宣"),
        }
        for mode, rows in (
            ("default", self.default_rows),
            ("assisted", self.assisted_rows),
        ):
            geometries = {
                (
                    row["para_id"],
                    row["start"],
                    row["end"],
                    row["surface"],
                    row["rule"],
                )
                for row in rows
                if row.get("field") == "main"
            }
            truncated = {geometry[:4] for geometry in geometries}
            with self.subTest(mode=mode):
                self.assertTrue(expected <= geometries)
                self.assertTrue(rejected.isdisjoint(truncated))

    def test_real_paragraph_keeps_nearby_empress_control(self):
        expected = (11, 40, 42, "皇后", "empress_title")
        for mode, rows in (
            ("default", self.default_rows),
            ("assisted", self.assisted_rows),
        ):
            geometries = {
                (
                    row["para_id"],
                    row["start"],
                    row["end"],
                    row["surface"],
                    row["rule"],
                )
                for row in rows
                if row.get("field") == "main"
            }
            with self.subTest(mode=mode):
                self.assertIn(expected, geometries)


if __name__ == "__main__":
    unittest.main()
