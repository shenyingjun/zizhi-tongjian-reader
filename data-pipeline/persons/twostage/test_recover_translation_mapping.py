from __future__ import annotations

import unittest

import recover_translation_mapping as R
import rules


class RecoverTranslationMappingTest(unittest.TestCase):
    def setUp(self):
        self.jies = R._jies(
            [
                {"id": 0, "main": "周纪一"},
                {"id": 1, "main": "①魏斯、赵籍至。"},
                {"id": 2, "main": "魏斯曰。"},
                {"id": 3, "main": "②韩虔至。"},
                {"id": 4, "main": "③魏斯归。"},
            ]
        )

    def test_aligns_and_maps_to_one_numbered_jie(self):
        original = "魏斯、赵籍至。魏斯曰。"
        pair = R.SourcePair(0, original * 3, "魏斯和赵籍到达。")
        rows = R.map_pair(
            1,
            pair,
            self.jies,
            (R.PersonEntity("魏斯", 0.99),),
            "https://example.test/juan-1",
        )

        self.assertEqual(2, len(rows))
        self.assertTrue(
            all(row["mapping_status"] == "mapped_exact_unique_jie" for row in rows)
        )
        self.assertEqual({1}, {row["repo_jie_index"] for row in rows})

    def test_flags_identity_that_spans_two_aligned_jies(self):
        original = ("魏斯、赵籍至。魏斯曰。韩虔至。魏斯归。") * 3
        pair = R.SourcePair(0, original, "魏斯后来归来。")
        rows = R.map_pair(
            1,
            pair,
            self.jies,
            (R.PersonEntity("魏斯", 0.99),),
            "https://example.test/juan-1",
        )

        self.assertGreaterEqual(len(rows), 2)
        self.assertTrue(
            all(
                row["mapping_status"] == "flagged_multi_jie_identity"
                for row in rows
            )
        )

    def test_translation_only_expansion_is_not_mapped(self):
        pair = R.SourcePair(
            0,
            ("魏斯、赵籍至。魏斯曰。") * 3,
            "司马光说魏斯到了。",
        )
        rows = R.map_pair(
            1,
            pair,
            self.jies,
            (R.PersonEntity("司马光", 0.99),),
            "https://example.test/juan-1",
        )

        self.assertEqual([], rows)

    def test_flags_person_title_prefix(self):
        jies = R._jies(
            [
                {"id": 0, "main": "周纪一"},
                {"id": 1, "main": "①智宣子至。"},
            ]
        )
        pair = R.SourcePair(0, ("智宣子至。") * 3, "智宣子到达。")
        rows = R.map_pair(
            1,
            pair,
            jies,
            (R.PersonEntity("智宣", 0.99),),
            "https://example.test/juan-1",
        )

        self.assertEqual(1, len(rows))
        self.assertEqual(
            "flagged_person_title_continuation",
            rows[0]["mapping_status"],
        )

    def test_finds_separate_translation_page(self):
        source = (
            '<html><body><a href="../translation-28.htm">'
            "\u537728\u8bd1\u6587</a></body></html>"
        ).encode("gb18030")

        url = R._alternate_translation_url(
            "http://example.test/source/juan-28.htm",
            source,
        )

        self.assertEqual(
            "http://example.test/translation-28.htm",
            url,
        )

    def test_recognizes_explicitly_missing_translation(self):
        source = (
            "<html><body>\u3010\u672c\u5377\u8bd1\u6587\u7f3a\u5931\uff0c"
            "\u9644\u201c\u80e1\u4e09\u7701\u97f3\u6ce8\u201d\u4ee5\u4f5c"
            "\u53c2\u8003\u3011</body></html>"
        ).encode("gb18030")

        self.assertTrue(R._translation_is_explicitly_missing(source))

    def test_translation_handle_does_not_turn_sui_adverb_into_person(self):
        text = "毛遂至。遂以毛遂为上客。"
        tokens = (
            rules.pos_giv.PosToken("毛", 0, 1, "PROPN", "PROPN|NameType=Sur", None, 0.99),
            rules.pos_giv.PosToken("遂", 1, 2, "PROPN", "PROPN|NameType=Giv", None, 0.99),
            rules.pos_giv.PosToken("至", 2, 3, "VERB", "VERB", None, 0.99),
            rules.pos_giv.PosToken("遂", 4, 5, "ADV", "ADV", None, 0.99),
            rules.pos_giv.PosToken("以", 5, 6, "VERB", "VERB", None, 0.99),
        )
        context = rules.Ctx(
            text,
            {1},
            rules.Corpus(set(), {}, set()),
            1,
            1,
            1,
            -1,
            gspans=((1, 2),),
            tokens=tokens,
        )
        context.translation_anchors = (
            {
                "start": 0,
                "end": len(text),
                "anchor_start": -1,
                "identity_surface": "毛遂",
                "handle": "遂",
            },
        )
        cards = [
            {
                "start": 0,
                "end": 2,
                "surface": "毛遂",
                "chunk_type": "translation_fullname",
            }
        ]

        hits = rules.detect_anaphora(context, cards)

        self.assertFalse(any(start == 4 and end == 5 for start, end, _, _ in hits))

    def test_translation_handle_stays_out_of_its_full_title_identity(self):
        text = "萧公角将兵。"
        token = rules.pos_giv.PosToken(
            "角", 2, 3, "PROPN", "PROPN|NameType=Giv", None, 0.99
        )
        context = rules.Ctx(
            text,
            {2},
            rules.Corpus(set(), {}, set()),
            1,
            1,
            1,
            -1,
            gspans=((2, 3),),
            tokens=(token,),
        )
        context.translation_anchors = (
            {
                "start": 0,
                "end": len(text),
                "anchor_start": -1,
                "identity_surface": "萧公角",
                "handle": "角",
            },
        )
        cards = [
            {
                "start": 0,
                "end": 3,
                "surface": "萧公角",
                "chunk_type": "translation_fullname",
            }
        ]

        hits = rules.detect_anaphora(context, cards)

        self.assertFalse(any(start == 2 and end == 3 for start, end, _, _ in hits))


if __name__ == "__main__":
    unittest.main()
