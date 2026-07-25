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
        jies = R._jies(
            [
                {"id": 0, "main": "周纪一"},
                {"id": 1, "main": "①魏斯至。"},
                {"id": 2, "main": "②魏斯至。"},
            ]
        )
        pair = R.SourcePair(0, "魏斯至。魏斯至。", "魏斯到达。")
        rows = R.map_pair(
            1,
            pair,
            jies,
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

    def test_maps_repeated_identity_to_translation_aligned_paragraph(self):
        pair = R.SourcePair(
            0,
            "魏斯、赵籍至。魏斯曰。韩虔至。魏斯归。",
            "韩虔到达。魏斯归。",
        )
        rows = R.map_pair(
            1,
            pair,
            self.jies,
            (R.PersonEntity("魏斯", 0.99),),
            "https://example.test/juan-1",
        )

        eligible = [
            row
            for row in rows
            if row["mapping_status"] == "mapped_exact_paragraph"
        ]
        self.assertEqual([(4, "魏斯")], [
            (row["repo_para_id"], row["original_surface"])
            for row in eligible
        ])
        self.assertTrue(
            all(
                row["mapping_status"] == "flagged_multi_jie_identity"
                for row in rows
                if row not in eligible
            )
        )

    def test_longer_matching_paragraph_beats_contained_decoy(self):
        jies = R._jies(
            [
                {"id": 0, "main": "周纪一"},
                {"id": 1, "main": "①魏斯守城，赵籍相助。"},
                {"id": 2, "main": "②魏斯守城。"},
            ]
        )
        pair = R.SourcePair(
            0,
            "魏斯守城，赵籍助之。",
            "魏斯守城，赵籍帮助他。",
        )

        rows = R.map_pair(
            1,
            pair,
            jies,
            (R.PersonEntity("魏斯", 0.99),),
            "https://example.test/juan-1",
        )

        self.assertEqual(
            [(1, "mapped_exact_paragraph")],
            [
                (row["repo_para_id"], row["mapping_status"])
                for row in rows
                if not row["mapping_status"].startswith("flagged_")
            ],
        )

    def test_independent_translation_sentences_can_map_separate_jies(self):
        jies = R._jies(
            [
                {"id": 0, "main": "周纪一"},
                {"id": 1, "main": "①魏斯守城。"},
                {"id": 2, "main": "②魏斯归国。"},
            ]
        )
        pair = R.SourcePair(
            0,
            "魏斯守城。魏斯归国。",
            "魏斯守城。魏斯归国。",
        )

        rows = R.map_pair(
            1,
            pair,
            jies,
            (R.PersonEntity("魏斯", 0.99),),
            "https://example.test/juan-1",
        )

        self.assertEqual(
            [(1, 1), (2, 2)],
            [
                (row["repo_para_id"], row["repo_jie_index"])
                for row in rows
                if row["mapping_status"] == "mapped_exact_paragraph"
            ],
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

    def test_maps_translation_expansion_to_later_same_jie_handle(self):
        jies = R._jies(
            [
                {"id": 0, "main": "晋纪一"},
                {"id": 1, "main": "①慕容垂至。"},
                {"id": 2, "main": "垂悉众攻邺，邺人拒之。"},
            ]
        )
        pair = R.SourcePair(
            0,
            "垂悉众攻邺，邺人拒之。",
            "慕容垂率领全部军队进攻邺城。",
        )

        rows = R.map_pair(
            90,
            pair,
            jies,
            (R.PersonEntity("慕容垂", 0.99),),
            "https://example.test/juan-90",
        )

        self.assertEqual(1, len(rows))
        self.assertEqual("垂", rows[0]["original_surface"])
        self.assertEqual("慕容垂", rows[0]["identity_surface"])
        self.assertEqual("anchor_given", rows[0]["transfer_mode"])
        self.assertEqual(
            "mapped_translation_coreference_paragraph",
            rows[0]["mapping_status"],
        )

    def test_does_not_map_handle_before_same_jie_identity_anchor(self):
        jies = R._jies(
            [
                {"id": 0, "main": "晋纪一"},
                {"id": 1, "main": "①垂悉众攻邺，邺人拒之。"},
                {"id": 2, "main": "慕容垂至。"},
            ]
        )
        pair = R.SourcePair(
            0,
            "垂悉众攻邺，邺人拒之。",
            "慕容垂率领全部军队进攻邺城。",
        )

        rows = R.map_pair(
            90,
            pair,
            jies,
            (R.PersonEntity("慕容垂", 0.99),),
            "https://example.test/juan-90",
        )

        self.assertEqual([], rows)

    def test_does_not_map_handle_without_local_source_context(self):
        jies = R._jies(
            [
                {"id": 0, "main": "晋纪一"},
                {"id": 1, "main": "①慕容垂至。"},
                {"id": 2, "main": "垂悉众攻邺，邺人拒之。"},
                {"id": 3, "main": "垂还中山，众皆贺之。"},
            ]
        )
        pair = R.SourcePair(
            0,
            "垂悉众攻邺，邺人拒之。",
            "慕容垂率领全部军队进攻邺城。",
        )

        rows = R.map_pair(
            90,
            pair,
            jies,
            (R.PersonEntity("慕容垂", 0.99),),
            "https://example.test/juan-90",
        )

        self.assertEqual(["垂"], [row["original_surface"] for row in rows])

    def test_prefers_longest_coreference_handle(self):
        jies = R._jies(
            [
                {"id": 0, "main": "宋纪一"},
                {"id": 1, "main": "①刘谦之至。"},
                {"id": 2, "main": "谦之讨贼，既而诛之。"},
            ]
        )
        pair = R.SourcePair(
            0,
            "谦之讨贼，既而诛之。",
            "刘谦之讨伐盗贼，随后杀了他。",
        )

        rows = R.map_pair(
            118,
            pair,
            jies,
            (R.PersonEntity("刘谦之", 0.99),),
            "https://example.test/juan-118",
        )

        self.assertEqual(["谦之"], [row["original_surface"] for row in rows])

    def test_maps_unique_translation_expansion_without_original_fullname(self):
        jies = R._jies(
            [
                {"id": 0, "main": "晋纪一"},
                {"id": 1, "main": "①垂悉众攻邺，邺人拒之。"},
            ]
        )
        pair = R.SourcePair(
            0,
            "垂悉众攻邺，邺人拒之。",
            "慕容垂率领全部军队进攻邺城。",
        )

        rows = R.map_pair(
            90,
            pair,
            jies,
            (R.PersonEntity("慕容垂", 0.99),),
            "https://example.test/juan-90",
        )

        self.assertEqual(["垂"], [row["original_surface"] for row in rows])
        self.assertEqual(
            "mapped_translation_expansion_paragraph",
            rows[0]["mapping_status"],
        )

    def test_translation_expansion_is_scoped_to_aligned_source_sentence(self):
        jies = R._jies(
            [
                {"id": 0, "main": "周纪一"},
                {"id": 1, "main": "①武攻城。"},
                {"id": 2, "main": "武悉众守邺。"},
            ]
        )
        pair = R.SourcePair(
            0,
            "武攻城。武悉众守邺。",
            "前军攻打城池。慕容武率领全部军队守卫邺城。",
        )

        rows = R.map_pair(
            90,
            pair,
            jies,
            (R.PersonEntity("慕容武", 0.99),),
            "https://example.test/juan-90",
        )

        self.assertEqual(
            [(2, "武")],
            [(row["repo_para_id"], row["original_surface"]) for row in rows],
        )

    def test_rejects_suffix_from_different_sentence_in_same_pair(self):
        jies = R._jies(
            [
                {"id": 0, "main": "周纪一"},
                {"id": 1, "main": "①士民不附，则汤、武不能胜。"},
                {"id": 2, "main": "孙、吴用之，无敌天下。"},
            ]
        )
        pair = R.SourcePair(
            0,
            (
                "士民不亲附，则汤、武不能以必胜也。"
                "孙、吴用之，无敌于天下。"
            ),
            (
                "百姓不归附，即使商汤、周武王也不能取胜。"
                "孙武、吴起采用这种战术，天下无敌。"
            ),
        )

        rows = R.map_pair(
            6,
            pair,
            jies,
            (R.PersonEntity("孙武", 0.99),),
            "https://example.test/juan-6",
        )

        self.assertEqual([], rows)

    def test_rejects_translation_name_with_speech_verb_suffix(self):
        jies = R._jies(
            [
                {"id": 0, "main": "唐纪一"},
                {"id": 1, "main": "①臣光曰：不可。"},
            ]
        )
        pair = R.SourcePair(
            0,
            "臣光曰：不可。",
            "臣司马光曰：不可以这样做。",
        )

        rows = R.map_pair(
            241,
            pair,
            jies,
            (R.PersonEntity("司马光曰", 0.99),),
            "https://example.test/juan-241",
        )

        self.assertEqual([], rows)

    def test_sentence_alignment_ignores_hu_sansheng_notes(self):
        pair = R.SourcePair(
            0,
            (
                "垂悉众攻邺，"
                "〔〖胡三省注〗邺，城名。〕"
                "邺人拒之。"
            ),
            "慕容垂率领全部军队进攻邺城，邺城的人抵抗他。",
        )

        aligned = R._aligned_source_sentences(pair, "慕容垂", "垂")

        self.assertEqual(("垂悉众攻邺邺人拒之",), aligned)

    def test_source_context_does_not_cross_sentence_boundary(self):
        text = "齐王以太史敫之女为后，生太子建。太史敫曰。"
        start = text.index("建")

        self.assertTrue(
            R._occurrence_matches_source_pair(
                text,
                start,
                start + 1,
                "齐王以太史敫之女为后生太子建",
            )
        )

    def test_rejects_ambiguous_translation_handle_owner(self):
        jies = R._jies(
            [
                {"id": 0, "main": "晋纪一"},
                {"id": 1, "main": "①垂悉众攻邺，邺人拒之。"},
            ]
        )
        pair = R.SourcePair(
            0,
            "垂悉众攻邺，邺人拒之。",
            "慕容垂与段垂率领军队进攻邺城。",
        )

        rows = R.map_pair(
            90,
            pair,
            jies,
            (
                R.PersonEntity("慕容垂", 0.99),
                R.PersonEntity("段垂", 0.99),
            ),
            "https://example.test/juan-90",
        )

        self.assertEqual([], rows)

    def test_drops_coreference_that_conflicts_with_existing_identity(self):
        exact = {
            "repo_para_id": 5,
            "repo_jie_index": 1,
            "original_start": 10,
            "original_end": 11,
            "original_surface": "汤",
            "identity_surface": "成汤",
            "mapping_status": "mapped_exact_unique_jie",
            "transfer_mode": "exact",
        }
        coreference = {
            **exact,
            "identity_surface": "商汤",
            "mapping_status": "mapped_translation_expansion_unique_jie",
            "transfer_mode": "anchor_given",
        }

        rows = R._drop_ambiguous_coreferences([exact, coreference])

        self.assertEqual([exact], rows)

    def test_drops_same_jie_coreference_with_another_handle_owner(self):
        exact = {
            "repo_para_id": 5,
            "repo_jie_index": 1,
            "original_start": 2,
            "original_end": 4,
            "original_surface": "成汤",
            "identity_surface": "成汤",
            "mapping_status": "mapped_exact_unique_jie",
            "transfer_mode": "exact",
        }
        coreference = {
            **exact,
            "repo_para_id": 6,
            "original_start": 8,
            "original_end": 9,
            "original_surface": "汤",
            "identity_surface": "商汤",
            "mapping_status": "mapped_translation_expansion_unique_jie",
            "transfer_mode": "anchor_given",
        }

        rows = R._drop_ambiguous_coreferences([exact, coreference])

        self.assertEqual([exact], rows)

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

    def test_translation_handle_stays_out_of_longer_known_name(self):
        text = "孔休至。休先作檄。"
        token = rules.pos_giv.PosToken(
            "休", 4, 5, "PROPN", "PROPN|NameType=Giv", None, 0.99
        )
        context = rules.Ctx(
            text,
            {1, 4},
            rules.Corpus({"孔休", "休先"}, {}, set()),
            1,
            1,
            1,
            -1,
            gspans=((1, 2), (4, 5)),
            tokens=(token,),
        )
        context.translation_anchors = (
            {
                "start": 0,
                "end": len(text),
                "anchor_start": 0,
                "identity_surface": "孔休",
                "handle": "休",
                "translation_coreference": True,
            },
        )
        cards = [
            {
                "start": 0,
                "end": 2,
                "surface": "孔休",
                "chunk_type": "translation_fullname",
            }
        ]

        hits = rules.detect_anaphora(context, cards)

        self.assertFalse(any(start == 4 and end == 5 for start, end, _, _ in hits))

    def test_direct_coreference_handle_stays_out_of_longer_known_name(self):
        text = "孔休至。休先作檄。"
        context = rules.Ctx(
            text,
            {1, 4},
            rules.Corpus({"孔休", "休先"}, {}, set()),
            1,
            1,
            1,
            -1,
            gspans=((1, 2), (4, 5)),
            tokens=(
                rules.pos_giv.PosToken(
                    "休", 4, 5, "PROPN", "PROPN|NameType=Giv", None, 0.99
                ),
            ),
        )
        context.translation_mentions = {
            4: {
                "start": 4,
                "end": 5,
                "surface": "休",
                "identity_surface": "孔休",
                "strict_identity": True,
                "strict_local_owner": True,
                "translation_coreference": True,
            }
        }

        hit = rules.rule_translation_given(context, 4)

        self.assertIsNone(hit)

    def test_sentence_aligned_two_char_handle_needs_no_extra_syntax(self):
        text = "男成素得众心。"
        context = rules.Ctx(
            text,
            {0, 1},
            rules.Corpus(set(), {}, set()),
            112,
            1,
            1,
            -1,
            gspans=((0, 2),),
            tokens=(
                rules.pos_giv.PosToken(
                    "男成", 0, 2, "PROPN", "PROPN|NameType=Prs", None, 0.99
                ),
            ),
        )
        context.translation_mentions = {
            0: {
                "start": 0,
                "end": 2,
                "surface": "男成",
                "identity_surface": "沮渠男成",
                "strict_identity": True,
                "strict_local_owner": True,
                "translation_coreference": True,
            }
        }

        hit = rules.rule_translation_given(context, 0)

        self.assertEqual((0, 2, "男成", "translation_anaphora"), hit)

    def test_sentence_aligned_two_char_handle_rejects_title_suffix(self):
        text = "渠侯谋曰。"
        context = rules.Ctx(
            text,
            {0, 1},
            rules.Corpus(set(), {}, set()),
            22,
            1,
            1,
            -1,
            gspans=((0, 2),),
            tokens=(
                rules.pos_giv.PosToken(
                    "渠侯", 0, 2, "PROPN", "PROPN|NameType=Prs", None, 0.99
                ),
            ),
        )
        context.translation_mentions = {
            0: {
                "start": 0,
                "end": 2,
                "surface": "渠侯",
                "identity_surface": "煇渠侯",
                "strict_identity": True,
                "strict_local_owner": True,
                "translation_coreference": True,
            }
        }

        hit = rules.rule_translation_given(context, 0)

        self.assertIsNone(hit)

    def test_sentence_aligned_two_char_handle_stops_before_predicate(self):
        text = "丰举杖击地曰。"
        context = rules.Ctx(
            text,
            {0},
            rules.Corpus(set(), {}, set()),
            63,
            1,
            30,
            200,
            gspans=((0, 1),),
            tokens=(
                rules.pos_giv.PosToken(
                    "丰", 0, 1, "PROPN", "PROPN|NameType=Giv", None, 0.997
                ),
                rules.pos_giv.PosToken(
                    "举", 1, 2, "VERB", "VERB", None, 0.998
                ),
                rules.pos_giv.PosToken(
                    "杖", 2, 3, "NOUN", "NOUN", None, 0.992
                ),
            ),
        )
        context.translation_mentions = {
            0: {
                "start": 0,
                "end": 2,
                "surface": "丰举",
                "identity_surface": "田丰举",
                "strict_identity": True,
                "strict_local_owner": True,
                "translation_coreference": True,
                "mapping_status": "mapped_translation_expansion_unique_jie",
            }
        }

        hit = rules.rule_translation_given(context, 0)

        self.assertIsNone(hit)

    def test_sentence_aligned_two_char_handle_rejects_geography(self):
        text = "武陵兄弟。"
        context = rules.Ctx(
            text,
            {0, 1},
            rules.Corpus(set(), {}, set()),
            89,
            1,
            1,
            -1,
            gspans=((0, 2),),
            tokens=(
                rules.pos_giv.PosToken(
                    "武陵", 0, 2, "PROPN", "PROPN|NameType=Geo", None, 0.99
                ),
            ),
        )
        context.translation_mentions = {
            0: {
                "start": 0,
                "end": 2,
                "surface": "武陵",
                "identity_surface": "刘武陵",
                "strict_identity": True,
                "strict_local_owner": True,
                "translation_coreference": True,
            }
        }

        hit = rules.rule_translation_given(context, 0)

        self.assertIsNone(hit)

    def test_anchored_two_char_handle_overrides_local_geo_tag(self):
        text = "汉宾心未可知。"
        context = rules.Ctx(
            text,
            {0, 1},
            rules.Corpus(set(), {}, set()),
            261,
            1,
            1,
            -1,
            gspans=((0, 2),),
            tokens=(
                rules.pos_giv.PosToken(
                    "汉", 0, 1, "PROPN", "PROPN|NameType=Nat", None, 0.99
                ),
                rules.pos_giv.PosToken(
                    "宾", 1, 2, "NOUN", "NOUN", None, 0.99
                ),
            ),
        )
        context.translation_mentions = {
            0: {
                "start": 0,
                "end": 2,
                "surface": "汉宾",
                "identity_surface": "陈汉宾",
                "strict_identity": True,
                "strict_local_owner": True,
                "translation_coreference": True,
                "mapping_status": "mapped_translation_coreference_unique_jie",
            }
        }

        hit = rules.rule_translation_given(context, 0)

        self.assertEqual((0, 2, "汉宾", "translation_anaphora"), hit)

    def test_paragraph_mapped_handle_overrides_local_geo_tag(self):
        text = "汉宾心未可知。"
        context = rules.Ctx(
            text,
            {0, 1},
            rules.Corpus(set(), {}, set()),
            261,
            1,
            1,
            -1,
            gspans=((0, 2),),
            tokens=(
                rules.pos_giv.PosToken(
                    "汉", 0, 1, "PROPN", "PROPN|NameType=Nat", None, 0.99
                ),
                rules.pos_giv.PosToken(
                    "宾", 1, 2, "NOUN", "NOUN", None, 0.99
                ),
            ),
        )
        context.translation_mentions = {
            0: {
                "start": 0,
                "end": 2,
                "surface": "汉宾",
                "identity_surface": "陈汉宾",
                "strict_identity": True,
                "strict_local_owner": True,
                "translation_coreference": True,
                "mapping_status": "mapped_translation_coreference_paragraph",
            }
        }

        hit = rules.rule_translation_given(context, 0)

        self.assertEqual((0, 2, "汉宾", "translation_anaphora"), hit)

    def test_translation_handle_rejects_office_title_continuation(self):
        text = "武衞将军王鉴"
        context = rules.Ctx(
            text,
            {0},
            rules.Corpus(set(), {}, set()),
            101,
            1,
            1,
            -1,
            gspans=((0, 1),),
            tokens=(
                rules.pos_giv.PosToken(
                    "武", 0, 1, "PROPN", "PROPN|NameType=Giv", None, 0.99
                ),
            ),
        )
        context.translation_mentions = {
            0: {
                "start": 0,
                "end": 1,
                "surface": "武",
                "identity_surface": "苻武",
                "strict_identity": True,
                "strict_local_owner": True,
                "translation_coreference": True,
                "mapping_status": "mapped_translation_expansion_paragraph",
            }
        }

        self.assertIsNone(rules.rule_translation_given(context, 0))

    def test_translation_fullname_rejects_bare_compound_surname(self):
        text = "出连辅政至"
        context = rules.Ctx(
            text,
            {0, 1},
            rules.Corpus({"出连"}, {}, set()),
            120,
            1,
            1,
            -1,
            gspans=((0, 2), (2, 4)),
            tokens=(
                rules.pos_giv.PosToken(
                    "出", 0, 1, "PROPN", "B-PROPN|NameType=Giv", "B", 0.99
                ),
                rules.pos_giv.PosToken(
                    "连", 1, 2, "PROPN", "I-PROPN|NameType=Giv", "I", 0.99
                ),
            ),
        )
        context.translation_fullnames = {
            0: {
                "start": 0,
                "end": 2,
                "surface": "出连",
                "identity_surface": "出连",
            }
        }

        self.assertIsNone(rules.rule_translation_fullname(context, 0))

    def test_same_jie_trusted_surface_overrides_one_local_geo_tag(self):
        text = "叔陵至。叔陵兵可千人。"
        context = rules.Ctx(
            text,
            {0, 1, 4, 5},
            rules.Corpus(set(), {}, set()),
            175,
            1,
            1,
            -1,
            gspans=((0, 2), (4, 6)),
            tokens=(
                rules.pos_giv.PosToken(
                    "叔陵", 0, 2, "PROPN", "PROPN|NameType=Prs", None, 0.99
                ),
                rules.pos_giv.PosToken(
                    "叔陵", 4, 6, "PROPN", "PROPN|NameType=Geo", None, 0.99
                ),
            ),
        )
        candidates = [
            {
                "start": start,
                "end": end,
                "surface": "叔陵",
                "identity_surface": "陈叔陵",
                "strict_identity": True,
                "strict_local_owner": True,
                "translation_coreference": True,
                "mapping_status": "mapped_translation_expansion_unique_jie",
            }
            for start, end in ((0, 2), (4, 6))
        ]
        rules._mark_translation_surface_propagation(context, candidates)
        context.translation_mentions = {4: candidates[1]}

        hit = rules.rule_translation_given(context, 4)

        self.assertEqual((4, 6, "叔陵", "translation_anaphora"), hit)


if __name__ == "__main__":
    unittest.main()
