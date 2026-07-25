from __future__ import annotations

import json
from pathlib import Path
import unittest

import rules as R


PosToken = R.pos_giv.PosToken
GivOffsets = R.pos_giv.GivOffsets


def given_token(text: str, surface: str, start: int, *, bio: bool = False):
    tokens = []
    for offset, char in enumerate(surface, start):
        marker = None
        prefix = ""
        if bio:
            marker = "B" if offset == start else "I"
            prefix = marker + "-"
        tokens.append(
            PosToken(
                char,
                offset,
                offset + 1,
                "PROPN",
                prefix + "PROPN|NameType=Giv",
                marker,
                0.99,
            )
        )
    return tokens


def evidence(text: str, spans: list[tuple[int, int]], extra_tokens=()):
    tokens = list(extra_tokens)
    for start, end in spans:
        tokens.extend(given_token(text, text[start:end], start, bio=end - start > 1))
    return GivOffsets(
        offsets={offset for start, end in spans for offset in range(start, end)},
        spans=spans,
        tokens=tuple(sorted(tokens, key=lambda token: token.start)),
    )


def detect(paragraphs, giv, ner=()):
    return R.detect_juan(
        1,
        paragraphs,
        giv,
        R.Corpus(set(ner), {}, set()),
        enabled=R.PRESET_RECALL,
    )


class GivenAuthorizationTest(unittest.TestCase):
    def test_unanchored_verb_pronoun_is_not_a_given_name(self):
        text = "①获三十人，释之使为前导。"
        start = text.index("释之")
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, [(start, start + 2)])},
            ner={"释之"},
        )
        self.assertNotIn("释之", {row["surface"] for row in rows})

    def test_unanchored_single_given_after_person_verb_is_rejected(self):
        text = "①韩逊奏克盐州，斩岐所署刺史李继直。"
        start = text.index("岐")
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, [(start, start + 1)])},
        )
        self.assertNotIn("岐", {row["surface"] for row in rows})

    def test_earlier_full_name_authorizes_given_anaphora(self):
        text = "①曹操至，操曰。"
        surname = text.index("曹")
        first_given = surname + 1
        repeated = text.rindex("操")
        surname_token = PosToken(
            "曹", surname, surname + 1, "PROPN", "PROPN|NameType=Sur", None, 0.99
        )
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {
                0: evidence(
                    text,
                    [(first_given, first_given + 1), (repeated, repeated + 1)],
                    [surname_token],
                )
            },
        )
        repeated_card = next(
            row for row in rows
            if row["start"] == repeated and row["surface"] == "操"
        )
        self.assertEqual("jie_anaphora", repeated_card["rule"])

    def test_translation_adverb_handle_requires_controlled_predicate(self):
        text = "①司徒胡广至，广求群议。"
        surname = text.index("胡")
        given = surname + 1
        repeated = text.rindex("广")
        tokens = (
            PosToken(
                "胡", surname, surname + 1,
                "PROPN", "PROPN|NameType=Sur", None, 0.99,
            ),
            PosToken(
                "广", given, given + 1,
                "PROPN", "PROPN|NameType=Giv", None, 0.99,
            ),
            PosToken(
                "广", repeated, repeated + 1,
                "ADV", "ADV|Degree=Pos", None, 0.99,
            ),
            PosToken(
                "求", repeated + 1, repeated + 2,
                "VERB", "VERB", None, 0.99,
            ),
        )
        context = R.Ctx(
            text,
            {given},
            R.Corpus(set(), {}, set()),
            53,
            1,
            None,
            1,
            gspans=((given, given + 1),),
            tokens=tokens,
        )
        context.translation_anchors = ({
            "start": 0,
            "end": len(text),
            "anchor_start": surname,
            "identity_surface": "胡广",
            "handle": "广",
        },)
        cards = [{
            "start": surname,
            "end": given + 1,
            "surface": "胡广",
            "chunk_type": "translation_fullname",
        }]

        rows = R.detect_anaphora(context, cards)

        self.assertFalse(any(start == repeated for start, *_ in rows))

    def test_translation_adverb_handle_allows_subject_predicate(self):
        text = "①李密至，密以祖母老固辞。"
        fullname = text.index("李密")
        repeated = text.rindex("密")
        tokens = (
            PosToken(
                "密", repeated, repeated + 1,
                "ADV", "ADV|Degree=Pos|VerbForm=Conv", None, 0.99,
            ),
            PosToken(
                "以", repeated + 1, repeated + 2,
                "VERB", "VERB", None, 0.99,
            ),
        )
        context = R.Ctx(
            text,
            {repeated},
            R.Corpus(set(), {}, set()),
            79,
            1,
            None,
            1,
            gspans=((repeated, repeated + 1),),
            tokens=tokens,
        )
        context.translation_anchors = ({
            "start": fullname,
            "end": len(text),
            "anchor_start": fullname,
            "identity_surface": "李密",
            "handle": "密",
        },)
        cards = [{
            "start": fullname,
            "end": fullname + 2,
            "surface": "李密",
            "chunk_type": "translation_fullname",
        }]

        rows = R.detect_anaphora(context, cards)

        self.assertIn(
            (repeated, repeated + 1, "密"),
            {(row[0], row[1], row[2]) for row in rows},
        )

    def test_title_character_inside_two_char_given_is_not_a_handle(self):
        text = "①平州刺史张公素，素有威望，公素来。"
        surname = text.index("张")
        given = text.index("公素")
        adverb = text.index("素有")
        repeated = text.rindex("公素")
        tokens = (
            PosToken(
                "张", surname, surname + 1,
                "PROPN", "PROPN|NameType=Sur", None, 0.99,
            ),
            *given_token(text, "公素", given, bio=True),
            PosToken(
                "素", adverb, adverb + 1,
                "ADV", "ADV|Degree=Pos", None, 0.99,
            ),
            PosToken(
                "有", adverb + 1, adverb + 2,
                "VERB", "VERB", None, 0.99,
            ),
            *given_token(text, "公素", repeated, bio=True),
        )
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: GivOffsets(
                offsets={given, given + 1, repeated, repeated + 1},
                spans=((given, given + 2), (repeated, repeated + 2)),
                tokens=tokens,
            )},
            ner={"张公素"},
        )

        geometries = {(row["start"], row["surface"]) for row in rows}
        self.assertNotIn((adverb, "素"), geometries)
        self.assertIn((repeated, "公素"), geometries)

    def test_fullname_backtrack_rejects_prefix_of_longer_given(self):
        text = "①夫昧爽丕显，后世犹怠，王世充来。"
        prefix = text.index("世犹")
        fullname = text.index("王世充")
        given = fullname + 1
        tokens = (
            PosToken(
                "世", prefix, prefix + 1,
                "PROPN", "PROPN|NameType=Giv", None, 0.55,
            ),
            PosToken(
                "王", fullname, fullname + 1,
                "PROPN", "PROPN|NameType=Sur", None, 0.99,
            ),
            *given_token(text, "世充", given, bio=True),
        )
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: GivOffsets(
                offsets={prefix, given, given + 1},
                spans=((prefix, prefix + 1), (given, given + 2)),
                tokens=tokens,
            )},
            ner={"王世充"},
        )

        self.assertNotIn("后世", {row["surface"] for row in rows})

    def test_unique_earlier_name_resolves_shared_future_handle(self):
        text = "①侯益至，景崇闻益尹开封，王益后至。"
        first = text.index("侯益")
        handle = text.index("益尹")
        later = text.index("王益")
        context = R.Ctx(
            text,
            {handle},
            R.Corpus(set(), {}, set()),
            288,
            1,
            None,
            1,
            gspans=((handle, handle + 1),),
            tokens=(
                PosToken(
                    "益", handle, handle + 1,
                    "PROPN", "PROPN|NameType=Giv", None, 0.99,
                ),
            ),
        )
        context.translation_anchors = (
            {
                "start": first,
                "end": len(text),
                "anchor_start": first,
                "identity_surface": "侯益",
                "handle": "益",
            },
            {
                "start": later,
                "end": len(text),
                "anchor_start": later,
                "identity_surface": "王益",
                "handle": "益",
            },
        )
        cards = [
            {
                "start": first,
                "end": first + 2,
                "surface": "侯益",
                "chunk_type": "translation_fullname",
            },
            {
                "start": later,
                "end": later + 2,
                "surface": "王益",
                "chunk_type": "translation_fullname",
            },
        ]

        rows = R.detect_anaphora(context, cards)

        self.assertIn(
            (handle, handle + 1, "益"),
            {(row[0], row[1], row[2]) for row in rows},
        )

    def test_title_given_is_full_geometry_and_authorizes_anaphora(self):
        text = "①吴越王镠欲归，镠曰。"
        first = text.index("镠")
        repeated = text.rindex("镠")
        title_tokens = (
            PosToken("吴", 1, 2, "PROPN", "PROPN|Case=Loc|NameType=Nat", None, 0.99),
            PosToken("越", 2, 3, "PROPN", "PROPN|Case=Loc|NameType=Nat", None, 0.99),
            PosToken("王", 3, 4, "NOUN", "NOUN", None, 0.99),
        )
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {
                0: evidence(
                    text,
                    [(first, first + 1), (repeated, repeated + 1)],
                    title_tokens,
                )
            },
        )
        geometries = {
            (row["start"], row["end"], row["surface"], row["rule"])
            for row in rows
        }
        self.assertIn((1, first + 1, "吴越王镠", "jue_name"), geometries)
        self.assertIn(
            (repeated, repeated + 1, "镠", "jie_anaphora"),
            geometries,
        )

    def test_polity_ruler_title_authorizes_given_anaphora(self):
        text = "①燕主垂还中山，垂为之大赦。秦主登进兵，登乃退。"
        spans = []
        for surface in ("垂", "垂", "登", "登"):
            start = text.index(surface, spans[-1][1] if spans else 0)
            spans.append((start, start + 1))
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, spans)},
        )
        geometries = {
            (row["start"], row["end"], row["surface"], row["rule"])
            for row in rows
        }
        first_chui, second_chui, first_deng, second_deng = spans
        self.assertIn(
            (text.index("燕主"), first_chui[1], "燕主垂", "jue_name"),
            geometries,
        )
        self.assertIn(
            (second_chui[0], second_chui[1], "垂", "jie_anaphora"),
            geometries,
        )
        self.assertIn(
            (text.index("秦主"), first_deng[1], "秦主登", "jue_name"),
            geometries,
        )
        self.assertIn(
            (second_deng[0], second_deng[1], "登", "jie_anaphora"),
            geometries,
        )

    def test_polity_ruler_title_does_not_swallow_following_verb(self):
        text = "①后秦主苌如阴密。"
        given = text.index("苌")
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, [(given, given + 1)])},
            ner={"苌如"},
        )
        surfaces = {row["surface"] for row in rows}
        self.assertIn("后秦主苌", surfaces)
        self.assertNotIn("后秦主苌如", surfaces)

    def test_title_given_authorizes_possessive_given_anaphora(self):
        text = "①吴越王镠之巡湖州也，时瓌已得镠密旨。"
        first = text.index("镠")
        repeated = text.rindex("镠")
        title_tokens = (
            PosToken("吴", 1, 2, "PROPN", "PROPN|Case=Loc|NameType=Nat", None, 0.99),
            PosToken("越", 2, 3, "PROPN", "PROPN|Case=Loc|NameType=Nat", None, 0.99),
            PosToken("王", 3, 4, "NOUN", "NOUN", None, 0.99),
        )
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {
                0: evidence(
                    text,
                    [(first, first + 1), (repeated, repeated + 1)],
                    title_tokens,
                )
            },
        )
        self.assertIn(
            (repeated, "镠"),
            {(row["start"], row["surface"]) for row in rows},
        )

    def test_exact_handle_survives_noun_verb_model_error_after_move(self):
        text = "①赵将石公立戍深州，赵王命开门，移公立于外以避之。"
        surname = text.index("石")
        first = text.index("公立")
        repeated = text.rindex("公立")
        tokens = (
            PosToken("石", surname, surname + 1, "PROPN", "PROPN|NameType=Sur", None, 0.99),
            *given_token(text, "公立", first, bio=True),
            PosToken("移", repeated - 1, repeated, "VERB", "VERB", None, 0.99),
            PosToken("公", repeated, repeated + 1, "NOUN", "NOUN", None, 0.99),
            PosToken("立", repeated + 1, repeated + 2, "VERB", "VERB", None, 0.99),
        )
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, [(first, first + 2)], tokens)},
            ner={"石公立", "公立"},
        )
        self.assertIn(
            (repeated, repeated + 2, "公立", "local_exact_surface"),
            {
                (row["start"], row["end"], row["surface"], row["rule"])
                for row in rows
            },
        )

    def test_exact_handle_survives_verb_model_error_after_visit(self):
        text = "①匡国节度使冯行袭疾笃，命李珽驰往视行袭病。"
        surname = text.index("冯")
        first = text.index("行袭")
        repeated = text.rindex("行袭")
        tokens = (
            PosToken("冯", surname, surname + 1, "PROPN", "PROPN|NameType=Sur", None, 0.99),
            *given_token(text, "行袭", first, bio=True),
            PosToken("视", repeated - 1, repeated, "VERB", "VERB", None, 0.99),
            PosToken("行", repeated, repeated + 1, "VERB", "VERB", None, 0.99),
            PosToken("袭", repeated + 1, repeated + 2, "VERB", "VERB", None, 0.99),
        )
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, [(first, first + 2)], tokens)},
            ner={"冯行袭", "行袭"},
        )
        self.assertIn(
            (repeated, repeated + 2, "行袭", "local_exact_surface"),
            {
                (row["start"], row["end"], row["surface"], row["rule"])
                for row in rows
            },
        )

    def test_anchored_handle_survives_geo_model_error_in_direct_appointment(self):
        text = "①其子延昌来，吴遣使拜延昌虔州刺史。"
        first = text.index("延昌")
        repeated = text.rindex("延昌")
        tokens = (
            PosToken(
                "延", repeated, repeated + 1,
                "PROPN", "B-PROPN|Case=Loc|NameType=Geo", "B", 0.99,
            ),
            PosToken(
                "昌", repeated + 1, repeated + 2,
                "PROPN", "I-PROPN|Case=Loc|NameType=Geo", "I", 0.99,
            ),
            PosToken(
                "虔", repeated + 2, repeated + 3,
                "PROPN", "PROPN|Case=Loc|NameType=Geo", None, 0.99,
            ),
            PosToken(
                "州", repeated + 3, repeated + 4,
                "NOUN", "NOUN|Case=Loc", None, 0.99,
            ),
        )
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, [(first, first + 2)], tokens)},
            ner={"延昌"},
        )
        self.assertIn(
            (repeated, repeated + 2, "延昌"),
            {
                (row["start"], row["end"], row["surface"])
                for row in rows
            },
        )

    def test_short_title_is_not_consumed_before_title_given(self):
        text = "①使人谓赵王镕及王处直曰。"
        given = text.index("镕")
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, [(given, given + 1)])},
            ner={"赵王"},
        )
        surfaces = {row["surface"] for row in rows}
        self.assertIn("赵王镕", surfaces)
        self.assertNotIn("镕", surfaces)

    def test_two_character_title_given_uses_full_geometry(self):
        text = "①王翦、蒙武虏楚王负刍，以其地置楚郡。"
        given = text.index("负刍")
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, [(given, given + 2)])},
        )
        self.assertIn("楚王负刍", {row["surface"] for row in rows})

    def test_office_and_single_given_use_full_geometry(self):
        text = "①始皇下其议。廷尉斯曰。"
        given = text.index("斯")
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, [(given, given + 1)])},
        )
        self.assertIn("廷尉斯", {row["surface"] for row in rows})

    def test_royal_kinship_and_single_given_use_full_geometry(self):
        text = "①立皇子据为太子。"
        given = text.index("据")
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, [(given, given + 1)])},
        )
        self.assertIn("皇子据", {row["surface"] for row in rows})

    def test_role_introduced_full_surface_anchors_exact_recurrence(self):
        text = "①盗杀韩相侠累。侠累与人有恶。"
        first = text.index("侠累")
        second = text.index("侠累", first + 2)
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, [(first, first + 2), (second, second + 2)])},
        )
        geometries = {(row["start"], row["surface"]) for row in rows}
        self.assertIn((first, "侠累"), geometries)
        self.assertIn((second, "侠累"), geometries)

    def test_full_name_with_personal_suffix_anchors_full_handle(self):
        text = "①遣人诣孟海公，待海公报。"
        surname = text.index("孟")
        given = surname + 1
        repeated = text.index("海公", given + 2)
        surname_token = PosToken(
            "孟", surname, surname + 1,
            "PROPN", "PROPN|NameType=Sur", None, 0.99,
        )
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {
                0: evidence(
                    text,
                    [(given, given + 1), (repeated, repeated + 2)],
                    [surname_token],
                )
            },
        )
        surfaces = {row["surface"] for row in rows}
        self.assertIn("孟海公", surfaces)
        self.assertIn("海公", surfaces)

    def test_affiliation_introduced_full_surface_anchors_recurrence(self):
        text = "①齐人少翁，以方见上。少翁复言。"
        first = text.index("少翁")
        second = text.index("少翁", first + 2)
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, [(first, first + 2), (second, second + 2)])},
        )
        geometries = {(row["start"], row["surface"]) for row in rows}
        self.assertIn((first, "少翁"), geometries)
        self.assertIn((second, "少翁"), geometries)

    def test_personal_title_suffix_is_full_geometry(self):
        text = "①皇泰主眉目如画。"
        start = text.index("皇泰")
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, [(start, start + 2)])},
        )
        self.assertIn("皇泰主", {row["surface"] for row in rows})
        self.assertNotIn("泰主", {row["surface"] for row in rows})

    def test_royal_title_does_not_consume_opening_quote(self):
        text = "①帝尝嫌太子「得汉家性质」。"
        quote = text.index("「")
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, [(quote, quote + 1)])},
        )
        self.assertNotIn("太子「", {row["surface"] for row in rows})

    def test_title_minister_given_introduces_identity(self):
        text = "①镇军将军臣颖胄、中领军臣详皆社稷之臣。"
        start = text.index("颖胄")
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, [(start, start + 2)])},
        )
        self.assertIn("颖胄", {row["surface"] for row in rows})

    def test_office_introduced_epithet_title_uses_full_geometry(self):
        text = "①以兖海留后惠王友能代为宋州留后。"
        hui = text.index("惠")
        given = text.index("友能")
        tokens = (
            PosToken(
                "惠", hui, hui + 1, "PROPN", "PROPN|NameType=Prs", None, 0.98
            ),
            PosToken("王", hui + 1, hui + 2, "NOUN", "NOUN", None, 0.9),
        )
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, [(given, given + 2)], tokens)},
        )
        self.assertIn("惠王友能", {row["surface"] for row in rows})

    def test_coordinated_sibling_name_is_kinship_authorized(self):
        text = "①楚王殷求为天策上将。殷始开天策府，以弟賨为左相，存为右相。"
        chu = text.index("殷")
        zong = text.index("賨")
        cun = text.index("存")
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {
                0: evidence(
                    text,
                    [
                        (chu, chu + 1),
                        (text.index("殷", chu + 1), text.index("殷", chu + 1) + 1),
                        (zong, zong + 1),
                        (cun, cun + 1),
                    ],
                )
            },
        )
        by_surface = {row["surface"]: row for row in rows}
        self.assertEqual("genealogy_given", by_surface["賨"]["rule"])
        self.assertEqual("coordinated_kinship_given", by_surface["存"]["rule"])

    def test_coordinated_adopted_children_all_remain_kinship_authorized(self):
        text = "①立假子宗裕为通王，宗范为夔王，宗鐬为昌王。"
        names = ("宗裕", "宗范", "宗鐬")
        spans = [
            (text.index(name), text.index(name) + len(name))
            for name in names
        ]
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, spans)},
        )
        by_surface = {row["surface"]: row for row in rows}
        self.assertEqual("genealogy_given", by_surface["宗裕"]["rule"])
        for name in names[1:]:
            self.assertEqual(
                "coordinated_kinship_given",
                by_surface[name]["rule"],
            )

    def test_punctuation_coordinated_children_remain_kinship_authorized(self):
        text = "①钱镠遣其子传璙、传瓘讨卢佶。"
        names = ("传璙", "传瓘")
        spans = [
            (text.index(name), text.index(name) + len(name))
            for name in names
        ]
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, spans)},
        )
        by_surface = {row["surface"]: row for row in rows}
        self.assertEqual("genealogy_given", by_surface["传璙"]["rule"])
        self.assertEqual(
            "coordinated_kinship_given",
            by_surface["传瓘"]["rule"],
        )

    def test_bare_child_prefix_continues_title_list(self):
        text = "①立子友文为博王，友珪为郢王。"
        names = ("友文", "友珪")
        spans = [
            (text.index(name), text.index(name) + len(name))
            for name in names
        ]
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, spans)},
        )
        by_surface = {row["surface"]: row for row in rows}
        self.assertEqual("genealogy_given", by_surface["友文"]["rule"])
        self.assertEqual(
            "coordinated_kinship_given",
            by_surface["友珪"]["rule"],
        )

    def test_bare_child_before_succession_predicate_is_kinship(self):
        text = "①去诸卒，子扫剌立。后复言扫剌。"
        first = text.index("扫剌")
        second = text.index("扫剌", first + 2)
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, [(first, first + 2), (second, second + 2)])},
        )
        geometries = {(row["start"], row["surface"]) for row in rows}
        self.assertIn((first, "扫剌"), geometries)
        self.assertIn((second, "扫剌"), geometries)

    def test_possessive_child_list_continues_after_punctuation(self):
        text = "①温韬之子延濬、延沼、延衮居许州。"
        names = ("延濬", "延沼", "延衮")
        spans = [
            (text.index(name), text.index(name) + len(name))
            for name in names
        ]
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, spans)},
        )
        surfaces = {row["surface"] for row in rows}
        for name in names:
            self.assertIn(name, surfaces)

    def test_long_coordinated_kinship_roles_continue(self):
        text = "①以皇子从荣为河南尹、判六军诸衞事，从厚为河东节度使。"
        first = text.index("从荣")
        second = text.index("从厚")
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, [(first, first + 2), (second, second + 2)])},
        )
        surfaces = {row["surface"] for row in rows}
        self.assertIn("皇子从荣", surfaces)
        self.assertIn("从厚", surfaces)

    def test_postposed_son_relation_authorizes_name(self):
        text = "①文超，子盖之子也。"
        start = text.index("子盖")
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, [(start, start + 2)])},
        )
        self.assertIn("子盖", {row["surface"] for row in rows})

    def test_earlier_full_name_authorizes_minister_self_reference(self):
        text = "①萧颖胄至。镇军将军臣颖胄、中领军臣详皆社稷之臣。"
        surname = text.index("萧")
        first_given = surname + 1
        repeated = text.index("颖胄", first_given + 2)
        surname_token = PosToken(
            "萧", surname, surname + 1,
            "PROPN", "PROPN|NameType=Sur", None, 0.99,
        )
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {
                0: evidence(
                    text,
                    [
                        (first_given, first_given + 2),
                        (repeated, repeated + 2),
                    ],
                    [surname_token],
                )
            },
        )
        self.assertIn(
            (repeated, "颖胄"),
            {(row["start"], row["surface"]) for row in rows},
        )

    def test_office_between_kinship_and_name_does_not_emit_office_word(self):
        text = "①以其子权知留后颢为节度使。"
        authority = text.index("权")
        name = text.index("颢")
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {
                0: evidence(
                    text,
                    [(authority, authority + 1), (name, name + 1)],
                )
            },
        )
        surfaces = {row["surface"] for row in rows}
        self.assertIn("颢", surfaces)
        self.assertNotIn("权", surfaces)

    def test_bestowed_name_is_explicit_identity_not_bare_anaphora(self):
        text = "①赐河南尹张全义名宗奭。"
        full_given = text.index("全义")
        alias = text.index("宗奭")
        surname = PosToken(
            "张",
            full_given - 1,
            full_given,
            "PROPN",
            "PROPN|NameType=Sur",
            None,
            0.99,
        )
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {
                0: evidence(
                    text,
                    [(full_given, full_given + 2), (alias, alias + 2)],
                    [surname],
                )
            },
        )
        self.assertIn("宗奭", {row["surface"] for row in rows})

    def test_postposed_kinship_authorizes_coordinated_names(self):
        text = "①惟宗懿等九人及宗特、宗平真其子。"
        names = ("宗特", "宗平")
        spans = [
            (text.index(name), text.index(name) + len(name))
            for name in names
        ]
        rows = detect(
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: evidence(text, spans)},
        )
        by_surface = {row["surface"]: row for row in rows}
        for name in names:
            self.assertIn(name, by_surface)

    def test_title_anchor_does_not_cross_numbered_jie(self):
        first = "①吴越王镠曰。"
        second = "②镠曰。"
        first_given = first.index("镠")
        second_given = second.index("镠")
        rows = detect(
            [
                {"id": 0, "main": first, "ce_year": 1},
                {"id": 1, "main": second, "ce_year": 1},
            ],
            {
                0: evidence(first, [(first_given, first_given + 1)]),
                1: evidence(second, [(second_given, second_given + 1)]),
            },
        )
        self.assertFalse(
            any(
                row["para_id"] == 1 and row["surface"] == "镠"
                for row in rows
            )
        )

    def test_real_juan267_keeps_anchored_given_recurrences(self):
        repo = Path(__file__).resolve().parents[3]
        text_dir = repo / "web" / "public" / "text"
        paragraphs = json.loads(
            (text_dir / "juan_267.json").read_text(encoding="utf-8")
        )["paragraphs"]
        giv = R.pos_giv.giv_for_juan(
            267,
            paragraphs,
            text_dir / "persons" / "pos_giv",
        )
        rows = R.detect_juan(
            267,
            paragraphs,
            giv,
            R.load_corpus(),
            enabled=R.PRESET_RECALL,
        )
        geometries = {
            (row["para_id"], row["start"], row["end"], row["surface"])
            for row in rows
            if row.get("field") == "main"
        }
        self.assertIn((111, 56, 57, "镠"), geometries)
        self.assertIn((116, 109, 111, "公立"), geometries)
        self.assertIn((98, 60, 62, "行袭"), geometries)
        self.assertIn((121, 51, 53, "延昌"), geometries)


if __name__ == "__main__":
    unittest.main()
