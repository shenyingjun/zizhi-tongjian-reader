from __future__ import annotations

import unittest

import evidence as E
import rules as R


class EvidencePolicyTest(unittest.TestCase):
    def test_requires_every_named_signal(self):
        candidate = E.Candidate(0, 4, "太平公主")
        candidate.add("inherent_person_title", "title_semantics")
        candidate.add("appointment_frame", "syntax")
        policy = E.AdmissionPolicy(
            "title-appointment",
            frozenset({
                "inherent_person_title",
                "appointment_frame",
                "human_appointment_role",
            }),
        )

        self.assertIsNone(E.decide(candidate, (policy,)))

    def test_requires_independent_families(self):
        candidate = E.Candidate(0, 4, "太平公主")
        candidate.add("inherent_person_title", "model")
        candidate.add("appointment_frame", "model")
        candidate.add("human_appointment_role", "role_semantics")
        policy = E.AdmissionPolicy(
            "title-appointment",
            frozenset({
                "inherent_person_title",
                "appointment_frame",
                "human_appointment_role",
            }),
        )

        self.assertIsNone(E.decide(candidate, (policy,)))

    def test_veto_overrides_independent_support(self):
        candidate = E.Candidate(0, 4, "太平公主")
        candidate.add("inherent_person_title", "title_semantics")
        candidate.add("appointment_frame", "syntax")
        candidate.add("human_appointment_role", "role_semantics")
        candidate.veto("local_polity_usage")
        policy = E.AdmissionPolicy(
            "title-appointment",
            frozenset({
                "inherent_person_title",
                "appointment_frame",
                "human_appointment_role",
            }),
        )

        self.assertIsNone(E.decide(candidate, (policy,)))

    def test_admits_three_independent_families(self):
        candidate = E.Candidate(0, 4, "太平公主")
        candidate.add("inherent_person_title", "title_semantics")
        candidate.add("appointment_frame", "syntax")
        candidate.add("human_appointment_role", "role_semantics")
        policy = E.AdmissionPolicy(
            "title-appointment",
            frozenset({
                "inherent_person_title",
                "appointment_frame",
                "human_appointment_role",
            }),
        )

        self.assertEqual(policy, E.decide(candidate, (policy,)))

    def test_prerequisite_may_share_a_support_family(self):
        candidate = E.Candidate(0, 3, "贺拔胜")
        candidate.add("model_ner_witness", "model", "candidate_contains", 1, 3)
        candidate.add("model_name_morphology", "model")
        candidate.add("exact_local_recurrence", "local_recurrence")
        candidate.add("hard_name_boundary", "syntax")
        policy = E.AdmissionPolicy(
            "fuzzy-model",
            frozenset({
                "model_name_morphology",
                "exact_local_recurrence",
                "hard_name_boundary",
            }),
            prerequisite_signals=frozenset({"model_ner_witness"}),
        )

        self.assertEqual(policy, E.decide(candidate, (policy,)))

    def test_fuzzy_relation_is_tight_and_overlapping(self):
        self.assertEqual(
            "candidate_contains",
            E.fuzzy_relation(4, 8, 5, 8),
        )
        self.assertEqual(
            "witness_contains",
            E.fuzzy_relation(5, 8, 4, 8),
        )
        self.assertIsNone(E.fuzzy_relation(4, 8, 8, 10))

    def test_policy_must_explicitly_allow_soft_conflict(self):
        candidate = E.Candidate(0, 3, "安禄山")
        candidate.add("partial_person_morphology", "model")
        candidate.add("surname_shape", "name_shape")
        candidate.add("exact_local_recurrence", "local_recurrence")
        candidate.add("person_occurrence_syntax", "syntax")
        candidate.conflict("geo_nat_morphology")
        strict = E.AdmissionPolicy(
            "strict",
            frozenset({
                "partial_person_morphology",
                "surname_shape",
                "exact_local_recurrence",
                "person_occurrence_syntax",
            }),
        )
        relaxed = E.AdmissionPolicy(
            "relaxed",
            strict.required_signals,
            allowed_soft_conflicts=frozenset({"geo_nat_morphology"}),
        )

        self.assertIsNone(E.decide(candidate, (strict,)))
        self.assertEqual(relaxed, E.decide(candidate, (relaxed,)))

    def test_unlisted_soft_conflict_still_rejects(self):
        candidate = E.Candidate(0, 2, "异人")
        candidate.add("genealogy_name_anchor", "genealogy_semantics")
        candidate.add("exact_local_recurrence", "local_recurrence")
        candidate.add("person_occurrence_syntax", "syntax")
        candidate.conflict("function_morphology")
        candidate.conflict("office_continuation")
        policy = E.AdmissionPolicy(
            "genealogy",
            frozenset({
                "genealogy_name_anchor",
                "exact_local_recurrence",
                "person_occurrence_syntax",
            }),
            allowed_soft_conflicts=frozenset({"function_morphology"}),
        )

        self.assertIsNone(E.decide(candidate, (policy,)))

    def test_cumulative_policy_caps_correlated_family_support(self):
        policy = E.CumulativeAdmissionPolicy(
            "cumulative",
            (
                E.FamilySupport(
                    "model",
                    2,
                    any_signals=frozenset({"pos", "bio"}),
                ),
                E.FamilySupport(
                    "syntax",
                    1,
                    all_signals=frozenset({"syntax"}),
                ),
            ),
            minimum_score=3,
            minimum_families=2,
        )
        candidate = E.Candidate(0, 2, "异人")
        candidate.add("pos", "model")
        candidate.add("bio", "model")

        self.assertIsNone(E.decide(candidate, (policy,)))
        candidate.add("syntax", "syntax")
        self.assertEqual(policy, E.decide(candidate, (policy,)))

    def test_cumulative_policy_penalizes_conflict_and_preserves_veto(self):
        policy = E.CumulativeAdmissionPolicy(
            "cumulative",
            (
                E.FamilySupport(
                    "recurrence",
                    2,
                    all_signals=frozenset({"recurrence"}),
                ),
                E.FamilySupport(
                    "syntax",
                    1,
                    all_signals=frozenset({"syntax"}),
                ),
            ),
            minimum_score=3,
            minimum_families=2,
            conflict_penalties=(("geo", 1),),
        )
        candidate = E.Candidate(0, 3, "安禄山")
        candidate.add("recurrence", "recurrence")
        candidate.add("syntax", "syntax")

        self.assertEqual(policy, E.decide(candidate, (policy,)))
        candidate.conflict("geo")
        self.assertIsNone(E.decide(candidate, (policy,)))
        candidate.soft_conflicts.clear()
        candidate.veto("polity")
        self.assertIsNone(E.decide(candidate, (policy,)))

    def test_cumulative_policy_uses_only_jie_evidence(self):
        policy = next(
            policy
            for policy in R.COMBINED_EVIDENCE_POLICIES
            if policy.name == "cumulative-family-score"
        )
        candidate = E.Candidate(0, 2, "望之")
        candidate.add("model_ner_witness", "model")
        candidate.add(
            "jie_person_morphology_anchor",
            "jie_morphology",
        )
        candidate.add(
            "jie_person_morphology_majority",
            "jie_morphology",
        )
        candidate.add("admitted_local_anchor", "local_anchor")
        candidate.add("exact_local_recurrence", "local_recurrence")
        candidate.add("person_occurrence_syntax", "syntax")

        self.assertIsNone(E.decide(candidate, (policy,)))
        candidate.add("strict_person_frame", "syntax")
        self.assertEqual(policy, E.decide(candidate, (policy,)))
        score, families = policy.score(candidate)
        self.assertEqual(6, score)
        self.assertEqual(
            frozenset({
                "jie_morphology",
                "local_anchor",
                "recurrence",
                "syntax",
            }),
            families,
        )

    def test_cumulative_policy_does_not_count_bare_surname_shape(self):
        policy = next(
            policy
            for policy in R.COMBINED_EVIDENCE_POLICIES
            if policy.name == "cumulative-family-score"
        )
        candidate = E.Candidate(0, 3, "江乃始")
        candidate.add("model_ner_witness", "model")
        candidate.add(
            "jie_person_morphology_anchor",
            "jie_morphology",
        )
        candidate.add(
            "jie_person_morphology_majority",
            "jie_morphology",
        )
        candidate.add("admitted_local_anchor", "local_anchor")
        candidate.add("exact_local_recurrence", "local_recurrence")
        candidate.add("surname_shape", "name_shape")

        self.assertIsNone(E.decide(candidate, (policy,)))
        candidate.add("local_surname_morphology", "model")
        self.assertEqual(policy, E.decide(candidate, (policy,)))

    def test_geo_conflict_requires_an_explicit_signal(self):
        policy = next(
            policy
            for policy in R.COMBINED_EVIDENCE_POLICIES
            if policy.name == "soft-surname-geo-recurrence-syntax"
        )
        candidate = E.Candidate(0, 3, "安禄山")
        candidate.add("model_ner_witness", "model")
        candidate.add(
            "jie_person_morphology_majority",
            "jie_morphology",
        )
        candidate.add("surname_shape", "name_shape")
        candidate.add("exact_local_recurrence", "local_recurrence")
        candidate.add("person_occurrence_syntax", "syntax")
        candidate.conflict("geo_nat_morphology")

        self.assertIsNone(E.decide(candidate, (policy,)))
        candidate.add("geo_nat_morphology", "model")
        self.assertEqual(policy, E.decide(candidate, (policy,)))

    def test_translation_requires_jie_morphology(self):
        policy = next(
            policy
            for policy in R.COMBINED_EVIDENCE_POLICIES
            if policy.name == "soft-translation-recurrence-syntax"
        )
        candidate = E.Candidate(0, 2, "常据")
        candidate.add("model_ner_witness", "model")
        candidate.add("translation_exact_identity", "translation")
        candidate.add("exact_local_recurrence", "local_recurrence")
        candidate.add("person_occurrence_syntax", "syntax")
        candidate.conflict("missing_person_morphology")

        self.assertIsNone(E.decide(candidate, (policy,)))
        candidate.add(
            "jie_person_morphology_anchor",
            "jie_morphology",
        )
        self.assertEqual(policy, E.decide(candidate, (policy,)))

    def test_geo_conflict_can_use_decisive_syntax_without_majority(self):
        policy = next(
            policy
            for policy in R.COMBINED_EVIDENCE_POLICIES
            if policy.name == "soft-surname-geo-decisive-syntax"
        )
        candidate = E.Candidate(0, 2, "盖吴")
        candidate.add("model_ner_witness", "model")
        candidate.add(
            "jie_person_morphology_anchor",
            "jie_morphology",
        )
        candidate.add("surname_shape", "name_shape")
        candidate.add("exact_local_recurrence", "local_recurrence")
        candidate.add("geo_nat_morphology", "model")
        candidate.conflict("geo_nat_morphology")

        self.assertIsNone(E.decide(candidate, (policy,)))
        candidate.add("decisive_person_syntax", "syntax")
        self.assertEqual(policy, E.decide(candidate, (policy,)))

    def test_nearest_policy_reports_missing_and_conflicting_evidence(self):
        policy = E.AdmissionPolicy(
            "cumulative",
            frozenset({"recurrence", "syntax", "shape"}),
            prerequisite_signals=frozenset({"model"}),
            allowed_soft_conflicts=frozenset({"geo"}),
        )
        candidate = E.Candidate(0, 2, "异人")
        candidate.add("model", "model")
        candidate.add("recurrence", "local_recurrence")
        candidate.add("syntax", "syntax")
        candidate.conflict("function")

        diagnostics = E.nearest_policy(candidate, (policy,))

        self.assertEqual("cumulative", diagnostics["policy"])
        self.assertEqual(["shape"], diagnostics["missing_required"])
        self.assertEqual(["function"], diagnostics["unallowed_soft_conflicts"])
        self.assertEqual([], diagnostics["missing_prerequisites"])

    def test_candidate_audit_preserves_rejection_vetoes(self):
        policy = E.AdmissionPolicy(
            "three-family",
            frozenset({"model", "recurrence", "syntax"}),
        )
        candidate = E.Candidate(0, 3, "高句丽")
        candidate.add("model", "model")
        candidate.add("recurrence", "local_recurrence")
        candidate.add("syntax", "syntax")
        candidate.veto("jie_polity_usage")

        metadata = E.candidate_audit_metadata(
            candidate,
            (policy,),
            E.decide(candidate, (policy,)),
        )

        self.assertFalse(metadata["admitted"])
        self.assertEqual(["jie_polity_usage"], metadata["evidence_vetoes"])
        self.assertEqual("three-family", metadata["nearest_policy"]["policy"])

    def test_cumulative_audit_reports_actual_score(self):
        policy = E.CumulativeAdmissionPolicy(
            "cumulative",
            (
                E.FamilySupport(
                    "recurrence",
                    2,
                    all_signals=frozenset({"recurrence"}),
                ),
                E.FamilySupport(
                    "syntax",
                    1,
                    all_signals=frozenset({"syntax"}),
                ),
            ),
            minimum_score=3,
            minimum_families=2,
        )
        candidate = E.Candidate(0, 2, "异人")
        candidate.add("recurrence", "recurrence")
        candidate.add("syntax", "syntax")

        metadata = E.candidate_audit_metadata(
            candidate,
            (policy,),
            E.decide(candidate, (policy,)),
        )

        self.assertEqual(3, metadata["policy_diagnostics"]["score"])
        self.assertEqual(
            ["recurrence", "syntax"],
            metadata["policy_diagnostics"]["supporting_families"],
        )


class CombinedEvidenceIntegrationTest(unittest.TestCase):
    @staticmethod
    def _detect(text):
        corpus = R.Corpus(set(), {}, set())
        context = R.Ctx(text, set(), corpus, 202, 1, None, 681)
        return R.detect_para(context, R.PRESET_RECALL)

    def test_admits_inherent_title_in_human_appointment_frame(self):
        text = "\u5929\u540e\u8bf7\u4ee5\u592a\u5e73\u516c\u4e3b\u4e3a\u5973\u5b98"
        rows = self._detect(text)

        row = next(row for row in rows if row["surface"] == "\u592a\u5e73\u516c\u4e3b")
        self.assertEqual("combined_evidence", row["rule"])
        self.assertEqual(
            ["role_semantics", "syntax", "title_semantics"],
            row["evidence_families"],
        )

    def test_rejects_title_shape_without_appointment_frame(self):
        rows = self._detect("\u9063\u957f\u516c\u4e3b\u5165\u671d")

        self.assertFalse(any(row["surface"] == "\u957f\u516c\u4e3b" for row in rows))

    def test_rejects_appointment_without_human_role(self):
        rows = self._detect(
            "\u4ee5\u592a\u5e73\u516c\u4e3b\u4e3a\u6b64\u4e8b"
        )

        self.assertFalse(any(row["surface"] == "\u592a\u5e73\u516c\u4e3b" for row in rows))

    def test_translation_exact_cannot_override_office_continuation(self):
        context = R.Ctx(
            "容管经略使阳旻",
            set(),
            R.Corpus({"容管"}, {}, set()),
            241,
            1,
            None,
            681,
        )
        context.translation_fullnames = {
            0: {
                "start": 0,
                "end": 2,
                "surface": "容管",
                "identity_surface": "容管",
            }
        }

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "容管"
        )

        self.assertIn("office_continuation", candidate.vetoes)

    def test_jie_collective_frame_is_a_generic_veto_source(self):
        corpus = R.Corpus(set(), {}, set())
        context = R.Ctx(
            "民谓之山棚，请募山棚以守城",
            set(),
            corpus,
            202,
            1,
            None,
            681,
        )

        self.assertTrue(
            R._surface_has_jie_collective_frame(context, "山棚")
        )

    def test_direct_troop_count_is_collective_but_possession_is_not(self):
        corpus = R.Corpus(set(), {}, set())
        context = R.Ctx(
            "铁勒数千骑奄至，今唐兵破贺鲁诸部",
            set(),
            corpus,
            149,
            1,
            None,
            523,
        )

        self.assertTrue(
            R._surface_has_jie_collective_frame(context, "铁勒")
        )
        self.assertFalse(
            R._surface_has_jie_collective_frame(context, "贺鲁")
        )

    def test_person_morphology_does_not_cross_jie_boundary(self):
        corpus = R.Corpus({"异人"}, {}, set())
        person_token = R.pos_giv.PosToken(
            "异人", 1, 3, "PROPN", "PROPN|NameType=Prs", "B-PER"
        )
        noun_token = R.pos_giv.PosToken(
            "异人", 1, 3, "NOUN", "NOUN", "O"
        )
        paras = [
            {"id": 0, "main": "①异人曰", "ce_year": 1},
            {"id": 1, "main": "②异人曰", "ce_year": 1},
        ]
        giv = {
            0: R.pos_giv.GivOffsets(tokens=(person_token,)),
            1: R.pos_giv.GivOffsets(tokens=(noun_token,)),
        }
        audit = []

        R.detect_juan(
            1,
            paras,
            giv,
            corpus,
            enabled=R.PRESET_RECALL,
            evidence_audit=audit,
        )

        candidate = next(
            row
            for row in audit
            if row["para_id"] == 1 and row["surface"] == "异人"
        )
        self.assertNotIn(
            "jie_person_morphology_anchor",
            candidate["evidence_signals"],
        )
        self.assertNotIn(
            "exact_local_recurrence",
            candidate["evidence_signals"],
        )

    def test_person_morphology_can_support_within_same_jie(self):
        corpus = R.Corpus({"异人"}, {}, set())
        person_token = R.pos_giv.PosToken(
            "异人", 1, 3, "PROPN", "PROPN|NameType=Prs", "B-PER"
        )
        noun_token = R.pos_giv.PosToken(
            "异人", 0, 2, "NOUN", "NOUN", "O"
        )
        paras = [
            {"id": 0, "main": "①异人曰", "ce_year": 1},
            {"id": 1, "main": "异人曰", "ce_year": 1},
        ]
        giv = {
            0: R.pos_giv.GivOffsets(tokens=(person_token,)),
            1: R.pos_giv.GivOffsets(tokens=(noun_token,)),
        }
        audit = []

        R.detect_juan(
            1,
            paras,
            giv,
            corpus,
            enabled=R.PRESET_RECALL,
            evidence_audit=audit,
        )

        candidate = next(
            row
            for row in audit
            if row["para_id"] == 1 and row["surface"] == "异人"
        )
        self.assertIn(
            "jie_person_morphology_anchor",
            candidate["evidence_signals"],
        )
        self.assertIn(
            "exact_local_recurrence",
            candidate["evidence_signals"],
        )

    def test_anaphora_does_not_tag_suffix_inside_repeated_full_name(self):
        corpus = R.Corpus({"常炜"}, {}, set())
        context = R.Ctx(
            "①常炜入朝，常炜上疏",
            {2, 3, 7, 8},
            corpus,
            99,
            1,
            None,
            1,
        )
        cards = [{
            "start": 1,
            "end": 3,
            "surface": "常炜",
            "chunk_type": "pos_fullname",
        }]

        rows = R.detect_anaphora(context, cards)

        self.assertFalse(any(surface == "炜" for _, _, surface, _ in rows))

    def test_anaphora_does_not_tag_tail_inside_longer_name_surface(self):
        corpus = R.Corpus({"拓跋沙漠汗", "沙漠汗"}, {}, set())
        text = "①拓跋沙漠汗归国，沙漠汗入质"
        repeated_start = text.rfind("沙漠汗")
        context = R.Ctx(
            text,
            {4, 5, 12, 13},
            corpus,
            80,
            1,
            None,
            1,
            gspans=((1, 6), (repeated_start, repeated_start + 3)),
        )
        cards = [{
            "start": 1,
            "end": 6,
            "surface": "拓跋沙漠汗",
            "chunk_type": "foreign_suffix_name",
        }]

        rows = R.detect_anaphora(context, cards)

        self.assertFalse(any(surface == "漠汗" for _, _, surface, _ in rows))

    def test_anaphora_does_not_tag_bio_continuation(self):
        corpus = R.Corpus({"斛拔弥俄突"}, {}, set())
        text = "①贺拔公至，斛拔弥俄突来"
        candidate_start = text.index("拔", text.index("斛"))
        tokens = (
            R.pos_giv.PosToken(
                "斛", candidate_start - 1, candidate_start,
                "PROPN", "B-PROPN|NameType=Sur", "B",
            ),
            R.pos_giv.PosToken(
                "拔", candidate_start, candidate_start + 1,
                "PROPN", "I-PROPN|NameType=Giv", "I",
            ),
        )
        context = R.Ctx(
            text,
            {candidate_start},
            corpus,
            156,
            1,
            None,
            1,
            tokens=tokens,
        )
        cards = [{
            "start": 1,
            "end": 3,
            "surface": "贺拔",
            "chunk_type": "pos_fullname",
        }]

        rows = R.detect_anaphora(context, cards)

        self.assertFalse(any(surface == "拔" for _, _, surface, _ in rows))

    def test_foreign_title_accepts_strict_local_components(self):
        corpus = R.Corpus(set(), {}, set())
        cases = (
            (
                "号曰老上单于",
                2,
                (
                    R.pos_giv.PosToken("老", 2, 3, "VERB", "VERB", None, 0.4),
                    R.pos_giv.PosToken("上", 3, 4, "NOUN", "NOUN", None, 0.9),
                ),
                "老上单于",
            ),
            (
                "击北单于弟",
                1,
                (
                    R.pos_giv.PosToken(
                        "北", 1, 2, "NOUN", "NOUN|Case=Loc", None, 0.99
                    ),
                ),
                "北单于",
            ),
            (
                "号颉利可汗",
                1,
                (
                    R.pos_giv.PosToken(
                        "颉", 1, 2, "PROPN", "B-PROPN|NameType=Giv", "B", 0.95
                    ),
                    R.pos_giv.PosToken(
                        "利", 2, 3, "PROPN", "I-PROPN|NameType=Giv", "I", 0.95
                    ),
                ),
                "颉利可汗",
            ),
        )
        for text, start, tokens, expected in cases:
            with self.subTest(expected):
                context = R.Ctx(
                    text,
                    {offset for token in tokens for offset in range(token.start, token.end)},
                    corpus,
                    1,
                    1,
                    None,
                    1,
                    tokens=tokens,
                    gspans=((start, start + 2),) if len(expected) == 4 else (),
                )
                self.assertEqual(
                    expected,
                    R.rule_foreign_title_name(context, start)[2],
                )

    def test_foreign_title_rejects_collective_and_name_continuations(self):
        corpus = R.Corpus(set(), {}, set())
        for text in ("击北单于部", "号颉利可汗弥"):
            with self.subTest(text):
                start = 1
                component_end = text.index("单于") if "单于" in text else text.index("可汗")
                component = text[start:component_end]
                tokens = tuple(
                    R.pos_giv.PosToken(
                        char,
                        start + offset,
                        start + offset + 1,
                        "PROPN",
                        ("B-" if offset == 0 else "I-") + "PROPN|NameType=Giv",
                        "B" if offset == 0 else "I",
                        0.95,
                    )
                    for offset, char in enumerate(component)
                )
                if text.endswith("弥"):
                    tokens += (
                        R.pos_giv.PosToken(
                            "弥", len(text) - 1, len(text),
                            "PROPN", "PROPN|NameType=Giv", None, 0.95,
                        ),
                    )
                context = R.Ctx(
                    text,
                    set(range(start, component_end)),
                    corpus,
                    1,
                    1,
                    None,
                    1,
                    tokens=tokens,
                    gspans=((start, component_end),),
                )
                self.assertIsNone(R.rule_foreign_title_name(context, start))

    def test_foreign_title_rejects_punctuation_component(self):
        corpus = R.Corpus(set(), {}, set())
        token = R.pos_giv.PosToken(
            "「", 0, 1, "NOUN", "NOUN", None, 0.4
        )
        context = R.Ctx(
            "「单于曰", set(), corpus, 1, 1, None, 1, tokens=(token,)
        )
        self.assertIsNone(R.rule_foreign_title_name(context, 0))

    def test_foreign_title_rejects_function_component(self):
        corpus = R.Corpus(set(), {}, set())
        tokens = (
            R.pos_giv.PosToken("册", 0, 1, "VERB", "VERB", None, 0.99),
            R.pos_giv.PosToken("使", 3, 4, "VERB", "VERB", None, 0.99),
        )
        context = R.Ctx(
            "册可汗使", set(), corpus, 1, 1, None, 1, tokens=tokens
        )
        self.assertIsNone(R.rule_foreign_title_name(context, 0))

    def test_foreign_title_rejects_envoy_designation_continuation(self):
        corpus = R.Corpus(set(), {}, set())
        tokens = (
            R.pos_giv.PosToken(
                "册", 0, 1, "PROPN", "PROPN|NameType=Giv", None, 0.98
            ),
            R.pos_giv.PosToken("使", 3, 4, "VERB", "VERB", None, 0.99),
            R.pos_giv.PosToken(
                "源", 4, 5, "PROPN", "PROPN|NameType=Sur", None, 0.99
            ),
        )
        context = R.Ctx(
            "册可汗使源休", {0}, corpus, 1, 1, None, 1, tokens=tokens
        )
        self.assertIsNone(R.rule_foreign_title_name(context, 0))

    def test_foreign_title_allows_ruler_predicate_shi(self):
        corpus = R.Corpus(set(), {}, set())
        tokens = (
            R.pos_giv.PosToken(
                "真", 0, 1, "PROPN", "B-PROPN|NameType=Giv", "B", 0.98
            ),
            R.pos_giv.PosToken(
                "珠", 1, 2, "PROPN", "I-PROPN|NameType=Giv", "I", 0.98
            ),
            R.pos_giv.PosToken("使", 4, 5, "VERB", "VERB", None, 0.99),
            R.pos_giv.PosToken("其", 5, 6, "PRON", "PRON", None, 0.99),
        )
        context = R.Ctx(
            "真珠可汗使其姪",
            {0, 1},
            corpus,
            1,
            1,
            None,
            1,
            tokens=tokens,
            gspans=((0, 2),),
        )
        self.assertEqual(
            "真珠可汗",
            R.rule_foreign_title_name(context, 0)[2],
        )

    def test_foreign_title_allows_shi_before_person_object(self):
        corpus = R.Corpus(set(), {}, set())
        tokens = (
            R.pos_giv.PosToken(
                "屠", 0, 1, "PROPN", "B-PROPN|NameType=Giv", "B", 0.98
            ),
            R.pos_giv.PosToken(
                "耆", 1, 2, "PROPN", "I-PROPN|NameType=Giv", "I", 0.98
            ),
            R.pos_giv.PosToken("使", 4, 5, "VERB", "VERB", None, 0.99),
            R.pos_giv.PosToken(
                "先", 5, 6, "PROPN", "B-PROPN|NameType=Giv", "B", 0.7
            ),
        )
        context = R.Ctx(
            "屠耆单于使先贤",
            {0, 1, 5},
            corpus,
            1,
            1,
            None,
            1,
            tokens=tokens,
            gspans=((0, 2), (5, 6)),
        )
        self.assertEqual(
            "屠耆单于",
            R.rule_foreign_title_name(context, 0)[2],
        )

    def test_foreign_title_accepts_complete_bio_component(self):
        corpus = R.Corpus(set(), {}, set())
        tokens = (
            R.pos_giv.PosToken(
                "狐", 0, 1, "PROPN", "B-PROPN|NameType=Sur", "B", 0.55
            ),
            R.pos_giv.PosToken(
                "鹿", 1, 2, "PROPN", "I-PROPN|NameType=Giv", "I", 0.51
            ),
            R.pos_giv.PosToken(
                "孤", 2, 3, "PROPN", "I-PROPN|NameType=Giv", "I", 0.84
            ),
            R.pos_giv.PosToken("有", 5, 6, "VERB", "VERB", None, 0.98),
        )
        context = R.Ctx(
            "狐鹿孤单于有弟", set(), corpus, 1, 1, None, 1, tokens=tokens
        )
        self.assertEqual(
            "狐鹿孤单于",
            R.rule_foreign_title_name(context, 0)[2],
        )

    def test_foreign_title_combines_following_complete_name(self):
        cases = (
            ("柔然可汗阿那瓌将至", ("柔", "然"), ("阿", "那", "瓌")),
            ("突骑施可汗苏禄，", ("突", "骑", "施"), ("苏", "禄")),
        )
        corpus = R.Corpus(set(), {}, set())
        for text, component, name in cases:
            with self.subTest(text):
                title_start = len(component)
                name_start = title_start + 2
                tokens = []
                for offset, char in enumerate(component):
                    tokens.append(R.pos_giv.PosToken(
                        char,
                        offset,
                        offset + 1,
                        "PROPN" if offset else "NOUN",
                        ("B-" if offset == 0 else "I-") + (
                            "NOUN" if offset == 0 else "PROPN|NameType=Giv"
                        ),
                        "B" if offset == 0 else "I",
                        0.6,
                    ))
                for offset, char in enumerate(name):
                    tokens.append(R.pos_giv.PosToken(
                        char,
                        name_start + offset,
                        name_start + offset + 1,
                        "PROPN",
                        ("B-" if offset == 0 else "I-") + "PROPN|NameType=Giv",
                        "B" if offset == 0 else "I",
                        0.9,
                    ))
                if text.endswith("将至"):
                    tokens.extend((
                        R.pos_giv.PosToken(
                            "将", len(text) - 2, len(text) - 1,
                            "ADV", "ADV", None, 0.99,
                        ),
                        R.pos_giv.PosToken(
                            "至", len(text) - 1, len(text),
                            "VERB", "VERB", None, 0.99,
                        ),
                    ))
                context = R.Ctx(
                    text, set(), corpus, 1, 1, None, 1, tokens=tuple(tokens)
                )
                self.assertEqual(
                    text[:name_start + len(name)],
                    R.rule_foreign_title_name(context, 0)[2],
                )

    def test_surname_honorific_requires_person_syntax(self):
        corpus = R.Corpus(set(), {}, set())
        positives = ("董君贵", "荀卿论", "萧郎出")
        for text in positives:
            with self.subTest(text):
                tokens = (
                    R.pos_giv.PosToken(
                        text[0], 0, 1, "PROPN", "PROPN|NameType=Sur", None, 0.98
                    ),
                    R.pos_giv.PosToken(text[1], 1, 2, "NOUN", "NOUN", None, 0.98),
                    R.pos_giv.PosToken(text[2], 2, 3, "VERB", "VERB", None, 0.98),
                )
                context = R.Ctx(
                    text, set(), corpus, 1, 1, None, 1, tokens=tokens
                )
                self.assertEqual(
                    text[:2],
                    R.rule_surname_honorific(context, 0)[2],
                )

    def test_surname_honorific_rejects_rank_and_name_continuations(self):
        corpus = R.Corpus(set(), {}, set())
        for text in ("王公侯", "王公弥"):
            with self.subTest(text):
                tokens = (
                    R.pos_giv.PosToken(
                        "王", 0, 1, "PROPN", "PROPN|NameType=Sur", None, 0.98
                    ),
                    R.pos_giv.PosToken("公", 1, 2, "NOUN", "NOUN", None, 0.98),
                    R.pos_giv.PosToken(
                        text[2], 2, 3,
                        "NOUN" if text[2] == "侯" else "PROPN",
                        "NOUN" if text[2] == "侯" else "PROPN|NameType=Giv",
                        None,
                        0.98,
                    ),
                )
                context = R.Ctx(
                    text, set(), corpus, 1, 1, None, 1, tokens=tokens
                )
                self.assertIsNone(R.rule_surname_honorific(context, 0))

    def test_surname_honorific_rejects_princess_and_longer_name(self):
        cases = (
            ("朱公主", {"朱公主"}),
            ("石君立", {"石君立"}),
        )
        for text, ner in cases:
            with self.subTest(text):
                tokens = (
                    R.pos_giv.PosToken(
                        text[0], 0, 1, "PROPN", "PROPN|NameType=Sur", None, 0.98
                    ),
                    R.pos_giv.PosToken(text[1], 1, 2, "NOUN", "NOUN", None, 0.98),
                    R.pos_giv.PosToken(text[2], 2, 3, "VERB", "VERB", None, 0.98),
                )
                context = R.Ctx(
                    text, set(), R.Corpus(ner, {}, set()), 1, 1, None, 1,
                    tokens=tokens,
                )
                self.assertIsNone(R.rule_surname_honorific(context, 0))

    def test_foreign_title_prefix_is_not_retagged_as_person_name(self):
        corpus = R.Corpus({"柔然"}, {}, set())
        tokens = (
            R.pos_giv.PosToken(
                "柔", 0, 1, "PROPN", "B-PROPN|NameType=Giv", "B", 0.98
            ),
            R.pos_giv.PosToken(
                "然", 1, 2, "PROPN", "I-PROPN|NameType=Giv", "I", 0.98
            ),
            R.pos_giv.PosToken(
                "社", 4, 5, "PROPN", "B-PROPN|NameType=Giv", "B", 0.98
            ),
        )
        context = R.Ctx(
            "柔然可汗社仑",
            {0, 1, 4},
            corpus,
            1,
            1,
            None,
            1,
            tokens=tokens,
            gspans=((0, 2), (4, 5)),
        )
        self.assertIsNone(R.rule_model_ner_given_boundary(context, 0))
        self.assertIsNone(R.rule_known_fullname_pos(context, 0))

    def test_surname_honorific_exact_surface_repeats_within_jie(self):
        text = "杨公曰，归罪杨公乎"
        tokens = (
            R.pos_giv.PosToken(
                "杨", 0, 1, "PROPN", "PROPN|NameType=Sur", None, 0.98
            ),
            R.pos_giv.PosToken("公", 1, 2, "NOUN", "NOUN", None, 0.98),
            R.pos_giv.PosToken("曰", 2, 3, "VERB", "VERB", None, 0.98),
            R.pos_giv.PosToken("罪", 5, 6, "VERB", "VERB", None, 0.98),
            R.pos_giv.PosToken(
                "杨", 6, 7, "PROPN", "PROPN|NameType=Sur", None, 0.98
            ),
            R.pos_giv.PosToken("公", 7, 8, "NOUN", "NOUN", None, 0.98),
            R.pos_giv.PosToken("乎", 8, 9, "PART", "PART", None, 0.99),
        )
        corpus = R.Corpus({"杨公", "杨公乎"}, {}, set())
        cards = R.detect_juan(
            1,
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: R.pos_giv.GivOffsets(tokens=tokens)},
            corpus,
            enabled=R.PRESET_RECALL,
        )
        self.assertEqual(
            [(0, 2), (6, 8)],
            [
                (card["start"], card["end"])
                for card in cards
                if card["surface"] == "杨公"
            ],
        )

    def test_female_court_titles_require_local_morphology_and_frame(self):
        corpus = R.Corpus(set(), {}, set())
        cases = (
            (
                "妃曰华阳夫人",
                2,
                (
                    R.pos_giv.PosToken(
                        "华", 2, 3, "PROPN", "B-PROPN|NameType=Geo", "B", 0.4
                    ),
                    R.pos_giv.PosToken(
                        "阳", 3, 4, "PROPN", "I-PROPN|NameType=Giv", "I", 0.5
                    ),
                    R.pos_giv.PosToken("夫", 4, 5, "NOUN", "NOUN", None, 0.98),
                    R.pos_giv.PosToken("人", 5, 6, "NOUN", "NOUN", None, 0.98),
                ),
                ((2, 4),),
            ),
            (
                "萧淑妃有宠",
                0,
                (
                    R.pos_giv.PosToken(
                        "萧", 0, 1, "PROPN", "PROPN|NameType=Sur", None, 0.98
                    ),
                    R.pos_giv.PosToken("淑", 1, 2, "VERB", "VERB", None, 0.4),
                    R.pos_giv.PosToken("妃", 2, 3, "NOUN", "NOUN", None, 0.98),
                    R.pos_giv.PosToken("有", 3, 4, "VERB", "VERB", None, 0.98),
                ),
                (),
            ),
        )
        for text, start, tokens, gspans in cases:
            with self.subTest(text):
                context = R.Ctx(
                    text, set(), corpus, 1, 1, None, 1,
                    tokens=tokens, gspans=gspans,
                )
                hit = R.rule_female_court_title(context, start)
                self.assertIsNotNone(hit)
                self.assertIn(hit[2], {"华阳夫人", "萧淑妃"})

    def test_female_court_title_rejects_generic_usage(self):
        corpus = R.Corpus(set(), {}, set())
        tokens = (
            R.pos_giv.PosToken(
                "华", 0, 1, "PROPN", "B-PROPN|NameType=Geo", "B", 0.4
            ),
            R.pos_giv.PosToken(
                "阳", 1, 2, "PROPN", "I-PROPN|NameType=Giv", "I", 0.5
            ),
            R.pos_giv.PosToken("夫", 2, 3, "NOUN", "NOUN", None, 0.98),
            R.pos_giv.PosToken("人", 3, 4, "NOUN", "NOUN", None, 0.98),
        )
        context = R.Ctx(
            "华阳夫人", set(), corpus, 1, 1, None, 1,
            tokens=tokens, gspans=((0, 2),),
        )
        self.assertIsNone(R.rule_female_court_title(context, 0))

    def test_female_court_title_rejects_punctuation_component(self):
        corpus = R.Corpus(set(), {}, set())
        tokens = (
            R.pos_giv.PosToken("」", 0, 1, "NOUN", "NOUN", None, 0.4),
            R.pos_giv.PosToken("夫", 1, 2, "NOUN", "NOUN", None, 0.98),
            R.pos_giv.PosToken("人", 2, 3, "NOUN", "NOUN", None, 0.98),
        )
        context = R.Ctx(
            "」夫人曰", set(), corpus, 1, 1, None, 1, tokens=tokens
        )
        self.assertIsNone(R.rule_female_court_title(context, 0))

    def test_female_court_title_exact_surface_repeats_within_jie(self):
        text = "潘夫人曰，潘夫人之计"
        tokens = (
            R.pos_giv.PosToken(
                "潘", 0, 1, "PROPN", "PROPN|NameType=Sur", None, 0.98
            ),
            R.pos_giv.PosToken("夫", 1, 2, "NOUN", "NOUN", None, 0.98),
            R.pos_giv.PosToken("人", 2, 3, "NOUN", "NOUN", None, 0.98),
            R.pos_giv.PosToken("曰", 3, 4, "VERB", "VERB", None, 0.98),
            R.pos_giv.PosToken(
                "潘", 5, 6, "PROPN", "PROPN|NameType=Sur", None, 0.98
            ),
            R.pos_giv.PosToken("夫", 6, 7, "NOUN", "NOUN", None, 0.98),
            R.pos_giv.PosToken("人", 7, 8, "NOUN", "NOUN", None, 0.98),
        )
        corpus = R.Corpus({"潘夫人"}, {}, set())
        cards = R.detect_juan(
            1,
            [{"id": 0, "main": text, "ce_year": 1}],
            {0: R.pos_giv.GivOffsets(tokens=tokens)},
            corpus,
            enabled=R.PRESET_RECALL,
        )
        self.assertEqual(
            [(0, 3), (5, 8)],
            [
                (card["start"], card["end"])
                for card in cards
                if card["surface"] == "潘夫人"
            ],
        )

    def test_anaphora_lineage_propagates_only_later_strict_given(self):
        corpus = R.Corpus(set(), {}, set())
        text = "吴起曰，起请，起曰"
        tokens = (
            R.pos_giv.PosToken(
                "起", 4, 5, "PROPN", "PROPN|NameType=Giv", None, 0.98
            ),
            R.pos_giv.PosToken(
                "起", 7, 8, "PROPN", "PROPN|NameType=Giv", None, 0.98
            ),
        )
        context = R.Ctx(
            text, {4, 7}, corpus, 1, 1, None, 1, tokens=tokens
        )
        context.consumed[4] = True
        cards = [{
            "start": 4,
            "end": 5,
            "surface": "起",
            "chunk_type": "anaphora",
            "anchor_start": 0,
            "anchor_surface": "吴起",
        }]
        rows = R.detect_local_exact_given(context, cards)
        self.assertEqual([(7, 8, "起", "local_exact_given")], rows)

    def test_anaphora_lineage_rejects_later_only_or_cross_jie_source(self):
        corpus = R.Corpus(set(), {}, set())
        token = R.pos_giv.PosToken(
            "操", 0, 1, "PROPN", "PROPN|NameType=Giv", None, 0.98
        )
        context = R.Ctx(
            "操曰", {0}, corpus, 1, 2, None, 1, tokens=(token,)
        )
        cards = [{
            "start": 0,
            "end": 1,
            "surface": "操",
            "chunk_type": "anaphora",
            "anchor_start": 2,
            "anchor_surface": "曹操",
        }]
        self.assertEqual([], R.detect_local_exact_given(context, cards))

    def test_supernatural_frame_is_a_generic_veto_source(self):
        corpus = R.Corpus(set(), {}, set())
        context = R.Ctx(
            "玉皇授白云先生，将补真官，鸾鹤不日当降",
            set(),
            corpus,
            254,
            1,
            None,
            881,
        )

        self.assertTrue(
            R._surface_has_supernatural_frame(context, "玉皇")
        )

    def test_collective_name_before_chieftain_is_vetoed(self):
        corpus = R.Corpus({"室韦"}, {}, set())
        context = R.Ctx(
            "室韦酋长妻子",
            set(),
            corpus,
            246,
            1,
            None,
            848,
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "室韦"
        )

        self.assertIn("collective_role_continuation", candidate.vetoes)

    def test_bare_title_before_person_name_is_vetoed(self):
        corpus = R.Corpus({"赞普"}, {}, set())
        name = R.pos_giv.PosToken(
            "器弩悉弄", 2, 6, "PROPN", "PROPN|NameType=Prs"
        )
        context = R.Ctx(
            "赞普器弩悉弄",
            set(),
            corpus,
            206,
            1,
            None,
            691,
            tokens=(name,),
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "赞普"
        )

        self.assertIn("title_continuation", candidate.vetoes)

    def test_polity_veto_is_occurrence_local(self):
        text = "始毕部众至，始毕解围去"
        second = text.rfind("始毕")
        tokens = (
            R.pos_giv.PosToken(
                "始毕", 0, 2, "PROPN", "PROPN|NameType=Nat", "B"
            ),
            R.pos_giv.PosToken(
                "始毕", second, second + 2,
                "PROPN", "PROPN|NameType=Giv", "B",
            ),
        )
        context = R.Ctx(
            text, set(), R.Corpus({"始毕"}, {}, set()), 182, 1, None, 1,
            tokens=tokens,
        )

        candidates = {
            row.start: row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "始毕"
        }

        self.assertIn("local_polity_usage", candidates[0].vetoes)
        self.assertNotIn("local_polity_usage", candidates[second].vetoes)
        self.assertNotIn("jie_polity_usage", candidates[second].vetoes)

    def test_no_longer_name_veto_from_ner_only(self):
        text = "穆之乃止"
        tokens = (
            R.pos_giv.PosToken(
                "穆之", 0, 2, "PROPN", "PROPN|NameType=Giv", "B"
            ),
            R.pos_giv.PosToken("乃", 2, 3, "ADV", "ADV", "B"),
        )
        context = R.Ctx(
            text, set(), R.Corpus({"穆之", "穆之乃"}, {}, set()),
            117, 1, None, 1, tokens=tokens,
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "穆之"
        )

        self.assertNotIn("longer_name_right_continuation", candidate.vetoes)
        self.assertNotIn("proper_name_continuation", candidate.vetoes)

    def test_real_bio_name_continuation_remains_vetoed(self):
        text = "沙漠汗来"
        tokens = (
            R.pos_giv.PosToken(
                "沙", 0, 1, "PROPN", "PROPN|NameType=Sur", "B"
            ),
            R.pos_giv.PosToken(
                "漠汗", 1, 3, "PROPN", "PROPN|NameType=Giv", "I"
            ),
        )
        context = R.Ctx(
            text, set(), R.Corpus({"沙漠汗", "漠汗"}, {}, set()),
            80, 1, None, 1, tokens=tokens,
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "漠汗"
        )

        self.assertIn("longer_name_left_continuation", candidate.vetoes)

    def test_mistagged_left_verb_is_not_name_continuation(self):
        tokens = (
            R.pos_giv.PosToken(
                "将", 0, 1, "PROPN", "PROPN|NameType=Geo", "B"
            ),
            R.pos_giv.PosToken(
                "卑", 1, 2, "PROPN", "PROPN|NameType=Geo", "I"
            ),
            R.pos_giv.PosToken("君", 2, 3, "NOUN", "NOUN", "B"),
        )
        context = R.Ctx(
            "将卑君还",
            set(),
            R.Corpus({"卑君"}, {}, set()),
            53,
            1,
            None,
            1,
            tokens=tokens,
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "卑君"
        )

        self.assertNotIn("incomplete_bio_left_continuation", candidate.vetoes)

    def test_named_entity_token_boundary_remains_vetoed(self):
        token = R.pos_giv.PosToken(
            "高句丽", 0, 3, "PROPN", "PROPN|NameType=Nat", "B"
        )
        context = R.Ctx(
            "高句丽入贡", set(), R.Corpus({"高句", "高句丽"}, {}, set()),
            91, 1, None, 1, tokens=(token,),
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "高句"
        )

        self.assertIn("longer_name_right_continuation", candidate.vetoes)

    def test_standalone_handle_is_not_a_token_continuation(self):
        token = R.pos_giv.PosToken(
            "拔陵", 0, 2, "PROPN", "PROPN|NameType=Giv", "B"
        )
        context = R.Ctx(
            "拔陵进兵", set(), R.Corpus({"拔陵", "破六韩拔陵"}, {}, set()),
            150, 1, None, 1, tokens=(token,),
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "拔陵"
        )

        self.assertNotIn("longer_name_left_continuation", candidate.vetoes)
        self.assertNotIn("longer_name_right_continuation", candidate.vetoes)

    def test_geo_tag_alone_does_not_veto_person_occurrence(self):
        token = R.pos_giv.PosToken(
            "贺鲁", 0, 2, "PROPN", "PROPN|NameType=Geo", "B"
        )
        context = R.Ctx(
            "贺鲁为贼", set(), R.Corpus({"贺鲁"}, {}, set()),
            200, 1, None, 1, tokens=(token,),
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "贺鲁"
        )

        self.assertNotIn("local_polity_usage", candidate.vetoes)
        self.assertNotIn("jie_polity_usage", candidate.vetoes)

    def test_embedded_title_character_does_not_veto_person_name(self):
        token = R.pos_giv.PosToken(
            "王敬则", 0, 3, "PROPN", "PROPN|NameType=Prs", "B"
        )
        context = R.Ctx(
            "王敬则出外", set(), R.Corpus({"王敬则"}, {}, set()),
            134, 1, None, 1, tokens=(token,),
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "王敬则"
        )

        self.assertNotIn("nonperson_lexical_class", candidate.vetoes)

    def test_exact_office_title_remains_lexically_vetoed(self):
        token = R.pos_giv.PosToken("丞相", 0, 2, "NOUN", "NOUN", "B")
        context = R.Ctx(
            "丞相在内", set(), R.Corpus({"丞相"}, {}, set()),
            29, 1, None, 1, tokens=(token,),
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "丞相"
        )

        self.assertIn("nonperson_lexical_class", candidate.vetoes)

    def test_verbal_令_is_not_office_continuation(self):
        tokens = (
            R.pos_giv.PosToken(
                "菩萨", 0, 2, "PROPN", "PROPN|NameType=Giv", "B"
            ),
            R.pos_giv.PosToken("令", 2, 3, "VERB", "VERB", "B"),
        )
        context = R.Ctx(
            "菩萨令省事", set(), R.Corpus({"菩萨"}, {}, set()),
            154, 1, None, 1, tokens=tokens,
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "菩萨"
        )

        self.assertNotIn("office_continuation", candidate.vetoes)

    def test_nominal_office_continuation_remains_vetoed(self):
        token = R.pos_giv.PosToken(
            "崇政学士", 0, 4, "NOUN", "NOUN", "B"
        )
        context = R.Ctx(
            "崇政学士刘光素", set(), R.Corpus({"崇政"}, {}, set()),
            272, 1, None, 1, tokens=(token,),
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "崇政"
        )

        self.assertIn("office_continuation", candidate.vetoes)

    def test_kinship_after_title_is_not_office_continuation(self):
        context = R.Ctx(
            "潞王长子重吉",
            set(),
            R.Corpus({"潞王"}, {}, set()),
            278,
            1,
            None,
            1,
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "潞王"
        )

        self.assertNotIn("office_continuation", candidate.vetoes)

    def test_ruler_title_does_not_veto_preceding_name(self):
        for surface, title in (
            ("车犂", "单于"),
            ("始毕", "可汗"),
            ("颉利", "可汗"),
            ("继往绝", "可汗"),
        ):
            with self.subTest(surface=surface):
                token = R.pos_giv.PosToken(
                    surface, 0, len(surface),
                    "PROPN", "PROPN|NameType=Giv", "B",
                )
                context = R.Ctx(
                    surface + title + "出",
                    set(),
                    R.Corpus({surface}, {}, set()),
                    27,
                    1,
                    None,
                    1,
                    tokens=(token,),
                )

                candidate = next(
                    row
                    for row in R._combined_candidate_lattice(context, [])
                    if row.surface == surface
                )

                self.assertNotIn("title_continuation", candidate.vetoes)

    def test_title_suffix_absorbed_by_candidate_remains_vetoed(self):
        for text, surface in (
            ("史良娣生子", "史良"),
            ("慕容镇军枕戈", "慕容镇"),
        ):
            with self.subTest(surface=surface):
                token = R.pos_giv.PosToken(
                    text[:len(surface) + 1],
                    0,
                    len(surface) + 1,
                    "NOUN",
                    "NOUN",
                    "B",
                )
                context = R.Ctx(
                    text,
                    set(),
                    R.Corpus({surface}, {}, set()),
                    96,
                    1,
                    None,
                    1,
                    tokens=(token,),
                )

                candidate = next(
                    row
                    for row in R._combined_candidate_lattice(context, [])
                    if row.surface == surface
                )

                self.assertIn("title_continuation", candidate.vetoes)

    def test_title_with_following_name_remains_vetoed(self):
        context = R.Ctx(
            "武平君畔为将军",
            set(),
            R.Corpus({"武平"}, {}, set()),
            8,
            1,
            None,
            1,
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "武平"
        )

        self.assertIn("title_continuation", candidate.vetoes)

    def test_split_bio_continuation_remains_vetoed(self):
        tokens = (
            R.pos_giv.PosToken(
                "高", 0, 1, "PROPN", "PROPN|NameType=Geo", "B"
            ),
            R.pos_giv.PosToken("句", 1, 2, "NOUN", "NOUN", "B"),
            R.pos_giv.PosToken("丽", 2, 3, "NOUN", "NOUN", "I"),
        )
        context = R.Ctx(
            "高句丽将至",
            set(),
            R.Corpus({"高句"}, {}, set()),
            91,
            1,
            None,
            1,
            tokens=tokens,
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "高句"
        )

        self.assertIn("proper_name_continuation", candidate.vetoes)

    def test_mixed_morphology_left_continuation_remains_vetoed(self):
        tokens = (
            R.pos_giv.PosToken(
                "乙", 0, 1, "PROPN", "PROPN|NameType=Giv", "B"
            ),
            R.pos_giv.PosToken(
                "咄", 1, 2, "PROPN", "PROPN|NameType=Geo", "I"
            ),
            R.pos_giv.PosToken(
                "陆", 2, 3, "PROPN", "PROPN|NameType=Giv", "I"
            ),
        )
        context = R.Ctx(
            "乙咄陆立",
            set(),
            R.Corpus({"咄陆", "乙咄陆"}, {}, set()),
            195,
            1,
            None,
            1,
            tokens=tokens,
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "咄陆"
        )

        self.assertIn("longer_name_left_continuation", candidate.vetoes)

    def test_repeated_model_extension_remains_vetoed(self):
        text = "兴昔亡、继往绝二可汗至，兴昔亡去"
        context = R.Ctx(
            text,
            set(),
            R.Corpus({"兴昔", "兴昔亡"}, {}, set()),
            201,
            1,
            None,
            1,
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "兴昔"
        )

        self.assertIn("longer_name_right_continuation", candidate.vetoes)

    def test_repeated_unit_or_function_tail_is_not_name_continuation(self):
        for text, surface in (
            ("马通军至，马通军还", "马通"),
            ("杨郎何在，杨郎何为", "杨郎"),
        ):
            with self.subTest(surface=surface):
                context = R.Ctx(
                    text,
                    set(),
                    R.Corpus({surface, surface + text[len(surface):len(surface) + 1]}, {}, set()),
                    168,
                    1,
                    None,
                    1,
                )

                candidate = next(
                    row
                    for row in R._combined_candidate_lattice(context, [])
                    if row.surface == surface
                )

                self.assertNotIn(
                    "longer_name_right_continuation", candidate.vetoes
                )

    def test_bio_tagged_closing_quote_is_not_name_continuation(self):
        tokens = (
            R.pos_giv.PosToken(
                "陈", 0, 1, "PROPN", "PROPN|NameType=Sur", "B"
            ),
            R.pos_giv.PosToken("夜", 1, 2, "NOUN", "NOUN", "B"),
            R.pos_giv.PosToken("叉", 2, 3, "NOUN", "NOUN", "I"),
            R.pos_giv.PosToken(
                "」", 3, 4, "PROPN", "PROPN|NameType=Giv", "I"
            ),
        )
        context = R.Ctx(
            "陈夜叉」为前锋",
            set(),
            R.Corpus({"陈夜叉"}, {}, set()),
            261,
            1,
            None,
            1,
            tokens=tokens,
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "陈夜叉"
        )

        self.assertNotIn("proper_name_continuation", candidate.vetoes)

    def test_nominal_attack_target_is_polity_vetoed(self):
        tokens = (
            R.pos_giv.PosToken("夫", 1, 2, "NOUN", "NOUN", "B"),
            R.pos_giv.PosToken("余", 2, 3, "NOUN", "NOUN", "I"),
        )
        context = R.Ctx(
            "袭夫余，虏其王及部落",
            set(),
            R.Corpus({"夫余"}, {}, set()),
            97,
            1,
            None,
            1,
            tokens=tokens,
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "夫余"
        )

        self.assertIn("local_polity_usage", candidate.vetoes)

    def test_attack_verb_does_not_alone_make_a_person_a_polity(self):
        token = R.pos_giv.PosToken(
            "贺鲁", 1, 3, "PROPN", "PROPN|NameType=Geo", "B"
        )
        context = R.Ctx(
            "击贺鲁",
            set(),
            R.Corpus({"贺鲁"}, {}, set()),
            200,
            1,
            None,
            1,
            tokens=(token,),
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "贺鲁"
        )

        self.assertNotIn("local_polity_usage", candidate.vetoes)

    def test_person_morphology_survives_realm_suffix(self):
        token = R.pos_giv.PosToken(
            "衍辰", 0, 2, "PROPN", "PROPN|NameType=Giv", "B"
        )
        context = R.Ctx(
            "衍辰国",
            set(),
            R.Corpus({"衍辰"}, {}, set()),
            107,
            1,
            None,
            1,
            tokens=(token,),
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "衍辰"
        )

        self.assertNotIn("local_polity_usage", candidate.vetoes)

    def test_current_collective_frames_remain_vetoed(self):
        for text, surface, expected in (
            ("柔然部人至，柔然附魏", "柔然", "local_polity_usage"),
            ("宇文强盛，必为国患", "宇文", "local_polity_usage"),
            ("尔朱酷逆，痛结人神", "尔朱", "local_polity_usage"),
            ("楼船将军至，击楼船；楼船军败", "楼船", "office_continuation"),
        ):
            with self.subTest(surface=surface):
                context = R.Ctx(
                    text,
                    set(),
                    R.Corpus({surface}, {}, set()),
                    107,
                    1,
                    None,
                    1,
                )

                candidate = next(
                    row
                    for row in R._combined_candidate_lattice(context, [])
                    if row.surface == surface
                )

                self.assertTrue(candidate.vetoes)

    def test_geo_title_modifier_remains_vetoed(self):
        tokens = (
            R.pos_giv.PosToken(
                "突骑施", 0, 3, "PROPN", "PROPN|NameType=Prs", "B"
            ),
            R.pos_giv.PosToken(
                "交", 3, 4, "PROPN", "PROPN|Case=Loc|NameType=Geo", "B"
            ),
            R.pos_giv.PosToken(
                "河", 4, 5, "PROPN", "PROPN|Case=Loc|NameType=Geo", "I"
            ),
        )
        context = R.Ctx(
            "突骑施交河公主遣使",
            set(),
            R.Corpus({"突骑施"}, {}, set()),
            213,
            1,
            None,
            1,
            tokens=tokens,
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "突骑施"
        )

        self.assertIn("polity_title_continuation", candidate.vetoes)

    def test_princess_title_accepts_complete_local_components(self):
        cases = (
            (
                "鲁元公主薨",
                (
                    R.pos_giv.PosToken(
                        "鲁", 0, 1, "PROPN", "PROPN|Case=Loc|NameType=Nat"
                    ),
                    R.pos_giv.PosToken(
                        "元", 1, 2, "PROPN", "PROPN|NameType=Prs"
                    ),
                ),
                (0, 4, "鲁元公主", "princess_title"),
            ),
            (
                "太平公主与上官昭容谋",
                (
                    R.pos_giv.PosToken(
                        "太", 0, 1, "PROPN", "B-PROPN|NameType=Giv", "B"
                    ),
                    R.pos_giv.PosToken(
                        "平", 1, 2, "PROPN",
                        "I-PROPN|Case=Loc|NameType=Geo", "I"
                    ),
                ),
                (0, 4, "太平公主", "princess_title"),
            ),
            (
                "平阳昭公主薨",
                (
                    R.pos_giv.PosToken(
                        "平", 0, 1, "PROPN",
                        "B-PROPN|Case=Loc|NameType=Geo", "B"
                    ),
                    R.pos_giv.PosToken(
                        "阳", 1, 2, "PROPN",
                        "I-PROPN|Case=Loc|NameType=Geo", "I"
                    ),
                    R.pos_giv.PosToken(
                        "昭", 2, 3, "PROPN", "PROPN|NameType=Prs"
                    ),
                ),
                (0, 5, "平阳昭公主", "princess_title"),
            ),
            (
                "为千金公主，妻之",
                (
                    R.pos_giv.PosToken("为", 0, 1, "AUX", "AUX|VerbType=Cop"),
                    R.pos_giv.PosToken("千", 1, 2, "NUM", "NUM"),
                    R.pos_giv.PosToken("金", 2, 3, "NOUN", "NOUN"),
                ),
                (1, 5, "千金公主", "princess_title"),
            ),
        )
        suffix = (
            R.pos_giv.PosToken("公", 0, 1, "NOUN", "NOUN"),
            R.pos_giv.PosToken("主", 1, 2, "NOUN", "NOUN"),
        )
        for text, component_tokens, expected in cases:
            with self.subTest(text=text):
                start = expected[0]
                component_end = expected[1] - 2
                tokens = component_tokens + tuple(
                    R.pos_giv.PosToken(
                        token.text,
                        component_end + token.start,
                        component_end + token.end,
                        token.pos,
                        token.tag,
                        token.bio,
                    )
                    for token in suffix
                )
                context = R.Ctx(
                    text, set(), R.Corpus(set(), {}, set()), 1, 1, None, 1,
                    tokens=tokens,
                )
                self.assertEqual(R.rule_princess_title(context, start), expected)

    def test_princess_title_does_not_absorb_realm_or_role_prefixes(self):
        tokens = (
            R.pos_giv.PosToken(
                "隋", 0, 1, "PROPN", "PROPN|Case=Loc|NameType=Nat"
            ),
            R.pos_giv.PosToken(
                "南", 1, 2, "PROPN", "B-PROPN|Case=Loc|NameType=Geo", "B"
            ),
            R.pos_giv.PosToken(
                "阳", 2, 3, "PROPN", "I-PROPN|Case=Loc|NameType=Geo", "I"
            ),
            R.pos_giv.PosToken("公", 3, 4, "NOUN", "NOUN"),
            R.pos_giv.PosToken("主", 4, 5, "NOUN", "NOUN"),
        )
        context = R.Ctx(
            "隋南阳公主有子", set(), R.Corpus(set(), {}, set()),
            181, 1, None, 1, tokens=tokens,
        )

        self.assertIsNone(R.rule_princess_title(context, 0))
        self.assertEqual(
            R.rule_princess_title(context, 1),
            (1, 5, "南阳公主", "princess_title"),
        )

        tokens = (
            R.pos_giv.PosToken(
                "隋", 0, 1, "PROPN", "PROPN|Case=Loc|NameType=Nat"
            ),
            R.pos_giv.PosToken(
                "义", 1, 2, "PROPN", "B-PROPN|NameType=Prs", "B"
            ),
            R.pos_giv.PosToken(
                "成", 2, 3, "PROPN", "I-PROPN|NameType=Prs", "I"
            ),
            R.pos_giv.PosToken("公", 3, 4, "NOUN", "NOUN"),
            R.pos_giv.PosToken("主", 4, 5, "NOUN", "NOUN"),
        )
        context = R.Ctx(
            "隋义成公主遣使", set(), R.Corpus(set(), {}, set()),
            187, 1, None, 1, tokens=tokens,
        )

        self.assertIsNone(R.rule_princess_title(context, 0))
        self.assertEqual(
            R.rule_princess_title(context, 1),
            (1, 5, "义成公主", "princess_title"),
        )

    def test_princess_title_rejects_numeric_count_and_verb_fragment(self):
        for text, tokens in (
            (
                "十公主矺死",
                (
                    R.pos_giv.PosToken("十", 0, 1, "NUM", "NUM"),
                    R.pos_giv.PosToken("公", 1, 2, "NOUN", "NOUN"),
                    R.pos_giv.PosToken("主", 2, 3, "NOUN", "NOUN"),
                ),
            ),
            (
                "迎公主归",
                (
                    R.pos_giv.PosToken("迎", 0, 1, "VERB", "VERB"),
                    R.pos_giv.PosToken("公", 1, 2, "NOUN", "NOUN"),
                    R.pos_giv.PosToken("主", 2, 3, "NOUN", "NOUN"),
                ),
            ),
            (
                "朕当许公主入觐",
                (
                    R.pos_giv.PosToken(
                        "许", 2, 3, "PROPN", "PROPN|NameType=Sur", None, 0.26
                    ),
                    R.pos_giv.PosToken("公", 3, 4, "NOUN", "NOUN"),
                    R.pos_giv.PosToken("主", 4, 5, "NOUN", "NOUN"),
                ),
            ),
        ):
            with self.subTest(text=text):
                context = R.Ctx(
                    text, set(), R.Corpus(set(), {}, set()),
                    1, 1, None, 1, tokens=tokens,
                )
                start = 2 if text.startswith("朕") else 0
                self.assertIsNone(R.rule_princess_title(context, start))

    def test_princess_title_rejects_surname_given_verb_sequence(self):
        tokens = (
            R.pos_giv.PosToken(
                "王", 0, 1, "PROPN", "PROPN|NameType=Sur", None, 0.83
            ),
            R.pos_giv.PosToken(
                "信", 1, 2, "PROPN", "PROPN|NameType=Giv", None, 0.93
            ),
            R.pos_giv.PosToken("公", 2, 3, "NOUN", "NOUN"),
            R.pos_giv.PosToken("主", 3, 4, "NOUN", "NOUN"),
        )
        context = R.Ctx(
            "王信公主之谗", set(), R.Corpus(set(), {}, set()),
            154, 1, None, 1, tokens=tokens,
        )

        self.assertIsNone(R.rule_princess_title(context, 0))

    def test_female_title_accepts_complete_surname_morphology(self):
        cases = (
            (
                "独孤后劳之",
                (
                    R.pos_giv.PosToken(
                        "独", 0, 1, "PROPN", "B-PROPN|NameType=Sur", "B", 0.97
                    ),
                    R.pos_giv.PosToken(
                        "孤", 1, 2, "PROPN", "I-PROPN|NameType=Sur", "I", 0.98
                    ),
                    R.pos_giv.PosToken("后", 2, 3, "NOUN", "NOUN", None, 0.99),
                ),
                (0, 3, "独孤后", "empress_title"),
            ),
            (
                "韦后日夜",
                (
                    R.pos_giv.PosToken(
                        "韦", 0, 1, "PROPN", "PROPN|NameType=Sur", None, 0.98
                    ),
                    R.pos_giv.PosToken("后", 1, 2, "NOUN", "NOUN", None, 0.99),
                ),
                (0, 2, "韦后", "empress_title"),
            ),
            (
                "其母栗姬，齐人也",
                (
                    R.pos_giv.PosToken(
                        "栗", 2, 3, "PROPN", "PROPN|NameType=Sur", None, 0.42
                    ),
                    R.pos_giv.PosToken("姬", 3, 4, "NOUN", "NOUN", None, 0.61),
                ),
                (2, 4, "栗姬", "consort_title"),
            ),
        )
        for text, tokens, expected in cases:
            with self.subTest(text=text):
                context = R.Ctx(
                    text, set(), R.Corpus(set(), {}, set()),
                    1, 1, None, 1, tokens=tokens,
                )
                self.assertEqual(
                    R.rule_surname_empress(context, expected[0]),
                    expected,
                )

    def test_female_title_rejects_incomplete_or_weak_surname_readings(self):
        cases = (
            (
                "孤后劳之",
                (
                    R.pos_giv.PosToken(
                        "孤", 0, 1, "PROPN", "I-PROPN|NameType=Sur", "I", 0.98
                    ),
                    R.pos_giv.PosToken("后", 1, 2, "NOUN", "NOUN", None, 0.99),
                ),
            ),
            (
                "栗姬至",
                (
                    R.pos_giv.PosToken(
                        "栗", 0, 1, "PROPN", "PROPN|NameType=Sur", None, 0.42
                    ),
                    R.pos_giv.PosToken("姬", 1, 2, "NOUN", "NOUN", None, 0.61),
                ),
            ),
            (
                "杨后起卒",
                (
                    R.pos_giv.PosToken(
                        "杨", 0, 1, "PROPN", "PROPN|NameType=Sur", None, 0.99
                    ),
                    R.pos_giv.PosToken("后", 1, 2, "NOUN", "NOUN", None, 0.99),
                    R.pos_giv.PosToken("起", 2, 3, "VERB", "VERB", None, 0.84),
                ),
            ),
        )
        for text, tokens in cases:
            with self.subTest(text=text):
                context = R.Ctx(
                    text, set(), R.Corpus(set(), {}, set()),
                    1, 1, None, 1, tokens=tokens,
                )
                self.assertIsNone(R.rule_surname_empress(context, 0))

    def test_polity_ruler_title_accepts_complete_local_morphology(self):
        cases = (
            (
                "齐上皇如晋阳",
                (
                    R.pos_giv.PosToken(
                        "齐", 0, 1, "PROPN",
                        "PROPN|Case=Loc|NameType=Nat", None, 0.99
                    ),
                    R.pos_giv.PosToken(
                        "上", 1, 2, "NOUN", "NOUN|Case=Loc", None, 0.99
                    ),
                    R.pos_giv.PosToken("皇", 2, 3, "NOUN", "NOUN", None, 0.99),
                ),
                (0, 3, "齐上皇", "title_appellation"),
            ),
            (
                "周高祖谋伐齐",
                (
                    R.pos_giv.PosToken(
                        "周", 0, 1, "PROPN",
                        "PROPN|Case=Loc|NameType=Nat", None, 0.99
                    ),
                    R.pos_giv.PosToken(
                        "高", 1, 2, "PROPN", "PROPN|NameType=Prs", None, 0.97
                    ),
                    R.pos_giv.PosToken("祖", 2, 3, "NOUN", "NOUN", None, 0.99),
                ),
                (0, 3, "周高祖", "title_appellation"),
            ),
        )
        for text, tokens, expected in cases:
            with self.subTest(text=text):
                context = R.Ctx(
                    text, set(), R.Corpus(set(), {}, set()),
                    1, 1, None, 1, tokens=tokens,
                )
                self.assertEqual(R.rule_title_appellation(context, 0), expected)

    def test_polity_ruler_title_rejects_ordinary_ancestral_noun(self):
        tokens = (
            R.pos_giv.PosToken(
                "周", 0, 1, "PROPN",
                "PROPN|Case=Loc|NameType=Nat", None, 0.99
            ),
            R.pos_giv.PosToken("祖", 1, 2, "NOUN", "NOUN", None, 0.99),
            R.pos_giv.PosToken("先", 2, 3, "NOUN", "NOUN", None, 0.99),
        )
        context = R.Ctx(
            "周祖先之法", set(), R.Corpus(set(), {}, set()),
            1, 1, None, 1, tokens=tokens,
        )

        self.assertIsNone(R.rule_title_appellation(context, 0))

    def test_temporal_case_token_is_not_geo_title_continuation(self):
        tokens = (
            R.pos_giv.PosToken(
                "宪宗", 0, 2, "PROPN", "PROPN|NameType=Prs", "B"
            ),
            R.pos_giv.PosToken(
                "朝", 2, 3, "NOUN", "NOUN|Case=Loc", "B"
            ),
        )
        context = R.Ctx(
            "宪宗朝公卿",
            set(),
            R.Corpus({"宪宗"}, {}, set()),
            248,
            1,
            None,
            1,
            tokens=tokens,
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "宪宗"
        )

        self.assertNotIn("polity_title_continuation", candidate.vetoes)

    def test_exact_person_surface_survives_following_title(self):
        text = "号多等至，赐号多侯印"
        second = text.rfind("号多")
        tokens = (
            R.pos_giv.PosToken(
                "号多", 0, 2, "PROPN", "PROPN|NameType=Prs", "B", 0.9
            ),
            R.pos_giv.PosToken(
                "号多", second, second + 2,
                "PROPN", "PROPN|NameType=Prs", "B", 0.9,
            ),
            R.pos_giv.PosToken(
                "侯", second + 2, second + 3,
                "NOUN", "NOUN", "I", 0.9,
            ),
        )
        context = R.Ctx(
            text,
            set(),
            R.Corpus({"号多"}, {}, set()),
            49,
            1,
            None,
            1,
            tokens=tokens,
        )
        self.assertTrue(
            R._has_person_name_before_rank_title(
                context, second, second + 2
            )
        )

    def test_location_office_modifier_remains_vetoed(self):
        tokens = (
            R.pos_giv.PosToken(
                "宇文", 1, 3, "PROPN", "PROPN|NameType=Sur", "B"
            ),
            R.pos_giv.PosToken(
                "夏", 3, 4, "PROPN", "PROPN|Case=Loc|NameType=Geo", "B"
            ),
            R.pos_giv.PosToken(
                "州", 4, 5, "NOUN", "NOUN|Case=Loc", "I"
            ),
        )
        context = R.Ctx(
            "奉宇文夏州以来",
            set(),
            R.Corpus({"宇文"}, {}, set()),
            156,
            1,
            None,
            1,
            tokens=tokens,
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "宇文"
        )

        self.assertIn("location_office_continuation", candidate.vetoes)

    def test_polity_title_fragment_on_left_remains_vetoed(self):
        tokens = (
            R.pos_giv.PosToken(
                "楚", 0, 1, "PROPN", "PROPN|NameType=Nat", "B"
            ),
            R.pos_giv.PosToken("王", 1, 2, "NOUN", "NOUN", "B"),
            R.pos_giv.PosToken(
                "戊", 2, 3, "PROPN", "PROPN|NameType=Giv", "B"
            ),
        )
        context = R.Ctx(
            "楚王戊来朝",
            set(),
            R.Corpus({"王戊"}, {}, set()),
            16,
            1,
            None,
            1,
            tokens=tokens,
        )

        candidate = next(
            row
            for row in R._combined_candidate_lattice(context, [])
            if row.surface == "王戊"
        )

        self.assertIn("title_left_continuation", candidate.vetoes)


if __name__ == "__main__":
    unittest.main()
