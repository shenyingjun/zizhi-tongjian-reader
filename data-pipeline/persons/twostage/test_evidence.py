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

    def test_geo_conflict_requires_an_explicit_signal(self):
        policy = next(
            policy
            for policy in R.COMBINED_EVIDENCE_POLICIES
            if policy.name == "soft-surname-geo-recurrence-syntax"
        )
        candidate = E.Candidate(0, 3, "安禄山")
        candidate.add("model_ner_witness", "model")
        candidate.add(
            "document_person_morphology_majority",
            "document_morphology",
        )
        candidate.add("surname_shape", "name_shape")
        candidate.add("exact_document_recurrence", "document_recurrence")
        candidate.add("person_occurrence_syntax", "syntax")
        candidate.conflict("geo_nat_morphology")

        self.assertIsNone(E.decide(candidate, (policy,)))
        candidate.add("geo_nat_morphology", "model")
        self.assertEqual(policy, E.decide(candidate, (policy,)))

    def test_translation_requires_document_morphology(self):
        policy = next(
            policy
            for policy in R.COMBINED_EVIDENCE_POLICIES
            if policy.name == "soft-translation-recurrence-syntax"
        )
        candidate = E.Candidate(0, 2, "常据")
        candidate.add("model_ner_witness", "model")
        candidate.add("translation_exact_identity", "translation")
        candidate.add("exact_document_recurrence", "document_recurrence")
        candidate.add("person_occurrence_syntax", "syntax")
        candidate.conflict("missing_person_morphology")

        self.assertIsNone(E.decide(candidate, (policy,)))
        candidate.add(
            "document_person_morphology_anchor",
            "document_morphology",
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
            "document_person_morphology_anchor",
            "document_morphology",
        )
        candidate.add("surname_shape", "name_shape")
        candidate.add("exact_document_recurrence", "document_recurrence")
        candidate.add("geo_nat_morphology", "model")
        candidate.conflict("geo_nat_morphology")

        self.assertIsNone(E.decide(candidate, (policy,)))
        candidate.add("decisive_person_syntax", "syntax")
        self.assertEqual(policy, E.decide(candidate, (policy,)))


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


if __name__ == "__main__":
    unittest.main()
