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
