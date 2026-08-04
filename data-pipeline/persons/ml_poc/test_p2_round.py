import unittest

from core import Span
from p2_round import (
    aggregate_candidate_metrics,
    candidate_metrics,
    validate_copilot_pack,
)


class P2RoundTest(unittest.TestCase):
    def test_requires_complete_copilot_provenance(self):
        digest = "a" * 64
        pack = {
            "phase": "assisted",
            "juan": 12,
            "candidate_teacher": {
                "version": "copilot_v1",
                "input_sha256": {
                    name: digest for name in (
                        "prompt", "spec", "boundary_guide", "task",
                        "ml_seed", "teacher_evidence", "note_source",
                        "translation_mapping", "translation_source",
                    )
                },
                "demonstration_sha256": {
                    name: digest for name in (
                        "train.jsonl", "dev.jsonl", "pilot_holdout.jsonl"
                    )
                },
                "target_scope": {
                    "unit": "numbered_jie",
                    "jie_indexes": [0, 1],
                    "cross_jie_authorization": False,
                },
            },
            "provenance_contract": {
                "v1_used": False,
                "rules_used": False,
                "identity_fields_present": False,
                "full_juan_context_visible": True,
                "target_jie_authorization_only": True,
                "hu_notes_used": True,
                "translation_prose_transient": True,
                "human_review_required": True,
                "juan_76_labels_used": False,
            },
        }

        validate_copilot_pack(pack, 12)
        pack["candidate_teacher"]["input_sha256"].pop("prompt")
        with self.assertRaisesRegex(ValueError, "incomplete input hashes"):
            validate_copilot_pack(pack, 12)

    def test_pack_must_match_task_hash_scope_and_geometry(self):
        digest = "a" * 64
        task_digest = "b" * 64
        task = {
            "jies": [{
                "jie_index": 4,
                "text": "①曹操至。",
                "segments": [{
                    "para_id": 2,
                    "assembled_start": 0,
                    "assembled_end": 5,
                }],
            }],
        }
        pack = {
            "phase": "assisted",
            "juan": 12,
            "candidates": [{
                "id": "2:1:3",
                "para_id": 2,
                "start": 1,
                "end": 3,
                "surface": "曹操",
            }],
            "candidate_teacher": {
                "version": "copilot_v1",
                "input_sha256": {
                    name: (task_digest if name == "task" else digest)
                    for name in (
                        "prompt", "spec", "boundary_guide", "task",
                        "ml_seed", "teacher_evidence", "note_source",
                        "translation_mapping", "translation_source",
                    )
                },
                "demonstration_sha256": {
                    name: digest for name in (
                        "train.jsonl", "dev.jsonl", "pilot_holdout.jsonl"
                    )
                },
                "target_scope": {
                    "unit": "numbered_jie",
                    "jie_indexes": [4],
                    "cross_jie_authorization": False,
                },
            },
            "provenance_contract": {
                "v1_used": False,
                "rules_used": False,
                "identity_fields_present": False,
                "full_juan_context_visible": True,
                "target_jie_authorization_only": True,
                "hu_notes_used": True,
                "translation_prose_transient": True,
                "human_review_required": True,
                "juan_76_labels_used": False,
            },
        }

        validate_copilot_pack(
            pack, 12, task=task, task_sha256=task_digest
        )
        pack["candidates"][0]["surface"] = "刘备"
        with self.assertRaisesRegex(ValueError, "outside its task"):
            validate_copilot_pack(
                pack, 12, task=task, task_sha256=task_digest
            )

    def test_candidate_metrics_separate_replacement_and_addition(self):
        reference = [
            Span(1, 0, 2, "甲乙"),
            Span(1, 3, 4, "丙"),
        ]
        candidates = [
            Span(1, 0, 1, "甲"),
            Span(1, 3, 4, "丙"),
            Span(1, 5, 6, "丁"),
        ]

        report = candidate_metrics(reference, candidates)

        self.assertEqual(1, report["exact"]["true_positive"])
        self.assertEqual(
            1, report["geometry_delta"]["geometry_replacements"]
        )
        self.assertEqual(0, report["geometry_delta"]["pure_additions"])
        self.assertEqual(1, report["geometry_delta"]["pure_removals"])

    def test_aggregate_keeps_same_paragraph_geometry_in_distinct_juans(self):
        per_juan = {
            str(juan): candidate_metrics(
                [Span(1, 0, 1, surface)],
                [],
            )
            for juan, surface in ((12, "甲"), (44, "乙"))
        }

        report = aggregate_candidate_metrics(per_juan)

        self.assertEqual(2, report["reference_spans"])
        self.assertEqual(2, report["geometry_delta"]["raw_additions"])
        self.assertEqual(2, report["geometry_delta"]["pure_additions"])


if __name__ == "__main__":
    unittest.main()
