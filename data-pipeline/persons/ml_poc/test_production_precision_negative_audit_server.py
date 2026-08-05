from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from production_precision_lexical_safe_review import SAFE_REVIEW_STATUS
from production_precision_negative_audit_server import (
    STATIC,
    SafeNegativeAuditStore,
)


def write(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SafeNegativeAuditStoreTest(unittest.TestCase):
    def test_page_does_not_reveal_intended_negative_label(self):
        page = (STATIC / "safe_negative_audit.html").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("负例", page)
        self.assertNotIn("audit", page.lower())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.review = root / "review"
        self.state = root / "state"
        self.task_id = "0123456789abcdefabcd"
        self.candidate_id = "fedcba9876543210abcd"
        task = {
            "schema_version": 1,
            "status": SAFE_REVIEW_STATUS,
            "phase": "revision-9-blind-negative-audit",
            "task_id": self.task_id,
            "juan": 1,
            "jie_index": 2,
            "jie": {
                "text": "②曹操至。",
                "segments": [{
                    "para_id": 7,
                    "assembled_start": 0,
                    "assembled_end": 5,
                }],
            },
            "candidates": [{
                "candidate_id": self.candidate_id,
                "para_id": 7,
                "start": 1,
                "end": 3,
                "surface": "曹操",
            }],
        }
        task_path = self.review / "tasks" / f"task_{self.task_id}.json"
        task_hash = write(task_path, task)
        rationales = {
            "schema_version": 1,
            "phase": "revision-9-post-judgment-rationales",
            "task_id": self.task_id,
            "task_sha256": task_hash,
            "candidates": [{
                "candidate_id": self.candidate_id,
                "judgments": [{
                    "teacher": teacher,
                    "model": model,
                    "rationale": f"reason {teacher}",
                } for teacher, model in [
                    ("A", "claude-sonnet-5"),
                    ("B", "claude-sonnet-5"),
                    ("C", "claude-sonnet-5"),
                    ("D", "gpt-5.6-sol"),
                ]],
            }],
        }
        rationale_path = (
            self.review / "rationales" / f"task_{self.task_id}.json"
        )
        rationale_hash = write(rationale_path, rationales)
        write(self.review / "manifest.json", {
            "schema_version": 1,
            "status": SAFE_REVIEW_STATUS,
            "revision": 9,
            "confirmation_read": False,
            "candidate_model_blind": True,
            "model_predictions_used": False,
            "audit": {"sample_size": 1},
            "counts": {"audit_candidates": 1},
            "selected": [{
                "task_id": self.task_id,
                "task": f"tasks\\task_{self.task_id}.json",
                "task_sha256": task_hash,
                "rationales": f"rationales\\task_{self.task_id}.json",
                "rationales_sha256": rationale_hash,
                "candidates": 1,
            }],
        })
        self.store = SafeNegativeAuditStore(self.review, self.state)

    def tearDown(self):
        self.temp.cleanup()

    def test_rationales_are_hidden_until_immutable_initial_judgment(self):
        self.assertEqual(
            {}, self.store.payload(self.task_id)["revealed_rationales"]
        )
        with self.assertRaisesRegex(PermissionError, "initial judgment"):
            self.store.reveal(self.task_id, self.candidate_id)

        self.store.initial(self.task_id, self.candidate_id, "not_person")
        with self.assertRaisesRegex(PermissionError, "immutable"):
            self.store.initial(
                self.task_id, self.candidate_id,
                "exclude_from_negative_training",
            )
        judgments = self.store.reveal(self.task_id, self.candidate_id)

        self.assertEqual(4, len(judgments))
        self.assertIn(
            self.candidate_id,
            self.store.payload(self.task_id)["revealed_rationales"],
        )

    def test_exclusion_stops_the_full_audit(self):
        decision = self.store.initial(
            self.task_id, self.candidate_id,
            "exclude_from_negative_training",
        )

        self.assertEqual(
            "exclude_from_negative_training", decision["final"]
        )
        self.assertTrue(self.store.index()["stopped"])
        with self.assertRaisesRegex(PermissionError, "stopped"):
            self.store.initial(
                self.task_id, self.candidate_id, "not_person"
            )

    def test_zero_error_task_completes_with_receipt(self):
        self.store.initial(
            self.task_id, self.candidate_id, "not_person"
        )
        self.store.reveal(self.task_id, self.candidate_id)
        self.store.final(
            self.task_id, self.candidate_id, "not_person"
        )

        completed = self.store.complete(self.task_id)

        self.assertTrue(completed["complete"])
        self.assertEqual(64, len(completed["completion_receipt"]))
        self.assertTrue(self.store.payload(self.task_id)["state"]["complete"])


if __name__ == "__main__":
    unittest.main()
