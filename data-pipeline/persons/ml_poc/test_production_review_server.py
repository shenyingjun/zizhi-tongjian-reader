from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from production_review_server import ProductionReviewStore


def write(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProductionReviewStoreTest(unittest.TestCase):
    def setUp(self):
        self.task_count = patch(
            "production_review_server.EXPECTED_TASKS", 1
        )
        self.task_count.start()
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.review = root / "review"
        self.state = root / "state"
        self.task_id = "0123456789abcdefabcd"
        self.task_path = self.review / "tasks" / f"task_{self.task_id}.json"
        self.pack_path = self.review / "review" / f"task_{self.task_id}.json"
        self.task = {
            "schema_version": 1,
            "phase": "copilot_double_pass",
            "candidate_model_blind": True,
            "juan": 1,
            "jies": [{
                "jie_index": 3,
                "jie_number": 2,
                "text": "②曹操与帝至。",
                "segments": [{
                    "para_id": 7,
                    "assembled_start": 0,
                    "assembled_end": 7,
                }],
                "annotations": [],
            }],
        }
        self.auto_id = f"copilot:{self.task_id}:7:1:3"
        self.audit_id = f"copilot:{self.task_id}:7:4:5"
        self.pack = {
            "schema_version": 1,
            "phase": "assisted",
            "training_only": True,
            "candidate_model_blind": True,
            "task_id": self.task_id,
            "juan": 1,
            "jie_index": 3,
            "initial_annotations": [{
                "para_id": 7,
                "start": 1,
                "end": 3,
                "surface": "曹操",
            }],
            "initial_decisions": {self.auto_id: "accept"},
            "candidates": [{
                "id": self.auto_id,
                "para_id": 7,
                "start": 1,
                "end": 3,
                "surface": "曹操",
                "channels": ["copilot_independent_a", "copilot_independent_b"],
                "confidence": "high",
                "review_reason": "",
            }, {
                "id": self.audit_id,
                "para_id": 7,
                "start": 4,
                "end": 5,
                "surface": "帝",
                "channels": ["copilot_independent_a", "copilot_independent_b"],
                "confidence": "low",
                "review_reason": (
                    "Predeclared 20% audit of exact non-low A/B consensus."
                ),
            }],
        }
        task_hash = write(self.task_path, self.task)
        review_hash = write(self.pack_path, self.pack)
        self.manifest_path = self.review / "manifest.json"
        write(self.manifest_path, {
            "schema_version": 1,
            "status": "ml_production_focused_review_with_negative_audit",
            "candidate_model_blind": True,
            "model_predictions_used": False,
            "negative_audit_inventory": {
                self.task_id: {"sha256": "a" * 64, "candidates": 0}
            },
            "counts": {
                "negative_jie_third_pass": 1,
                "negative_audit_review": 0,
            },
            "selected": [{
                "task_id": self.task_id,
                "task": f"tasks\\task_{self.task_id}.json",
                "task_sha256": task_hash,
                "review": f"review\\task_{self.task_id}.json",
                "review_sha256": review_hash,
            }],
        })
        self.store = ProductionReviewStore(self.review, self.state)

    def tearDown(self):
        self.temp.cleanup()
        self.task_count.stop()

    def test_initializes_from_auto_accept_without_mutating_source(self):
        source_before = self.pack_path.read_bytes()

        payload = self.store.payload(self.task_id)

        self.assertEqual(["曹操"], [
            row["surface"] for row in payload["state"]["annotations"]
        ])
        self.assertEqual([self.audit_id], payload["state"]["required_ids"])
        self.assertEqual(
            {self.auto_id: "accept"},
            payload["state"]["effective_decisions"],
        )
        self.assertEqual(source_before, self.pack_path.read_bytes())
        self.assertEqual([], list(self.state.iterdir()))

    def test_accepts_audit_candidate_and_locks(self):
        saved = self.store.save(self.task_id, {
            "annotations": self.pack["initial_annotations"] + [{
                "para_id": 7,
                "start": 4,
                "end": 5,
                "surface": "帝",
            }],
            "human_decisions": {self.audit_id: "accept"},
        })

        self.assertFalse(saved["expanded_full_union"])
        self.assertTrue(self.store.complete(self.task_id)["complete"])
        with self.assertRaisesRegex(PermissionError, "locked"):
            self.store.save(self.task_id, {
                "annotations": [],
                "human_decisions": {},
            })

    def test_failed_consensus_audit_expands_to_full_union(self):
        saved = self.store.save(self.task_id, {
            "annotations": self.pack["initial_annotations"],
            "human_decisions": {self.audit_id: "reject"},
        })

        self.assertTrue(saved["expanded_full_union"])
        self.assertEqual(
            sorted([self.audit_id, self.auto_id]),
            saved["required_ids"],
        )
        with self.assertRaisesRegex(ValueError, "1 unresolved"):
            self.store.complete(self.task_id)
        self.store.save(self.task_id, {
            "annotations": self.pack["initial_annotations"],
            "human_decisions": {
                self.audit_id: "reject",
                self.auto_id: "accept",
            },
        })
        self.assertTrue(self.store.complete(self.task_id)["complete"])

    def test_rejecting_auto_accept_requires_expanded_review(self):
        saved = self.store.save(self.task_id, {
            "annotations": [],
            "human_decisions": {
                self.auto_id: "reject",
                self.audit_id: "reject",
            },
        })

        self.assertTrue(saved["expanded_full_union"])
        self.assertTrue(self.store.complete(self.task_id)["complete"])

    def test_source_hash_change_fails_closed(self):
        self.pack["candidates"][0]["surface"] = "刘备"
        write(self.pack_path, self.pack)

        with self.assertRaisesRegex(PermissionError, "hash differs"):
            self.store.payload(self.task_id)

    def test_intermediate_review_pack_is_rejected(self):
        manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        manifest["status"] = "ml_production_focused_review"
        write(self.manifest_path, manifest)

        with self.assertRaisesRegex(ValueError, "unsupported"):
            ProductionReviewStore(self.review, self.state)

    def test_tampered_completed_state_is_revalidated(self):
        self.store.save(self.task_id, {
            "annotations": self.pack["initial_annotations"] + [{
                "para_id": 7,
                "start": 4,
                "end": 5,
                "surface": "帝",
            }],
            "human_decisions": {self.audit_id: "accept"},
        })
        self.store.complete(self.task_id)
        state_path = self.state / f"task_{self.task_id}.json"
        persisted = json.loads(state_path.read_text(encoding="utf-8"))
        persisted["human_decisions"] = {}
        write(state_path, persisted)

        with self.assertRaisesRegex(ValueError, "completed review"):
            self.store.payload(self.task_id)

    def test_consistent_completed_state_edit_breaks_receipt(self):
        self.store.save(self.task_id, {
            "annotations": self.pack["initial_annotations"] + [{
                "para_id": 7,
                "start": 4,
                "end": 5,
                "surface": "帝",
            }],
            "human_decisions": {self.audit_id: "accept"},
        })
        self.store.complete(self.task_id)
        state_path = self.state / f"task_{self.task_id}.json"
        persisted = json.loads(state_path.read_text(encoding="utf-8"))
        persisted.update({
            "expanded_full_union": True,
            "annotations": [],
            "human_decisions": {
                self.auto_id: "reject",
                self.audit_id: "reject",
            },
        })
        write(state_path, persisted)

        with self.assertRaisesRegex(PermissionError, "receipt differs"):
            self.store.payload(self.task_id)

    def test_completed_state_cannot_be_reopened(self):
        self.store.save(self.task_id, {
            "annotations": self.pack["initial_annotations"] + [{
                "para_id": 7,
                "start": 4,
                "end": 5,
                "surface": "帝",
            }],
            "human_decisions": {self.audit_id: "accept"},
        })
        self.store.complete(self.task_id)
        state_path = self.state / f"task_{self.task_id}.json"
        persisted = json.loads(state_path.read_text(encoding="utf-8"))
        persisted["complete"] = False
        write(state_path, persisted)

        with self.assertRaisesRegex(PermissionError, "cannot be reopened"):
            self.store.payload(self.task_id)

    def test_index_does_not_expose_private_role(self):
        row = self.store.index()["tasks"][0]

        self.assertEqual(self.task_id, row["task_id"])
        self.assertNotIn("role", row)
        self.assertNotIn("stratum", row)


if __name__ == "__main__":
    unittest.main()
