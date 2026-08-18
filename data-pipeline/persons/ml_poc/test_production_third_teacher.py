from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from production_third_teacher import merge_outputs, prepare_tasks


def write(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProductionThirdTeacherTest(unittest.TestCase):
    def setUp(self):
        self.expected = patch("production_third_teacher.EXPECTED_TASKS", 1)
        self.expected.start()
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.review = root / "review-v3"
        self.tasks = root / "third-tasks"
        self.outputs = root / "third-outputs"
        self.task_id = "0123456789abcdefabcd"
        self.disagreement_id = f"copilot:{self.task_id}:7:1:3"
        self.audit_id = f"copilot:{self.task_id}:7:4:5"
        task = {
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
        task_path = self.review / "tasks" / f"task_{self.task_id}.json"
        review_path = self.review / "review" / f"task_{self.task_id}.json"
        task_hash = write(task_path, task)
        review_hash = write(review_path, {
            "schema_version": 1,
            "phase": "assisted",
            "training_only": True,
            "candidate_model_blind": True,
            "task_id": self.task_id,
            "juan": 1,
            "jie_index": 3,
            "initial_annotations": [],
            "initial_decisions": {},
            "candidates": [{
                "id": self.disagreement_id,
                "para_id": 7,
                "start": 1,
                "end": 3,
                "surface": "曹操",
                "channels": ["copilot_independent_a"],
                "confidence": "low",
                "review_reason": (
                    "Independent candidate-blind passes disagree on this geometry."
                ),
                "pass_confidence": {"a": "high", "b": None},
            }, {
                "id": self.audit_id,
                "para_id": 7,
                "start": 4,
                "end": 5,
                "surface": "帝",
                "channels": [
                    "copilot_independent_a", "copilot_independent_b"
                ],
                "confidence": "low",
                "review_reason": (
                    "Predeclared 20% audit of exact non-low A/B consensus."
                ),
                "pass_confidence": {"a": "high", "b": "high"},
            }],
        })
        write(self.review / "private" / "selection.json", {
            "schema_version": 1,
        })
        write(self.review / "manifest.json", {
            "schema_version": 1,
            "status": "ml_production_focused_review_with_negative_audit",
            "candidate_model_blind": True,
            "model_predictions_used": False,
            "negative_audit_inventory": {
                self.task_id: {"sha256": "a" * 64, "candidates": 0}
            },
            "counts": {
                "candidate_union": 2,
                "negative_jie_third_pass": 1,
                "negative_audit_review": 0,
            },
            "selected": [{
                "task_id": self.task_id,
                "task": f"tasks\\task_{self.task_id}.json",
                "task_sha256": task_hash,
                "review": f"review\\task_{self.task_id}.json",
                "review_sha256": review_hash,
                "review_candidates": 2,
            }],
        })

    def tearDown(self):
        self.temp.cleanup()
        self.expected.stop()

    def _prepare_output(
        self,
        decision: str,
        confidence: str = "high",
        additions: list[dict] | None = None,
    ):
        manifest = prepare_tasks(self.review, self.tasks)
        self.assertEqual(
            {"tasks": 1, "candidates": 1}, manifest["counts"]
        )
        task_path = self.tasks / "tasks" / f"task_{self.task_id}.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        self.assertEqual(
            [self.disagreement_id],
            [row["id"] for row in task["candidates"]],
        )
        reason = "" if confidence == "high" else "Boundary remains uncertain."
        write(self.outputs / f"task_{self.task_id}.json", {
            "schema_version": 1,
            "phase": "third-teacher-adjudication",
            "candidate_model_blind": True,
            "candidate_sources_hidden": True,
            "model_predictions_used": False,
            "task_id": self.task_id,
            "task_sha256": task["task_sha256"],
            "adjudication_task_sha256": hashlib.sha256(
                task_path.read_bytes()
            ).hexdigest(),
            "teacher_pass": "C-source-hidden-adjudication",
            "channel": "copilot_independent_c_adjudicator",
            "decisions": [{
                "id": self.disagreement_id,
                "decision": decision,
                "confidence": confidence,
                "review_reason": reason,
            }],
            "additions": additions or [],
        })

    def test_high_accept_passes_and_audit_remains_human(self):
        self._prepare_output("accept")
        output = Path(self.temp.name) / "review-v4"

        manifest = merge_outputs(
            self.review, self.tasks, self.outputs, output
        )

        pack = json.loads(
            (output / "review" / f"task_{self.task_id}.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(
            {self.disagreement_id: "accept"}, pack["initial_decisions"]
        )
        self.assertEqual(["曹操"], [
            row["surface"] for row in pack["initial_annotations"]
        ])
        self.assertEqual(1, manifest["counts"]["human_review_candidates"])
        self.assertEqual(1, manifest["counts"]["third_teacher_high_accept"])

    def test_high_reject_passes_without_annotation(self):
        self._prepare_output("reject")
        output = Path(self.temp.name) / "review-v4-reject"

        merge_outputs(self.review, self.tasks, self.outputs, output)

        pack = json.loads(
            (output / "review" / f"task_{self.task_id}.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(
            {self.disagreement_id: "reject"}, pack["initial_decisions"]
        )
        self.assertEqual([], pack["initial_annotations"])

    def test_non_high_decision_remains_human(self):
        self._prepare_output("accept", confidence="medium")
        output = Path(self.temp.name) / "review-v4-medium"

        manifest = merge_outputs(
            self.review, self.tasks, self.outputs, output
        )

        self.assertEqual(2, manifest["counts"]["human_review_candidates"])
        self.assertEqual(1, manifest["counts"]["third_teacher_human_review"])

    def test_addition_duplicate_of_human_audit_does_not_bypass_audit(self):
        self._prepare_output("accept", additions=[{
            "id": self.audit_id,
            "para_id": 7,
            "start": 4,
            "end": 5,
            "surface": "帝",
            "confidence": "high",
            "review_reason": "",
        }])
        output = Path(self.temp.name) / "review-v4-duplicate"

        manifest = merge_outputs(
            self.review, self.tasks, self.outputs, output
        )

        pack = json.loads(
            (output / "review" / f"task_{self.task_id}.json")
            .read_text(encoding="utf-8")
        )
        self.assertNotIn(self.audit_id, pack["initial_decisions"])
        self.assertEqual(2, len(pack["candidates"]))
        self.assertEqual(1, manifest["counts"]["human_review_candidates"])
        self.assertEqual(
            1, manifest["counts"]["third_teacher_duplicate_existing"]
        )

    def test_modified_task_and_self_updated_manifest_are_rejected(self):
        self._prepare_output("accept")
        task_path = self.tasks / "tasks" / f"task_{self.task_id}.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task["candidates"] = []
        task_path.chmod(stat.S_IWRITE)
        task_hash = write(task_path, task)
        manifest_path = self.tasks / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["selected"][0]["task_sha256"] = task_hash
        manifest["selected"][0]["candidates"] = 0
        manifest["counts"]["candidates"] = 0
        manifest_path.chmod(stat.S_IWRITE)
        write(manifest_path, manifest)

        with self.assertRaisesRegex(ValueError, "task inventory differs"):
            merge_outputs(
                self.review,
                self.tasks,
                self.outputs,
                Path(self.temp.name) / "tampered-output",
            )

    def test_source_review_path_cannot_escape_artifact(self):
        manifest_path = self.review / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["selected"][0]["review"] = "..\\outside.json"
        write(manifest_path, manifest)

        with self.assertRaisesRegex(ValueError, "escapes artifact root"):
            prepare_tasks(
                self.review, Path(self.temp.name) / "escaped-tasks"
            )

    def test_task_id_cannot_escape_task_directory(self):
        manifest_path = self.review / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["selected"][0]["task_id"] = "..\\..\\escaped"
        write(manifest_path, manifest)

        with self.assertRaisesRegex(ValueError, "task inventory differs"):
            prepare_tasks(
                self.review, Path(self.temp.name) / "escaped-task-id"
            )


if __name__ == "__main__":
    unittest.main()
