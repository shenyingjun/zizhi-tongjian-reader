from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from production_precision_lexical_teacher import (
    _batch,
    validate_teacher_output,
)


class TeacherBatchTest(unittest.TestCase):
    def test_batch_is_deterministic_and_bounded(self):
        first = _batch("0123456789abcdefabcd")
        self.assertEqual(first, _batch("0123456789abcdefabcd"))
        self.assertIn(first, range(8))


class TeacherValidationTest(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(tempfile.mkdtemp(dir=Path(__file__).parent))

    def tearDown(self):
        shutil.rmtree(self.scratch, ignore_errors=True)

    def _fixture(self, *, confidence=0.97, rationale="非人词。"):
        tasks = self.scratch / "tasks"
        batches = self.scratch / "batches"
        raw = self.scratch / "raw"
        for path in (tasks / "tasks", batches, raw):
            path.mkdir(parents=True)
        task = {
            "candidates": [{"candidate_id": "c1"}],
        }
        task_path = tasks / "tasks" / "task_t1.json"
        task_path.write_text(json.dumps(task), encoding="utf-8")
        import hashlib

        digest = hashlib.sha256(task_path.read_bytes()).hexdigest()
        task_manifest = {
            "status": "ml_production_precision_lexical_negative_tasks",
            "selected": [{
                "task_id": "t1",
                "task": "tasks/task_t1.json",
                "task_sha256": digest,
            }],
        }
        manifest_path = tasks / "manifest.json"
        manifest_path.write_text(json.dumps(task_manifest), encoding="utf-8")
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        (batches / "manifest.json").write_text(json.dumps({
            "status": "ml_production_precision_lexical_teacher_batches",
            "task_manifest_sha256": manifest_sha,
        }), encoding="utf-8")
        (raw / "t1.json").write_text(json.dumps({
            "schema_version": 1,
            "phase": "lexical-negative-verification",
            "teacher": "A",
            "task_id": "t1",
            "task_sha256": digest,
            "decisions": [{
                "candidate_id": "c1",
                "label": "definitely_not_person",
                "confidence": confidence,
                "rationale": rationale,
            }],
        }), encoding="utf-8")
        return tasks, batches, raw

    def test_accepts_complete_schema(self):
        tasks, batches, raw = self._fixture()
        result = validate_teacher_output(
            tasks, batches, raw, self.scratch / "out", teacher="A"
        )
        self.assertEqual(result["counts"], {"tasks": 1, "candidates": 1})

    def test_rejects_nonfinite_confidence(self):
        tasks, batches, raw = self._fixture(confidence=float("nan"))
        with self.assertRaises(ValueError):
            validate_teacher_output(
                tasks, batches, raw, self.scratch / "out", teacher="A"
            )

    def test_rejects_empty_rationale(self):
        tasks, batches, raw = self._fixture(rationale=" ")
        with self.assertRaises(ValueError):
            validate_teacher_output(
                tasks, batches, raw, self.scratch / "out", teacher="A"
            )


if __name__ == "__main__":
    unittest.main()
