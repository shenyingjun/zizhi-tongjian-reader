import json
import tempfile
import unittest
from pathlib import Path

from production_negative_audit import apply_negative_audit
from production_review import prepare_review


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class ProductionReviewTest(unittest.TestCase):
    def test_builds_focused_review_and_negative_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            round_root = root / "round"
            teachers = root / "teachers"
            output = root / "review"
            tasks = []
            private_rows = []
            for index in range(180):
                task_id = f"{index:020x}"
                name = f"task_{task_id}.json"
                task = {
                    "schema_version": 1,
                    "phase": "copilot_double_pass",
                    "candidate_model_blind": True,
                    "juan": index % 294 + 1,
                    "instructions": "blind",
                    "jies": [{
                        "jie_index": index,
                        "jie_number": 1,
                        "text": "①甲乙",
                        "segments": [{
                            "para_id": index,
                            "assembled_start": 0,
                            "assembled_end": 3,
                        }],
                        "annotations": [],
                    }],
                }
                task_path = round_root / "tasks" / name
                _write(task_path, task)
                task_hash = __import__("hashlib").sha256(
                    task_path.read_bytes()
                ).hexdigest()
                tasks.append({
                    "task_id": task_id,
                    "task": str(Path("tasks") / name),
                    "task_sha256": task_hash,
                    "source_sha256": "a" * 64,
                })
                private_rows.append({
                    "task_id": task_id,
                    "split": "train",
                    "stratum": "uniform_random",
                    "juan": task["juan"],
                    "jie_index": index,
                })
                for pass_name, teacher_pass, channel in (
                    ("pass-a", "A-recall-first", "copilot_independent_a"),
                    ("pass-b", "B-boundary-first", "copilot_independent_b"),
                ):
                    candidates = []
                    if index < 10:
                        end = 3 if pass_name == "pass-a" and index == 0 else 2
                        candidates = [{
                            "id": f"copilot:{index}:1:{end}",
                            "para_id": index,
                            "start": 1,
                            "end": end,
                            "surface": "甲乙" if end == 3 else "甲",
                            "confidence": "high",
                            "review_reason": "",
                        }]
                    _write(teachers / pass_name / name, {
                        "schema_version": 1,
                        "phase": "assisted",
                        "training_only": True,
                        "candidate_model_blind": True,
                        "task_id": task_id,
                        "task_sha256": task_hash,
                        "juan": task["juan"],
                        "jie_index": index,
                        "teacher_pass": teacher_pass,
                        "channel": channel,
                        "candidates": candidates,
                    })
            private_path = round_root / "private" / "selection.json"
            _write(private_path, {
                "schema_version": 1,
                "status": "ml_production_private_task_roles",
                "selection_seed": 1,
                "selected_jies": private_rows,
            })
            private_hash = __import__("hashlib").sha256(
                private_path.read_bytes()
            ).hexdigest()
            _write(round_root / "manifest.json", {
                "schema_version": 1,
                "status": "ml_production_round_tasks_before_labeling",
                "candidate_model_blind": True,
                "model_predictions_generated": False,
                "rules_loaded": False,
                "v1_loaded": False,
                "identity_data_loaded": False,
                "private_selection_sha256": private_hash,
                "tasks": tasks,
            })

            manifest = prepare_review(
                round_root, teachers, output, audit_seed=7
            )

            self.assertEqual(11, manifest["counts"]["candidate_union"])
            self.assertEqual(9, manifest["counts"]["exact_consensus"])
            self.assertEqual(
                2, manifest["counts"]["teacher_disagreement_review"]
            )
            self.assertEqual(
                2, manifest["counts"]["consensus_audit_review"]
            )
            self.assertEqual(
                34, manifest["counts"]["negative_jie_third_pass"]
            )
            self.assertEqual(
                34,
                len(list((output / "negative-audit-tasks").glob("*.json"))),
            )
            negative = root / "negative"
            negative_names = sorted(
                path.name
                for path in (output / "negative-audit-tasks").glob("*.json")
            )
            for offset, name in enumerate(negative_names):
                task = json.loads(
                    (output / "negative-audit-tasks" / name).read_text(
                        encoding="utf-8"
                    )
                )
                task_id = name.removeprefix("task_").removesuffix(".json")
                task_path = output / "negative-audit-tasks" / name
                candidates = []
                if offset == 0:
                    para_id = int(task["jies"][0]["segments"][0]["para_id"])
                    candidates = [{
                        "id": f"copilot:{para_id}:1:2",
                        "para_id": para_id,
                        "start": 1,
                        "end": 2,
                        "surface": "甲",
                        "confidence": "medium",
                        "review_reason": "Role use needs review.",
                    }]
                _write(negative / name, {
                    "schema_version": 1,
                    "phase": "negative-audit",
                    "training_only": True,
                    "candidate_model_blind": True,
                    "task_id": task_id,
                    "task_sha256": __import__("hashlib").sha256(
                        task_path.read_bytes()
                    ).hexdigest(),
                    "juan": task["juan"],
                    "jie_index": task["jies"][0]["jie_index"],
                    "teacher_pass": "C-negative-recall-audit",
                    "channel": "copilot_independent_c",
                    "candidates": candidates,
                })

            merged = apply_negative_audit(
                output, negative, root / "review-v2"
            )

            self.assertEqual(
                12, merged["counts"]["candidate_union"]
            )
            self.assertEqual(
                1, merged["counts"]["negative_audit_review"]
            )


if __name__ == "__main__":
    unittest.main()
