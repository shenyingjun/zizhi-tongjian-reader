import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from p3_active_finalize import finalize_active_review


def write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


class P3ActiveFinalizeTest(unittest.TestCase):
    def test_freezes_locked_consistent_decisions_as_bio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review = root / "review"
            selected = []
            for juan in range(1, 61):
                task_path = (
                    review / "tasks" / f"blind_juan_{juan:03d}.json"
                )
                task_sha256 = write(task_path, {
                    "phase": "assisted",
                    "juan": juan,
                    "jies": [{
                        "jie_index": 1,
                        "jie_number": 1,
                        "text": "①曹操与刘备。",
                        "segments": [{
                            "para_id": 2,
                            "assembled_start": 0,
                            "assembled_end": 7,
                        }],
                    }],
                })
                pack_path = (
                    review / "assisted"
                    / f"assisted_juan_{juan:03d}.json"
                )
                initial = []
                candidates = []
                annotations = []
                decisions = {}
                if juan == 1:
                    initial = [{
                        "para_id": 2,
                        "start": 1,
                        "end": 3,
                        "surface": "曹操",
                    }]
                    candidates = [{
                        "id": "copilot:2:1:3",
                        "para_id": 2,
                        "start": 1,
                        "end": 3,
                        "surface": "曹操",
                    }, {
                        "id": "copilot:2:4:6",
                        "para_id": 2,
                        "start": 4,
                        "end": 6,
                        "surface": "刘备",
                    }]
                    annotations = [{
                        "para_id": 2,
                        "start": 4,
                        "end": 6,
                        "surface": "刘备",
                    }]
                    decisions = {
                        "copilot:2:1:3": "reject",
                        "copilot:2:4:6": "accept",
                    }
                pack_sha256 = write(pack_path, {
                    "schema_version": 1,
                    "phase": "assisted",
                    "juan": juan,
                    "active_learning_round": 3,
                    "diagnostic_only": True,
                    "initial_annotations": initial,
                    "candidates": candidates,
                })
                write(
                    review / "state" / f"juan_{juan:03d}.json",
                    {
                        "assisted": {
                            "complete": True,
                            "pack_sha256": pack_sha256,
                            "annotations": annotations,
                            "decisions": decisions,
                        },
                    },
                )
                selected.append({
                    "juan": juan,
                    "mode": "active_assisted",
                    "task": task_path.name,
                    "task_sha256": task_sha256,
                    "pack_sha256": pack_sha256,
                })
            write(review / "tasks" / "manifest.json", {
                "status": "round3_active_learning_human_review",
                "formal_evaluation": False,
                "eligible_for_training_after_human_review": True,
                "selected": selected,
            })
            output = root / "frozen"

            with patch(
                "p3_active_finalize._git_commit_clean",
                return_value="commit",
            ):
                report = finalize_active_review(review, output)

            example = json.loads((output / "train_round3_active.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()[0])
            self.assertEqual(
                ["O", "O", "O", "O", "B-PER", "I-PER", "O"],
                example["labels"],
            )
            self.assertEqual({"reject": 1, "accept": 1}, report[
                "candidate_decisions"
            ])
            self.assertEqual(1, report["teacher_to_final_geometry"][
                "raw_additions"
            ])
            self.assertEqual(1, report["teacher_to_final_geometry"][
                "removals"
            ])
            self.assertFalse(report["formal_evaluation"])


if __name__ == "__main__":
    unittest.main()
