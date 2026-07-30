import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from p4_fresh_sealed import (
    EXCLUDED_JUANS,
    prepare_fresh_sealed,
)


class P4FreshSealedTest(unittest.TestCase):
    def test_freezes_candidate_blind_tasks_from_unused_juans(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            text_dir = root / "text"
            model_root = root / "candidate"
            output_dir = root / "sealed"
            text_dir.mkdir()
            (model_root / "model").mkdir(parents=True)
            for juan in range(1, 295):
                (text_dir / f"juan_{juan:03d}.json").write_text(
                    json.dumps({"paragraphs": [{
                        "id": juan,
                        "main": (
                            "①" + "甲" * 30
                            + "太后" * (juan % 11)
                            + "可汗" * (juan % 13)
                        ),
                    }]}, ensure_ascii=False),
                    encoding="utf-8",
                )
            artifact = {
                "files": {"model.safetensors": "abc"},
                "combined_sha256": "candidate-hash",
            }
            report = {
                "config": {"selected_epoch": 4},
                "round3_control": {
                    "dataset_manifest_sha256": "dataset-hash",
                    "model_artifact": artifact,
                },
            }
            report_path = model_root / "report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            report_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()

            with (
                patch("p4_fresh_sealed.TEXT", text_dir),
                patch(
                    "p4_fresh_sealed.EXPECTED_MODEL_REPORT_SHA256",
                    report_hash,
                ),
                patch(
                    "p4_fresh_sealed.EXPECTED_MODEL_ARTIFACT_SHA256",
                    "candidate-hash",
                ),
                patch(
                    "p4_fresh_sealed.EXPECTED_DATASET_MANIFEST_SHA256",
                    "dataset-hash",
                ),
                patch(
                    "p4_fresh_sealed._model_artifact",
                    return_value=artifact,
                ),
                patch(
                    "p4_fresh_sealed._git_commit_clean",
                    return_value="abc123",
                ),
            ):
                manifest = prepare_fresh_sealed(
                    output_dir, model_root, report_path, seed=9
                )

            private = manifest["private_selected_jies"]
            self.assertEqual(20, len(private))
            self.assertFalse(
                EXCLUDED_JUANS & {row["juan"] for row in private}
            )
            self.assertTrue(manifest["formal_evaluation"])
            self.assertTrue(manifest["candidate_blind"])
            self.assertFalse(manifest["model_predictions_generated"])
            self.assertEqual("candidate-hash", (
                manifest["selected_model"]["artifact_sha256"]
            ))
            expected = {
                (row["juan"], row["jie_index"]) for row in private
            }
            actual = set()
            for selected in manifest["selected"]:
                task_path = output_dir / selected["task"]
                task = json.loads(task_path.read_text(encoding="utf-8"))
                keys = set(task)
                keys.update(
                    key for jie in task["jies"] for key in jie
                )
                self.assertTrue(keys.isdisjoint({
                    "role", "scores", "model", "model_sha256", "seed",
                }))
                actual.update(
                    (task["juan"], jie["jie_index"]) for jie in task["jies"]
                )
                self.assertFalse(task_path.stat().st_mode & (
                    stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
                ))
            self.assertEqual(expected, actual)
            for path in output_dir.iterdir():
                path.chmod(stat.S_IWRITE)

    def test_rejects_unbound_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.json"
            report.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "report hash differs"):
                prepare_fresh_sealed(
                    root / "output", root / "model", report, seed=1
                )


if __name__ == "__main__":
    unittest.main()
