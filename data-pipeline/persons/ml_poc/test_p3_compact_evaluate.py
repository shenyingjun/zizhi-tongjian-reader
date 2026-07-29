import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from p3_compact_evaluate import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    _rule_span,
    bootstrap_probability,
    freeze_reference,
)


class P3CompactEvaluateTest(unittest.TestCase):
    def test_rule_surface_is_derived_from_frozen_geometry(self):
        span = _rule_span(
            {
                "para_id": 3,
                "start": 1,
                "end": 3,
                "surface": "corrupt",
            },
            {3: "①曹操至。"},
        )

        self.assertEqual("曹操", span.surface)

    def test_paired_bootstrap_is_deterministic_and_preserves_identity(self):
        rows = [
            {
                "model": {
                    "reference_spans": 10,
                    "prediction_spans": 10,
                    "true_positive": 9,
                },
                "rules": {
                    "reference_spans": 10,
                    "prediction_spans": 10,
                    "true_positive": 8,
                },
            }
            for _ in range(12)
        ]

        result = bootstrap_probability(rows, replicates=100)

        self.assertEqual(BOOTSTRAP_SEED, result["seed"])
        self.assertEqual(100, result["replicates"])
        self.assertEqual([0.9, 0.9], result["model"]["f1"]["ci95"])
        for endpoint in result["rules"]["f1"]["ci95"]:
            self.assertAlmostEqual(0.8, endpoint)
        delta = result["model_minus_rules"]["f1"]["ci95"]
        self.assertAlmostEqual(0.1, delta[0])
        self.assertAlmostEqual(0.1, delta[1])
        self.assertEqual(10_000, BOOTSTRAP_REPLICATES)

    def test_freezes_twenty_complete_hash_bound_jies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = root / "tasks"
            state = root / "state"
            model = root / "model"
            output = root / "frozen"
            tasks.mkdir()
            state.mkdir()
            model.mkdir()
            model_bytes = b"model"
            (model / "model.safetensors").write_bytes(model_bytes)
            (model / "config.json").write_text("{}", encoding="utf-8")
            model_sha256 = hashlib.sha256(model_bytes).hexdigest()
            selected = []
            private = []
            roles = (
                ["probability_random"] * 12
                + ["role_appellation_challenge"] * 4
                + ["foreign_title_challenge"] * 4
            )
            for index, role in enumerate(roles, start=1):
                task = {
                    "juan": index,
                    "jies": [{
                        "jie_index": 1,
                        "jie_number": 1,
                        "text": "①曹操至。",
                        "segments": [{
                            "para_id": index,
                            "assembled_start": 0,
                            "assembled_end": 5,
                        }],
                        "annotations": [],
                    }],
                }
                task_path = tasks / f"blind_juan_{index:03d}.json"
                task_path.write_text(
                    json.dumps(task, ensure_ascii=False), encoding="utf-8"
                )
                selected.append({
                    "juan": index,
                    "role": "compact_sealed",
                    "mode": "sealed_blind",
                    "task": task_path.name,
                    "task_sha256": hashlib.sha256(
                        task_path.read_bytes()
                    ).hexdigest(),
                })
                private.append({
                    "juan": index,
                    "jie_index": 1,
                    "role": role,
                })
                (state / f"juan_{index:03d}.json").write_text(
                    json.dumps({
                        "blind": {
                            "complete": True,
                            "annotations": [{
                                "para_id": index,
                                "start": 1,
                                "end": 3,
                                "surface": "曹操",
                            }],
                        },
                    }, ensure_ascii=False),
                    encoding="utf-8",
                )
            (tasks / "manifest.json").write_text(json.dumps({
                "formal_p3": True,
                "candidate_blind": True,
                "model_predictions_generated": False,
                "selected_model": {
                    "sha256": model_sha256,
                },
                "git_commit": "selection-commit",
                "selected": selected,
                "private_selected_jies": private,
            }), encoding="utf-8")

            with (
                patch(
                    "p3_compact_evaluate._git_commit_clean",
                    return_value="freeze-commit",
                ),
                patch(
                    "p3_compact_evaluate.EXPECTED_MODEL_SHA256",
                    model_sha256,
                ),
            ):
                report = freeze_reference(tasks, state, model, output)

            self.assertEqual(20, report["jies"])
            self.assertEqual(20, report["spans"])
            self.assertEqual("freeze-commit", report["freeze_git_commit"])
            reference = (
                output / "reference.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(20, len(reference))
            self.assertTrue(all(
                not path.stat().st_mode & (
                    stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
                )
                for path in output.iterdir()
            ))
            for path in output.iterdir():
                path.chmod(stat.S_IWRITE)


if __name__ == "__main__":
    unittest.main()
