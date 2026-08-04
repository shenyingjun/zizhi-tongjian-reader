import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from production_program import (
    DEV_COUNTS,
    TRAIN_COUNTS,
    load_exact_exclusions,
    prepare_program,
    select_program_rows,
)


NUMBERS = "①②③④⑤⑥⑦⑧⑨⑩"


class ProductionProgramTest(unittest.TestCase):
    def _write_sources(self, root: Path) -> Path:
        source_dir = root / "text"
        source_dir.mkdir()
        for juan in range(1, 295):
            paragraphs = [{"id": juan * 100, "main": "卷标题"}]
            for index, number in enumerate(NUMBERS[:6], 1):
                paragraphs.append({
                    "id": juan * 100 + index,
                    "main": (
                        number + "甲" * 24 + "太后使君可汗单于字名弟兄子父母氏王公主皇后"
                    ),
                })
            (source_dir / f"juan_{juan:03d}.json").write_text(
                json.dumps({"paragraphs": paragraphs}, ensure_ascii=False),
                encoding="utf-8",
            )
        return source_dir

    def _write_exclusions(self, root: Path, *, complete: bool = True) -> Path:
        path = root / "exclusions.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "status": "ml_production_exact_jie_exclusions",
            "complete": complete,
            "inputs": [{"path": "poc-artifacts", "sha256": "a" * 64}],
            "consumed": [
                {"juan": 1, "jie_index": 1, "reason": "poc_training"},
                {"juan": 2, "jie_index": 2, "reason": "round11_sealed"},
            ],
            "sealed": [
                {"juan": 2, "jie_index": 2, "reason": "round11_sealed"},
            ],
        }), encoding="utf-8")
        return path

    def test_rejects_incomplete_exact_exclusions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_exclusions(Path(temporary), complete=False)
            with self.assertRaisesRegex(ValueError, "not complete"):
                load_exact_exclusions(path)

    def test_refuses_existing_output_before_other_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_program(
                    output, root / "missing.json",
                    source_dir=root / "missing", seed=1,
                )

    def test_freezes_candidate_blind_train_and_dev_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = self._write_sources(root)
            exclusions = self._write_exclusions(root)
            output = root / "round"
            with patch(
                "production_program._git_commit_clean", return_value="abc123"
            ):
                manifest = prepare_program(
                    output, exclusions, source_dir=source_dir, seed=20260804
                )

            private = json.loads(
                (output / "private" / "selection.json").read_text(
                    encoding="utf-8"
                )
            )
            selected = private["selected_jies"]
            self.assertEqual(
                sum(TRAIN_COUNTS.values()),
                sum(row["split"] == "train" for row in selected),
            )
            self.assertEqual(
                sum(DEV_COUNTS.values()),
                sum(row["split"] == "development" for row in selected),
            )
            keys = {(row["juan"], row["jie_index"]) for row in selected}
            self.assertEqual(len(selected), len(keys))
            self.assertNotIn((1, 1), keys)
            self.assertNotIn((2, 2), keys)
            self.assertGreaterEqual(
                manifest["sampling_frame"]["eligible_jies"] - len(selected),
                (
                    manifest["sampling_frame"]["formal_reserve"]
                    + manifest["sampling_frame"]["replacement_round_reserve"]
                ),
            )
            self.assertGreaterEqual(
                manifest["sampling_frame"]["formal_foreign_reserve"],
                20,
            )
            for task_row in manifest["tasks"]:
                path = output / task_row["task"]
                task = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(1, len(task["jies"]))
                leaked = set(task)
                leaked.update(key for jie in task["jies"] for key in jie)
                self.assertTrue(leaked.isdisjoint({
                    "seed", "stratum", "term_score", "role", "model",
                    "rules", "v1", "person_id", "identity",
                }))
                self.assertFalse(
                    path.stat().st_mode
                    & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                )
                self.assertNotIn("split", task_row)
                self.assertNotIn("stratum", task_row)

    def test_replacement_preserves_exhausted_foreign_reserve(self):
        rows = [
            {
                "juan": index + 1,
                "jie_index": 1,
                "jie_number": 1,
                "text": ("可汗" if index < 20 else "甲") + "乙" * 30,
                "segments": [],
                "characters": 32,
            }
            for index in range(220)
        ]
        counts = {
            "uniform_random": 2,
            "role_appellation": 0,
            "foreign_title": 2,
            "boundary_anaphora": 0,
        }

        selected = select_program_rows(
            rows,
            seed=20260806,
            train_counts=counts,
            dev_counts=counts,
            replacement_round=True,
        )

        self.assertEqual(8, len(selected))
        self.assertFalse(any("可汗" in row["text"] for row in selected))
        self.assertTrue(all(
            row["stratum"] == "uniform_random" for row in selected
        ))


if __name__ == "__main__":
    unittest.main()
