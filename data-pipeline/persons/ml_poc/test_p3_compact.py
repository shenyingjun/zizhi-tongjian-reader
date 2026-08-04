import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from p3_compact import (
    CHALLENGE_COHORT,
    EXCLUDED_JUANS,
    EXPECTED_MODEL_SHA256,
    FOREIGN_JIES,
    FOREIGN_TERMS,
    RANDOM_JIES,
    ROLE_JIES,
    ROLE_TERMS,
    prepare_compact,
    select_compact,
)


class P3CompactTest(unittest.TestCase):
    def test_selects_disjoint_random_and_challenge_jies(self):
        rows = []
        for index in range(1, 201):
            rows.append({
                "juan": index,
                "jie_index": 1,
                "jie_number": 1,
                "text": (
                    "甲" * 20
                    + "太后" * (index % 17)
                    + "单于" * (index % 19)
                ),
                "segments": [],
                "characters": 20 + 2 * (
                    index % 17 + index % 19
                ),
            })

        selected = select_compact(rows, seed=7)

        self.assertEqual(
            RANDOM_JIES + ROLE_JIES + FOREIGN_JIES, len(selected)
        )
        self.assertEqual(len(selected), len({
            (row["juan"], row["jie_index"]) for row in selected
        }))
        self.assertEqual(RANDOM_JIES, sum(
            row["role"] == "probability_random" for row in selected
        ))
        self.assertEqual(ROLE_JIES, sum(
            row["role"] == "role_appellation_challenge"
            for row in selected
        ))
        self.assertEqual(FOREIGN_JIES, sum(
            row["role"] == "foreign_title_challenge"
            for row in selected
        ))
        for role, terms in (
            ("role_appellation_challenge", ROLE_TERMS),
            ("foreign_title_challenge", FOREIGN_TERMS),
        ):
            top_cohort = {
                row["juan"]
                for row in sorted(
                    rows,
                    key=lambda row: (
                        sum(row["text"].count(term) for term in terms),
                        row["characters"],
                        -row["juan"],
                        -row["jie_index"],
                    ),
                    reverse=True,
                )[:CHALLENGE_COHORT]
            }
            self.assertTrue({
                row["juan"] for row in selected if row["role"] == role
            }.issubset(top_cohort))

    def test_freeze_emits_only_selected_jies_without_role_leaks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            text_dir = root / "text"
            model_dir = root / "model"
            output_dir = root / "sealed"
            text_dir.mkdir()
            model_dir.mkdir()
            for juan in range(1, 90):
                paragraphs = [{
                    "id": juan * 10,
                    "main": (
                        "卷首标题"
                    ),
                }, {
                    "id": juan * 10 + 1,
                    "main": (
                        "①" + "甲" * 20
                        + "太后" * (juan % 17)
                        + "单于" * (juan % 19)
                    ),
                }, {
                    "id": juan * 10 + 2,
                    "main": "②" + "乙" * 30,
                }]
                (text_dir / f"juan_{juan:03d}.json").write_text(
                    json.dumps(
                        {"paragraphs": paragraphs}, ensure_ascii=False
                    ),
                    encoding="utf-8",
                )
            model_bytes = b"frozen-round-2"
            model_path = model_dir / "model.safetensors"
            model_path.write_bytes(model_bytes)
            actual_hash = hashlib.sha256(model_bytes).hexdigest()
            selected_model = root / "selected.json"
            selected_model.write_text(json.dumps({
                "model_sha256": actual_hash,
                "selection_basis": "dev only",
                "selected_mode": "target_only",
                "selected_epoch": 5,
            }), encoding="utf-8")

            with (
                patch("p3_compact.TEXT", text_dir),
                patch("p3_compact.EXPECTED_MODEL_SHA256", actual_hash),
                patch("p3_compact._git_commit_clean", return_value="abc123"),
            ):
                manifest = prepare_compact(
                    output_dir,
                    model_dir,
                    selected_model,
                    seed=11,
                )

            private = manifest["private_selected_jies"]
            self.assertEqual(20, len(private))
            self.assertEqual(20, len({
                (row["juan"], row["jie_index"]) for row in private
            }))
            self.assertFalse(
                EXCLUDED_JUANS & {row["juan"] for row in private}
            )
            expected_by_juan = {}
            for row in private:
                expected_by_juan.setdefault(row["juan"], set()).add(
                    row["jie_index"]
                )
            for public in manifest["selected"]:
                task_path = output_dir / public["task"]
                task = json.loads(task_path.read_text(encoding="utf-8"))
                task_keys = set(task)
                task_keys.update(
                    key
                    for jie in task["jies"]
                    for key in jie
                )
                self.assertTrue(
                    task_keys.isdisjoint({
                        "role", "score", "scores", "seed",
                        "model", "model_sha256",
                    })
                )
                self.assertEqual(
                    expected_by_juan[public["juan"]],
                    {jie["jie_index"] for jie in task["jies"]},
                )
                self.assertNotIn(0, {
                    jie["jie_index"] for jie in task["jies"]
                })
                self.assertEqual(
                    public["task_sha256"],
                    hashlib.sha256(task_path.read_bytes()).hexdigest(),
                )
                self.assertFalse(task_path.stat().st_mode & (
                    stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
                ))
            self.assertEqual(
                {"compact_sealed"},
                {row["role"] for row in manifest["selected"]},
            )
            self.assertEqual(
                EXPECTED_MODEL_SHA256,
                "2149e9283f239a02969b6d7663d64faf"
                "2dbb193832fe5e1bcd7a3c623aa7f90c",
            )
            for path in output_dir.iterdir():
                path.chmod(stat.S_IWRITE)


if __name__ == "__main__":
    unittest.main()
