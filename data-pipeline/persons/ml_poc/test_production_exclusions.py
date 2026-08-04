import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from production_exclusions import (
    REQUIRED_STATUSES,
    build_exclusion_inventory,
)


class ProductionExclusionsTest(unittest.TestCase):
    def test_builds_complete_exact_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            statuses = sorted(REQUIRED_STATUSES)
            for index, status in enumerate(statuses, 1):
                (artifacts / f"round{index}.json").write_text(json.dumps({
                    "status": status,
                    "selected_jies": [{
                        "juan": index,
                        "jie_index": index,
                    }],
                }), encoding="utf-8")
            (artifacts / "round11_reference.jsonl").write_text(
                json.dumps({"juan": 20, "jie_index": 3}) + "\n",
                encoding="utf-8",
            )
            output = root / "exclusions.json"
            hashes = {
                path.stem: __import__("hashlib").sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in artifacts.iterdir()
            }
            with patch("production_exclusions.REQUIRED_ROOT_HASHES", hashes):
                manifest = build_exclusion_inventory([artifacts], output)

            self.assertTrue(manifest["complete"])
            self.assertEqual([], manifest["completeness_checks"]["missing_statuses"])
            consumed = {
                (row["juan"], row["jie_index"])
                for row in manifest["consumed"]
            }
            self.assertIn((20, 3), consumed)
            self.assertIn((20, 3), {
                (row["juan"], row["jie_index"])
                for row in manifest["sealed"]
            })

    def test_reports_missing_status_and_unresolved_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "partial.json"
            artifact.write_text(json.dumps({
                "status": next(iter(REQUIRED_STATUSES)),
                "jies": 3,
                "selected_jies": [{"juan": 1, "jie_index": 2}],
            }), encoding="utf-8")
            output = root / "exclusions.json"

            with patch(
                "production_exclusions.REQUIRED_ROOT_HASHES",
                {"required": "f" * 64},
            ):
                manifest = build_exclusion_inventory([artifact], output)

            self.assertFalse(manifest["complete"])
            self.assertEqual(
                [],
                manifest["completeness_checks"]["unresolved_claimed_juans"],
            )
            self.assertTrue(
                manifest["completeness_checks"]["missing_statuses"]
            )
            self.assertEqual(
                ["required"],
                manifest["completeness_checks"]["missing_root_artifacts"],
            )

    def test_expands_historical_claim_to_every_numbered_jie(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "text"
            source_dir.mkdir()
            (source_dir / "juan_009.json").write_text(json.dumps({
                "paragraphs": [
                    {"id": 1, "main": "卷标题"},
                    {"id": 2, "main": "①甲"},
                    {"id": 3, "main": "②乙"},
                ],
            }, ensure_ascii=False), encoding="utf-8")
            artifacts = root / "artifacts"
            artifacts.mkdir()
            statuses = sorted(REQUIRED_STATUSES)
            for index, status in enumerate(statuses):
                (artifacts / f"{index}.json").write_text(json.dumps({
                    "status": status,
                    "excluded_juans": [9],
                }), encoding="utf-8")
            hashes = {
                path.stem: __import__("hashlib").sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in artifacts.iterdir()
            }
            output = root / "exclusions.json"

            with patch("production_exclusions.REQUIRED_ROOT_HASHES", hashes):
                manifest = build_exclusion_inventory(
                    [artifacts], output, source_dir=source_dir
                )

            self.assertTrue(manifest["complete"])
            self.assertEqual(
                {(9, 1), (9, 2)},
                {
                    (row["juan"], row["jie_index"])
                    for row in manifest["consumed"]
                },
            )
            self.assertEqual(
                "historical_conservative_whole_juan_exclusion",
                manifest["whole_juan_exclusions"][0]["reason"],
            )

    def test_refuses_existing_output_before_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "exclusions.json"
            output.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                build_exclusion_inventory([root / "missing"], output)


if __name__ == "__main__":
    unittest.main()
