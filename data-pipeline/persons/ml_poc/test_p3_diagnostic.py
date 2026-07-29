import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from p3_diagnostic import prepare_diagnostic


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class P3DiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tasks = self.root / "sealed"
        self.packs = self.root / "packs"
        selected = []
        for juan in range(1, 6):
            task_path = self.tasks / f"blind_juan_{juan:03d}.json"
            write(task_path, {
                "juan": juan,
                "jies": [{
                    "jie_index": 0,
                    "text": "①曹操至。",
                    "segments": [{
                        "para_id": juan,
                        "assembled_start": 0,
                        "assembled_end": 5,
                    }],
                }],
            })
            selected.append({
                "juan": juan,
                "task_sha256": hashlib.sha256(
                    task_path.read_bytes()
                ).hexdigest(),
            })
            write(self.packs / f"assisted_juan_{juan:03d}.json", {
                "phase": "assisted",
                "diagnostic_only": True,
                "juan": juan,
                "candidates": [{
                    "id": f"copilot:{juan}:1:3",
                    "para_id": juan,
                    "start": 1,
                    "end": 3,
                    "surface": "曹操",
                    "channels": ["copilot_diagnostic"],
                    "confidence": "low",
                    "review_reason": "test",
                }],
            })
        write(self.tasks / "manifest.json", {"selected": selected})

    def tearDown(self):
        self.temp.cleanup()

    def test_packages_five_validated_packs(self):
        manifest = prepare_diagnostic(
            self.tasks, self.packs, self.root / "output"
        )

        self.assertEqual(5, manifest["counts"]["total"])
        self.assertEqual(5, manifest["counts"]["low"])
        self.assertEqual(
            {"diagnostic_assisted"},
            {row["mode"] for row in manifest["selected"]},
        )

    def test_rejects_changed_sealed_task(self):
        task_path = self.tasks / "blind_juan_001.json"
        task_path.write_text("{}", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "sealed task hash differs"):
            prepare_diagnostic(
                self.tasks, self.packs, self.root / "output"
            )

    def test_requires_five_distinct_juans(self):
        manifest_path = self.tasks / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["selected"][-1] = manifest["selected"][0]
        write(manifest_path, manifest)

        with self.assertRaisesRegex(ValueError, "five distinct"):
            prepare_diagnostic(
                self.tasks, self.packs, self.root / "output"
            )


if __name__ == "__main__":
    unittest.main()
