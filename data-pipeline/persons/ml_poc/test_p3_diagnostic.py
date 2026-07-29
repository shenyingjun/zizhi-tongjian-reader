import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from p3_diagnostic import prepare_diagnostic
from p3_diagnostic_freeze import freeze_diagnostic


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

    def test_freezes_locked_diagnostic_labels(self):
        packaged = self.root / "packaged"
        manifest = prepare_diagnostic(
            self.tasks, self.packs, packaged
        )
        state = packaged / "state"
        for selection in manifest["selected"]:
            juan = int(selection["juan"])
            write(state / f"juan_{juan:03d}.json", {
                "assisted": {
                    "complete": True,
                    "pack_sha256": selection["pack_sha256"],
                    "annotations": [{
                        "para_id": juan,
                        "start": 1,
                        "end": 3,
                        "surface": "曹操",
                    }],
                    "decisions": {
                        f"copilot:{juan}:1:3": "accept",
                    },
                },
            })
        base_train = packaged / "tasks" / "base.jsonl"
        base_train.write_text(json.dumps({
            "id": "juan-099-jie-0000",
            "juan": 99,
            "jie_index": 0,
            "jie_number": 1,
            "text": "曹操",
            "labels": ["B-PER", "I-PER"],
            "span_count": 1,
            "label_provenance": "human_assisted_copilot",
            "segments": [{
                "para_id": 1,
                "assembled_start": 0,
                "assembled_end": 2,
            }],
        }, ensure_ascii=False) + "\n", encoding="utf-8")

        report = freeze_diagnostic(
            packaged / "tasks",
            packaged / "assisted",
            state,
            self.root / "frozen",
            base_train=base_train,
        )

        self.assertFalse(report["formal_p3"])
        self.assertEqual(5, report["spans"])
        self.assertEqual(6, report["combined_train"]["spans"])
        self.assertEqual([1, 2, 3, 4, 5, 99],
                         report["combined_train"]["juans"])
        rows = [
            json.loads(line)
            for line in (
                self.root / "frozen" / "train_diagnostic.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            {"human_assisted_copilot_diagnostic"},
            {row["label_provenance"] for row in rows},
        )


if __name__ == "__main__":
    unittest.main()
