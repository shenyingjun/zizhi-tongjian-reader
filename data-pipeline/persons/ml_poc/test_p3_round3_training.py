import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from p3_round3_training import prepare_round3_training


def write_json(path: Path, value: object) -> str:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, juan: int, surface: str) -> str:
    text = f"①{surface}"
    row = {
        "id": f"juan-{juan:03d}-jie-0001",
        "juan": juan,
        "jie_index": 1,
        "text": text,
        "labels": ["O", "B-PER", "I-PER"],
        "span_count": 1,
    }
    path.write_text(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Round3TrainingTest(unittest.TestCase):
    def test_combines_disjoint_training_and_copies_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.jsonl"
            active = root / "active.jsonl"
            dev = root / "dev.jsonl"
            evaluation = root / "evaluation.jsonl"
            write_jsonl(base, 1, "曹操")
            active_sha256 = write_jsonl(active, 2, "刘备")
            write_jsonl(dev, 3, "孙权")
            write_jsonl(evaluation, 4, "周瑜")
            base_report = root / "base-report.json"
            active_report = root / "active-report.json"
            base_report_sha256 = write_json(base_report, {
                "model": "KoichiYasuoka/roberta-classical-chinese-base-char",
                "evaluation": {"name": "locked_blind_anchor_diagnostic"},
                "config": {
                    **__import__("p3_round3_training").EXPECTED_CONFIG,
                },
                "inputs": {
                    "train": str(base),
                    "dev": str(dev),
                    "evaluation": str(evaluation),
                },
            })
            active_report_sha256 = write_json(active_report, {
                "status": "frozen_round3_active_training",
                "formal_evaluation": False,
                "examples": 60,
                "spans": 1460,
                "candidate_decisions": {"accept": 1453, "reject": 183},
                "outputs": {
                    "train_round3_active_sha256": active_sha256,
                },
            })
            guides = []
            for name in ("guide.md", "spec.md", "spec-zh.md"):
                path = root / name
                path.write_text(name, encoding="utf-8")
                guides.append(path)
            output = root / "output"

            with patch.multiple(
                "p3_round3_training",
                _git_commit_clean=lambda: "commit",
                EXPECTED_BASE_TRAIN_SHA256=hashlib.sha256(
                    base.read_bytes()
                ).hexdigest(),
                EXPECTED_DEV_SHA256=hashlib.sha256(
                    dev.read_bytes()
                ).hexdigest(),
                EXPECTED_EVALUATION_SHA256=hashlib.sha256(
                    evaluation.read_bytes()
                ).hexdigest(),
                EXPECTED_BASE_REPORT_SHA256=base_report_sha256,
                EXPECTED_ACTIVE_REPORT_SHA256=active_report_sha256,
            ):
                manifest = prepare_round3_training(
                    base,
                    active,
                    dev,
                    evaluation,
                    base_report,
                    active_report,
                    *guides,
                    output,
                )

            self.assertEqual(2, manifest["splits"]["train"]["examples"])
            self.assertEqual([1, 2], manifest["splits"]["train"]["juans"])
            self.assertEqual(
                hashlib.sha256((output / "dev.jsonl").read_bytes()).hexdigest(),
                manifest["outputs"]["dev.jsonl"],
            )
            self.assertFalse(manifest["formal_evaluation"])

    def test_rejects_whole_juan_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {}
            for name, juan in (
                ("base", 1),
                ("active", 2),
                ("dev", 1),
                ("evaluation", 4),
            ):
                files[name] = root / f"{name}.jsonl"
                write_jsonl(files[name], juan, "曹操")
            active_sha256 = hashlib.sha256(
                files["active"].read_bytes()
            ).hexdigest()
            base_report = root / "base-report.json"
            active_report = root / "active-report.json"
            base_report_sha256 = write_json(base_report, {
                "model": "KoichiYasuoka/roberta-classical-chinese-base-char",
                "evaluation": {"name": "locked_blind_anchor_diagnostic"},
                "config": {
                    **__import__("p3_round3_training").EXPECTED_CONFIG,
                },
                "inputs": {
                    "train": str(files["base"]),
                    "dev": str(files["dev"]),
                    "evaluation": str(files["evaluation"]),
                },
            })
            active_report_sha256 = write_json(active_report, {
                "status": "frozen_round3_active_training",
                "formal_evaluation": False,
                "examples": 60,
                "spans": 1460,
                "candidate_decisions": {"accept": 1453, "reject": 183},
                "outputs": {
                    "train_round3_active_sha256": active_sha256,
                },
            })
            guide = root / "guide.md"
            guide.write_text("guide", encoding="utf-8")

            with patch.multiple(
                "p3_round3_training",
                _git_commit_clean=lambda: "commit",
                EXPECTED_BASE_TRAIN_SHA256=hashlib.sha256(
                    files["base"].read_bytes()
                ).hexdigest(),
                EXPECTED_DEV_SHA256=hashlib.sha256(
                    files["dev"].read_bytes()
                ).hexdigest(),
                EXPECTED_EVALUATION_SHA256=hashlib.sha256(
                    files["evaluation"].read_bytes()
                ).hexdigest(),
                EXPECTED_BASE_REPORT_SHA256=base_report_sha256,
                EXPECTED_ACTIVE_REPORT_SHA256=active_report_sha256,
            ), self.assertRaisesRegex(ValueError, "split overlap"):
                prepare_round3_training(
                    files["base"],
                    files["active"],
                    files["dev"],
                    files["evaluation"],
                    base_report,
                    active_report,
                    guide,
                    guide,
                    guide,
                    root / "output",
                )


if __name__ == "__main__":
    unittest.main()
