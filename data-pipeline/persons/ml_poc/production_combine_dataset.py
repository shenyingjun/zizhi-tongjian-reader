from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _summary(rows: list[dict]) -> dict:
    return {
        "examples": len(rows),
        "characters": sum(len(row["text"]) for row in rows),
        "spans": sum(int(row["span_count"]) for row in rows),
        "juans": len({int(row["juan"]) for row in rows}),
    }


def combine_datasets(
    round1_dir: Path,
    round2_dir: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"cumulative dataset exists: {output_dir}")
    manifests = {
        1: _load_json(round1_dir / "manifest.json"),
        2: _load_json(round2_dir / "manifest.json"),
    }
    roots = {1: round1_dir, 2: round2_dir}
    rows = {}
    for number in (1, 2):
        manifest = manifests[number]
        root = roots[number]
        train_path = root / "train.jsonl"
        development_path = root / "development.jsonl"
        if (
            manifest.get("status")
            != f"ml_production_round{number}_frozen_dataset"
            or manifest.get("round", number) != number
            or manifest.get("formal_evaluation") is not False
            or manifest.get("eligible_for_training") is not True
            or manifest.get("eligible_for_checkpoint_selection") is not True
            or manifest.get("outputs", {}).get("train_sha256")
            != _sha256(train_path)
            or manifest.get("outputs", {}).get("development_sha256")
            != _sha256(development_path)
        ):
            raise ValueError(f"Round {number} dataset binding differs")
        rows[number] = {
            "train": _load_jsonl(train_path),
            "development": _load_jsonl(development_path),
        }
        if (
            len(rows[number]["train"]) != 140
            or len(rows[number]["development"]) != 40
        ):
            raise ValueError(f"Round {number} dataset inventory differs")
    round1_keys = {
        (int(row["juan"]), int(row["jie_index"]))
        for split in rows[1].values() for row in split
    }
    round2_keys = {
        (int(row["juan"]), int(row["jie_index"]))
        for split in rows[2].values() for row in split
    }
    if len(round1_keys) != 180 or len(round2_keys) != 180 or round1_keys & round2_keys:
        raise ValueError("production round geometry overlaps")
    train = sorted(
        [*rows[1]["train"], *rows[2]["train"]],
        key=lambda row: (int(row["juan"]), int(row["jie_index"])),
    )
    development = rows[2]["development"]
    train_keys = {
        (int(row["juan"]), int(row["jie_index"])) for row in train
    }
    development_keys = {
        (int(row["juan"]), int(row["jie_index"])) for row in development
    }
    if len(train) != 280 or len(development) != 40 or train_keys & development_keys:
        raise ValueError("cumulative production split geometry differs")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        train_path = staging / "train.jsonl"
        development_path = staging / "development.jsonl"
        _write_jsonl(train_path, train)
        _write_jsonl(development_path, development)
        manifest = {
            "schema_version": 1,
            "status": "ml_production_round2_cumulative_frozen_dataset",
            "round": 2,
            "training_only": False,
            "formal_evaluation": False,
            "eligible_for_training": True,
            "eligible_for_checkpoint_selection": True,
            "eligible_for_promotion": False,
            "training_composition": "round1_train_plus_round2_train",
            "development_composition": "round2_fresh_development_only",
            "splits": {
                "train": _summary(train),
                "development": _summary(development),
            },
            "inputs": {
                "round1_manifest_sha256": _sha256(round1_dir / "manifest.json"),
                "round2_manifest_sha256": _sha256(round2_dir / "manifest.json"),
            },
            "outputs": {
                "train_sha256": _sha256(train_path),
                "development_sha256": _sha256(development_path),
            },
            "git_commit": _git_commit_clean(),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in staging.iterdir():
            path.chmod(0o444)
        staging.chmod(0o555)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze cumulative Round 2 production train/development data."
    )
    parser.add_argument("--round1", type=Path, required=True)
    parser.add_argument("--round2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = combine_datasets(args.round1, args.round2, args.output)
    print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
