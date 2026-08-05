from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_grouped_data import GROUPED_DATA_STATUS
from production_precision_lexical_mining import MINING_STATUS, _sha256
from production_train import _make_read_only


TASK_STATUS = "ml_production_precision_lexical_negative_tasks"


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def prepare_tasks(
    mining_root: Path,
    grouped_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"lexical review task output exists: {output_dir}")
    mining_manifest_path = mining_root / "manifest.json"
    mining_manifest = _read(mining_manifest_path)
    retained_path = mining_root / "retained.jsonl"
    retained = _read_jsonl(retained_path)
    if (
        mining_manifest.get("status") != MINING_STATUS
        or mining_manifest.get("confirmation_read") is not False
        or mining_manifest.get("outputs", {}).get("retained_sha256")
        != _sha256(retained_path)
    ):
        raise ValueError("lexical review mining binding differs")
    grouped_manifest_path = grouped_root / "manifest.json"
    grouped_manifest = _read(grouped_manifest_path)
    examples_path = grouped_root / "examples.jsonl"
    examples = _read_jsonl(examples_path)
    if (
        grouped_manifest.get("status") != GROUPED_DATA_STATUS
        or grouped_manifest.get("outputs", {}).get("examples_sha256")
        != _sha256(examples_path)
        or mining_manifest.get("bindings", {}).get("grouped_manifest_sha256")
        != _sha256(grouped_manifest_path)
    ):
        raise ValueError("lexical review grouped binding differs")
    examples_by_id = {str(row["id"]): row for row in examples}
    by_id: dict[str, list[dict]] = {}
    candidate_ids = set()
    for row in retained:
        candidate_id = str(row["candidate_id"])
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate lexical candidate id: {candidate_id}")
        candidate_ids.add(candidate_id)
        by_id.setdefault(str(row["id"]), []).append({
            "candidate_id": candidate_id,
            "para_id": int(row["para_id"]),
            "start": int(row["start"]),
            "end": int(row["end"]),
            "surface": str(row["surface"]),
        })

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        task_dir = staging / "tasks"
        task_dir.mkdir()
        selected = []
        for identity in sorted(by_id):
            example = examples_by_id[identity]
            task_id = hashlib.sha256(
                f"revision7:{identity}".encode("ascii")
            ).hexdigest()[:20]
            task = {
                "schema_version": 1,
                "status": TASK_STATUS,
                "phase": "lexical-negative-verification",
                "task_id": task_id,
                "candidate_model_blind": True,
                "candidate_scores_hidden": True,
                "translation_hidden": True,
                "person_kb_hidden": True,
                "identity_kb_hidden": True,
                "juan": int(example["juan"]),
                "jie_index": int(example["jie_index"]),
                "instructions": (
                    "Using only this numbered jie, classify each highlighted exact "
                    "span as definitely_not_person only when it is unambiguously not "
                    "a person occurrence. Otherwise use possible_person_or_boundary. "
                    "Do not resolve identity and do not use any other jie."
                ),
                "jie": {
                    "id": identity,
                    "text": str(example["text"]),
                    "segments": example["segments"],
                },
                "candidates": sorted(
                    by_id[identity],
                    key=lambda row: (
                        row["para_id"], row["start"], row["end"]
                    ),
                ),
                "response_schema": {
                    "labels": [
                        "definitely_not_person",
                        "possible_person_or_boundary",
                    ],
                    "confidence": "finite number in [0,1]",
                    "rationale": "non-empty source-grounded string",
                },
            }
            target = task_dir / f"task_{task_id}.json"
            target.write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            selected.append({
                "task_id": task_id,
                "juan": int(example["juan"]),
                "jie_index": int(example["jie_index"]),
                "task": str(Path("tasks") / target.name),
                "task_sha256": _sha256(target),
                "candidates": len(task["candidates"]),
            })
        git_commit = _git_commit_clean()
        manifest = {
            "schema_version": 1,
            "status": TASK_STATUS,
            "revision": 7,
            "candidate_model_blind": True,
            "candidate_scores_hidden": True,
            "confirmation_read": False,
            "git_commit": git_commit,
            "mining_manifest_sha256": _sha256(mining_manifest_path),
            "grouped_manifest_sha256": _sha256(grouped_manifest_path),
            "counts": {
                "tasks": len(selected),
                "candidates": sum(row["candidates"] for row in selected),
            },
            "selected": selected,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare source-hidden revision-7 negative-review tasks."
    )
    parser.add_argument("--mining", type=Path, required=True)
    parser.add_argument("--grouped-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_tasks(args.mining, args.grouped_data, args.output)
    print(json.dumps({
        "status": manifest["status"],
        "counts": manifest["counts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
