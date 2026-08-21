from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_revision17_plan import _read, _sha256
from production_precision_revision17_tasks import (
    LABELS,
    TASKS_STATUS,
    TASK_STATUS,
)
from production_train import _make_read_only


REVISION = 17
FROZEN_STATUS = "ml_production_precision_revision17_review_frozen"
RAW_KEYS = {
    "task_id",
    "candidate_id",
    "label",
    "rationale",
    "reviewer",
    "model",
}


def _read_jsonl(path: Path) -> list[dict]:
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


def normalize_raw(value: dict) -> dict:
    if set(value) != RAW_KEYS:
        raise ValueError("Revision-17 raw review fields differ")
    if (
        value.get("label") not in LABELS
        or value.get("reviewer") != "copilot-teacher"
        or value.get("model") != "gpt-5.6-sol"
        or not isinstance(value.get("rationale"), str)
        or not value["rationale"].strip()
    ):
        raise ValueError("Revision-17 raw review decision differs")
    return {
        "task_id": str(value["task_id"]),
        "candidate_id": str(value["candidate_id"]),
        "label": str(value["label"]),
        "rationale": value["rationale"].strip(),
        "reviewer": "copilot-teacher",
        "model": "gpt-5.6-sol",
    }


def freeze_decisions(
    task_root: Path,
    raw_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Revision-17 decision freeze exists: {output_dir}")
    task_manifest_path = task_root / "manifest.json"
    task_hashes_path = task_root / "task-hashes.jsonl"
    task_manifest = _read(task_manifest_path)
    if (
        task_manifest.get("status") != TASKS_STATUS
        or task_manifest.get("confirmation_read") is not False
        or task_manifest.get("formal_reserve_text_read") is not False
        or task_manifest.get("one_candidate_per_task") is not True
        or task_manifest.get("reviewer_progress_disclosure") is not False
        or task_manifest.get("outputs", {}).get("task_hashes_sha256")
        != _sha256(task_hashes_path)
    ):
        raise ValueError("Revision-17 task manifest differs")

    task_hash_rows = _read_jsonl(task_hashes_path)
    expected = {
        str(row["task_id"]): str(row["task_sha256"])
        for row in task_hash_rows
    }
    expected_count = int(task_manifest.get("counts", {}).get("tasks", -1))
    if (
        expected_count < 1
        or len(task_hash_rows) != expected_count
        or len(expected) != expected_count
    ):
        raise ValueError("Revision-17 task hash inventory differs")

    raw_paths = sorted(raw_root.glob("*.json"))
    if len(raw_paths) != expected_count:
        raise ValueError("Revision-17 raw review coverage differs")
    decisions = []
    raw_hashes = []
    seen = set()
    for path in raw_paths:
        raw = normalize_raw(_read(path))
        task_id = raw["task_id"]
        if (
            path.stem != task_id
            or task_id in seen
            or task_id not in expected
        ):
            raise ValueError(f"Revision-17 raw review task differs: {path.name}")
        task_path = task_root / "reviewer-tasks" / f"task_{task_id}.json"
        task = _read(task_path)
        if (
            _sha256(task_path) != expected[task_id]
            or task.get("status") != TASK_STATUS
            or task.get("task_id") != task_id
            or task.get("candidate", {}).get("candidate_id")
            != raw["candidate_id"]
            or task.get("allowed_labels") != list(LABELS)
        ):
            raise ValueError(f"Revision-17 raw review binding differs: {task_id}")
        decisions.append({
            "task_id": task_id,
            "candidate_id": raw["candidate_id"],
            "label": raw["label"],
            "rationale": raw["rationale"],
            "task_sha256": expected[task_id],
            "raw_sha256": _sha256(path),
        })
        raw_hashes.append({"path": path.name, "sha256": _sha256(path)})
        seen.add(task_id)
    if seen != set(expected):
        raise ValueError("Revision-17 raw review tasks are missing")

    counts = {
        label: sum(row["label"] == label for row in decisions)
        for label in LABELS
    }
    git_commit = _git_commit_clean()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        decisions_path = staging / "decisions.jsonl"
        _write_jsonl(
            decisions_path,
            sorted(decisions, key=lambda row: row["task_id"]),
        )
        raw_hashes_path = staging / "raw-hashes.jsonl"
        _write_jsonl(raw_hashes_path, raw_hashes)
        manifest = {
            "schema_version": 1,
            "status": FROZEN_STATUS,
            "revision": REVISION,
            "fit_only": True,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "formal_reserve_text_read": False,
            "first_judgments_immutable": True,
            "selection_metadata_joined": False,
            "git_commit": git_commit,
            "bindings": {
                "task_manifest_sha256": _sha256(task_manifest_path),
                "task_hashes_sha256": _sha256(task_hashes_path),
            },
            "counts": {"tasks": len(decisions), **counts},
            "outputs": {
                "decisions_sha256": _sha256(decisions_path),
                "raw_hashes_sha256": _sha256(raw_hashes_path),
            },
            "next_action": (
                "join_frozen_decisions_to_sealed_selection_then_build_blind_"
                "targets_for_wrong_boundaries"
            ),
            "claim_limit": (
                "Candidate-blind fit-only first judgments; not formal-grade "
                "evidence."
            ),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze Revision-17 blind first judgments."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_decisions(args.tasks, args.raw, args.output)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
