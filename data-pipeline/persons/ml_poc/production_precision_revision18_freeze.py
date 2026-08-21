from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_revision17_plan import _read, _sha256
from production_precision_revision17_target_freeze import (
    normalize_target_raw,
)
from production_precision_revision18_adjudication import (
    EXPECTED_TASKS,
    TASKS_STATUS,
    TASK_STATUS,
)
from production_train import _make_read_only


REVISION = 18
FROZEN_STATUS = "ml_production_precision_revision18_adjudication_frozen"
BLOCKED_STATUS = "ml_production_precision_revision18_adjudication_blocked"
RAW_KEYS = {
    "adjudication_task_id",
    "candidate_id",
    "label",
    "targets",
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


def normalize_raw(value: dict, task: dict) -> dict:
    if set(value) != RAW_KEYS:
        raise ValueError("Revision-18 raw fields differ")
    label = value.get("label")
    targets = value.get("targets")
    if (
        value.get("adjudication_task_id")
        != task.get("adjudication_task_id")
        or value.get("candidate_id") != task.get("candidate_id")
        or label not in task.get("allowed_labels", [])
        or not isinstance(targets, list)
        or not isinstance(value.get("rationale"), str)
        or not value["rationale"].strip()
        or value.get("reviewer") != "copilot-teacher"
        or value.get("model") != "gpt-5.6-sol"
        or (label != "wrong_boundary" and targets)
    ):
        raise ValueError("Revision-18 raw decision differs")
    if label == "wrong_boundary":
        target_value = {
            "target_task_id": task["adjudication_task_id"],
            "candidate_id": task["candidate_id"],
            "uncertain": False,
            "targets": targets,
            "rationale": value["rationale"],
            "reviewer": value["reviewer"],
            "model": value["model"],
        }
        target_task = {
            "target_task_id": task["adjudication_task_id"],
            "candidate_id": task["candidate_id"],
            "jie": task["jie"],
            "wrong_candidate": task["candidate"],
        }
        normalized_targets = normalize_target_raw(
            target_value, target_task
        )
        contradiction = normalized_targets["contradiction"]
        targets = normalized_targets["targets"]
    else:
        contradiction = False
    return {
        "adjudication_task_id": str(value["adjudication_task_id"]),
        "candidate_id": str(value["candidate_id"]),
        "label": str(label),
        "targets": targets,
        "contradiction": contradiction,
        "rationale": value["rationale"].strip(),
    }


def freeze_decisions(
    task_root: Path,
    raw_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Revision-18 freeze exists: {output_dir}")
    manifest_path = task_root / "manifest.json"
    hashes_path = task_root / "task-hashes.jsonl"
    manifest = _read(manifest_path)
    if (
        manifest.get("status") != TASKS_STATUS
        or manifest.get("prior_judgments_visible") is not False
        or manifest.get("outputs", {}).get("task_hashes_sha256")
        != _sha256(hashes_path)
    ):
        raise ValueError("Revision-18 task manifest differs")
    hash_rows = _read_jsonl(hashes_path)
    expected = {
        str(row["adjudication_task_id"]): row for row in hash_rows
    }
    if len(hash_rows) != EXPECTED_TASKS or len(expected) != EXPECTED_TASKS:
        raise ValueError("Revision-18 task hashes differ")
    raw_paths = sorted(raw_root.glob("*.json"))
    if len(raw_paths) != EXPECTED_TASKS:
        raise ValueError("Revision-18 raw coverage differs")

    decisions = []
    raw_hashes = []
    seen = set()
    for path in raw_paths:
        value = _read(path)
        task_id = str(value.get("adjudication_task_id"))
        expected_row = expected.get(task_id)
        task_path = task_root / "reviewer-tasks" / f"task_{task_id}.json"
        if (
            path.stem != task_id
            or task_id in seen
            or expected_row is None
            or _sha256(task_path) != expected_row["task_sha256"]
        ):
            raise ValueError("Revision-18 raw task differs")
        task = _read(task_path)
        if task.get("status") != TASK_STATUS:
            raise ValueError("Revision-18 task status differs")
        decisions.append({
            **normalize_raw(value, task),
            "task_sha256": expected_row["task_sha256"],
            "raw_sha256": _sha256(path),
        })
        raw_hashes.append({"path": path.name, "sha256": _sha256(path)})
        seen.add(task_id)
    if seen != set(expected):
        raise ValueError("Revision-18 tasks are missing")

    unresolved = sum(
        row["label"] == "uncertain" or row["contradiction"]
        for row in decisions
    )
    git_commit = _git_commit_clean()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        decisions_path = staging / "decisions.jsonl"
        raw_hashes_path = staging / "raw-hashes.jsonl"
        _write_jsonl(
            decisions_path,
            sorted(decisions, key=lambda row: row["adjudication_task_id"]),
        )
        _write_jsonl(raw_hashes_path, raw_hashes)
        counts = {
            label: sum(row["label"] == label for row in decisions)
            for label in ("exact_person", "wrong_boundary", "not_person", "uncertain")
        }
        frozen = {
            "schema_version": 1,
            "status": BLOCKED_STATUS if unresolved else FROZEN_STATUS,
            "revision": REVISION,
            "fit_only": True,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "formal_reserve_text_read": False,
            "first_judgments_immutable": True,
            "git_commit": git_commit,
            "bindings": {
                "task_manifest_sha256": _sha256(manifest_path),
                "task_hashes_sha256": _sha256(hashes_path),
            },
            "counts": {
                "tasks": len(decisions),
                **counts,
                "contradictions": sum(
                    row["contradiction"] for row in decisions
                ),
                "exact_targets": sum(
                    len(row["targets"]) for row in decisions
                ),
                "surface_corrections": sum(
                    target["surface_corrected"]
                    for row in decisions
                    for target in row["targets"]
                ),
                "unresolved": unresolved,
            },
            "outputs": {
                "decisions_sha256": _sha256(decisions_path),
                "raw_hashes_sha256": _sha256(raw_hashes_path),
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(frozen, indent=2) + "\n", encoding="utf-8"
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return frozen


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze Revision-18 blind adjudication decisions."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_decisions(args.tasks, args.raw, args.output)
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["status"] == FROZEN_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
