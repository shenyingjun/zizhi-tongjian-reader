from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_revision17_freeze import FROZEN_STATUS
from production_precision_revision17_plan import _read, _sha256
from production_precision_revision17_tasks import TASKS_STATUS
from production_train import _make_read_only


REVISION = 17
TARGET_TASK_STATUS = "ml_production_precision_revision17_target_task"
TARGETS_STATUS = "ml_production_precision_revision17_target_tasks"
BLOCKED_STATUS = "ml_production_precision_revision17_insufficient_semantic_negatives"
MIN_NOT_PERSON = 300
MIN_NOT_PERSON_JUANS = 20


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


def _target_task_id(task_id: str) -> str:
    return hashlib.sha256(
        f"revision-17-target:{task_id}".encode("ascii")
    ).hexdigest()[:24]


def build_target_inventory(
    decisions: list[dict],
    selection: list[dict],
) -> tuple[list[dict], dict]:
    by_task = {str(row["task_id"]): row for row in selection}
    if len(by_task) != len(selection):
        raise ValueError("Revision-17 sealed selection has duplicate tasks")
    joined = []
    for decision in decisions:
        task_id = str(decision["task_id"])
        selected = by_task.get(task_id)
        if (
            selected is None
            or str(selected["candidate_id"]) != str(decision["candidate_id"])
        ):
            raise ValueError("Revision-17 frozen decision join differs")
        joined.append({**selected, **decision})
    if len(joined) != len(decisions) or {
        str(row["task_id"]) for row in joined
    } != set(by_task):
        raise ValueError("Revision-17 frozen decision coverage differs")
    not_person = [row for row in joined if row["label"] == "not_person"]
    counts = {
        "not_person": len(not_person),
        "not_person_juans": len({int(row["juan"]) for row in not_person}),
        "wrong_boundary": sum(
            row["label"] == "wrong_boundary" for row in joined
        ),
    }
    return joined, counts


def freeze_target_tasks(
    task_root: Path,
    decision_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Revision-17 target tasks exist: {output_dir}")
    task_manifest_path = task_root / "manifest.json"
    task_hashes_path = task_root / "task-hashes.jsonl"
    selection_path = task_root / "sealed-selection" / "selection.jsonl"
    decision_manifest_path = decision_root / "manifest.json"
    decisions_path = decision_root / "decisions.jsonl"
    task_manifest = _read(task_manifest_path)
    decision_manifest = _read(decision_manifest_path)
    if (
        task_manifest.get("status") != TASKS_STATUS
        or task_manifest.get("outputs", {}).get("task_hashes_sha256")
        != _sha256(task_hashes_path)
        or task_manifest.get("outputs", {}).get("selection_sha256")
        != _sha256(selection_path)
        or decision_manifest.get("status") != FROZEN_STATUS
        or decision_manifest.get("selection_metadata_joined") is not False
        or decision_manifest.get("bindings", {}).get("task_manifest_sha256")
        != _sha256(task_manifest_path)
        or decision_manifest.get("bindings", {}).get("task_hashes_sha256")
        != _sha256(task_hashes_path)
        or decision_manifest.get("outputs", {}).get("decisions_sha256")
        != _sha256(decisions_path)
    ):
        raise ValueError("Revision-17 target source binding differs")

    task_hash_rows = _read_jsonl(task_hashes_path)
    task_hashes = {
        str(row["task_id"]): str(row["task_sha256"])
        for row in task_hash_rows
    }
    if (
        len(task_hashes) != len(task_hash_rows)
        or len(task_hashes)
        != int(task_manifest.get("counts", {}).get("tasks", -1))
    ):
        raise ValueError("Revision-17 source task hash inventory differs")
    decisions = _read_jsonl(decisions_path)
    selection = _read_jsonl(selection_path)
    joined, counts = build_target_inventory(decisions, selection)
    sufficient = (
        counts["not_person"] >= MIN_NOT_PERSON
        and counts["not_person_juans"] >= MIN_NOT_PERSON_JUANS
    )
    wrong_rows = [
        row for row in joined if row["label"] == "wrong_boundary"
    ]
    git_commit = _git_commit_clean()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        tasks_dir = staging / "reviewer-tasks"
        sealed_dir = staging / "sealed-join"
        tasks_dir.mkdir()
        sealed_dir.mkdir()
        joined_path = sealed_dir / "joined-decisions.jsonl"
        _write_jsonl(
            joined_path,
            sorted(joined, key=lambda row: str(row["task_id"])),
        )
        target_task_hashes = []
        if sufficient:
            for row in wrong_rows:
                source_task_path = (
                    task_root
                    / "reviewer-tasks"
                    / f"task_{row['task_id']}.json"
                )
                if _sha256(source_task_path) != task_hashes.get(
                    str(row["task_id"])
                ):
                    raise ValueError("Revision-17 source review task hash differs")
                source_task = _read(source_task_path)
                target_task_id = _target_task_id(str(row["task_id"]))
                target_task = {
                    "schema_version": 1,
                    "status": TARGET_TASK_STATUS,
                    "phase": "revision-17-blind-exact-target-review",
                    "target_task_id": target_task_id,
                    "candidate_id": str(row["candidate_id"]),
                    "review_scope": "current-numbered-jie-only",
                    "protocol": {
                        "decision": (
                            "Return every exact individual-person geometry that "
                            "overlaps the marked wrong-boundary candidate."
                        ),
                        "evidence": (
                            "Use only the complete numbered jie in this task."
                        ),
                        "independence": (
                            "Return one first judgment without seeking sibling "
                            "tasks or prior rationales."
                        ),
                    },
                    "jie": source_task["jie"],
                    "wrong_candidate": source_task["candidate"],
                    "response_schema": {
                        "uncertain": "boolean",
                        "targets": [
                            {
                                "para_id": "integer",
                                "start": "integer",
                                "end": "integer",
                                "surface": "string",
                            }
                        ],
                        "rationale": "nonempty string",
                    },
                }
                path = tasks_dir / f"task_{target_task_id}.json"
                path.write_text(
                    json.dumps(target_task, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                target_task_hashes.append({
                    "target_task_id": target_task_id,
                    "candidate_id": str(row["candidate_id"]),
                    "source_task_id": str(row["task_id"]),
                    "task_sha256": _sha256(path),
                })
        hashes_path = staging / "task-hashes.jsonl"
        _write_jsonl(
            hashes_path,
            sorted(target_task_hashes, key=lambda row: row["target_task_id"]),
        )
        manifest = {
            "schema_version": 1,
            "status": TARGETS_STATUS if sufficient else BLOCKED_STATUS,
            "revision": REVISION,
            "fit_only": True,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "formal_reserve_text_read": False,
            "selection_join_after_decision_freeze": True,
            "one_candidate_per_task": True,
            "reviewer_progress_disclosure": False,
            "git_commit": git_commit,
            "bindings": {
                "task_manifest_sha256": _sha256(task_manifest_path),
                "task_hashes_sha256": _sha256(task_hashes_path),
                "selection_sha256": _sha256(selection_path),
                "decision_manifest_sha256": _sha256(decision_manifest_path),
                "decisions_sha256": _sha256(decisions_path),
            },
            "counts": {
                **counts,
                "target_tasks": len(target_task_hashes),
            },
            "gate": {
                "minimum_not_person": MIN_NOT_PERSON,
                "minimum_not_person_juans": MIN_NOT_PERSON_JUANS,
                "passed": sufficient,
            },
            "outputs": {
                "joined_decisions_sha256": _sha256(joined_path),
                "target_task_hashes_sha256": _sha256(hashes_path),
            },
            "next_action": (
                "complete_blind_exact_target_review"
                if sufficient and wrong_rows
                else "build_revision17_training_overlay"
                if sufficient
                else "stop_insufficient_semantic_negative_diversity"
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
        description="Join frozen Revision-17 decisions and build target tasks."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_target_tasks(args.tasks, args.decisions, args.output)
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["status"] == TARGETS_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
