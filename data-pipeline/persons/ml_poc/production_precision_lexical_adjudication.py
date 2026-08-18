from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_lexical_mining import _sha256
from production_precision_lexical_review import TASK_STATUS
from production_precision_lexical_teacher import OUTPUT_STATUS
from production_train import _make_read_only


ROUTING_STATUS = "ml_production_precision_lexical_third_teacher_tasks"
HIGH_CONFIDENCE = 0.95
AUDIT_RATE = 0.10


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line
    ]


def _teacher_decisions(root: Path, teacher: str) -> tuple[dict, dict]:
    manifest_path = root / "manifest.json"
    manifest = _read(manifest_path)
    decisions_path = root / "decisions.jsonl"
    if (
        manifest.get("status") != OUTPUT_STATUS
        or manifest.get("teacher") != teacher
        or manifest.get("candidate_model_blind") is not True
        or manifest.get("candidate_scores_hidden") is not True
        or manifest.get("decisions_sha256") != _sha256(decisions_path)
    ):
        raise ValueError(f"lexical {teacher} teacher binding differs")
    decisions = {}
    for task in _read_jsonl(decisions_path):
        task_id = str(task["task_id"])
        if task_id in decisions:
            raise ValueError(f"duplicate lexical {teacher} task: {task_id}")
        decisions[task_id] = {
            str(row["candidate_id"]): row for row in task["decisions"]
        }
        if len(decisions[task_id]) != len(task["decisions"]):
            raise ValueError(f"duplicate lexical {teacher} candidate: {task_id}")
    return manifest, decisions


def _audit_digest(task: dict, candidate: dict) -> str:
    key = ":".join(str(value) for value in (
        int(task["juan"]),
        int(task["jie_index"]),
        int(candidate["para_id"]),
        int(candidate["start"]),
        int(candidate["end"]),
    ))
    return hashlib.sha256(key.encode("ascii")).hexdigest()


def prepare_third_tasks(
    task_root: Path,
    teacher_a_root: Path,
    teacher_b_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"third-teacher task output exists: {output_dir}")
    task_manifest_path = task_root / "manifest.json"
    task_manifest = _read(task_manifest_path)
    manifest_a, decisions_a = _teacher_decisions(teacher_a_root, "A")
    manifest_b, decisions_b = _teacher_decisions(teacher_b_root, "B")
    if (
        task_manifest.get("status") != TASK_STATUS
        or task_manifest.get("candidate_scores_hidden") is not True
        or manifest_a.get("task_manifest_sha256") != _sha256(task_manifest_path)
        or manifest_b.get("task_manifest_sha256") != _sha256(task_manifest_path)
    ):
        raise ValueError("lexical third-teacher source binding differs")

    tasks = {}
    provisional = []
    routed = []
    for selected in task_manifest["selected"]:
        task_id = str(selected["task_id"])
        task_path = task_root / selected["task"]
        task = _read(task_path)
        if (
            _sha256(task_path) != selected["task_sha256"]
            or set(decisions_a.get(task_id, {}))
            != {str(row["candidate_id"]) for row in task["candidates"]}
            or set(decisions_b.get(task_id, {}))
            != {str(row["candidate_id"]) for row in task["candidates"]}
        ):
            raise ValueError(f"lexical third-teacher inventory differs: {task_id}")
        tasks[task_id] = (selected, task)
        for candidate in task["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            a = decisions_a[task_id][candidate_id]
            b = decisions_b[task_id][candidate_id]
            row = {
                "task_id": task_id,
                "candidate_id": candidate_id,
                "audit_digest": _audit_digest(task, candidate),
                "a_label": a["label"],
                "a_confidence": float(a["confidence"]),
                "b_label": b["label"],
                "b_confidence": float(b["confidence"]),
            }
            if (
                a["label"] == "definitely_not_person"
                and b["label"] == "definitely_not_person"
                and float(a["confidence"]) >= HIGH_CONFIDENCE
                and float(b["confidence"]) >= HIGH_CONFIDENCE
            ):
                provisional.append(row)
            else:
                row["route"] = "teacher_disagreement_or_low_confidence"
                routed.append(row)

    audit_count = math.ceil(AUDIT_RATE * len(provisional))
    audited_ids = {
        (row["task_id"], row["candidate_id"])
        for row in sorted(
            provisional,
            key=lambda row: (
                row["audit_digest"], row["task_id"], row["candidate_id"]
            ),
        )[:audit_count]
    }
    passed = []
    for row in provisional:
        if (row["task_id"], row["candidate_id"]) in audited_ids:
            row["route"] = "provisional_agreement_audit"
            routed.append(row)
        else:
            row["route"] = "provisional_agreement"
            passed.append(row)

    routed_by_task: dict[str, set[str]] = {}
    for row in routed:
        routed_by_task.setdefault(row["task_id"], set()).add(row["candidate_id"])

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        task_dir = staging / "tasks"
        task_dir.mkdir()
        selected_rows = []
        for task_id in sorted(routed_by_task):
            source_selected, source_task = tasks[task_id]
            candidate_ids = routed_by_task[task_id]
            third_task = {
                **source_task,
                "status": ROUTING_STATUS,
                "teacher_pass": "C",
                "source_task_sha256": source_selected["task_sha256"],
                "candidates": [
                    row for row in source_task["candidates"]
                    if str(row["candidate_id"]) in candidate_ids
                ],
            }
            target = task_dir / f"task_{task_id}.json"
            target.write_text(
                json.dumps(third_task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            selected_rows.append({
                "task_id": task_id,
                "juan": int(source_task["juan"]),
                "jie_index": int(source_task["jie_index"]),
                "task": str(Path("tasks") / target.name),
                "task_sha256": _sha256(target),
                "candidates": len(third_task["candidates"]),
            })
        routing_path = staging / "routing.jsonl"
        routing_path.write_text(
            "".join(
                json.dumps(row, separators=(",", ":")) + "\n"
                for row in sorted(
                    passed + routed,
                    key=lambda row: (row["task_id"], row["candidate_id"]),
                )
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "status": TASK_STATUS,
            "routing_status": ROUTING_STATUS,
            "revision": 7,
            "teacher_pass": "C",
            "candidate_model_blind": True,
            "candidate_scores_hidden": True,
            "confirmation_read": False,
            "git_commit": _git_commit_clean(),
            "source_task_manifest_sha256": _sha256(task_manifest_path),
            "teacher_a_manifest_sha256": _sha256(
                teacher_a_root / "manifest.json"
            ),
            "teacher_b_manifest_sha256": _sha256(
                teacher_b_root / "manifest.json"
            ),
            "high_confidence": HIGH_CONFIDENCE,
            "audit_rate": AUDIT_RATE,
            "audit_count": audit_count,
            "counts": {
                "source_candidates": len(provisional) + (
                    len(routed) - audit_count
                ),
                "provisional_agreements": len(provisional),
                "provisional_passed_without_audit": len(passed),
                "third_teacher_candidates": len(routed),
                "third_teacher_tasks": len(selected_rows),
            },
            "routing_sha256": _sha256(routing_path),
            "selected": selected_rows,
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
        description="Route Revision-7 lexical candidates to the third teacher."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--teacher-a", type=Path, required=True)
    parser.add_argument("--teacher-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_third_tasks(
        args.tasks, args.teacher_a, args.teacher_b, args.output
    )
    print(json.dumps({
        "status": manifest["routing_status"],
        "counts": manifest["counts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
