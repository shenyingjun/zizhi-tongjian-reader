from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_lexical_adjudication import (
    _audit_digest,
    _read,
    _teacher_decisions,
)
from production_precision_lexical_mining import _sha256
from production_precision_lexical_review import TASK_STATUS
from production_train import _make_read_only


FINAL_REVIEW_STATUS = "ml_production_precision_lexical_final_human_review"
TEACHER_MODELS = {
    "A": "claude-sonnet-5",
    "B": "claude-sonnet-5",
    "C": "claude-sonnet-5",
    "D": "gpt-5.6-sol",
}
TEACHER_RUNS = {
    teacher: [f"teacher-{teacher.lower()}-{index}" for index in range(8)]
    for teacher in TEACHER_MODELS
}
AUDIT_RATE = 0.10
MIN_VERIFIED = 2000
EXPECTED_JUANS = 28


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _schema_sha256() -> str:
    schema = {
        "labels": ["definitely_not_person", "possible_person_or_boundary"],
        "decision_keys": [
            "candidate_id", "label", "confidence", "rationale"
        ],
        "confidence": "finite_number_0_to_1",
    }
    raw = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def prepare_final_review(
    task_root: Path,
    third_task_root: Path,
    teacher_a_root: Path,
    teacher_b_root: Path,
    teacher_c_root: Path,
    teacher_d_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"final lexical review output exists: {output_dir}")
    task_manifest_path = task_root / "manifest.json"
    task_manifest = _read(task_manifest_path)
    third_manifest_path = third_task_root / "manifest.json"
    third_manifest = _read(third_manifest_path)
    routing_path = third_task_root / "routing.jsonl"
    if (
        task_manifest.get("status") != TASK_STATUS
        or third_manifest.get("status") != TASK_STATUS
        or third_manifest.get("routing_sha256") != _sha256(routing_path)
        or third_manifest.get("source_task_manifest_sha256")
        != _sha256(task_manifest_path)
    ):
        raise ValueError("final lexical review task binding differs")

    roots = {
        "A": teacher_a_root,
        "B": teacher_b_root,
        "C": teacher_c_root,
        "D": teacher_d_root,
    }
    manifests = {}
    decisions = {}
    for teacher, root in roots.items():
        manifests[teacher], decisions[teacher] = _teacher_decisions(root, teacher)
    expected_task_sha = _sha256(task_manifest_path)
    if any(
        manifests[teacher].get("task_manifest_sha256")
        != (
            expected_task_sha
            if teacher != "C"
            else _sha256(third_manifest_path)
        )
        for teacher in manifests
    ):
        raise ValueError("final lexical teacher task binding differs")

    routing_rows = [
        json.loads(line)
        for line in routing_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    routing_by_candidate = {
        (str(row["task_id"]), str(row["candidate_id"])): row
        for row in routing_rows
    }
    if len(routing_by_candidate) != len(routing_rows):
        raise ValueError("duplicate final lexical routing candidate")

    tasks = {}
    candidates = {}
    for selected in task_manifest["selected"]:
        task_id = str(selected["task_id"])
        task_path = task_root / selected["task"]
        task = _read(task_path)
        if _sha256(task_path) != selected["task_sha256"]:
            raise ValueError(f"final lexical source task differs: {task_id}")
        tasks[task_id] = (selected, task)
        for candidate in task["candidates"]:
            key = (task_id, str(candidate["candidate_id"]))
            if key in candidates:
                raise ValueError(f"duplicate final lexical candidate: {key}")
            candidates[key] = candidate
    if set(candidates) != set(routing_by_candidate):
        raise ValueError("final lexical routing inventory differs")

    full_candidate_keys = set(candidates)
    for teacher in ("A", "B", "D"):
        teacher_keys = {
            (task_id, candidate_id)
            for task_id, rows in decisions[teacher].items()
            for candidate_id in rows
        }
        if teacher_keys != full_candidate_keys:
            raise ValueError(
                f"final lexical {teacher} inventory differs"
            )
    c_keys = {
        (task_id, candidate_id)
        for task_id, rows in decisions["C"].items()
        for candidate_id in rows
    }
    expected_c_keys = {
        key for key, row in routing_by_candidate.items()
        if row["route"] != "provisional_agreement"
    }
    if c_keys != expected_c_keys:
        raise ValueError("final lexical C routing differs")
    provisional = []
    disagreement = []
    all_routes = []
    for key in sorted(candidates):
        task_id, candidate_id = key
        route = routing_by_candidate[key]
        a = decisions["A"][task_id][candidate_id]
        b = decisions["B"][task_id][candidate_id]
        d = decisions["D"][task_id][candidate_id]
        c = decisions["C"].get(task_id, {}).get(candidate_id)
        if c is None:
            if not (
                route["route"] == "provisional_agreement"
                and a["label"] == "definitely_not_person"
                and b["label"] == "definitely_not_person"
                and float(a["confidence"]) >= 0.95
                and float(b["confidence"]) >= 0.95
            ):
                raise ValueError(
                    f"non-C partition is not exact A/B agreement: {key}"
                )
            original_support = 2
        else:
            original_support = sum(
                row["label"] == "definitely_not_person" for row in (a, b, c)
            )
        passed = (
            d["label"] == "definitely_not_person"
            and original_support >= 2
        )
        final = {
            "task_id": task_id,
            "candidate_id": candidate_id,
            "audit_digest": route["audit_digest"],
            "read_by_c": c is not None,
            "original_definite_support": original_support,
            "d_label": d["label"],
            "provisional_pass": passed,
        }
        (provisional if passed else disagreement).append(final)
        all_routes.append(final)

    audit_count = math.ceil(AUDIT_RATE * len(provisional))
    audit_keys = {
        (row["task_id"], row["candidate_id"])
        for row in sorted(
            provisional,
            key=lambda row: (
                row["audit_digest"], row["task_id"], row["candidate_id"]
            ),
        )[:audit_count]
    }
    human_rows = []
    human_by_task: dict[str, list[dict]] = {}
    for row in all_routes:
        key = (row["task_id"], row["candidate_id"])
        if not row["provisional_pass"]:
            reason = "cross_family_or_majority_disagreement"
        elif key in audit_keys:
            reason = "deterministic_zero_error_audit"
        else:
            continue
        task_id, candidate_id = key
        candidate = candidates[key]
        judgments = []
        for teacher in ("A", "B", "C", "D"):
            decision = decisions[teacher].get(task_id, {}).get(candidate_id)
            if decision is not None:
                judgments.append({
                    "teacher": teacher,
                    "model": TEACHER_MODELS[teacher],
                    "label": decision["label"],
                    "confidence": float(decision["confidence"]),
                    "rationale": decision["rationale"],
                })
        human = {
            **candidate,
            "review_reason": reason,
            "judgments": judgments,
        }
        human_by_task.setdefault(task_id, []).append(human)
        human_rows.append({
            **row,
            "review_reason": reason,
        })

    provisional_juans = {
        int(tasks[row["task_id"]][1]["juan"]) for row in provisional
    }
    floors_passed = (
        len(provisional) >= MIN_VERIFIED
        and len(provisional_juans) == EXPECTED_JUANS
    )
    if not floors_passed:
        raise RuntimeError(
            "revision-8 provisional negative floors not met before audit"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        task_dir = staging / "tasks"
        state_dir = staging / "state"
        task_dir.mkdir()
        state_dir.mkdir()
        selected_rows = []
        for task_id in sorted(human_by_task):
            source_selected, source_task = tasks[task_id]
            review_task = {
                "schema_version": 1,
                "status": FINAL_REVIEW_STATUS,
                "phase": "revision-8-human-negative-audit",
                "task_id": task_id,
                "source_task_sha256": source_selected["task_sha256"],
                "juan": int(source_task["juan"]),
                "jie_index": int(source_task["jie_index"]),
                "instructions": (
                    "Review each highlighted span using only this numbered jie, all "
                    "available AI rationales, and approved same-jie translation when "
                    "available. Choose not_person only when the exact span is safely "
                    "negative; otherwise choose exclude_from_negative_training. Any "
                    "exclude stops the full revision."
                ),
                "jie": source_task["jie"],
                "approved_translation": {
                    "available": False,
                    "reason": "No approved translation prose is bound to this task.",
                },
                "candidates": sorted(
                    human_by_task[task_id],
                    key=lambda row: (
                        row["para_id"], row["start"], row["end"]
                    ),
                ),
            }
            target = task_dir / f"task_{task_id}.json"
            target.write_text(
                json.dumps(review_task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            state = {
                "schema_version": 1,
                "phase": "revision-8-human-negative-audit",
                "task_id": task_id,
                "task_sha256": _sha256(target),
                "decisions": {},
                "complete": False,
            }
            state_path = state_dir / f"task_{task_id}.json"
            state_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            selected_rows.append({
                "task_id": task_id,
                "juan": int(source_task["juan"]),
                "jie_index": int(source_task["jie_index"]),
                "task": str(Path("tasks") / target.name),
                "task_sha256": _sha256(target),
                "state": str(Path("state") / state_path.name),
                "state_sha256": _sha256(state_path),
                "candidates": len(review_task["candidates"]),
            })
        routes_path = staging / "final-routing.jsonl"
        _write_jsonl(routes_path, all_routes)
        human_path = staging / "human-routing.jsonl"
        _write_jsonl(human_path, human_rows)
        manifest = {
            "schema_version": 1,
            "status": FINAL_REVIEW_STATUS,
            "revision": 8,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "git_commit": _git_commit_clean(),
            "teacher_models": TEACHER_MODELS,
            "teacher_runs": TEACHER_RUNS,
            "teacher_response_schema_sha256": _schema_sha256(),
            "bindings": {
                "source_task_manifest_sha256": _sha256(task_manifest_path),
                "third_task_manifest_sha256": _sha256(third_manifest_path),
                **{
                    f"teacher_{teacher.lower()}_manifest_sha256": _sha256(
                        root / "manifest.json"
                    )
                    for teacher, root in roots.items()
                },
            },
            "audit_rate": AUDIT_RATE,
            "counts": {
                "source_candidates": len(candidates),
                "provisional_passes": len(provisional),
                "cross_family_or_majority_disagreements": len(disagreement),
                "audit_candidates": audit_count,
                "human_candidates": len(human_rows),
                "human_tasks": len(selected_rows),
                "provisional_juans": len(provisional_juans),
            },
            "floors": {
                "minimum_verified": MIN_VERIFIED,
                "required_juans": EXPECTED_JUANS,
                "passed_before_zero_error_audit": floors_passed,
            },
            "stop_rule": (
                "Any exclude_from_negative_training human decision stops the "
                "entire revision before training."
            ),
            "final_routing_sha256": _sha256(routes_path),
            "human_routing_sha256": _sha256(human_path),
            "selected": selected_rows,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        # Task definitions are immutable; state remains writable for the reviewer.
        for path in task_dir.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Revision-8 final cross-family human audit."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--third-tasks", type=Path, required=True)
    parser.add_argument("--teacher-a", type=Path, required=True)
    parser.add_argument("--teacher-b", type=Path, required=True)
    parser.add_argument("--teacher-c", type=Path, required=True)
    parser.add_argument("--teacher-d", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_final_review(
        args.tasks,
        args.third_tasks,
        args.teacher_a,
        args.teacher_b,
        args.teacher_c,
        args.teacher_d,
        args.output,
    )
    print(json.dumps({
        "status": manifest["status"],
        "counts": manifest["counts"],
        "floors": manifest["floors"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
