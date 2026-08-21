from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_revision17_plan import _read, _sha256
from production_precision_revision17_target_freeze import BLOCKED_STATUS
from production_precision_revision17_targets import TARGETS_STATUS
from production_precision_revision17_tasks import (
    LABELS,
    TASKS_STATUS as REVIEW_TASKS_STATUS,
)
from production_train import _make_read_only


REVISION = 18
TASK_STATUS = "ml_production_precision_revision18_adjudication_task"
TASKS_STATUS = "ml_production_precision_revision18_adjudication_tasks"
EXPECTED_UNCERTAIN = 11
EXPECTED_CONTRADICTIONS = 2
EXPECTED_TASKS = EXPECTED_UNCERTAIN + EXPECTED_CONTRADICTIONS


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


def unresolved_rows(rows: list[dict]) -> list[dict]:
    selected = [
        row
        for row in rows
        if bool(row.get("uncertain")) or bool(row.get("contradiction"))
    ]
    if (
        sum(bool(row.get("uncertain")) for row in selected)
        != EXPECTED_UNCERTAIN
        or sum(bool(row.get("contradiction")) for row in selected)
        != EXPECTED_CONTRADICTIONS
        or len(selected) != EXPECTED_TASKS
    ):
        raise ValueError("Revision-18 unresolved inventory differs")
    return sorted(selected, key=lambda row: str(row["candidate_id"]))


def _task_id(candidate_id: str, targets_sha256: str) -> str:
    return hashlib.sha256(
        f"revision-18:{targets_sha256}:{candidate_id}".encode("ascii")
    ).hexdigest()[:24]


def freeze_tasks(
    review_task_root: Path,
    target_task_root: Path,
    frozen_target_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Revision-18 tasks exist: {output_dir}")

    review_manifest_path = review_task_root / "manifest.json"
    review_hashes_path = review_task_root / "task-hashes.jsonl"
    target_manifest_path = target_task_root / "manifest.json"
    target_hashes_path = target_task_root / "task-hashes.jsonl"
    frozen_manifest_path = frozen_target_root / "manifest.json"
    targets_path = frozen_target_root / "targets.jsonl"
    review_manifest = _read(review_manifest_path)
    target_manifest = _read(target_manifest_path)
    frozen_manifest = _read(frozen_manifest_path)
    if (
        review_manifest.get("status") != REVIEW_TASKS_STATUS
        or review_manifest.get("outputs", {}).get("task_hashes_sha256")
        != _sha256(review_hashes_path)
        or target_manifest.get("status") != TARGETS_STATUS
        or target_manifest.get("outputs", {}).get("target_task_hashes_sha256")
        != _sha256(target_hashes_path)
        or frozen_manifest.get("status") != BLOCKED_STATUS
        or frozen_manifest.get("counts", {}).get("uncertain")
        != EXPECTED_UNCERTAIN
        or frozen_manifest.get("counts", {}).get("contradictions")
        != EXPECTED_CONTRADICTIONS
        or frozen_manifest.get("outputs", {}).get("targets_sha256")
        != _sha256(targets_path)
    ):
        raise ValueError("Revision-18 source binding differs")

    review_hash_rows = _read_jsonl(review_hashes_path)
    review_hashes = {
        str(row["task_id"]): str(row["task_sha256"])
        for row in review_hash_rows
    }
    target_hash_rows = _read_jsonl(target_hashes_path)
    source_task_by_candidate = {
        str(row["candidate_id"]): str(row["source_task_id"])
        for row in target_hash_rows
    }
    if (
        len(review_hashes) != len(review_hash_rows)
        or len(source_task_by_candidate) != len(target_hash_rows)
    ):
        raise ValueError("Revision-18 source hash inventory differs")

    targets_sha256 = _sha256(targets_path)
    unresolved = unresolved_rows(_read_jsonl(targets_path))
    prepared = []
    for row in unresolved:
        candidate_id = str(row["candidate_id"])
        source_task_id = source_task_by_candidate.get(candidate_id)
        source_path = (
            review_task_root
            / "reviewer-tasks"
            / f"task_{source_task_id}.json"
        )
        if (
            source_task_id is None
            or _sha256(source_path) != review_hashes.get(source_task_id)
        ):
            raise ValueError("Revision-18 source review task hash differs")
        source = _read(source_path)
        if source.get("candidate", {}).get("candidate_id") != candidate_id:
            raise ValueError("Revision-18 source candidate binding differs")
        prepared.append((
            _task_id(candidate_id, targets_sha256),
            source_task_id,
            source,
        ))
    if len({item[0] for item in prepared}) != EXPECTED_TASKS:
        raise ValueError("Revision-18 opaque task ID collision")
    prepared.sort(key=lambda item: item[0])

    git_commit = _git_commit_clean()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        tasks_dir = staging / "reviewer-tasks"
        sealed_dir = staging / "sealed-source"
        tasks_dir.mkdir()
        sealed_dir.mkdir()
        task_hashes = []
        sealed_rows = []
        for task_id, source_task_id, source in prepared:
            candidate_id = str(source["candidate"]["candidate_id"])
            task = {
                "schema_version": 1,
                "status": TASK_STATUS,
                "phase": "revision-18-blind-unresolved-target-adjudication",
                "adjudication_task_id": task_id,
                "candidate_id": candidate_id,
                "review_scope": "current-numbered-jie-only",
                "protocol": {
                    "decision": (
                        "Classify the marked candidate and, only for a "
                        "wrong-boundary person, return every exact overlapping "
                        "individual-person geometry."
                    ),
                    "evidence": "Use only the complete numbered jie in this task.",
                    "independence": (
                        "Return one first judgment without seeking prior or "
                        "sibling tasks."
                    ),
                },
                "jie": source["jie"],
                "candidate": source["candidate"],
                "allowed_labels": list(LABELS),
                "response_schema": {
                    "label": "exact_person|wrong_boundary|not_person|uncertain",
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
            path = tasks_dir / f"task_{task_id}.json"
            path.write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            task_hashes.append({
                "adjudication_task_id": task_id,
                "candidate_id": candidate_id,
                "task_sha256": _sha256(path),
            })
            sealed_rows.append({
                "adjudication_task_id": task_id,
                "candidate_id": candidate_id,
                "source_task_id": source_task_id,
            })
        hashes_path = staging / "task-hashes.jsonl"
        sealed_path = sealed_dir / "source-map.jsonl"
        _write_jsonl(hashes_path, task_hashes)
        _write_jsonl(sealed_path, sealed_rows)
        manifest = {
            "schema_version": 1,
            "status": TASKS_STATUS,
            "revision": REVISION,
            "fit_only": True,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "formal_reserve_text_read": False,
            "one_candidate_per_task": True,
            "reviewer_progress_disclosure": False,
            "prior_judgments_visible": False,
            "git_commit": git_commit,
            "bindings": {
                "review_task_manifest_sha256": _sha256(review_manifest_path),
                "review_task_hashes_sha256": _sha256(review_hashes_path),
                "target_task_manifest_sha256": _sha256(target_manifest_path),
                "target_task_hashes_sha256": _sha256(target_hashes_path),
                "frozen_target_manifest_sha256": _sha256(
                    frozen_manifest_path
                ),
                "targets_sha256": targets_sha256,
            },
            "counts": {
                "tasks": len(task_hashes),
                "source_uncertain": EXPECTED_UNCERTAIN,
                "source_contradictions": EXPECTED_CONTRADICTIONS,
            },
            "outputs": {
                "task_hashes_sha256": _sha256(hashes_path),
                "sealed_source_map_sha256": _sha256(sealed_path),
            },
            "claim_limit": (
                "Fit-only blind adjudication of frozen unresolved targets; not "
                "formal-grade evidence."
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
        description="Freeze Revision-18 unresolved-target adjudication tasks."
    )
    parser.add_argument("--review-tasks", type=Path, required=True)
    parser.add_argument("--target-tasks", type=Path, required=True)
    parser.add_argument("--frozen-targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_tasks(
        args.review_tasks,
        args.target_tasks,
        args.frozen_targets,
        args.output,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
