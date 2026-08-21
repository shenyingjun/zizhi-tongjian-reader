from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_revision17_plan import _read, _sha256
from production_precision_revision17_targets import (
    TARGETS_STATUS,
    TARGET_TASK_STATUS,
)
from production_train import _make_read_only


REVISION = 17
FROZEN_STATUS = "ml_production_precision_revision17_targets_frozen"
BLOCKED_STATUS = "ml_production_precision_revision17_targets_blocked_unresolved"
RAW_KEYS = {
    "target_task_id",
    "candidate_id",
    "uncertain",
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


def _overlaps(left: dict, right: dict) -> bool:
    return (
        int(left["para_id"]) == int(right["para_id"])
        and int(left["start"]) < int(right["end"])
        and int(right["start"]) < int(left["end"])
    )


def normalize_target_raw(value: dict, task: dict) -> dict:
    if set(value) != RAW_KEYS:
        raise ValueError("Revision-17 target raw fields differ")
    if (
        str(value.get("target_task_id")) != str(task["target_task_id"])
        or str(value.get("candidate_id")) != str(task["candidate_id"])
        or not isinstance(value.get("uncertain"), bool)
        or not isinstance(value.get("targets"), list)
        or not isinstance(value.get("rationale"), str)
        or not value["rationale"].strip()
        or value.get("reviewer") != "copilot-teacher"
        or value.get("model") != "gpt-5.6-sol"
    ):
        raise ValueError("Revision-17 target raw decision differs")
    uncertain = value["uncertain"]
    targets = value["targets"]
    if uncertain and targets:
        raise ValueError("Revision-17 target uncertainty/geometry differs")
    contradiction = not uncertain and not targets

    wrong = task["wrong_candidate"]
    segments = {
        int(row["para_id"]): row for row in task["jie"]["segments"]
    }
    normalized = []
    seen = set()
    for target in targets:
        if set(target) != {"para_id", "start", "end", "surface"}:
            raise ValueError("Revision-17 target geometry fields differ")
        para_id = int(target["para_id"])
        start = int(target["start"])
        end = int(target["end"])
        segment = segments.get(para_id)
        if (
            segment is None
            or not 0 <= start < end
            <= int(segment["assembled_end"]) - int(segment["assembled_start"])
        ):
            raise ValueError("Revision-17 target geometry bounds differ")
        assembled_start = int(segment["assembled_start"]) + start
        assembled_end = int(segment["assembled_start"]) + end
        surface = str(target["surface"])
        normalized_target = {
            "para_id": para_id,
            "start": start,
            "end": end,
            "surface": surface,
        }
        key = para_id, start, end
        if (
            key in seen
            or task["jie"]["text"][assembled_start:assembled_end] != surface
            or not _overlaps(normalized_target, wrong)
            or (
                para_id == int(wrong["para_id"])
                and start == int(wrong["start"])
                and end == int(wrong["end"])
            )
        ):
            raise ValueError("Revision-17 target geometry validity differs")
        seen.add(key)
        normalized.append(normalized_target)
    normalized.sort(key=lambda row: (
        row["para_id"], row["start"], row["end"]
    ))
    if any(
        _overlaps(left, right)
        for index, left in enumerate(normalized)
        for right in normalized[index + 1:]
    ):
        raise ValueError("Revision-17 exact targets overlap each other")
    if len(normalized) > 1 and any(
        int(target["para_id"]) != int(wrong["para_id"])
        or int(target["start"]) < int(wrong["start"])
        or int(target["end"]) > int(wrong["end"])
        for target in normalized
    ):
        raise ValueError("Revision-17 merged-candidate targets are not contained")
    return {
        "target_task_id": str(value["target_task_id"]),
        "candidate_id": str(value["candidate_id"]),
        "uncertain": uncertain,
        "contradiction": contradiction,
        "targets": normalized,
        "rationale": value["rationale"].strip(),
    }


def freeze_targets(
    task_root: Path,
    raw_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Revision-17 target freeze exists: {output_dir}")
    manifest_path = task_root / "manifest.json"
    hashes_path = task_root / "task-hashes.jsonl"
    manifest = _read(manifest_path)
    if (
        manifest.get("status") != TARGETS_STATUS
        or manifest.get("outputs", {}).get("target_task_hashes_sha256")
        != _sha256(hashes_path)
    ):
        raise ValueError("Revision-17 target task manifest differs")
    hashes = _read_jsonl(hashes_path)
    expected = {
        str(row["target_task_id"]): row for row in hashes
    }
    if len(expected) != len(hashes):
        raise ValueError("Revision-17 target task hashes differ")
    raw_paths = sorted(raw_root.glob("*.json"))
    if len(raw_paths) != len(expected):
        raise ValueError("Revision-17 target raw coverage differs")

    decisions = []
    seen = set()
    raw_hashes = []
    for path in raw_paths:
        value = _read(path)
        task_id = str(value.get("target_task_id"))
        expected_row = expected.get(task_id)
        task_path = task_root / "reviewer-tasks" / f"task_{task_id}.json"
        if (
            path.stem != task_id
            or task_id in seen
            or expected_row is None
            or _sha256(task_path) != expected_row["task_sha256"]
        ):
            raise ValueError("Revision-17 target raw task differs")
        task = _read(task_path)
        if task.get("status") != TARGET_TASK_STATUS:
            raise ValueError("Revision-17 target task status differs")
        decision = normalize_target_raw(value, task)
        decisions.append({
            **decision,
            "task_sha256": expected_row["task_sha256"],
            "raw_sha256": _sha256(path),
        })
        raw_hashes.append({"path": path.name, "sha256": _sha256(path)})
        seen.add(task_id)
    if seen != set(expected):
        raise ValueError("Revision-17 target tasks are missing")

    uncertain_count = sum(row["uncertain"] for row in decisions)
    contradiction_count = sum(row["contradiction"] for row in decisions)
    git_commit = _git_commit_clean()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        decisions_path = staging / "targets.jsonl"
        raw_hashes_path = staging / "raw-hashes.jsonl"
        _write_jsonl(
            decisions_path,
            sorted(decisions, key=lambda row: row["target_task_id"]),
        )
        _write_jsonl(raw_hashes_path, raw_hashes)
        frozen = {
            "schema_version": 1,
            "status": (
                BLOCKED_STATUS
                if uncertain_count or contradiction_count
                else FROZEN_STATUS
            ),
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
                "uncertain": uncertain_count,
                "contradictions": contradiction_count,
                "exact_targets": sum(
                    len(row["targets"]) for row in decisions
                ),
            },
            "outputs": {
                "targets_sha256": _sha256(decisions_path),
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
        description="Freeze Revision-17 blind exact-target judgments."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_targets(args.tasks, args.raw, args.output)
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["status"] == FROZEN_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
