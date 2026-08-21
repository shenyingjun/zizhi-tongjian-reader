from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_revision17_plan import _read, _sha256
from production_precision_revision18_overlay import _overlaps
from production_precision_revision19_conflicts import (
    EXPECTED_COMPONENTS,
    TASKS_STATUS,
    TASK_STATUS,
)
from production_train import _make_read_only


REVISION = 19
FROZEN_STATUS = "ml_production_precision_revision19_conflicts_frozen"
BLOCKED_STATUS = "ml_production_precision_revision19_conflicts_blocked"
RAW_KEYS = {
    "conflict_task_id",
    "uncertain",
    "exact_people",
    "rationale",
    "reviewer",
    "model",
}
PERSON_KEYS = {"para_id", "start", "end", "surface"}


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


def _paragraph(task: dict, para_id: int) -> str:
    segment = next(
        (
            row for row in task["jie"]["segments"]
            if int(row["para_id"]) == para_id
        ),
        None,
    )
    if segment is None:
        raise ValueError("Revision-19 exact geometry paragraph differs")
    return task["jie"]["text"][
        int(segment["assembled_start"]):int(segment["assembled_end"])
    ]


def normalize_raw(value: dict, task: dict) -> dict:
    if set(value) != RAW_KEYS:
        raise ValueError("Revision-19 raw fields differ")
    uncertain = value.get("uncertain")
    people = value.get("exact_people")
    if (
        value.get("conflict_task_id") != task.get("conflict_task_id")
        or not isinstance(uncertain, bool)
        or not isinstance(people, list)
        or any(
            not isinstance(row, dict) or set(row) != PERSON_KEYS
            for row in people
        )
        or (uncertain and people)
        or not isinstance(value.get("rationale"), str)
        or not value["rationale"].strip()
        or value.get("reviewer") != "copilot-teacher"
        or value.get("model") != "gpt-5.6-sol"
    ):
        raise ValueError("Revision-19 raw decision differs")

    shown = [
        (
            str(task["jie"].get("id", "")),
            int(row["para_id"]),
            int(row["start"]),
            int(row["end"]),
        )
        for row in task["shown_geometries"]
    ]
    normalized = []
    seen = set()
    for row in people:
        para_id = row["para_id"]
        start = row["start"]
        end = row["end"]
        surface = row["surface"]
        if (
            isinstance(para_id, bool)
            or not isinstance(para_id, int)
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not isinstance(surface, str)
        ):
            raise ValueError("Revision-19 exact geometry types differ")
        paragraph = _paragraph(task, para_id)
        if not (0 <= start < end <= len(paragraph)):
            raise ValueError("Revision-19 exact geometry bounds differ")
        canonical_surface = paragraph[start:end]
        geometry = (para_id, start, end)
        comparable = ("", para_id, start, end)
        if (
            geometry in seen
            or not any(_overlaps(comparable, item) for item in shown)
        ):
            raise ValueError("Revision-19 exact geometry coverage differs")
        seen.add(geometry)
        normalized.append({
            "para_id": para_id,
            "start": start,
            "end": end,
            "surface": canonical_surface,
            "reported_surface": surface,
            "surface_corrected": surface != canonical_surface,
        })
    normalized.sort(
        key=lambda row: (row["para_id"], row["start"], row["end"])
    )
    for index, left in enumerate(normalized):
        left_geometry = ("", left["para_id"], left["start"], left["end"])
        for right in normalized[index + 1:]:
            right_geometry = (
                "",
                right["para_id"],
                right["start"],
                right["end"],
            )
            if _overlaps(left_geometry, right_geometry):
                raise ValueError("Revision-19 exact geometries overlap")
    return {
        "conflict_task_id": str(value["conflict_task_id"]),
        "uncertain": uncertain,
        "exact_people": normalized,
        "rationale": value["rationale"].strip(),
    }


def freeze_decisions(
    task_root: Path,
    raw_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Revision-19 freeze exists: {output_dir}")
    manifest_path = task_root / "manifest.json"
    hashes_path = task_root / "task-hashes.jsonl"
    manifest = _read(manifest_path)
    if (
        manifest.get("status") != TASKS_STATUS
        or manifest.get("prior_judgments_visible") is not False
        or manifest.get("counts", {}).get("tasks") != EXPECTED_COMPONENTS
        or manifest.get("outputs", {}).get("task_hashes_sha256")
        != _sha256(hashes_path)
    ):
        raise ValueError("Revision-19 task manifest differs")
    hash_rows = _read_jsonl(hashes_path)
    expected = {
        str(row["conflict_task_id"]): row for row in hash_rows
    }
    if (
        len(hash_rows) != EXPECTED_COMPONENTS
        or len(expected) != EXPECTED_COMPONENTS
    ):
        raise ValueError("Revision-19 task hash inventory differs")
    raw_paths = sorted(raw_root.glob("*.json"))
    if len(raw_paths) != EXPECTED_COMPONENTS:
        raise ValueError("Revision-19 raw coverage differs")

    decisions = []
    raw_hashes = []
    seen = set()
    for path in raw_paths:
        value = _read(path)
        task_id = str(value.get("conflict_task_id"))
        expected_row = expected.get(task_id)
        task_path = task_root / "reviewer-tasks" / f"task_{task_id}.json"
        if (
            path.stem != task_id
            or task_id in seen
            or expected_row is None
            or _sha256(task_path) != expected_row["task_sha256"]
        ):
            raise ValueError("Revision-19 raw task differs")
        task = _read(task_path)
        if task.get("status") != TASK_STATUS:
            raise ValueError("Revision-19 task status differs")
        decisions.append({
            **normalize_raw(value, task),
            "task_sha256": expected_row["task_sha256"],
            "raw_sha256": _sha256(path),
        })
        raw_hashes.append({"path": path.name, "sha256": _sha256(path)})
        seen.add(task_id)
    if seen != set(expected):
        raise ValueError("Revision-19 tasks are missing")

    unresolved = sum(row["uncertain"] for row in decisions)
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
            sorted(decisions, key=lambda row: row["conflict_task_id"]),
        )
        _write_jsonl(raw_hashes_path, raw_hashes)
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
                "uncertain": unresolved,
                "resolved_empty": sum(
                    not row["uncertain"] and not row["exact_people"]
                    for row in decisions
                ),
                "exact_people": sum(
                    len(row["exact_people"]) for row in decisions
                ),
                "surface_corrections": sum(
                    person["surface_corrected"]
                    for row in decisions
                    for person in row["exact_people"]
                ),
            },
            "outputs": {
                "decisions_sha256": _sha256(decisions_path),
                "raw_hashes_sha256": _sha256(raw_hashes_path),
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(frozen, indent=2) + "\n",
            encoding="utf-8",
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return frozen


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze Revision-19 blind conflict decisions."
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
