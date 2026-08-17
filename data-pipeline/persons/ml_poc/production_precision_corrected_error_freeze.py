from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_corrected_error_audit import (
    ARTIFACT_STATUS,
    EXPECTED_TASKS,
    LABELS,
    TASK_STATUS,
)
from production_precision_grouped_verifier import _read, _read_jsonl, _sha256
from production_train import _make_read_only


REVISION = 14
FROZEN_STATUS = "ml_production_precision_corrected_error_audit_frozen"
BLOCKED_STATUS = "ml_production_precision_corrected_error_audit_blocked_uncertain"
CANONICAL_KEYS = {
    "task_id",
    "candidate_id",
    "label",
    "rationale",
    "reviewer",
    "model",
}
ALIASED_KEYS = (CANONICAL_KEYS - {"rationale"}) | {"brief source rationale"}


def _read_raw(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"corrected error audit raw output is not an object: {path}")
    return value


def _normalize_raw(value: dict) -> dict:
    keys = set(value)
    if keys == CANONICAL_KEYS:
        rationale = value["rationale"]
    elif keys == ALIASED_KEYS:
        rationale = value["brief source rationale"]
    else:
        raise ValueError("corrected error audit raw fields differ")
    if (
        value.get("label") not in LABELS
        or value.get("reviewer") != "copilot-teacher"
        or value.get("model") != "gpt-5.6-sol"
        or not isinstance(rationale, str)
        or not rationale.strip()
    ):
        raise ValueError("corrected error audit raw decision differs")
    return {
        "task_id": str(value["task_id"]),
        "candidate_id": str(value["candidate_id"]),
        "label": str(value["label"]),
        "rationale": rationale.strip(),
        "reviewer": "copilot-teacher",
        "model": "gpt-5.6-sol",
    }


def freeze_decisions(task_root: Path, raw_root: Path, output_dir: Path) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"corrected error audit freeze exists: {output_dir}")
    task_manifest_path = task_root / "manifest.json"
    task_hashes_path = task_root / "task-hashes.jsonl"
    task_manifest = _read(task_manifest_path)
    if (
        task_manifest.get("status") != ARTIFACT_STATUS
        or task_manifest.get("confirmation_read") is not False
        or task_manifest.get("one_candidate_per_task") is not True
        or task_manifest.get("reviewer_progress_disclosure") is not False
        or int(task_manifest.get("counts", {}).get("tasks", -1))
        != EXPECTED_TASKS
        or task_manifest.get("outputs", {}).get("task_hashes_sha256")
        != _sha256(task_hashes_path)
    ):
        raise ValueError("corrected error audit task manifest differs")
    expected_rows = _read_jsonl(task_hashes_path)
    expected = {str(row["task_id"]): str(row["task_sha256"]) for row in expected_rows}
    if len(expected_rows) != EXPECTED_TASKS or len(expected) != EXPECTED_TASKS:
        raise ValueError("corrected error audit task hash inventory differs")

    raw_paths = sorted(raw_root.glob("*.json"))
    if len(raw_paths) != EXPECTED_TASKS:
        raise ValueError("corrected error audit raw coverage differs")
    decisions = []
    raw_hashes = []
    seen = set()
    for path in raw_paths:
        raw = _normalize_raw(_read_raw(path))
        task_id = raw["task_id"]
        if path.stem != task_id or task_id in seen or task_id not in expected:
            raise ValueError(f"corrected error audit raw task differs: {path.name}")
        task_path = task_root / "reviewer-tasks" / f"task_{task_id}.json"
        task = _read(task_path)
        if (
            _sha256(task_path) != expected[task_id]
            or task.get("status") != TASK_STATUS
            or task.get("task_id") != task_id
            or task.get("candidate", {}).get("candidate_id") != raw["candidate_id"]
            or task.get("allowed_labels") != list(LABELS)
        ):
            raise ValueError(f"corrected error audit task binding differs: {task_id}")
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
        raise ValueError("corrected error audit tasks are missing")

    counts = {
        label: sum(row["label"] == label for row in decisions) for label in LABELS
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        decisions_path = staging / "decisions.jsonl"
        decisions_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in sorted(decisions, key=lambda item: item["task_id"])
            ),
            encoding="utf-8",
        )
        raw_hashes_path = staging / "raw-hashes.jsonl"
        raw_hashes_path.write_text(
            "".join(
                json.dumps(row, separators=(",", ":")) + "\n"
                for row in raw_hashes
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "status": (
                BLOCKED_STATUS if counts["uncertain"] else FROZEN_STATUS
            ),
            "revision": REVISION,
            "formal_grade": False,
            "fit_only": True,
            "eligible_for_production": False,
            "confirmation_read": False,
            "first_judgments_immutable": True,
            "git_commit": _git_commit_clean(),
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
                "stop_on_uncertainty"
                if counts["uncertain"]
                else "blind_exact_target_audit_for_every_wrong_boundary"
            ),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze Revision-14 corrected-OOF error-audit judgments."
    )
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        freeze_decisions(args.task_root, args.raw_root, args.output_dir),
        indent=2,
    ))


if __name__ == "__main__":
    main()
