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
from production_train import _make_read_only


OUTPUT_STATUS = "ml_production_precision_lexical_teacher_outputs"
LABELS = {"definitely_not_person", "possible_person_or_boundary"}
TEACHERS = {"A", "B"}
BATCHES = 8


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _batch(task_id: str) -> int:
    return int(hashlib.sha256(task_id.encode("ascii")).hexdigest(), 16) % BATCHES


def prepare_batches(task_root: Path, output_dir: Path) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"teacher batch output exists: {output_dir}")
    manifest_path = task_root / "manifest.json"
    manifest = _read(manifest_path)
    if (
        manifest.get("status") != TASK_STATUS
        or manifest.get("candidate_model_blind") is not True
        or manifest.get("candidate_scores_hidden") is not True
        or manifest.get("confirmation_read") is not False
    ):
        raise ValueError("lexical teacher task binding differs")
    batches = {str(index): [] for index in range(BATCHES)}
    for row in manifest["selected"]:
        task_path = task_root / row["task"]
        if _sha256(task_path) != row["task_sha256"]:
            raise ValueError(f"lexical teacher task hash differs: {row['task_id']}")
        index = _batch(str(row["task_id"]))
        batches[str(index)].append({
            "task_id": str(row["task_id"]),
            "task": str(task_path.resolve()),
            "task_sha256": row["task_sha256"],
            "candidates": int(row["candidates"]),
        })
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        counts = {}
        for index in range(BATCHES):
            rows = sorted(batches[str(index)], key=lambda row: row["task_id"])
            path = staging / f"batch-{index}.json"
            path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            counts[str(index)] = {
                "tasks": len(rows),
                "candidates": sum(row["candidates"] for row in rows),
                "sha256": _sha256(path),
            }
        frozen = {
            "schema_version": 1,
            "status": "ml_production_precision_lexical_teacher_batches",
            "git_commit": _git_commit_clean(),
            "task_manifest_sha256": _sha256(manifest_path),
            "batches": BATCHES,
            "counts": counts,
        }
        (staging / "manifest.json").write_text(
            json.dumps(frozen, indent=2) + "\n", encoding="utf-8"
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return frozen


def validate_teacher_output(
    task_root: Path,
    batch_root: Path,
    raw_root: Path,
    output_dir: Path,
    *,
    teacher: str,
) -> dict:
    if teacher not in TEACHERS:
        raise ValueError(f"unsupported teacher: {teacher}")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"teacher output exists: {output_dir}")
    task_manifest_path = task_root / "manifest.json"
    task_manifest = _read(task_manifest_path)
    batch_manifest_path = batch_root / "manifest.json"
    batch_manifest = _read(batch_manifest_path)
    if (
        task_manifest.get("status") != TASK_STATUS
        or batch_manifest.get("status")
        != "ml_production_precision_lexical_teacher_batches"
        or batch_manifest.get("task_manifest_sha256")
        != _sha256(task_manifest_path)
    ):
        raise ValueError("lexical teacher validation binding differs")
    expected_tasks = {
        str(row["task_id"]): row for row in task_manifest["selected"]
    }
    validated = []
    seen = set()
    source_hashes = {}
    for path in sorted(raw_root.rglob("*.json")):
        payload = _read(path)
        task_id = str(payload.get("task_id", ""))
        source = expected_tasks.get(task_id)
        if source is None or task_id in seen:
            raise ValueError(f"unexpected lexical teacher task: {task_id}")
        task_path = task_root / source["task"]
        task = _read(task_path)
        if (
            payload.get("schema_version") != 1
            or payload.get("phase") != "lexical-negative-verification"
            or payload.get("teacher") != teacher
            or payload.get("task_sha256") != source["task_sha256"]
            or _sha256(task_path) != source["task_sha256"]
        ):
            raise ValueError(f"lexical teacher provenance differs: {task_id}")
        expected_candidates = {
            str(row["candidate_id"]): row for row in task["candidates"]
        }
        decisions = {}
        for row in payload.get("decisions", []):
            candidate_id = str(row.get("candidate_id", ""))
            confidence = row.get("confidence")
            rationale = row.get("rationale")
            if (
                set(row) != {
                    "candidate_id", "label", "confidence", "rationale"
                }
                or candidate_id not in expected_candidates
                or candidate_id in decisions
                or row.get("label") not in LABELS
                or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
                or not 0 <= float(confidence) <= 1
                or not isinstance(rationale, str)
                or not rationale.strip()
            ):
                raise ValueError(
                    f"invalid lexical teacher decision: {task_id} {candidate_id}"
                )
            decisions[candidate_id] = {
                "candidate_id": candidate_id,
                "label": row["label"],
                "confidence": float(confidence),
                "rationale": rationale.strip(),
            }
        if set(decisions) != set(expected_candidates):
            raise ValueError(
                f"lexical teacher candidate inventory differs: {task_id}"
            )
        validated.append({
            "task_id": task_id,
            "task_sha256": source["task_sha256"],
            "decisions": [
                decisions[candidate_id]
                for candidate_id in sorted(decisions)
            ],
        })
        seen.add(task_id)
        source_hashes[str(path.relative_to(raw_root))] = _sha256(path)
    if seen != set(expected_tasks):
        missing = sorted(set(expected_tasks) - seen)
        raise ValueError(f"lexical teacher tasks missing: {missing[:5]}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        decisions_path = staging / "decisions.jsonl"
        decisions_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in sorted(validated, key=lambda row: row["task_id"])
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "status": OUTPUT_STATUS,
            "teacher": teacher,
            "candidate_model_blind": True,
            "candidate_scores_hidden": True,
            "task_manifest_sha256": _sha256(task_manifest_path),
            "batch_manifest_sha256": _sha256(batch_manifest_path),
            "counts": {
                "tasks": len(validated),
                "candidates": sum(
                    len(row["decisions"]) for row in validated
                ),
            },
            "raw_outputs": source_hashes,
            "decisions_sha256": _sha256(decisions_path),
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
        description="Prepare or validate revision-7 lexical teacher batches."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--tasks", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--tasks", type=Path, required=True)
    validate.add_argument("--batches", type=Path, required=True)
    validate.add_argument("--raw", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--teacher", choices=sorted(TEACHERS), required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_batches(args.tasks, args.output)
    else:
        result = validate_teacher_output(
            args.tasks,
            args.batches,
            args.raw,
            args.output,
            teacher=args.teacher,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
