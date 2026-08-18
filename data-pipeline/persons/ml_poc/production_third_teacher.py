from __future__ import annotations

import argparse
import json
import shutil
import stat
import tempfile
from collections import Counter
from pathlib import Path

from production_review import _read, _sha256, _task_paragraphs


SOURCE_STATUS = "ml_production_focused_review_with_negative_audit"
OUTPUT_STATUS = "ml_production_focused_review_with_third_teacher"
TASK_STATUS = "ml_production_third_teacher_tasks"
EXPECTED_TASKS = 180
THIRD_PASS = "C-source-hidden-adjudication"
THIRD_CHANNEL = "copilot_independent_c_adjudicator"
BOUNDARY_GUIDE = Path(__file__).with_name("BOUNDARY_GUIDE.md")


def _artifact_path(root: Path, value: object, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise ValueError(f"{label} path must be relative")
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes artifact root") from error
    return path


def _valid_task_id(task_id: str) -> bool:
    return (
        len(task_id) == 20
        and all(char in "0123456789abcdef" for char in task_id)
    )


def _validate_source(review_root: Path) -> tuple[dict, Path]:
    manifest_path = review_root / "manifest.json"
    manifest = _read(manifest_path)
    selected = manifest.get("selected")
    audit_inventory = manifest.get("negative_audit_inventory")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != SOURCE_STATUS
        or manifest.get("candidate_model_blind") is not True
        or manifest.get("model_predictions_used") is not False
        or not isinstance(selected, list)
        or len(selected) != EXPECTED_TASKS
        or not isinstance(audit_inventory, dict)
        or not audit_inventory
    ):
        raise ValueError("third-teacher source review differs")
    task_ids = {str(row.get("task_id")) for row in selected}
    if (
        len(task_ids) != EXPECTED_TASKS
        or any(not _valid_task_id(task_id) for task_id in task_ids)
    ):
        raise ValueError("third-teacher source task inventory differs")
    for row in selected:
        for key, hash_key in (
            ("task", "task_sha256"),
            ("review", "review_sha256"),
        ):
            path = _artifact_path(
                review_root, row[key], f"source {key}"
            )
            if _sha256(path) != row.get(hash_key):
                raise ValueError(f"third-teacher source {key} hash differs")
    return manifest, manifest_path


def _needs_third_teacher(candidate: dict, initial: dict) -> bool:
    reason = str(candidate.get("review_reason", ""))
    return (
        candidate["id"] not in initial
        and not (
            reason.startswith("Predeclared ")
            and " audit of exact non-low A/B consensus." in reason
        )
        and not reason.startswith("Independent negative-jie recall audit")
    )


def _task_payload(
    row: dict, task: dict, review: dict
) -> dict | None:
    task_id = str(row["task_id"])
    initial = review.get("initial_decisions", {})
    candidates = [
        {
            key: candidate[key]
            for key in ("id", "para_id", "start", "end", "surface")
        }
        for candidate in review.get("candidates", [])
        if _needs_third_teacher(candidate, initial)
    ]
    if not candidates:
        return None
    return {
        "schema_version": 1,
        "status": TASK_STATUS,
        "phase": "third-teacher-adjudication",
        "candidate_model_blind": True,
        "candidate_sources_hidden": True,
        "model_predictions_used": False,
        "task_id": task_id,
        "task_sha256": row["task_sha256"],
        "review_sha256": row["review_sha256"],
        "juan": int(task["juan"]),
        "jie_index": int(task["jies"][0]["jie_index"]),
        "instructions": (
            "Using only this numbered jie and the frozen boundary policy, "
            "adjudicate every anonymous candidate as an exact person span "
            "accept or reject. Candidate sources and prior decisions are "
            "hidden. Mark confidence high only when the decision and exact "
            "boundary are unambiguous. Also add any omitted person spans. "
            "Do not use identities, translations, notes, other jies, rules, "
            "v1, or model predictions."
        ),
        "jies": task["jies"],
        "candidates": candidates,
    }


def _expected_tasks(review_root: Path, source: dict) -> dict[str, dict]:
    expected = {}
    for row in source["selected"]:
        task_id = str(row["task_id"])
        task = _read(_artifact_path(
            review_root, row["task"], "source task"
        ))
        review = _read(_artifact_path(
            review_root, row["review"], "source review"
        ))
        payload = _task_payload(row, task, review)
        if payload is not None:
            expected[task_id] = payload
    return expected


def prepare_tasks(review_root: Path, output_dir: Path) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"third-teacher task output exists: {output_dir}")
    source, source_manifest_path = _validate_source(review_root)
    manifest = {
        "schema_version": 1,
        "status": TASK_STATUS,
        "candidate_model_blind": True,
        "candidate_sources_hidden": True,
        "model_predictions_used": False,
        "source_review_manifest_sha256": _sha256(source_manifest_path),
        "boundary_guide": "BOUNDARY_GUIDE.md",
        "boundary_guide_sha256": _sha256(BOUNDARY_GUIDE),
        "selected": [],
        "counts": {},
    }
    totals = Counter()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        task_dir = staging / "tasks"
        task_dir.mkdir()
        shutil.copyfile(BOUNDARY_GUIDE, staging / "BOUNDARY_GUIDE.md")
        for row in source["selected"]:
            task_id = str(row["task_id"])
            task_path = _artifact_path(
                review_root, row["task"], "source task"
            )
            review_path = _artifact_path(
                review_root, row["review"], "source review"
            )
            task = _read(task_path)
            review = _read(review_path)
            payload = _task_payload(row, task, review)
            if payload is None:
                continue
            target = task_dir / f"task_{task_id}.json"
            target.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest["selected"].append({
                "task_id": task_id,
                "task": str(Path("tasks") / target.name),
                "task_sha256": _sha256(target),
                "candidates": len(payload["candidates"]),
            })
            totals["tasks"] += 1
            totals["candidates"] += len(payload["candidates"])
        manifest["counts"] = dict(totals)
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in staging.rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return manifest


def _validate_third_output(task: dict, payload: dict) -> tuple[dict, list[dict]]:
    task_id = str(task["task_id"])
    if (
        payload.get("schema_version") != 1
        or payload.get("phase") != "third-teacher-adjudication"
        or payload.get("candidate_model_blind") is not True
        or payload.get("candidate_sources_hidden") is not True
        or payload.get("model_predictions_used") is not False
        or payload.get("task_id") != task_id
        or payload.get("task_sha256") != task["task_sha256"]
        or payload.get("adjudication_task_sha256") != task["_sha256"]
        or payload.get("teacher_pass") != THIRD_PASS
        or payload.get("channel") != THIRD_CHANNEL
    ):
        raise ValueError(f"third-teacher provenance differs: {task_id}")
    expected = {candidate["id"]: candidate for candidate in task["candidates"]}
    decisions = {}
    for row in payload.get("decisions", []):
        candidate_id = str(row.get("id", ""))
        confidence = row.get("confidence")
        reason = row.get("review_reason")
        if (
            set(row) != {"id", "decision", "confidence", "review_reason"}
            or candidate_id not in expected
            or candidate_id in decisions
            or row.get("decision") not in {"accept", "reject"}
            or confidence not in {"high", "medium", "low"}
            or (confidence == "high" and reason != "")
            or (confidence != "high" and not str(reason).strip())
        ):
            raise ValueError(
                f"invalid third-teacher decision: {task_id} {candidate_id}"
            )
        decisions[candidate_id] = row
    if set(decisions) != set(expected):
        raise ValueError(f"third-teacher decision inventory differs: {task_id}")

    source_task = {
        "jies": task["jies"],
    }
    paragraphs = _task_paragraphs(source_task)
    existing_geometry = {
        (
            int(candidate["para_id"]),
            int(candidate["start"]),
            int(candidate["end"]),
        )
        for candidate in task["candidates"]
    }
    additions = []
    addition_ids = set()
    for row in payload.get("additions", []):
        required = {
            "id", "para_id", "start", "end", "surface",
            "confidence", "review_reason",
        }
        para_id = int(row.get("para_id", -1))
        start = int(row.get("start", -1))
        end = int(row.get("end", -1))
        geometry = (para_id, start, end)
        paragraph = paragraphs.get(para_id)
        confidence = row.get("confidence")
        reason = row.get("review_reason")
        candidate_id = f"copilot:{task_id}:{para_id}:{start}:{end}"
        if (
            set(row) != required
            or row.get("id") != candidate_id
            or candidate_id in addition_ids
            or geometry in existing_geometry
            or paragraph is None
            or not 0 <= start < end <= len(paragraph)
            or paragraph[start:end] != row.get("surface")
            or confidence not in {"high", "medium", "low"}
            or (confidence == "high" and reason != "")
            or (confidence != "high" and not str(reason).strip())
        ):
            raise ValueError(
                f"invalid third-teacher addition: {task_id} {candidate_id}"
            )
        addition_ids.add(candidate_id)
        additions.append(row)
    additions.sort(key=lambda row: (
        int(row["para_id"]), int(row["start"]), int(row["end"])
    ))
    return decisions, additions


def _overlaps(annotation: dict, accepted: list[dict]) -> bool:
    return any(
        int(annotation["para_id"]) == int(other["para_id"])
        and int(annotation["start"]) < int(other["end"])
        and int(other["start"]) < int(annotation["end"])
        for other in accepted
    )


def merge_outputs(
    review_root: Path,
    task_root: Path,
    output_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"third-teacher review output exists: {output_dir}")
    source, source_manifest_path = _validate_source(review_root)
    task_manifest_path = task_root / "manifest.json"
    task_manifest = _read(task_manifest_path)
    if (
        task_manifest.get("schema_version") != 1
        or task_manifest.get("status") != TASK_STATUS
        or task_manifest.get("candidate_sources_hidden") is not True
        or task_manifest.get("source_review_manifest_sha256")
        != _sha256(source_manifest_path)
        or task_manifest.get("boundary_guide") != "BOUNDARY_GUIDE.md"
        or task_manifest.get("boundary_guide_sha256")
        != _sha256(task_root / "BOUNDARY_GUIDE.md")
        or task_manifest.get("boundary_guide_sha256")
        != _sha256(BOUNDARY_GUIDE)
    ):
        raise ValueError("third-teacher task manifest differs")
    expected_tasks = _expected_tasks(review_root, source)
    selected_tasks = {
        str(row["task_id"]): row for row in task_manifest.get("selected", [])
    }
    if (
        set(selected_tasks) != set(expected_tasks)
        or len(selected_tasks) != int(task_manifest["counts"]["tasks"])
        or sum(
            len(task["candidates"]) for task in expected_tasks.values()
        )
        != int(task_manifest["counts"]["candidates"])
    ):
        raise ValueError("third-teacher task inventory differs")
    expected_names = {
        f"task_{task_id}.json" for task_id in selected_tasks
    }
    output_paths = list(output_root.glob("*.json"))
    output_by_name = {path.name: path for path in output_paths}
    if (
        len(output_paths) != len(output_by_name)
        or set(output_by_name) != expected_names
    ):
        raise ValueError("third-teacher output inventory differs")

    validated = {}
    for task_id, selection in selected_tasks.items():
        task_path = _artifact_path(
            task_root, selection["task"], "third-teacher task"
        )
        task_sha256 = _sha256(task_path)
        task = _read(task_path)
        if (
            task_sha256 != selection["task_sha256"]
            or int(selection.get("candidates", -1))
            != len(expected_tasks[task_id]["candidates"])
            or task != expected_tasks[task_id]
        ):
            raise ValueError(f"third-teacher task hash differs: {task_id}")
        task["_sha256"] = task_sha256
        output_path = output_by_name[f"task_{task_id}.json"]
        decisions, additions = _validate_third_output(
            task, _read(output_path)
        )
        validated[task_id] = (
            output_path, decisions, additions
        )

    manifest = {
        **source,
        "status": OUTPUT_STATUS,
        "source_review_manifest_sha256": _sha256(source_manifest_path),
        "third_teacher_task_manifest_sha256": _sha256(task_manifest_path),
        "third_teacher_inventory": {},
        "counts": dict(source["counts"]),
    }
    totals = Counter()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        for name in ("tasks", "review", "private"):
            shutil.copytree(review_root / name, staging / name)
        selections = {
            str(row["task_id"]): row for row in manifest["selected"]
        }
        for task_id, (output_path, decisions, additions) in validated.items():
            selection = selections[task_id]
            review_relative = Path(str(selection["review"]))
            if review_relative.is_absolute() or ".." in review_relative.parts:
                raise ValueError("source review path escapes artifact root")
            review_path = staging / review_relative
            review = _read(review_path)
            candidates = {
                str(candidate["id"]): candidate
                for candidate in review["candidates"]
            }
            source_candidate_ids = set(candidates)
            initial_decisions = dict(review["initial_decisions"])
            initial_annotations = list(review["initial_annotations"])
            for candidate_id, decision in decisions.items():
                candidate = candidates[candidate_id]
                candidate["channels"].append(THIRD_CHANNEL)
                candidate["third_teacher"] = {
                    key: decision[key]
                    for key in ("decision", "confidence", "review_reason")
                }
                totals["third_teacher_decisions"] += 1
                if decision["confidence"] != "high":
                    candidate["review_reason"] = (
                        "Third teacher did not make a high-confidence decision. "
                        + str(decision["review_reason"]).strip()
                    )
                    totals["third_teacher_human_review"] += 1
                    continue
                if decision["decision"] == "reject":
                    initial_decisions[candidate_id] = "reject"
                    totals["third_teacher_high_reject"] += 1
                    continue
                annotation = {
                    key: candidate[key]
                    for key in ("para_id", "start", "end", "surface")
                }
                if _overlaps(annotation, initial_annotations):
                    candidate["review_reason"] = (
                        "High-confidence third-teacher acceptance overlaps an "
                        "existing auto-accepted geometry."
                    )
                    totals["third_teacher_geometry_conflict_review"] += 1
                    continue
                initial_decisions[candidate_id] = "accept"
                initial_annotations.append(annotation)
                totals["third_teacher_high_accept"] += 1

            for addition in additions:
                candidate_id = str(addition["id"])
                totals["third_teacher_additions"] += 1
                if candidate_id in candidates:
                    totals["third_teacher_duplicate_existing"] += 1
                    continue
                candidate = {
                    "id": candidate_id,
                    "para_id": int(addition["para_id"]),
                    "start": int(addition["start"]),
                    "end": int(addition["end"]),
                    "surface": addition["surface"],
                    "channels": [THIRD_CHANNEL],
                    "confidence": addition["confidence"],
                    "review_reason": str(addition["review_reason"]),
                    "pass_confidence": {
                        "a": None,
                        "b": None,
                        "c": addition["confidence"],
                    },
                    "third_teacher": {
                        "decision": "accept",
                        "confidence": addition["confidence"],
                        "review_reason": addition["review_reason"],
                    },
                }
                annotation = {
                    key: candidate[key]
                    for key in ("para_id", "start", "end", "surface")
                }
                overlaps_existing_candidate = any(
                    int(annotation["para_id"]) == int(other["para_id"])
                    and int(annotation["start"]) < int(other["end"])
                    and int(other["start"]) < int(annotation["end"])
                    for other in candidates.values()
                )
                if (
                    addition["confidence"] == "high"
                    and not _overlaps(annotation, initial_annotations)
                    and not overlaps_existing_candidate
                ):
                    initial_decisions[candidate_id] = "accept"
                    initial_annotations.append(annotation)
                    totals["third_teacher_high_addition"] += 1
                else:
                    if addition["confidence"] == "high":
                        candidate["review_reason"] = (
                            "High-confidence third-teacher addition overlaps an "
                            "existing auto-accepted geometry."
                        )
                        totals[
                            "third_teacher_geometry_conflict_review"
                        ] += 1
                    else:
                        candidate["review_reason"] = (
                            "Third-teacher addition requires human review. "
                            + str(addition["review_reason"]).strip()
                        )
                        totals["third_teacher_human_review"] += 1
                review["candidates"].append(candidate)
                candidates[candidate_id] = candidate
                totals["third_teacher_novel_additions"] += 1

            initial_annotations.sort(key=lambda row: (
                int(row["para_id"]), int(row["start"]), int(row["end"])
            ))
            review["initial_annotations"] = initial_annotations
            review["initial_decisions"] = initial_decisions
            review["human_review_scope"] = (
                "third_teacher_non_high_consensus_audit_and_negative_recall_audit"
            )
            review["third_teacher_sha256"] = _sha256(output_path)
            review_path.chmod(stat.S_IWRITE)
            review_path.write_text(
                json.dumps(review, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            selection["review_sha256"] = _sha256(review_path)
            selection["review_candidates"] = (
                len(review["candidates"]) - len(initial_decisions)
            )
            manifest["third_teacher_inventory"][task_id] = {
                "sha256": _sha256(output_path),
                "decisions": len(decisions),
                "additions": len(additions),
                "novel_additions": sum(
                    addition["id"] not in source_candidate_ids
                    for addition in additions
                ),
                "duplicate_existing": sum(
                    addition["id"] in source_candidate_ids
                    for addition in additions
                ),
            }
        manifest["counts"].update(totals)
        manifest["counts"]["candidate_union"] = (
            int(source["counts"]["candidate_union"])
            + int(totals["third_teacher_novel_additions"])
        )
        manifest["counts"]["human_review_candidates"] = sum(
            int(row["review_candidates"]) for row in manifest["selected"]
        )
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in staging.rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or merge source-hidden third-teacher adjudication."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--review", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--review", type=Path, required=True)
    merge.add_argument("--tasks", type=Path, required=True)
    merge.add_argument("--adjudications", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare_tasks(args.review, args.output)
    else:
        manifest = merge_outputs(
            args.review, args.tasks, args.adjudications, args.output
        )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
