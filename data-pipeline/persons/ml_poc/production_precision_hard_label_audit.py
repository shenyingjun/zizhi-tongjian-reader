from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_grouped_data import GROUPED_DATA_STATUS
from production_precision_grouped_verifier import _read, _read_jsonl, _sha256
from production_train import _make_read_only


TASK_STATUS = "ml_production_hard_label_audit_tasks"
LABEL_STATUS = "ml_production_hard_label_audit_original_labels"
OUTPUT_STATUS = "ml_production_hard_label_audit_outputs"
ROUTING_STATUS = "ml_production_hard_label_audit_routing"
REVISION = 12
EXPECTED_REAL_NEGATIVES = 171
EXPECTED_BOUNDARIES = 132
HIGH_CONFIDENCE = 0.90
LABELS = {"exact_person", "wrong_boundary", "not_person", "uncertain"}
REVIEWERS = {"A", "B"}
MODEL_FAMILIES = {"anthropic", "openai"}


def _candidate_id(grouped_sha256: str, row: dict) -> str:
    value = ":".join(str(item) for item in (
        grouped_sha256,
        int(row["juan"]),
        int(row["jie_index"]),
        int(row["para_id"]),
        int(row["start"]),
        int(row["end"]),
    ))
    return hashlib.sha256(value.encode("ascii")).hexdigest()[:20]


def _task_id(juan: int, jie_index: int) -> str:
    return hashlib.sha256(f"{juan}:{jie_index}".encode("ascii")).hexdigest()[:20]


def build_tasks(grouped_root: Path, output_dir: Path) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"hard-label task output exists: {output_dir}")
    manifest_path = grouped_root / "manifest.json"
    manifest = _read(manifest_path)
    examples_path = grouped_root / "examples.jsonl"
    existence_path = grouped_root / "existence.jsonl"
    if (
        manifest.get("status") != GROUPED_DATA_STATUS
        or manifest.get("confirmation_read") is not False
        or manifest.get("outputs", {}).get("examples_sha256")
        != _sha256(examples_path)
        or manifest.get("outputs", {}).get("existence_sha256")
        != _sha256(existence_path)
    ):
        raise ValueError("hard-label grouped-data binding differs")
    grouped_sha256 = _sha256(manifest_path)
    examples = {
        str(row["id"]): row for row in _read_jsonl(examples_path)
    }
    selected = []
    original_labels = {}
    for row in _read_jsonl(existence_path):
        if not int(row["label"]):
            original = "real_not_person"
        else:
            exact = any(
                int(reference["para_id"]) == int(row["para_id"])
                and int(reference["start"]) == int(row["start"])
                and int(reference["end"]) == int(row["end"])
                for reference in row["overlapping_references"]
            )
            if exact:
                continue
            original = "boundary_alternative"
        candidate_id = _candidate_id(grouped_sha256, row)
        selected.append({
            "candidate_id": candidate_id,
            "id": str(row["id"]),
            "juan": int(row["juan"]),
            "jie_index": int(row["jie_index"]),
            "para_id": int(row["para_id"]),
            "start": int(row["start"]),
            "end": int(row["end"]),
            "surface": str(row["surface"]),
        })
        original_labels[candidate_id] = original
        example = examples[str(row["id"])]
        segments = [
            value for value in example["segments"]
            if int(value["para_id"]) == int(row["para_id"])
        ]
        if len(segments) != 1:
            raise ValueError("hard-label paragraph geometry missing")
        segment = segments[0]
        paragraph_length = (
            int(segment["assembled_end"]) - int(segment["assembled_start"])
        )
        start = int(row["start"])
        end = int(row["end"])
        if not 0 <= start < end <= paragraph_length:
            raise ValueError("hard-label paragraph-local bounds differ")
        assembled_start = int(segment["assembled_start"]) + start
        assembled_end = int(segment["assembled_start"]) + end
        if str(example["text"])[assembled_start:assembled_end] != str(
            row["surface"]
        ):
            raise ValueError("hard-label candidate source mismatch")
    counts = {
        "real_not_person": sum(
            value == "real_not_person" for value in original_labels.values()
        ),
        "boundary_alternative": sum(
            value == "boundary_alternative"
            for value in original_labels.values()
        ),
    }
    if (
        counts["real_not_person"] != EXPECTED_REAL_NEGATIVES
        or counts["boundary_alternative"] != EXPECTED_BOUNDARIES
        or len({row["candidate_id"] for row in selected}) != len(selected)
    ):
        raise ValueError("hard-label selected inventory differs")

    by_id: dict[str, list[dict]] = {}
    for row in selected:
        by_id.setdefault(row["id"], []).append(row)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        reviewer_root = staging / "reviewer-tasks"
        reviewer_root.mkdir()
        tasks_dir = reviewer_root / "tasks"
        tasks_dir.mkdir()
        selected_tasks = []
        for identity in sorted(by_id):
            example = examples[identity]
            rows = sorted(
                by_id[identity],
                key=lambda row: row["candidate_id"],
            )
            task_id = _task_id(int(example["juan"]), int(example["jie_index"]))
            task = {
                "schema_version": 1,
                "status": TASK_STATUS,
                "phase": "revision-12-blind-hard-label-audit",
                "task_id": task_id,
                "juan": int(example["juan"]),
                "jie_index": int(example["jie_index"]),
                "review_scope": "current-numbered-jie-only",
                "jie": {
                    "text": str(example["text"]),
                    "segments": [
                        {
                            "para_id": int(segment["para_id"]),
                            "assembled_start": int(segment["assembled_start"]),
                            "assembled_end": int(segment["assembled_end"]),
                        }
                        for segment in example["segments"]
                    ],
                },
                "candidates": [
                    {
                        key: row[key]
                        for key in (
                            "candidate_id", "para_id", "start", "end", "surface"
                        )
                    }
                    for row in rows
                ],
                "allowed_labels": sorted(LABELS),
            }
            path = tasks_dir / f"task_{task_id}.json"
            path.write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            selected_tasks.append({
                "task_id": task_id,
                "juan": int(example["juan"]),
                "jie_index": int(example["jie_index"]),
                "task": str(Path("tasks") / path.name),
                "task_sha256": _sha256(path),
                "candidates": len(rows),
            })
        frozen = {
            "schema_version": 1,
            "status": TASK_STATUS,
            "revision": REVISION,
            "candidate_model_blind": True,
            "candidate_scores_hidden": True,
            "original_labels_hidden": True,
            "confirmation_read": False,
            "git_commit": _git_commit_clean(),
            "bindings": {
                "grouped_manifest_sha256": grouped_sha256,
                "examples_sha256": _sha256(examples_path),
                "existence_sha256": _sha256(existence_path),
            },
            "counts": {
                **counts,
                "candidates": len(selected),
                "tasks": len(selected_tasks),
            },
            "selected": selected_tasks,
        }
        task_manifest_path = reviewer_root / "manifest.json"
        task_manifest_path.write_text(
            json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        labels_root = staging / "sealed-original-labels"
        labels_root.mkdir()
        labels_path = labels_root / "original-labels.jsonl"
        labels_path.write_text(
            "".join(
                json.dumps(
                    {
                        "candidate_id": candidate_id,
                        "original_label": original_labels[candidate_id],
                    },
                    separators=(",", ":"),
                ) + "\n"
                for candidate_id in sorted(original_labels)
            ),
            encoding="utf-8",
        )
        labels_manifest = {
            "schema_version": 1,
            "status": LABEL_STATUS,
            "revision": REVISION,
            "reviewer_visible": False,
            "task_manifest_sha256": _sha256(task_manifest_path),
            "counts": {
                **counts,
                "candidates": len(original_labels),
            },
            "original_labels_sha256": _sha256(labels_path),
        }
        (labels_root / "manifest.json").write_text(
            json.dumps(labels_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return frozen


def validate_outputs(
    task_root: Path,
    raw_root: Path,
    output_dir: Path,
    reviewer: str,
    model_family: str,
) -> dict:
    if reviewer not in REVIEWERS:
        raise ValueError(f"unsupported hard-label reviewer: {reviewer}")
    model_family = model_family.strip().lower()
    if model_family not in MODEL_FAMILIES:
        raise ValueError("unsupported hard-label model family")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"hard-label output exists: {output_dir}")
    task_manifest_path = task_root / "manifest.json"
    task_manifest = _read(task_manifest_path)
    if (
        task_manifest.get("status") != TASK_STATUS
        or task_manifest.get("candidate_model_blind") is not True
        or task_manifest.get("candidate_scores_hidden") is not True
        or task_manifest.get("original_labels_hidden") is not True
        or task_manifest.get("confirmation_read") is not False
    ):
        raise ValueError("hard-label task binding differs")
    expected = {
        str(row["task_id"]): row for row in task_manifest["selected"]
    }
    validated = []
    source_hashes = {}
    seen = set()
    for path in sorted(raw_root.glob("*.json")):
        payload = _read(path)
        task_id = str(payload.get("task_id", ""))
        selected = expected.get(task_id)
        if selected is None or task_id in seen:
            raise ValueError(f"unexpected hard-label task: {task_id}")
        task_path = task_root / selected["task"]
        task = _read(task_path)
        if (
            payload.get("schema_version") != 1
            or payload.get("phase") != "revision-12-blind-hard-label-audit"
            or payload.get("reviewer") != reviewer
            or payload.get("model_family") != model_family
            or payload.get("task_sha256") != selected["task_sha256"]
            or _sha256(task_path) != selected["task_sha256"]
        ):
            raise ValueError(f"hard-label provenance differs: {task_id}")
        candidate_ids = {
            str(row["candidate_id"]) for row in task["candidates"]
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
                or candidate_id not in candidate_ids
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
                    f"invalid hard-label decision: {task_id} {candidate_id}"
                )
            decisions[candidate_id] = {
                "candidate_id": candidate_id,
                "label": str(row["label"]),
                "confidence": float(confidence),
                "rationale": rationale.strip(),
            }
        if set(decisions) != candidate_ids:
            raise ValueError(f"hard-label coverage differs: {task_id}")
        validated.append({
            "task_id": task_id,
            "task_sha256": selected["task_sha256"],
            "decisions": [
                decisions[candidate_id] for candidate_id in sorted(decisions)
            ],
        })
        seen.add(task_id)
        source_hashes[path.name] = _sha256(path)
    if seen != set(expected):
        raise ValueError("hard-label reviewer tasks missing")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        decisions_path = staging / "decisions.jsonl"
        decisions_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
                for row in sorted(validated, key=lambda row: row["task_id"])
            ),
            encoding="utf-8",
        )
        frozen = {
            "schema_version": 1,
            "status": OUTPUT_STATUS,
            "revision": REVISION,
            "reviewer": reviewer,
            "model_family": model_family,
            "candidate_model_blind": True,
            "candidate_scores_hidden": True,
            "original_labels_hidden": True,
            "task_manifest_sha256": _sha256(task_manifest_path),
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
            json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return frozen


def route_reviews(
    task_root: Path,
    labels_root: Path,
    reviewer_a_root: Path,
    reviewer_b_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"hard-label routing output exists: {output_dir}")
    task_manifest_path = task_root / "manifest.json"
    task_manifest = _read(task_manifest_path)

    def load_reviewer(root: Path, reviewer: str) -> tuple[dict, dict]:
        manifest = _read(root / "manifest.json")
        path = root / "decisions.jsonl"
        if (
            manifest.get("status") != OUTPUT_STATUS
            or manifest.get("reviewer") != reviewer
            or manifest.get("task_manifest_sha256")
            != _sha256(task_manifest_path)
            or manifest.get("decisions_sha256") != _sha256(path)
        ):
            raise ValueError(f"hard-label reviewer {reviewer} binding differs")
        decisions = {}
        for task in _read_jsonl(path):
            for row in task["decisions"]:
                candidate_id = str(row["candidate_id"])
                if candidate_id in decisions:
                    raise ValueError("duplicate hard-label candidate decision")
                decisions[candidate_id] = row
        return manifest, decisions

    manifest_a, decisions_a = load_reviewer(reviewer_a_root, "A")
    manifest_b, decisions_b = load_reviewer(reviewer_b_root, "B")
    if (
        manifest_a.get("model_family") not in MODEL_FAMILIES
        or manifest_b.get("model_family") not in MODEL_FAMILIES
        or manifest_a["model_family"] == manifest_b["model_family"]
    ):
        raise ValueError("hard-label reviewer families must differ")
    labels_manifest_path = labels_root / "manifest.json"
    labels_manifest = _read(labels_manifest_path)
    original_path = labels_root / "original-labels.jsonl"
    if (
        task_manifest.get("status") != TASK_STATUS
        or labels_manifest.get("status") != LABEL_STATUS
        or labels_manifest.get("reviewer_visible") is not False
        or labels_manifest.get("task_manifest_sha256")
        != _sha256(task_manifest_path)
        or labels_manifest.get("original_labels_sha256")
        != _sha256(original_path)
    ):
        raise ValueError("hard-label routing task binding differs")
    original = {
        str(row["candidate_id"]): str(row["original_label"])
        for row in _read_jsonl(original_path)
    }
    if set(decisions_a) != set(original) or set(decisions_b) != set(original):
        raise ValueError("hard-label routing candidate inventory differs")
    candidate_tasks = {}
    tasks_by_id = {}
    for selected in task_manifest["selected"]:
        task_path = task_root / selected["task"]
        task = _read(task_path)
        if (
            _sha256(task_path) != selected["task_sha256"]
            or task.get("task_id") != selected["task_id"]
            or int(task.get("juan", -1)) != int(selected["juan"])
            or int(task.get("jie_index", -1)) != int(selected["jie_index"])
            or len(task.get("candidates", [])) != int(selected["candidates"])
        ):
            raise ValueError("hard-label routing task provenance differs")
        task_candidate_ids = {
            str(candidate["candidate_id"]) for candidate in task["candidates"]
        }
        if (
            set(
                candidate_id for candidate_id in task_candidate_ids
                if candidate_id in decisions_a
            ) != task_candidate_ids
            or set(
                candidate_id for candidate_id in task_candidate_ids
                if candidate_id in decisions_b
            ) != task_candidate_ids
        ):
            raise ValueError("hard-label per-task reviewer coverage differs")
        tasks_by_id[str(selected["task_id"])] = task
        for candidate in task["candidates"]:
            candidate_tasks[str(candidate["candidate_id"])] = {
                "task_id": str(selected["task_id"]),
                "juan": int(selected["juan"]),
                "jie_index": int(selected["jie_index"]),
            }
    if set(candidate_tasks) != set(original):
        raise ValueError("hard-label routing task map differs")
    routed = []
    accepted = []
    for candidate_id in sorted(original):
        a = decisions_a[candidate_id]
        b = decisions_b[candidate_id]
        agreement = (
            a["label"] == b["label"]
            and a["label"] != "uncertain"
            and float(a["confidence"]) >= HIGH_CONFIDENCE
            and float(b["confidence"]) >= HIGH_CONFIDENCE
        )
        row = {
            "candidate_id": candidate_id,
            **candidate_tasks[candidate_id],
            "original_label": original[candidate_id],
            "a_label": a["label"],
            "a_confidence": float(a["confidence"]),
            "b_label": b["label"],
            "b_confidence": float(b["confidence"]),
            "route": (
                "accepted_cross_family_agreement"
                if agreement
                else "human_review"
            ),
            "audited_label": a["label"] if agreement else None,
        }
        (accepted if agreement else routed).append(row)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        sealed_root = staging / "sealed-routing"
        sealed_root.mkdir()
        routing_path = sealed_root / "routing.jsonl"
        routing_path.write_text(
            "".join(
                json.dumps(row, separators=(",", ":")) + "\n"
                for row in accepted + routed
            ),
            encoding="utf-8",
        )
        frozen = {
            "schema_version": 1,
            "status": ROUTING_STATUS,
            "revision": REVISION,
            "confirmation_read": False,
            "high_confidence": HIGH_CONFIDENCE,
            "task_manifest_sha256": _sha256(task_manifest_path),
            "original_labels_manifest_sha256": _sha256(labels_manifest_path),
            "reviewer_a_manifest_sha256": _sha256(
                reviewer_a_root / "manifest.json"
            ),
            "reviewer_b_manifest_sha256": _sha256(
                reviewer_b_root / "manifest.json"
            ),
            "counts": {
                "candidates": len(original),
                "accepted": len(accepted),
                "human_review": len(routed),
            },
            "routing_sha256": _sha256(routing_path),
        }
        routing_manifest_path = sealed_root / "manifest.json"
        routing_manifest_path.write_text(
            json.dumps(frozen, indent=2) + "\n",
            encoding="utf-8",
        )
        human_root = staging / "human-review"
        human_root.mkdir()
        task_dir = human_root / "tasks"
        task_dir.mkdir()
        routed_by_task: dict[str, set[str]] = {}
        for row in routed:
            routed_by_task.setdefault(row["task_id"], set()).add(
                row["candidate_id"]
            )
        human_selected = []
        for task_id in sorted(routed_by_task):
            source = tasks_by_id[task_id]
            candidate_ids = routed_by_task[task_id]
            human_task = {
                **source,
                "status": "ml_production_hard_label_human_tasks",
                "phase": "revision-12-blind-human-hard-label-audit",
                "candidates": [
                    candidate for candidate in source["candidates"]
                    if str(candidate["candidate_id"]) in candidate_ids
                ],
            }
            path = task_dir / f"task_{task_id}.json"
            path.write_text(
                json.dumps(human_task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            human_selected.append({
                "task_id": task_id,
                "juan": int(source["juan"]),
                "jie_index": int(source["jie_index"]),
                "task": str(Path("tasks") / path.name),
                "task_sha256": _sha256(path),
                "candidates": len(human_task["candidates"]),
            })
        human_manifest = {
            "schema_version": 1,
            "status": "ml_production_hard_label_human_tasks",
            "revision": REVISION,
            "candidate_model_blind": True,
            "candidate_scores_hidden": True,
            "original_labels_hidden": True,
            "ai_judgments_hidden": True,
            "confirmation_read": False,
            "source_routing_manifest_sha256": _sha256(
                routing_manifest_path
            ),
            "counts": {
                "tasks": len(human_selected),
                "candidates": len(routed),
            },
            "selected": human_selected,
        }
        (human_root / "manifest.json").write_text(
            json.dumps(human_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return frozen


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and validate Revision-12 hard-label audit."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--grouped-data", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--tasks", type=Path, required=True)
    validate.add_argument("--raw", type=Path, required=True)
    validate.add_argument("--reviewer", choices=sorted(REVIEWERS), required=True)
    validate.add_argument("--model-family", required=True)
    validate.add_argument("--output", type=Path, required=True)
    route = sub.add_parser("route")
    route.add_argument("--tasks", type=Path, required=True)
    route.add_argument("--original-labels", type=Path, required=True)
    route.add_argument("--reviewer-a", type=Path, required=True)
    route.add_argument("--reviewer-b", type=Path, required=True)
    route.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_tasks(args.grouped_data, args.output)
    elif args.command == "validate":
        result = validate_outputs(
            args.tasks,
            args.raw,
            args.output,
            args.reviewer,
            args.model_family,
        )
    else:
        result = route_reviews(
            args.tasks,
            args.original_labels,
            args.reviewer_a,
            args.reviewer_b,
            args.output,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
