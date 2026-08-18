from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_grouped_data import GROUPED_DATA_STATUS
from production_precision_grouped_verifier import _read, _read_jsonl, _sha256
from production_precision_hard_label_freeze import FROZEN_STATUS
from production_precision_hard_label_audit import _candidate_id
from production_train import _make_read_only


TASK_STATUS = "ml_production_boundary_target_tasks"
FROZEN_TARGET_STATUS = "ml_production_boundary_targets_frozen"
EXPECTED_BOUNDARIES = 170
DECISIONS = {"targets", "uncertain"}


def _task_id(candidate_id: str) -> str:
    return hashlib.sha256(
        f"revision-13:{candidate_id}".encode("ascii")
    ).hexdigest()[:20]


def build_tasks(
    grouped_root: Path, audit_root: Path, output_dir: Path
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"boundary-target tasks exist: {output_dir}")
    grouped_manifest_path = grouped_root / "manifest.json"
    grouped_manifest = _read(grouped_manifest_path)
    examples_path = grouped_root / "examples.jsonl"
    existence_path = grouped_root / "existence.jsonl"
    audit_manifest_path = audit_root / "manifest.json"
    audit_manifest = _read(audit_manifest_path)
    audit_path = audit_root / "decisions.jsonl"
    if (
        grouped_manifest.get("status") != GROUPED_DATA_STATUS
        or grouped_manifest.get("confirmation_read") is not False
        or grouped_manifest.get("outputs", {}).get("examples_sha256")
        != _sha256(examples_path)
        or grouped_manifest.get("outputs", {}).get("existence_sha256")
        != _sha256(existence_path)
        or audit_manifest.get("status") != FROZEN_STATUS
        or audit_manifest.get("confirmation_read") is not False
        or audit_manifest.get("decisions_sha256") != _sha256(audit_path)
        or audit_manifest.get("bindings", {}).get("task_manifest_sha256")
        is None
    ):
        raise ValueError("boundary-target source binding differs")
    examples = {
        str(row["id"]): row for row in _read_jsonl(examples_path)
    }
    boundaries = [
        row for row in _read_jsonl(audit_path)
        if row["audited_label"] == "wrong_boundary"
    ]
    if (
        len(boundaries) != EXPECTED_BOUNDARIES
        or len({row["candidate_id"] for row in boundaries})
        != EXPECTED_BOUNDARIES
        or any(
            _candidate_id(_sha256(grouped_manifest_path), row)
            != row["candidate_id"]
            for row in boundaries
        )
    ):
        raise ValueError("boundary-target inventory differs")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        task_dir = staging / "tasks"
        task_dir.mkdir()
        selected = []
        for row in sorted(boundaries, key=lambda item: item["candidate_id"]):
            example = examples.get(
                f"juan-{int(row['juan']):03d}-jie-{int(row['jie_index']):04d}"
            )
            if example is None:
                raise ValueError("boundary-target example missing")
            segments = [
                segment for segment in example["segments"]
                if int(segment["para_id"]) == int(row["para_id"])
            ]
            if len(segments) != 1:
                raise ValueError("boundary-target paragraph missing")
            segment = segments[0]
            paragraph = str(example["text"])[
                int(segment["assembled_start"]):int(segment["assembled_end"])
            ]
            start = int(row["start"])
            end = int(row["end"])
            if (
                not 0 <= start < end <= len(paragraph)
                or paragraph[start:end] != row["surface"]
            ):
                raise ValueError("boundary-target source geometry differs")
            task_id = _task_id(str(row["candidate_id"]))
            task = {
                "schema_version": 1,
                "status": TASK_STATUS,
                "phase": "revision-13-blind-exact-target-audit",
                "task_id": task_id,
                "candidate_id": str(row["candidate_id"]),
                "juan": int(row["juan"]),
                "jie_index": int(row["jie_index"]),
                "review_scope": "current-numbered-jie-only",
                "jie": {
                    "text": str(example["text"]),
                    "segments": [
                        {
                            "para_id": int(value["para_id"]),
                            "assembled_start": int(value["assembled_start"]),
                            "assembled_end": int(value["assembled_end"]),
                        }
                        for value in example["segments"]
                    ],
                },
                "wrong_candidate": {
                    "para_id": int(row["para_id"]),
                    "start": start,
                    "end": end,
                    "surface": str(row["surface"]),
                },
                "allowed_decisions": sorted(DECISIONS),
            }
            path = task_dir / f"task_{task_id}.json"
            path.write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            selected.append({
                "task_id": task_id,
                "candidate_id": str(row["candidate_id"]),
                "juan": int(row["juan"]),
                "jie_index": int(row["jie_index"]),
                "task": str(Path("tasks") / path.name),
                "task_sha256": _sha256(path),
            })
        if (
            len({row["task_id"] for row in selected}) != EXPECTED_BOUNDARIES
            or len({row["task"] for row in selected}) != EXPECTED_BOUNDARIES
        ):
            raise ValueError("boundary-target task identity collision")
        manifest = {
            "schema_version": 1,
            "status": TASK_STATUS,
            "revision": 13,
            "candidate_model_blind": True,
            "candidate_scores_hidden": True,
            "old_references_hidden": True,
            "prior_judgments_hidden": True,
            "confirmation_read": False,
            "git_commit": _git_commit_clean(),
            "bindings": {
                "grouped_manifest_sha256": _sha256(grouped_manifest_path),
                "audit_manifest_sha256": _sha256(audit_manifest_path),
                "examples_sha256": _sha256(examples_path),
                "existence_sha256": _sha256(existence_path),
            },
            "counts": {"tasks": len(selected), "candidates": len(selected)},
            "selected": selected,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return manifest


def freeze_targets(
    task_root: Path, raw_root: Path, output_dir: Path
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"boundary-target freeze exists: {output_dir}")
    manifest_path = task_root / "manifest.json"
    manifest = _read(manifest_path)
    if (
        manifest.get("status") != TASK_STATUS
        or manifest.get("candidate_model_blind") is not True
        or manifest.get("old_references_hidden") is not True
        or manifest.get("prior_judgments_hidden") is not True
        or manifest.get("confirmation_read") is not False
        or int(manifest.get("counts", {}).get("tasks", -1))
        != EXPECTED_BOUNDARIES
    ):
        raise ValueError("boundary-target task binding differs")
    expected = {
        str(row["task_id"]): row for row in manifest["selected"]
    }
    if (
        len(expected) != EXPECTED_BOUNDARIES
        or len({str(row["candidate_id"]) for row in manifest["selected"]})
        != EXPECTED_BOUNDARIES
        or len({str(row["task"]) for row in manifest["selected"]})
        != EXPECTED_BOUNDARIES
    ):
        raise ValueError("boundary-target frozen task inventory differs")
    frozen = []
    raw_hashes = {}
    seen = set()
    for path in sorted(raw_root.glob("*.json")):
        payload = _read(path)
        task_id = str(payload.get("task_id", ""))
        selected = expected.get(task_id)
        if selected is None or task_id in seen:
            raise ValueError(f"unexpected boundary-target task: {task_id}")
        task_path = task_root / selected["task"]
        task = _read(task_path)
        if (
            payload.get("schema_version") != 1
            or payload.get("phase")
            != "revision-13-blind-exact-target-audit"
            or payload.get("adjudicator") != "copilot_teacher"
            or payload.get("task_sha256") != selected["task_sha256"]
            or _sha256(task_path) != selected["task_sha256"]
            or payload.get("candidate_id") != selected["candidate_id"]
            or payload.get("decision") not in DECISIONS
            or task.get("status") != TASK_STATUS
            or task.get("task_id") != task_id
            or task.get("candidate_id") != selected["candidate_id"]
        ):
            raise ValueError(f"boundary-target provenance differs: {task_id}")
        targets = payload.get("targets")
        rationale = payload.get("rationale")
        if (
            not isinstance(targets, list)
            or not isinstance(rationale, str)
            or not rationale.strip()
            or (
                payload["decision"] == "uncertain" and targets
            )
            or (
                payload["decision"] == "targets" and not targets
            )
        ):
            raise ValueError(f"invalid boundary-target decision: {task_id}")
        wrong = task["wrong_candidate"]
        segments = {
            int(value["para_id"]): value for value in task["jie"]["segments"]
        }
        normalized = []
        geometries = set()
        for target in targets:
            if set(target) != {"para_id", "start", "end", "surface"}:
                raise ValueError("invalid boundary-target keys")
            para_id = int(target["para_id"])
            start = int(target["start"])
            end = int(target["end"])
            segment = segments.get(para_id)
            if segment is None:
                raise ValueError("boundary-target paragraph differs")
            paragraph = str(task["jie"]["text"])[
                int(segment["assembled_start"]):int(segment["assembled_end"])
            ]
            geometry = (para_id, start, end)
            overlaps = (
                para_id == int(wrong["para_id"])
                and start < int(wrong["end"])
                and int(wrong["start"]) < end
            )
            if (
                geometry in geometries
                or not 0 <= start < end <= len(paragraph)
                or paragraph[start:end] != target["surface"]
                or not overlaps
                or (
                    start == int(wrong["start"])
                    and end == int(wrong["end"])
                )
            ):
                raise ValueError(f"invalid exact target: {task_id}")
            geometries.add(geometry)
            normalized.append({
                "para_id": para_id,
                "start": start,
                "end": end,
                "surface": str(target["surface"]),
            })
        if len(normalized) > 1:
            ordered = sorted(
                normalized,
                key=lambda row: (row["para_id"], row["start"], row["end"]),
            )
            if any(
                target["para_id"] != int(wrong["para_id"])
                or target["start"] < int(wrong["start"])
                or target["end"] > int(wrong["end"])
                for target in ordered
            ) or any(
                left["para_id"] == right["para_id"]
                and left["start"] < right["end"]
                and right["start"] < left["end"]
                for left, right in zip(ordered, ordered[1:])
            ):
                raise ValueError(f"invalid merged exact targets: {task_id}")
        frozen.append({
            "task_id": task_id,
            "candidate_id": selected["candidate_id"],
            "juan": int(selected["juan"]),
            "jie_index": int(selected["jie_index"]),
            "wrong_candidate": wrong,
            "decision": payload["decision"],
            "targets": sorted(
                normalized,
                key=lambda row: (row["para_id"], row["start"], row["end"]),
            ),
            "rationale": rationale.strip(),
            "decision_source": "user_authorized_copilot_teacher",
        })
        seen.add(task_id)
        raw_hashes[path.name] = _sha256(path)
    if seen != set(expected) or len(frozen) != EXPECTED_BOUNDARIES:
        raise ValueError("boundary-target coverage differs")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        targets_path = staging / "targets.jsonl"
        targets_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
                for row in sorted(frozen, key=lambda row: row["candidate_id"])
            ),
            encoding="utf-8",
        )
        counts = {
            "tasks": len(frozen),
            "target_decisions": sum(
                row["decision"] == "targets" for row in frozen
            ),
            "uncertain": sum(
                row["decision"] == "uncertain" for row in frozen
            ),
            "exact_targets": sum(len(row["targets"]) for row in frozen),
            "multi_target_decisions": sum(
                len(row["targets"]) > 1 for row in frozen
            ),
        }
        result = {
            "schema_version": 1,
            "status": FROZEN_TARGET_STATUS,
            "revision": 13,
            "formal_grade": False,
            "confirmation_read": False,
            "git_commit": _git_commit_clean(),
            "task_manifest_sha256": _sha256(manifest_path),
            "bindings": dict(manifest["bindings"]),
            "counts": counts,
            "raw_outputs": raw_hashes,
            "targets_sha256": _sha256(targets_path),
            "claim_limit": (
                "Fit-only geometry audit by a user-authorized Copilot "
                "teacher; not formal-grade evidence."
            ),
        }
        (staging / "manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or freeze Revision-13 exact-target geometry audit."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--grouped-data", type=Path, required=True)
    build.add_argument("--audit", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--tasks", type=Path, required=True)
    freeze.add_argument("--raw", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_tasks(args.grouped_data, args.audit, args.output)
    else:
        result = freeze_targets(args.tasks, args.raw, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
