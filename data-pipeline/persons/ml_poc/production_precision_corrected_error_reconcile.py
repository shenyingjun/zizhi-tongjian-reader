from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_corrected_error_audit import TASK_STATUS as ERROR_TASK_STATUS
from production_precision_corrected_error_freeze import FROZEN_STATUS
from production_precision_corrected_error_target_audit import (
    ARTIFACT_STATUS as TARGET_TASK_STATUS,
    TASK_STATUS as TARGET_STATUS,
)
from production_precision_grouped_verifier import _read, _read_jsonl, _sha256
from production_train import _make_read_only


REVISION = 14
STATUS = "ml_production_precision_corrected_error_reconciled"
EXPECTED_DECISIONS = 97
EXPECTED_INITIAL_WRONG = 6
EXPECTED_HUMAN_RECLASSIFICATIONS = 3
EXPECTED_TARGET_CORRECTIONS = 3
LABELS = ("exact_person", "not_person", "uncertain", "wrong_boundary")
RAW_KEYS = {
    "task_id",
    "candidate_id",
    "decision",
    "targets",
    "rationale",
    "reviewer",
    "model",
}


def _read_raw(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"corrected error target raw output differs: {path}")
    return value


def _validate_target_output(task: dict, raw: dict) -> tuple[str, list[dict], str]:
    if (
        set(raw) != RAW_KEYS
        or raw.get("task_id") != task.get("task_id")
        or raw.get("candidate_id")
        != task.get("source_candidate", {}).get("candidate_id")
        or raw.get("decision") not in {"targets", "uncertain"}
        or raw.get("reviewer") != "copilot-teacher"
        or raw.get("model") != "gpt-5.6-sol"
        or not isinstance(raw.get("rationale"), str)
        or not raw["rationale"].strip()
        or not isinstance(raw.get("targets"), list)
    ):
        raise ValueError("corrected error target raw schema differs")
    if raw["decision"] == "uncertain":
        if raw["targets"]:
            raise ValueError("uncertain corrected error target has geometry")
        return "uncertain", [], raw["rationale"].strip()
    if not raw["targets"]:
        raise ValueError("corrected error target geometry is empty")

    source = task["source_candidate"]
    segments = {
        int(row["para_id"]): row for row in task["jie"]["segments"]
    }
    normalized = []
    seen = set()
    unchanged = False
    for target in raw["targets"]:
        if set(target) != {"para_id", "start", "end", "surface"}:
            raise ValueError("corrected error target geometry fields differ")
        para_id = int(target["para_id"])
        start = int(target["start"])
        end = int(target["end"])
        segment = segments.get(para_id)
        if segment is None:
            raise ValueError("corrected error target paragraph differs")
        paragraph = str(task["jie"]["text"])[
            int(segment["assembled_start"]):int(segment["assembled_end"])
        ]
        geometry = (para_id, start, end)
        if (
            geometry in seen
            or not 0 <= start < end <= len(paragraph)
            or paragraph[start:end] != target["surface"]
            or para_id != int(source["para_id"])
            or not (
                start < int(source["end"]) and int(source["start"]) < end
            )
        ):
            raise ValueError("corrected error target geometry differs")
        if start == int(source["start"]) and end == int(source["end"]):
            unchanged = True
        seen.add(geometry)
        normalized.append({
            "para_id": para_id,
            "start": start,
            "end": end,
            "surface": str(target["surface"]),
        })
    if unchanged:
        if len(normalized) != 1:
            raise ValueError("unchanged target is mixed with replacements")
        return "unchanged", normalized, raw["rationale"].strip()
    ordered = sorted(
        normalized, key=lambda row: (row["para_id"], row["start"], row["end"])
    )
    if any(
        left["para_id"] == right["para_id"]
        and left["start"] < right["end"]
        and right["start"] < left["end"]
        for left, right in zip(ordered, ordered[1:])
    ):
        raise ValueError("corrected error targets overlap each other")
    return "targets", ordered, raw["rationale"].strip()


def reconcile(
    error_task_root: Path,
    frozen_root: Path,
    target_task_root: Path,
    initial_raw_root: Path,
    rereview_raw_root: Path,
    human_decisions_path: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"corrected error reconciliation exists: {output_dir}")
    frozen_manifest_path = frozen_root / "manifest.json"
    frozen_decisions_path = frozen_root / "decisions.jsonl"
    target_manifest_path = target_task_root / "manifest.json"
    mapping_path = target_task_root / "sealed-mapping" / "mapping.jsonl"
    frozen_manifest = _read(frozen_manifest_path)
    target_manifest = _read(target_manifest_path)
    if (
        frozen_manifest.get("status") != FROZEN_STATUS
        or int(frozen_manifest.get("counts", {}).get("tasks", -1))
        != EXPECTED_DECISIONS
        or int(frozen_manifest.get("counts", {}).get("wrong_boundary", -1))
        != EXPECTED_INITIAL_WRONG
        or frozen_manifest.get("outputs", {}).get("decisions_sha256")
        != _sha256(frozen_decisions_path)
        or target_manifest.get("status") != TARGET_TASK_STATUS
        or int(target_manifest.get("counts", {}).get("tasks", -1))
        != EXPECTED_INITIAL_WRONG
        or target_manifest.get("bindings", {}).get("frozen_manifest_sha256")
        != _sha256(frozen_manifest_path)
        or target_manifest.get("outputs", {}).get("mapping_sha256")
        != _sha256(mapping_path)
    ):
        raise ValueError("corrected error reconciliation binding differs")

    decisions = _read_jsonl(frozen_decisions_path)
    by_source_task = {str(row["task_id"]): row for row in decisions}
    if len(decisions) != EXPECTED_DECISIONS or len(by_source_task) != len(decisions):
        raise ValueError("corrected error reconciliation decisions differ")
    mapping_rows = _read_jsonl(mapping_path)
    mapping = {str(row["task_id"]): row for row in mapping_rows}
    if len(mapping_rows) != EXPECTED_INITIAL_WRONG or len(mapping) != len(mapping_rows):
        raise ValueError("corrected error reconciliation target mapping differs")
    human_rows = _read_jsonl(human_decisions_path)
    human = {str(row["task_id"]): row for row in human_rows}
    if len(human_rows) != EXPECTED_HUMAN_RECLASSIFICATIONS or len(human) != len(
        human_rows
    ):
        raise ValueError("corrected error human decision inventory differs")

    target_corrections = []
    resolutions = {}
    raw_hashes = []
    for target_task_id, map_row in mapping.items():
        target_path = (
            target_task_root / "reviewer-tasks" / f"task_{target_task_id}.json"
        )
        target_task = _read(target_path)
        source_task_id = str(map_row["source_task_id"])
        source_decision = by_source_task.get(source_task_id)
        if (
            target_task.get("status") != TARGET_STATUS
            or target_task.get("task_id") != target_task_id
            or target_task.get("source_candidate", {}).get("candidate_id")
            != map_row["candidate_id"]
            or source_decision is None
            or source_decision["label"] != "wrong_boundary"
            or source_decision["candidate_id"] != map_row["source_candidate_id"]
        ):
            raise ValueError("corrected error reconciliation source differs")
        initial_path = initial_raw_root / f"{target_task_id}.json"
        initial_raw = _read_raw(initial_path)
        initial_kind, initial_targets, initial_rationale = _validate_target_output(
            target_task, initial_raw
        )
        raw_hashes.append({
            "stage": "initial",
            "path": initial_path.name,
            "sha256": _sha256(initial_path),
        })
        selected_kind = initial_kind
        selected_targets = initial_targets
        selected_rationale = initial_rationale
        selected_stage = "initial"
        rereview_path = rereview_raw_root / f"{target_task_id}.json"
        if rereview_path.exists():
            if initial_kind != "unchanged":
                raise ValueError("unexpected corrected error target rereview")
            rereview_raw = _read_raw(rereview_path)
            try:
                selected_kind, selected_targets, selected_rationale = (
                    _validate_target_output(target_task, rereview_raw)
                )
            except ValueError:
                if target_task_id not in human:
                    raise
                selected_kind = "invalid"
                selected_targets = []
                selected_rationale = str(rereview_raw.get("rationale", "")).strip()
            selected_stage = "rereview"
            raw_hashes.append({
                "stage": "rereview",
                "path": rereview_path.name,
                "sha256": _sha256(rereview_path),
            })

        human_row = human.get(target_task_id)
        if human_row is not None:
            if (
                selected_kind not in {"invalid", "uncertain"}
                or set(human_row) != {
                    "task_id",
                    "candidate_id",
                    "decision",
                    "rationale",
                    "adjudicator",
                    "authority",
                }
                or human_row["candidate_id"] != map_row["candidate_id"]
                or human_row["decision"] != "reclassify_exact_person"
                or human_row["adjudicator"] != "user"
                or human_row["authority"]
                != "explicit_interactive_human_adjudication"
                or not str(human_row["rationale"]).strip()
            ):
                raise ValueError("corrected error human adjudication differs")
            resolutions[source_task_id] = {
                "final_label": "exact_person",
                "resolution": "human_reclassification",
                "resolution_rationale": str(human_row["rationale"]).strip(),
                "target_task_id": target_task_id,
            }
        elif selected_kind == "targets":
            resolutions[source_task_id] = {
                "final_label": "wrong_boundary",
                "resolution": f"{selected_stage}_exact_targets",
                "resolution_rationale": selected_rationale,
                "target_task_id": target_task_id,
            }
            source_task_path = (
                error_task_root
                / "reviewer-tasks"
                / f"task_{source_task_id}.json"
            )
            source_task = _read(source_task_path)
            if source_task.get("status") != ERROR_TASK_STATUS:
                raise ValueError("corrected error source task status differs")
            target_corrections.append({
                "source_task_id": source_task_id,
                "source_candidate": source_task["candidate"],
                "targets": selected_targets,
                "resolution": f"{selected_stage}_exact_targets",
                "rationale": selected_rationale,
            })
        else:
            raise ValueError("corrected error target remains unresolved")
    if (
        set(human) != {
            row["target_task_id"]
            for row in resolutions.values()
            if row["resolution"] == "human_reclassification"
        }
        or len(target_corrections) != EXPECTED_TARGET_CORRECTIONS
        or len(resolutions) != EXPECTED_INITIAL_WRONG
    ):
        raise ValueError("corrected error resolution coverage differs")

    reconciled = []
    for row in decisions:
        resolution = resolutions.get(str(row["task_id"]))
        reconciled.append({
            **row,
            "initial_label": str(row["label"]),
            "final_label": (
                resolution["final_label"] if resolution else str(row["label"])
            ),
            "resolution": (
                resolution["resolution"] if resolution else "unchanged"
            ),
            "resolution_rationale": (
                resolution["resolution_rationale"] if resolution else None
            ),
            "target_task_id": (
                resolution["target_task_id"] if resolution else None
            ),
        })
    counts = {
        label: sum(row["final_label"] == label for row in reconciled)
        for label in LABELS
    }
    if counts != {
        "exact_person": 69,
        "not_person": 25,
        "uncertain": 0,
        "wrong_boundary": 3,
    }:
        raise ValueError("corrected error reconciled labels differ")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        reconciled_path = staging / "decisions.jsonl"
        reconciled_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in sorted(reconciled, key=lambda item: item["task_id"])
            ),
            encoding="utf-8",
        )
        targets_path = staging / "targets.jsonl"
        targets_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in sorted(
                    target_corrections, key=lambda item: item["source_task_id"]
                )
            ),
            encoding="utf-8",
        )
        raw_hashes_path = staging / "target-raw-hashes.jsonl"
        raw_hashes_path.write_text(
            "".join(
                json.dumps(row, separators=(",", ":")) + "\n"
                for row in sorted(raw_hashes, key=lambda item: (
                    item["stage"], item["path"]
                ))
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "status": STATUS,
            "revision": REVISION,
            "formal_grade": False,
            "fit_only": True,
            "confirmation_read": False,
            "git_commit": _git_commit_clean(),
            "bindings": {
                "frozen_manifest_sha256": _sha256(frozen_manifest_path),
                "frozen_decisions_sha256": _sha256(frozen_decisions_path),
                "target_manifest_sha256": _sha256(target_manifest_path),
                "target_mapping_sha256": _sha256(mapping_path),
                "human_decisions_sha256": _sha256(human_decisions_path),
            },
            "counts": {
                "decisions": len(reconciled),
                "human_reclassifications": EXPECTED_HUMAN_RECLASSIFICATIONS,
                "target_corrections": len(target_corrections),
                **counts,
            },
            "outputs": {
                "decisions_sha256": _sha256(reconciled_path),
                "targets_sha256": _sha256(targets_path),
                "target_raw_hashes_sha256": _sha256(raw_hashes_path),
            },
            "next_action": "rerun_conflict_closure_and_corrected_inventory",
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
        description="Reconcile Revision-14 labels and exact targets."
    )
    parser.add_argument("--error-task-root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--target-task-root", type=Path, required=True)
    parser.add_argument("--initial-raw-root", type=Path, required=True)
    parser.add_argument("--rereview-raw-root", type=Path, required=True)
    parser.add_argument("--human-decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reconcile(
        args.error_task_root,
        args.frozen_root,
        args.target_task_root,
        args.initial_raw_root,
        args.rereview_raw_root,
        args.human_decisions,
        args.output_dir,
    ), indent=2))


if __name__ == "__main__":
    main()
