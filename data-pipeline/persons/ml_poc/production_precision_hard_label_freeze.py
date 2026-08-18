from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import Counter
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_grouped_verifier import _read, _read_jsonl, _sha256
from production_precision_hard_label_audit import (
    LABELS,
    LABEL_STATUS,
    ROUTING_STATUS,
)
from production_train import _make_read_only


FROZEN_STATUS = "ml_production_hard_label_audit_frozen"
HUMAN_STATUS = "ml_production_hard_label_human_tasks"
EXPECTED_CANDIDATES = 303
EXPECTED_HUMAN = 298
EXPECTED_AGREEMENTS = 5
EXPECTED_REAL_NEGATIVES = 171
EXPECTED_BOUNDARIES = 132


def _confusion(rows: list[dict]) -> dict:
    originals = ("real_not_person", "boundary_alternative")
    return {
        original: {
            label: sum(
                row["original_label"] == original
                and row["audited_label"] == label
                for row in rows
            )
            for label in sorted(LABELS)
        }
        for original in originals
    }


def freeze_audit(
    tasks_root: Path,
    labels_root: Path,
    routing_root: Path,
    state_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"hard-label freeze output exists: {output_dir}")
    task_manifest_path = tasks_root / "manifest.json"
    task_manifest = _read(task_manifest_path)
    labels_manifest_path = labels_root / "manifest.json"
    labels_manifest = _read(labels_manifest_path)
    original_path = labels_root / "original-labels.jsonl"
    routing_manifest_path = routing_root / "sealed-routing" / "manifest.json"
    routing_manifest = _read(routing_manifest_path)
    routing_path = routing_root / "sealed-routing" / "routing.jsonl"
    human_manifest_path = routing_root / "human-review" / "manifest.json"
    human_manifest = _read(human_manifest_path)
    if (
        task_manifest.get("status")
        != "ml_production_hard_label_audit_tasks"
        or task_manifest.get("confirmation_read") is not False
        or labels_manifest.get("status") != LABEL_STATUS
        or labels_manifest.get("task_manifest_sha256")
        != _sha256(task_manifest_path)
        or labels_manifest.get("original_labels_sha256")
        != _sha256(original_path)
        or routing_manifest.get("status") != ROUTING_STATUS
        or routing_manifest.get("confirmation_read") is not False
        or routing_manifest.get("task_manifest_sha256")
        != _sha256(task_manifest_path)
        or routing_manifest.get("original_labels_manifest_sha256")
        != _sha256(labels_manifest_path)
        or routing_manifest.get("routing_sha256") != _sha256(routing_path)
        or human_manifest.get("status") != HUMAN_STATUS
        or human_manifest.get("confirmation_read") is not False
        or human_manifest.get("source_routing_manifest_sha256")
        != _sha256(routing_manifest_path)
        or int(human_manifest.get("counts", {}).get("candidates", -1))
        != EXPECTED_HUMAN
        or int(routing_manifest.get("counts", {}).get("accepted", -1))
        != EXPECTED_AGREEMENTS
        or int(routing_manifest.get("counts", {}).get("human_review", -1))
        != EXPECTED_HUMAN
    ):
        raise ValueError("hard-label freeze bindings differ")
    original = {
        str(row["candidate_id"]): str(row["original_label"])
        for row in _read_jsonl(original_path)
    }
    routing = {
        str(row["candidate_id"]): row for row in _read_jsonl(routing_path)
    }
    if (
        len(original) != EXPECTED_CANDIDATES
        or set(routing) != set(original)
        or len(routing) != EXPECTED_CANDIDATES
        or Counter(original.values()) != Counter({
            "real_not_person": EXPECTED_REAL_NEGATIVES,
            "boundary_alternative": EXPECTED_BOUNDARIES,
        })
    ):
        raise ValueError("hard-label freeze candidate inventory differs")

    candidates = {}
    human_by_task = {}
    for selected in task_manifest["selected"]:
        path = tasks_root / selected["task"]
        task = _read(path)
        if (
            _sha256(path) != selected["task_sha256"]
            or task.get("task_id") != selected["task_id"]
        ):
            raise ValueError("hard-label freeze task provenance differs")
        for candidate in task["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            if candidate_id in candidates:
                raise ValueError("duplicate hard-label freeze candidate")
            candidates[candidate_id] = {
                "task_id": str(selected["task_id"]),
                "juan": int(selected["juan"]),
                "jie_index": int(selected["jie_index"]),
                "para_id": int(candidate["para_id"]),
                "start": int(candidate["start"]),
                "end": int(candidate["end"]),
                "surface": str(candidate["surface"]),
            }
    for selected in human_manifest["selected"]:
        path = routing_root / "human-review" / selected["task"]
        if _sha256(path) != selected["task_sha256"]:
            raise ValueError("hard-label human task hash differs")
        human_by_task[str(selected["task_id"])] = {
            "candidate_ids": {
                str(row["candidate_id"]) for row in _read(path)["candidates"]
            },
            "task_sha256": str(selected["task_sha256"]),
        }
    expected_state_files = {
        f"task_{task_id}.json" for task_id in human_by_task
    }
    state_files = {path.name for path in state_root.glob("task_*.json")}
    if state_files != expected_state_files:
        raise ValueError("hard-label state-file inventory differs")
    human_decisions = {}
    state_hashes = {}
    for task_id, human_task in human_by_task.items():
        candidate_ids = human_task["candidate_ids"]
        path = state_root / f"task_{task_id}.json"
        state = _read(path)
        decisions = state.get("decisions", {})
        if (
            state.get("status") != HUMAN_STATUS
            or state.get("task_id") != task_id
            or state.get("task_sha256") != human_task["task_sha256"]
            or state.get("complete") is not True
            or set(decisions) != candidate_ids
            or any(label not in LABELS for label in decisions.values())
        ):
            raise ValueError(f"incomplete hard-label state: {task_id}")
        for candidate_id, label in decisions.items():
            if candidate_id in human_decisions:
                raise ValueError("duplicate hard-label human decision")
            human_decisions[candidate_id] = str(label)
        state_hashes[path.name] = _sha256(path)
    if len(human_decisions) != EXPECTED_HUMAN:
        raise ValueError("hard-label human decision count differs")

    frozen_rows = []
    source_counts = Counter()
    for candidate_id in sorted(original):
        route = routing[candidate_id]
        if route["route"] == "accepted_cross_family_agreement":
            label = str(route["audited_label"])
            a_confidence = route.get("a_confidence")
            b_confidence = route.get("b_confidence")
            if (
                route.get("a_label") != label
                or route.get("b_label") != label
                or label == "uncertain"
                or isinstance(a_confidence, bool)
                or not isinstance(a_confidence, (int, float))
                or not math.isfinite(float(a_confidence))
                or float(a_confidence) < 0.90
                or isinstance(b_confidence, bool)
                or not isinstance(b_confidence, (int, float))
                or not math.isfinite(float(b_confidence))
                or float(b_confidence) < 0.90
            ):
                raise ValueError("invalid cross-family accepted route")
            source = "cross_family_high_confidence_agreement"
        elif route["route"] == "human_review":
            label = human_decisions.get(candidate_id, "")
            source = "user_authorized_copilot_teacher"
        else:
            raise ValueError("unsupported hard-label route")
        if label not in LABELS or candidate_id not in candidates:
            raise ValueError("hard-label frozen decision differs")
        frozen_rows.append({
            "candidate_id": candidate_id,
            **candidates[candidate_id],
            "original_label": original[candidate_id],
            "audited_label": label,
            "decision_source": source,
        })
        source_counts[source] += 1
    if source_counts != Counter({
        "cross_family_high_confidence_agreement": EXPECTED_AGREEMENTS,
        "user_authorized_copilot_teacher": EXPECTED_HUMAN,
    }):
        raise ValueError("hard-label decision-source counts differ")
    counts = Counter(row["audited_label"] for row in frozen_rows)
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
                for row in frozen_rows
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "status": FROZEN_STATUS,
            "revision": 12,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "git_commit": _git_commit_clean(),
            "bindings": {
                "task_manifest_sha256": _sha256(task_manifest_path),
                "original_labels_manifest_sha256": _sha256(
                    labels_manifest_path
                ),
                "routing_manifest_sha256": _sha256(routing_manifest_path),
                "human_manifest_sha256": _sha256(human_manifest_path),
                "human_state_sha256": state_hashes,
            },
            "counts": {
                "candidates": len(frozen_rows),
                "cross_family_agreements": (
                    source_counts["cross_family_high_confidence_agreement"]
                ),
                "copilot_teacher_decisions": source_counts[
                    "user_authorized_copilot_teacher"
                ],
                "audited_labels": dict(sorted(counts.items())),
            },
            "confusion": _confusion(frozen_rows),
            "decisions_sha256": _sha256(decisions_path),
            "claim_limit": (
                "Fit-label diagnostic audit with a user-authorized Copilot "
                "teacher; not formal-grade evaluation evidence."
            ),
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
        description="Freeze the completed Revision-12 hard-label audit."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--original-labels", type=Path, required=True)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze_audit(
        args.tasks,
        args.original_labels,
        args.routing,
        args.state,
        args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
