from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_revision17_freeze import FROZEN_STATUS as REVIEW_FROZEN_STATUS
from production_precision_revision17_plan import PLAN_STATUS, _read, _sha256
from production_precision_revision17_target_freeze import (
    BLOCKED_STATUS as TARGET_BLOCKED_STATUS,
)
from production_precision_revision17_tasks import TASKS_STATUS as REVIEW_TASKS_STATUS
from production_precision_revision18_freeze import (
    FROZEN_STATUS as ADJUDICATION_FROZEN_STATUS,
)
from production_train import _make_read_only


REVISION = 18
READY_STATUS = "ml_production_precision_revision18_overlay_ready"
BLOCKED_STATUS = "ml_production_precision_revision18_overlay_blocked_conflicts"
EXPECTED_DECISIONS = 1093
EXPECTED_ADJUDICATIONS = 13


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


def _geometry(row: dict) -> tuple[str, int, int, int]:
    return (
        str(row["id"]),
        int(row["para_id"]),
        int(row["start"]),
        int(row["end"]),
    )


def _overlaps(left: tuple, right: tuple) -> bool:
    return (
        left[0] == right[0]
        and left[1] == right[1]
        and left[2] < right[3]
        and right[2] < left[3]
    )


def construct_overlay(
    selection: list[dict],
    first_decisions: list[dict],
    target_decisions: list[dict],
    adjudications: list[dict],
) -> dict:
    selected_by_candidate = {
        str(row["candidate_id"]): row for row in selection
    }
    first_by_candidate = {
        str(row["candidate_id"]): row for row in first_decisions
    }
    targets_by_candidate = {
        str(row["candidate_id"]): row for row in target_decisions
    }
    adjudication_by_candidate = {
        str(row["candidate_id"]): row for row in adjudications
    }
    if (
        len(selected_by_candidate) != EXPECTED_DECISIONS
        or len(first_by_candidate) != EXPECTED_DECISIONS
        or set(selected_by_candidate) != set(first_by_candidate)
        or len(adjudication_by_candidate) != EXPECTED_ADJUDICATIONS
        or not set(adjudication_by_candidate).issubset(first_by_candidate)
    ):
        raise ValueError("Revision-18 overlay decision inventory differs")

    final_decisions = []
    exact_owners: dict[tuple, list[str]] = {}
    exact_rows: dict[tuple, dict] = {}
    semantic_rows = []
    rank_pairs = []
    transitions = []
    for candidate_id in sorted(first_by_candidate):
        selected = selected_by_candidate[candidate_id]
        first = first_by_candidate[candidate_id]
        adjudication = adjudication_by_candidate.get(candidate_id)
        if adjudication is not None:
            label = str(adjudication["label"])
            targets = list(adjudication["targets"])
            source = "revision18_adjudication"
            transitions.append({
                "candidate_id": candidate_id,
                "from_label": str(first["label"]),
                "to_label": label,
            })
            if label == "wrong_boundary" and not targets:
                raise ValueError(
                    "Revision-18 overlay has contradictory adjudication target"
                )
        else:
            label = str(first["label"])
            source = "revision17_first_judgment"
            if label == "wrong_boundary":
                target = targets_by_candidate.get(candidate_id)
                if (
                    target is None
                    or target.get("uncertain") is not False
                    or target.get("contradiction") is not False
                    or not target.get("targets")
                ):
                    raise ValueError(
                        "Revision-18 overlay has unresolved Revision-17 target"
                    )
                targets = list(target["targets"])
            else:
                targets = []
        candidate = {
            key: selected[key]
            for key in (
                "id",
                "juan",
                "jie_index",
                "para_id",
                "start",
                "end",
                "surface",
            )
        }
        final_decisions.append({
            "candidate_id": candidate_id,
            "label": label,
            "decision_source": source,
            "candidate": candidate,
            "targets": targets,
        })
        if label == "exact_person":
            additions = [candidate]
        elif label == "wrong_boundary":
            additions = [
                {
                    "id": candidate["id"],
                    "juan": candidate["juan"],
                    "jie_index": candidate["jie_index"],
                    "para_id": int(target["para_id"]),
                    "start": int(target["start"]),
                    "end": int(target["end"]),
                    "surface": str(target["surface"]),
                }
                for target in targets
            ]
        elif label == "not_person":
            additions = []
            semantic_rows.append({
                **candidate,
                "candidate_id": candidate_id,
                "class": "semantic_negative",
                "existence_label": 0,
            })
        elif label == "uncertain":
            raise ValueError("Revision-18 overlay contains uncertainty")
        else:
            raise ValueError("Revision-18 overlay label differs")

        for addition in additions:
            key = _geometry(addition)
            exact_rows.setdefault(key, {
                **addition,
                "class": "mined_exact_reference",
                "existence_label": 1,
            })
            exact_owners.setdefault(key, []).append(candidate_id)
            if label == "wrong_boundary":
                rank_pairs.append({
                    "positive": addition,
                    "negative": candidate,
                    "candidate_id": candidate_id,
                })

    exact_keys = sorted(exact_rows)
    conflicts = []
    for index, left in enumerate(exact_keys):
        for right in exact_keys[index + 1:]:
            if _overlaps(left, right):
                conflicts.append({
                    "type": "overlapping_exact_additions",
                    "left": list(left),
                    "right": list(right),
                    "left_owners": exact_owners[left],
                    "right_owners": exact_owners[right],
                })
    for semantic in semantic_rows:
        semantic_key = _geometry(semantic)
        for exact_key in exact_keys:
            if _overlaps(semantic_key, exact_key):
                conflicts.append({
                    "type": "semantic_overlaps_exact_addition",
                    "semantic": list(semantic_key),
                    "semantic_candidate_id": semantic["candidate_id"],
                    "exact": list(exact_key),
                    "exact_owners": exact_owners[exact_key],
                })

    return {
        "final_decisions": final_decisions,
        "exact_rows": [exact_rows[key] for key in exact_keys],
        "exact_owners": [
            {"geometry": list(key), "candidate_ids": sorted(owners)}
            for key, owners in sorted(exact_owners.items())
        ],
        "semantic_rows": sorted(semantic_rows, key=_geometry),
        "rank_pairs": sorted(
            rank_pairs,
            key=lambda row: (
                _geometry(row["positive"]),
                _geometry(row["negative"]),
            ),
        ),
        "transitions": sorted(
            transitions, key=lambda row: row["candidate_id"]
        ),
        "conflicts": conflicts,
    }


def freeze_overlay(
    plan_root: Path,
    review_task_root: Path,
    review_root: Path,
    target_root: Path,
    adjudication_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Revision-18 overlay exists: {output_dir}")
    plan_manifest_path = plan_root / "manifest.json"
    examples_path = plan_root / "mining.jsonl"
    review_task_manifest_path = review_task_root / "manifest.json"
    selection_path = review_task_root / "sealed-selection" / "selection.jsonl"
    review_manifest_path = review_root / "manifest.json"
    decisions_path = review_root / "decisions.jsonl"
    target_manifest_path = target_root / "manifest.json"
    targets_path = target_root / "targets.jsonl"
    adjudication_manifest_path = adjudication_root / "manifest.json"
    adjudications_path = adjudication_root / "decisions.jsonl"
    plan_manifest = _read(plan_manifest_path)
    review_task_manifest = _read(review_task_manifest_path)
    review_manifest = _read(review_manifest_path)
    target_manifest = _read(target_manifest_path)
    adjudication_manifest = _read(adjudication_manifest_path)
    if (
        plan_manifest.get("status") != PLAN_STATUS
        or plan_manifest.get("outputs", {}).get("mining_sha256")
        != _sha256(examples_path)
        or review_task_manifest.get("status") != REVIEW_TASKS_STATUS
        or review_task_manifest.get("outputs", {}).get("selection_sha256")
        != _sha256(selection_path)
        or review_manifest.get("status") != REVIEW_FROZEN_STATUS
        or review_manifest.get("selection_metadata_joined") is not False
        or review_manifest.get("outputs", {}).get("decisions_sha256")
        != _sha256(decisions_path)
        or target_manifest.get("status") != TARGET_BLOCKED_STATUS
        or target_manifest.get("outputs", {}).get("targets_sha256")
        != _sha256(targets_path)
        or adjudication_manifest.get("status") != ADJUDICATION_FROZEN_STATUS
        or adjudication_manifest.get("outputs", {}).get("decisions_sha256")
        != _sha256(adjudications_path)
    ):
        raise ValueError("Revision-18 overlay source binding differs")

    overlay = construct_overlay(
        _read_jsonl(selection_path),
        _read_jsonl(decisions_path),
        _read_jsonl(targets_path),
        _read_jsonl(adjudications_path),
    )
    example_rows = _read_jsonl(examples_path)
    examples_by_id = {str(row["id"]): row for row in example_rows}
    used_ids = {
        str(row["candidate"]["id"]) for row in overlay["final_decisions"]
    }
    if len(examples_by_id) != len(example_rows) or not used_ids.issubset(examples_by_id):
        raise ValueError("Revision-18 overlay example inventory differs")
    used_examples = [
        examples_by_id[example_id] for example_id in sorted(used_ids)
    ]
    git_commit = _git_commit_clean()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        outputs = {
            "decisions": ("decisions.jsonl", overlay["final_decisions"]),
            "examples": ("examples.jsonl", used_examples),
            "exact_additions": ("exact-additions.jsonl", overlay["exact_rows"]),
            "exact_owners": ("exact-owners.jsonl", overlay["exact_owners"]),
            "semantic_negatives": (
                "semantic-negatives.jsonl", overlay["semantic_rows"]
            ),
            "rank_pairs": ("rank-pairs.jsonl", overlay["rank_pairs"]),
            "transitions": ("transitions.jsonl", overlay["transitions"]),
            "conflicts": ("conflicts.jsonl", overlay["conflicts"]),
        }
        output_hashes = {}
        for name, (filename, rows) in outputs.items():
            path = staging / filename
            _write_jsonl(path, rows)
            output_hashes[f"{name}_sha256"] = _sha256(path)
        status = BLOCKED_STATUS if overlay["conflicts"] else READY_STATUS
        manifest = {
            "schema_version": 1,
            "status": status,
            "revision": REVISION,
            "fit_only": True,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "formal_reserve_text_read": False,
            "git_commit": git_commit,
            "bindings": {
                "plan_manifest_sha256": _sha256(plan_manifest_path),
                "mining_examples_sha256": _sha256(examples_path),
                "review_task_manifest_sha256": _sha256(
                    review_task_manifest_path
                ),
                "selection_sha256": _sha256(selection_path),
                "review_manifest_sha256": _sha256(review_manifest_path),
                "review_decisions_sha256": _sha256(decisions_path),
                "target_manifest_sha256": _sha256(target_manifest_path),
                "targets_sha256": _sha256(targets_path),
                "adjudication_manifest_sha256": _sha256(
                    adjudication_manifest_path
                ),
                "adjudications_sha256": _sha256(adjudications_path),
            },
            "counts": {
                "decisions": len(overlay["final_decisions"]),
                "examples": len(used_examples),
                "exact_person_decisions": sum(
                    row["label"] == "exact_person"
                    for row in overlay["final_decisions"]
                ),
                "wrong_boundary_decisions": sum(
                    row["label"] == "wrong_boundary"
                    for row in overlay["final_decisions"]
                ),
                "not_person_decisions": len(overlay["semantic_rows"]),
                "unique_exact_additions": len(overlay["exact_rows"]),
                "rank_pairs": len(overlay["rank_pairs"]),
                "transitions": {
                    f"{source}->{target}": sum(
                        row["from_label"] == source
                        and row["to_label"] == target
                        for row in overlay["transitions"]
                    )
                    for source in (
                        "exact_person",
                        "wrong_boundary",
                        "not_person",
                        "uncertain",
                    )
                    for target in (
                        "exact_person",
                        "wrong_boundary",
                        "not_person",
                        "uncertain",
                    )
                },
                "conflicts": len(overlay["conflicts"]),
            },
            "outputs": output_hashes,
            "next_action": (
                "build_revision18_oof_augmentation"
                if status == READY_STATUS
                else "blind_conflict_adjudication"
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
        description="Construct the Revision-18 reviewed training overlay."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--review-tasks", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_overlay(
        args.plan,
        args.review_tasks,
        args.review,
        args.targets,
        args.adjudication,
        args.output,
    )
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["status"] == READY_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
