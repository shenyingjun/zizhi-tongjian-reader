from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_revision17_plan import _read, _sha256
from production_precision_revision18_overlay import (
    BLOCKED_STATUS as REVISION18_BLOCKED_STATUS,
    _geometry,
    _overlaps,
)
from production_precision_revision19_conflicts import TASKS_STATUS
from production_precision_revision19_freeze import (
    FROZEN_STATUS as CONFLICT_FROZEN_STATUS,
)
from production_train import _make_read_only


REVISION = 19
READY_STATUS = "ml_production_precision_revision19_overlay_ready"
BLOCKED_STATUS = "ml_production_precision_revision19_overlay_blocked"
EXPECTED_DECISIONS = 1093
EXPECTED_COMPONENTS = 36


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


def reconcile_overlay(
    prior_decisions: list[dict],
    prior_exact: list[dict],
    prior_owners: list[dict],
    examples: list[dict],
    components: list[dict],
    adjudications: list[dict],
) -> dict:
    decision_by_id = {
        str(row["candidate_id"]): row for row in prior_decisions
    }
    example_by_id = {str(row["id"]): row for row in examples}
    component_by_task = {
        str(row["conflict_task_id"]): row for row in components
    }
    adjudication_by_task = {
        str(row["conflict_task_id"]): row for row in adjudications
    }
    if (
        len(decision_by_id) != EXPECTED_DECISIONS
        or len(example_by_id) != len(examples)
        or len(component_by_task) != EXPECTED_COMPONENTS
        or len(adjudication_by_task) != EXPECTED_COMPONENTS
        or set(component_by_task) != set(adjudication_by_task)
    ):
        raise ValueError("Revision-19 reconciliation inventory differs")

    candidate_to_task = {}
    component_candidate_ids = set()
    canonical_by_task = {}
    for task_id in sorted(component_by_task):
        component = component_by_task[task_id]
        adjudication = adjudication_by_task[task_id]
        if adjudication.get("uncertain"):
            raise ValueError("Revision-19 reconciliation contains uncertainty")
        source_id = str(component["geometries"][0][0])
        example = example_by_id.get(source_id)
        if example is None:
            raise ValueError("Revision-19 reconciliation source differs")
        canonical = []
        for person in adjudication["exact_people"]:
            canonical.append({
                "id": source_id,
                "juan": int(example["juan"]),
                "jie_index": int(example["jie_index"]),
                "para_id": int(person["para_id"]),
                "start": int(person["start"]),
                "end": int(person["end"]),
                "surface": str(person["surface"]),
            })
        canonical_by_task[task_id] = canonical
        for candidate_id in component["candidate_ids"]:
            candidate_id = str(candidate_id)
            if (
                candidate_id not in decision_by_id
                or candidate_id in candidate_to_task
            ):
                raise ValueError("Revision-19 component candidate differs")
            candidate_to_task[candidate_id] = task_id
            component_candidate_ids.add(candidate_id)

    final_decisions = []
    transitions = []
    for candidate_id in sorted(decision_by_id):
        prior = decision_by_id[candidate_id]
        task_id = candidate_to_task.get(candidate_id)
        if task_id is None:
            final_decisions.append(prior)
            continue
        candidate = prior["candidate"]
        candidate_geometry = _geometry(candidate)
        overlaps = [
            row for row in canonical_by_task[task_id]
            if _overlaps(candidate_geometry, _geometry(row))
        ]
        exact = next(
            (
                row for row in overlaps
                if _geometry(row) == candidate_geometry
            ),
            None,
        )
        if exact is not None:
            label = "exact_person"
            targets = []
        elif overlaps:
            label = "wrong_boundary"
            targets = [{
                "para_id": row["para_id"],
                "start": row["start"],
                "end": row["end"],
                "surface": row["surface"],
                "reported_surface": row["surface"],
                "surface_corrected": False,
            } for row in overlaps]
        else:
            label = "not_person"
            targets = []
        final_decisions.append({
            "candidate_id": candidate_id,
            "label": label,
            "decision_source": "revision19_conflict_adjudication",
            "candidate": candidate,
            "targets": targets,
        })
        transitions.append({
            "candidate_id": candidate_id,
            "conflict_task_id": task_id,
            "from_label": prior["label"],
            "to_label": label,
        })

    prior_owner_by_geometry = {
        tuple(row["geometry"]): [str(value) for value in row["candidate_ids"]]
        for row in prior_owners
    }
    prior_exact_by_geometry = {
        _geometry(row): row for row in prior_exact
    }
    if (
        len(prior_owner_by_geometry) != len(prior_owners)
        or len(prior_exact_by_geometry) != len(prior_exact)
        or set(prior_owner_by_geometry) != set(prior_exact_by_geometry)
    ):
        raise ValueError("Revision-19 prior exact inventory differs")

    exact_by_geometry = {}
    owners_by_geometry = {}
    for geometry, row in prior_exact_by_geometry.items():
        owners = prior_owner_by_geometry[geometry]
        if not component_candidate_ids.intersection(owners):
            exact_by_geometry[geometry] = row
            owners_by_geometry[geometry] = owners
    for task_id, canonical in canonical_by_task.items():
        component_ids = set(component_by_task[task_id]["candidate_ids"])
        for row in canonical:
            geometry = _geometry(row)
            owners = {
                candidate_id
                for candidate_id in component_ids
                if _overlaps(
                    _geometry(decision_by_id[candidate_id]["candidate"]),
                    geometry,
                )
            }
            owners.update(
                candidate_id
                for prior_geometry, prior_candidate_ids
                in prior_owner_by_geometry.items()
                if _overlaps(prior_geometry, geometry)
                for candidate_id in prior_candidate_ids
                if candidate_id in component_ids
            )
            if not owners:
                raise ValueError("Revision-19 canonical exact has no owner")
            exact_by_geometry.setdefault(geometry, {
                **row,
                "class": "mined_exact_reference",
                "existence_label": 1,
            })
            owners_by_geometry.setdefault(geometry, [])
            owners_by_geometry[geometry] = sorted(set(
                owners_by_geometry[geometry] + list(owners)
            ))

    semantic_rows = []
    rank_pairs = []
    for row in final_decisions:
        candidate = row["candidate"]
        if row["label"] == "not_person":
            semantic_rows.append({
                **candidate,
                "candidate_id": row["candidate_id"],
                "class": "semantic_negative",
                "existence_label": 0,
            })
        elif row["label"] == "wrong_boundary":
            for target in row["targets"]:
                rank_pairs.append({
                    "positive": {
                        "id": candidate["id"],
                        "juan": candidate["juan"],
                        "jie_index": candidate["jie_index"],
                        "para_id": target["para_id"],
                        "start": target["start"],
                        "end": target["end"],
                        "surface": target["surface"],
                    },
                    "negative": candidate,
                    "candidate_id": row["candidate_id"],
                })

    exact_keys = sorted(exact_by_geometry)
    conflicts = []
    for index, left in enumerate(exact_keys):
        for right in exact_keys[index + 1:]:
            if _overlaps(left, right):
                conflicts.append({
                    "type": "overlapping_exact_additions",
                    "left": list(left),
                    "right": list(right),
                    "left_owners": owners_by_geometry[left],
                    "right_owners": owners_by_geometry[right],
                })
    for semantic in semantic_rows:
        semantic_geometry = _geometry(semantic)
        for exact_geometry in exact_keys:
            if _overlaps(semantic_geometry, exact_geometry):
                conflicts.append({
                    "type": "semantic_overlaps_exact_addition",
                    "semantic": list(semantic_geometry),
                    "semantic_candidate_id": semantic["candidate_id"],
                    "exact": list(exact_geometry),
                    "exact_owners": owners_by_geometry[exact_geometry],
                })

    prior_keys = set(prior_exact_by_geometry)
    final_keys = set(exact_by_geometry)
    additions = sorted(final_keys - prior_keys)
    removals = sorted(prior_keys - final_keys)
    replacements = [
        {"removed": list(removed), "added": list(added)}
        for removed in removals
        for added in additions
        if _overlaps(removed, added)
    ]
    return {
        "final_decisions": final_decisions,
        "exact_rows": [exact_by_geometry[key] for key in exact_keys],
        "exact_owners": [{
            "geometry": list(key),
            "candidate_ids": owners_by_geometry[key],
        } for key in exact_keys],
        "semantic_rows": sorted(semantic_rows, key=_geometry),
        "rank_pairs": sorted(
            rank_pairs,
            key=lambda row: (
                _geometry(row["positive"]),
                _geometry(row["negative"]),
            ),
        ),
        "transitions": transitions,
        "conflicts": conflicts,
        "raw_additions": [exact_by_geometry[key] for key in additions],
        "raw_removals": [prior_exact_by_geometry[key] for key in removals],
        "geometry_replacements": replacements,
    }


def freeze_overlay(
    prior_root: Path,
    task_root: Path,
    frozen_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Revision-19 overlay exists: {output_dir}")
    prior_manifest_path = prior_root / "manifest.json"
    prior_decisions_path = prior_root / "decisions.jsonl"
    prior_exact_path = prior_root / "exact-additions.jsonl"
    prior_owners_path = prior_root / "exact-owners.jsonl"
    examples_path = prior_root / "examples.jsonl"
    task_manifest_path = task_root / "manifest.json"
    components_path = task_root / "sealed-source" / "components.jsonl"
    frozen_manifest_path = frozen_root / "manifest.json"
    adjudications_path = frozen_root / "decisions.jsonl"
    prior_manifest = _read(prior_manifest_path)
    task_manifest = _read(task_manifest_path)
    frozen_manifest = _read(frozen_manifest_path)
    if (
        prior_manifest.get("status") != REVISION18_BLOCKED_STATUS
        or prior_manifest.get("outputs", {}).get("decisions_sha256")
        != _sha256(prior_decisions_path)
        or prior_manifest.get("outputs", {}).get("exact_additions_sha256")
        != _sha256(prior_exact_path)
        or prior_manifest.get("outputs", {}).get("exact_owners_sha256")
        != _sha256(prior_owners_path)
        or prior_manifest.get("outputs", {}).get("examples_sha256")
        != _sha256(examples_path)
        or task_manifest.get("status") != TASKS_STATUS
        or task_manifest.get("bindings", {}).get("overlay_manifest_sha256")
        != _sha256(prior_manifest_path)
        or task_manifest.get("outputs", {}).get("sealed_components_sha256")
        != _sha256(components_path)
        or frozen_manifest.get("status") != CONFLICT_FROZEN_STATUS
        or frozen_manifest.get("bindings", {}).get("task_manifest_sha256")
        != _sha256(task_manifest_path)
        or frozen_manifest.get("outputs", {}).get("decisions_sha256")
        != _sha256(adjudications_path)
    ):
        raise ValueError("Revision-19 overlay source binding differs")

    overlay = reconcile_overlay(
        _read_jsonl(prior_decisions_path),
        _read_jsonl(prior_exact_path),
        _read_jsonl(prior_owners_path),
        _read_jsonl(examples_path),
        _read_jsonl(components_path),
        _read_jsonl(adjudications_path),
    )
    git_commit = _git_commit_clean()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        outputs = {
            "decisions": ("decisions.jsonl", overlay["final_decisions"]),
            "examples": ("examples.jsonl", _read_jsonl(examples_path)),
            "exact_additions": ("exact-additions.jsonl", overlay["exact_rows"]),
            "exact_owners": ("exact-owners.jsonl", overlay["exact_owners"]),
            "semantic_negatives": (
                "semantic-negatives.jsonl", overlay["semantic_rows"]
            ),
            "rank_pairs": ("rank-pairs.jsonl", overlay["rank_pairs"]),
            "transitions": ("transitions.jsonl", overlay["transitions"]),
            "conflicts": ("conflicts.jsonl", overlay["conflicts"]),
            "raw_additions": ("raw-additions.jsonl", overlay["raw_additions"]),
            "raw_removals": ("raw-removals.jsonl", overlay["raw_removals"]),
            "geometry_replacements": (
                "geometry-replacements.jsonl",
                overlay["geometry_replacements"],
            ),
        }
        output_hashes = {}
        for name, (filename, rows) in outputs.items():
            path = staging / filename
            _write_jsonl(path, rows)
            output_hashes[f"{name}_sha256"] = _sha256(path)
        status = BLOCKED_STATUS if overlay["conflicts"] else READY_STATUS
        labels = ("exact_person", "wrong_boundary", "not_person")
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
                "prior_overlay_manifest_sha256": _sha256(prior_manifest_path),
                "task_manifest_sha256": _sha256(task_manifest_path),
                "sealed_components_sha256": _sha256(components_path),
                "conflict_manifest_sha256": _sha256(frozen_manifest_path),
                "conflict_decisions_sha256": _sha256(adjudications_path),
            },
            "counts": {
                "decisions": len(overlay["final_decisions"]),
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
                    for source in labels
                    for target in labels
                },
                "raw_additions": len(overlay["raw_additions"]),
                "raw_removals": len(overlay["raw_removals"]),
                "net_growth": (
                    len(overlay["raw_additions"])
                    - len(overlay["raw_removals"])
                ),
                "geometry_replacements": len(
                    overlay["geometry_replacements"]
                ),
                "conflicts": len(overlay["conflicts"]),
            },
            "outputs": output_hashes,
            "next_action": (
                "build_revision19_oof_augmentation"
                if status == READY_STATUS
                else "stop_for_blind_conflict_revision"
            ),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile the Revision-19 reviewed training overlay."
    )
    parser.add_argument("--prior-overlay", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--conflicts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_overlay(
        args.prior_overlay,
        args.tasks,
        args.conflicts,
        args.output,
    )
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["status"] == READY_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
