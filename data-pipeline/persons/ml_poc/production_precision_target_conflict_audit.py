from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_grouped_data import GROUPED_DATA_STATUS
from production_precision_grouped_verifier import _read, _read_jsonl, _sha256
from production_precision_hard_label_audit import _candidate_id
from production_precision_hard_label_freeze import FROZEN_STATUS
from production_precision_boundary_target_audit import FROZEN_TARGET_STATUS
from production_train import _make_read_only


TASK_STATUS = "ml_production_precision_target_conflict_tasks"
FROZEN_CONFLICT_STATUS = "ml_production_precision_target_conflicts_frozen"
EXPECTED_AUDITED = 303
EXPECTED_CONFLICTS = 4
DECISIONS = {"targets", "none", "uncertain"}


def _example_id(row: dict) -> str:
    return (
        f"juan-{int(row['juan']):03d}-jie-{int(row['jie_index']):04d}"
    )


def _geometry(row: dict) -> tuple[int, int, int]:
    return int(row["para_id"]), int(row["start"]), int(row["end"])


def _full_geometry(row: dict) -> tuple[str, int, int, int]:
    return _example_id(row), *_geometry(row)


def _overlaps(
    left: tuple[str, int, int, int],
    right: tuple[str, int, int, int],
) -> bool:
    return (
        left[0] == right[0]
        and left[1] == right[1]
        and left[2] < right[3]
        and right[2] < left[3]
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _normalize_candidate(row: dict) -> dict:
    return {
        "candidate_id": str(row["candidate_id"]),
        "juan": int(row["juan"]),
        "jie_index": int(row["jie_index"]),
        "para_id": int(row["para_id"]),
        "start": int(row["start"]),
        "end": int(row["end"]),
        "surface": str(row["surface"]),
    }


def _normalize_addition(row: dict) -> dict:
    return {
        "juan": int(row["juan"]),
        "jie_index": int(row["jie_index"]),
        "para_id": int(row["para_id"]),
        "start": int(row["start"]),
        "end": int(row["end"]),
        "surface": str(row["surface"]),
    }


def derive_conflict_components(
    audit_rows: list[dict], target_rows: list[dict]
) -> list[dict]:
    candidates: dict[str, dict] = {}
    for row in audit_rows:
        candidate = _normalize_candidate(row)
        candidate_id = candidate["candidate_id"]
        if candidate_id in candidates:
            raise ValueError("duplicate conflict-audit candidate")
        candidates[candidate_id] = candidate

    targets_by_candidate: dict[str, dict] = {}
    for row in target_rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id in targets_by_candidate:
            raise ValueError("duplicate conflict-audit target decision")
        targets_by_candidate[candidate_id] = row

    additions: dict[tuple[str, int, int, int], dict] = {}
    additions_by_owner: dict[str, set[tuple[str, int, int, int]]] = {
        candidate_id: set() for candidate_id in candidates
    }
    wrong_ids = set()
    for candidate_id, candidate in candidates.items():
        audit = next(
            row for row in audit_rows
            if str(row["candidate_id"]) == candidate_id
        )
        label = str(audit["audited_label"])
        proposed: list[dict]
        if label == "exact_person":
            proposed = [candidate]
        elif label == "wrong_boundary":
            wrong_ids.add(candidate_id)
            target = targets_by_candidate.get(candidate_id)
            if (
                target is None
                or target.get("decision") != "targets"
                or not isinstance(target.get("targets"), list)
                or not target["targets"]
            ):
                raise ValueError("conflict-audit boundary target missing")
            wrong = target.get("wrong_candidate", {})
            if (
                int(wrong.get("para_id", -1)),
                int(wrong.get("start", -1)),
                int(wrong.get("end", -1)),
                str(wrong.get("surface", "")),
            ) != (
                candidate["para_id"],
                candidate["start"],
                candidate["end"],
                candidate["surface"],
            ):
                raise ValueError("conflict-audit target candidate differs")
            proposed = [
                {
                    **candidate,
                    "para_id": int(target_row["para_id"]),
                    "start": int(target_row["start"]),
                    "end": int(target_row["end"]),
                    "surface": str(target_row["surface"]),
                }
                for target_row in target["targets"]
            ]
            seen_targets = set()
            for proposed_row in proposed:
                geometry = _full_geometry(proposed_row)
                if (
                    geometry in seen_targets
                    or not _overlaps(
                        _full_geometry(candidate), geometry
                    )
                ):
                    raise ValueError(
                        "invalid conflict-audit boundary target"
                    )
                seen_targets.add(geometry)
            if any(
                _overlaps(
                    _full_geometry(left_target),
                    _full_geometry(right_target),
                )
                for index, left_target in enumerate(proposed)
                for right_target in proposed[index + 1:]
            ):
                raise ValueError(
                    "overlapping conflict-audit boundary targets"
                )
        elif label == "not_person":
            proposed = []
        else:
            raise ValueError("uncertain conflict-audit source label")

        for proposed_row in proposed:
            normalized = _normalize_addition(proposed_row)
            key = _full_geometry(normalized)
            prior = additions.get(key)
            if prior is None:
                additions[key] = {
                    **normalized,
                    "owners": set(),
                }
            elif prior["surface"] != normalized["surface"]:
                raise ValueError("conflict-audit addition surface differs")
            additions[key]["owners"].add(candidate_id)
            additions_by_owner[candidate_id].add(key)

    if set(targets_by_candidate) != wrong_ids:
        raise ValueError("conflict-audit target coverage differs")

    keys = sorted(additions)
    neighbors = {key: set() for key in keys}
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            if _overlaps(left, right):
                neighbors[left].add(right)
                neighbors[right].add(left)

    graph_components = []
    unseen = set(keys)
    while unseen:
        seed = min(unseen)
        stack = [seed]
        component = set()
        while stack:
            key = stack.pop()
            if key in component:
                continue
            component.add(key)
            unseen.discard(key)
            stack.extend(sorted(neighbors[key] - component, reverse=True))
        graph_components.append(component)

    closed_components: dict[
        tuple[tuple[str, ...], tuple[tuple[str, int, int, int], ...]], dict
    ] = {}
    candidate_geometries = {
        candidate_id: _full_geometry(candidate)
        for candidate_id, candidate in candidates.items()
    }
    for initial in graph_components:
        if not any(neighbors[key] & initial for key in initial):
            continue
        included_additions = set(initial)
        included_candidates: set[str] = set()
        changed = True
        while changed:
            changed = False
            for candidate_id, geometry in candidate_geometries.items():
                if (
                    candidate_id not in included_candidates
                    and any(
                        _overlaps(geometry, addition)
                        for addition in included_additions
                    )
                ):
                    included_candidates.add(candidate_id)
                    changed = True
            owned = set().union(
                *(
                    additions_by_owner[candidate_id]
                    for candidate_id in included_candidates
                ),
                set(),
            )
            if not owned.issubset(included_additions):
                included_additions.update(owned)
                changed = True

        key = (
            tuple(sorted(included_candidates)),
            tuple(sorted(included_additions)),
        )
        closed_components[key] = {
            "candidate_ids": list(key[0]),
            "candidates": [
                candidates[candidate_id]
                for candidate_id in key[0]
            ],
            "addition_membership": [
                {
                    **{
                        name: value
                        for name, value in additions[geometry].items()
                        if name != "owners"
                    },
                    "owner_candidate_ids": sorted(
                        additions[geometry]["owners"]
                    ),
                }
                for geometry in key[1]
            ],
        }
    return [
        closed_components[key] for key in sorted(closed_components)
    ]


def _validate_sources(
    examples: dict[str, dict], components: list[dict]
) -> None:
    for component in components:
        for row in [
            *component["candidates"],
            *component["addition_membership"],
        ]:
            example = examples.get(_example_id(row))
            if example is None:
                raise ValueError("conflict-audit example missing")
            segments = [
                segment for segment in example["segments"]
                if int(segment["para_id"]) == int(row["para_id"])
            ]
            if len(segments) != 1:
                raise ValueError("conflict-audit paragraph missing")
            segment = segments[0]
            paragraph = str(example["text"])[
                int(segment["assembled_start"]):
                int(segment["assembled_end"])
            ]
            start = int(row["start"])
            end = int(row["end"])
            if (
                not 0 <= start < end <= len(paragraph)
                or paragraph[start:end] != str(row["surface"])
            ):
                raise ValueError("conflict-audit source geometry differs")


def _load_validated_sources(
    grouped_root: Path, audit_root: Path, targets_root: Path
) -> tuple[dict, list[dict], list[dict], dict]:
    grouped_manifest_path = grouped_root / "manifest.json"
    audit_manifest_path = audit_root / "manifest.json"
    targets_manifest_path = targets_root / "manifest.json"
    examples_path = grouped_root / "examples.jsonl"
    existence_path = grouped_root / "existence.jsonl"
    audit_path = audit_root / "decisions.jsonl"
    targets_path = targets_root / "targets.jsonl"
    grouped_manifest = _read(grouped_manifest_path)
    audit_manifest = _read(audit_manifest_path)
    targets_manifest = _read(targets_manifest_path)
    grouped_hash = _sha256(grouped_manifest_path)
    audit_hash = _sha256(audit_manifest_path)
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
        or targets_manifest.get("status") != FROZEN_TARGET_STATUS
        or targets_manifest.get("confirmation_read") is not False
        or targets_manifest.get("counts", {}).get("uncertain") != 0
        or targets_manifest.get("targets_sha256") != _sha256(targets_path)
        or targets_manifest.get("bindings", {}).get(
            "grouped_manifest_sha256"
        ) != grouped_hash
        or targets_manifest.get("bindings", {}).get(
            "audit_manifest_sha256"
        ) != audit_hash
        or targets_manifest.get("bindings", {}).get("examples_sha256")
        != _sha256(examples_path)
        or targets_manifest.get("bindings", {}).get("existence_sha256")
        != _sha256(existence_path)
    ):
        raise ValueError("conflict-audit source binding differs")

    audit_rows = _read_jsonl(audit_path)
    target_rows = _read_jsonl(targets_path)
    if (
        len(audit_rows) != EXPECTED_AUDITED
        or len({str(row["candidate_id"]) for row in audit_rows})
        != EXPECTED_AUDITED
        or any(
            _candidate_id(grouped_hash, row) != str(row["candidate_id"])
            for row in audit_rows
        )
    ):
        raise ValueError("conflict-audit candidate inventory differs")
    examples = {
        str(row["id"]): row for row in _read_jsonl(examples_path)
    }
    bindings = {
        "grouped_manifest_sha256": grouped_hash,
        "audit_manifest_sha256": audit_hash,
        "targets_manifest_sha256": _sha256(targets_manifest_path),
        "examples_sha256": _sha256(examples_path),
        "existence_sha256": _sha256(existence_path),
        "audit_decisions_sha256": _sha256(audit_path),
        "boundary_targets_sha256": _sha256(targets_path),
    }
    return examples, audit_rows, target_rows, bindings


def build_tasks(
    grouped_root: Path,
    audit_root: Path,
    targets_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"conflict-audit tasks exist: {output_dir}")
    examples, audit_rows, target_rows, bindings = _load_validated_sources(
        grouped_root, audit_root, targets_root
    )
    components = derive_conflict_components(audit_rows, target_rows)
    if len(components) != EXPECTED_CONFLICTS:
        raise ValueError(
            "derived conflict-audit component count differs: "
            f"{len(components)}"
        )
    _validate_sources(examples, components)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        task_dir = staging / "tasks"
        task_dir.mkdir()
        selected = []
        sealed = []
        for component in components:
            membership_binding = {
                "bindings": bindings,
                "candidate_ids": component["candidate_ids"],
                "candidates": component["candidates"],
                "addition_membership": component["addition_membership"],
            }
            task_id = hashlib.sha256(
                b"revision-13-conflict:" + _canonical_json(membership_binding)
            ).hexdigest()[:24]
            candidates = []
            for candidate in component["candidates"]:
                neutral_id = hashlib.sha256(
                    (
                        f"{task_id}:{candidate['candidate_id']}:"
                        f"{candidate['para_id']}:{candidate['start']}:"
                        f"{candidate['end']}"
                    ).encode("ascii")
                ).hexdigest()[:16]
                candidates.append({
                    "neutral_id": neutral_id,
                    "para_id": int(candidate["para_id"]),
                    "start": int(candidate["start"]),
                    "end": int(candidate["end"]),
                    "surface": str(candidate["surface"]),
                })
            candidates.sort(key=lambda row: row["neutral_id"])
            first = component["candidates"][0]
            example = examples[_example_id(first)]
            paragraphs = {int(row["para_id"]) for row in candidates}
            if len(paragraphs) != 1:
                raise ValueError("conflict-audit component crosses paragraphs")
            task = {
                "schema_version": 1,
                "status": TASK_STATUS,
                "phase": "revision-13-candidate-closed-conflict-audit",
                "task_id": task_id,
                "juan": int(first["juan"]),
                "jie_index": int(first["jie_index"]),
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
                "included_candidates": candidates,
                "request": (
                    "Return every exact individual-person span in the same "
                    "paragraph that overlaps any included candidate."
                ),
                "allowed_decisions": sorted(DECISIONS),
                "component_binding_sha256": hashlib.sha256(
                    _canonical_json(membership_binding)
                ).hexdigest(),
            }
            path = task_dir / f"task_{task_id}.json"
            path.write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            selected.append({
                "task_id": task_id,
                "juan": int(first["juan"]),
                "jie_index": int(first["jie_index"]),
                "task": str(Path("tasks") / path.name),
                "task_sha256": _sha256(path),
                "component_binding_sha256": task[
                    "component_binding_sha256"
                ],
            })
            sealed.append({
                "task_id": task_id,
                "candidate_ids": component["candidate_ids"],
                "candidates": component["candidates"],
                "neutral_ids": {
                    row["neutral_id"]: candidate["candidate_id"]
                    for row, candidate in zip(
                        sorted(candidates, key=lambda row: (
                            row["para_id"], row["start"], row["end"],
                            row["surface"],
                        )),
                        sorted(component["candidates"], key=lambda row: (
                            row["para_id"], row["start"], row["end"],
                            row["surface"],
                        )),
                    )
                },
                "component_binding_sha256": task[
                    "component_binding_sha256"
                ],
            })
        if (
            len({row["task_id"] for row in selected})
            != EXPECTED_CONFLICTS
            or len({row["task"] for row in selected})
            != EXPECTED_CONFLICTS
        ):
            raise ValueError("conflict-audit task identity collision")
        sealed_path = staging / "sealed-component-membership.jsonl"
        sealed_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
                for row in sorted(sealed, key=lambda row: row["task_id"])
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "status": TASK_STATUS,
            "revision": 13,
            "candidate_closed": True,
            "candidate_model_blind": True,
            "baseline_references_hidden": True,
            "upstream_labels_hidden": True,
            "model_scores_hidden": True,
            "prior_targets_hidden": True,
            "neighboring_jies_hidden": True,
            "translations_hidden": True,
            "knowledge_bases_hidden": True,
            "confirmation_read": False,
            "git_commit": _git_commit_clean(),
            "bindings": bindings,
            "counts": {
                "derived_conflict_components": len(components),
                "tasks": len(selected),
                "included_candidates": sum(
                    len(row["candidate_ids"]) for row in sealed
                ),
            },
            "selected": sorted(selected, key=lambda row: row["task_id"]),
            "sealed_component_membership_sha256": _sha256(sealed_path),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return manifest


def _task_paragraph(task: dict, para_id: int) -> str:
    segments = [
        row for row in task["jie"]["segments"]
        if int(row["para_id"]) == para_id
    ]
    if len(segments) != 1:
        raise ValueError("conflict decision paragraph differs")
    segment = segments[0]
    return str(task["jie"]["text"])[
        int(segment["assembled_start"]):int(segment["assembled_end"])
    ]


def freeze_decisions(
    task_root: Path, raw_root: Path, output_dir: Path
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"conflict decision freeze exists: {output_dir}")
    manifest_path = task_root / "manifest.json"
    manifest = _read(manifest_path)
    sealed_path = task_root / "sealed-component-membership.jsonl"
    if (
        manifest.get("status") != TASK_STATUS
        or manifest.get("candidate_closed") is not True
        or manifest.get("candidate_model_blind") is not True
        or manifest.get("baseline_references_hidden") is not True
        or manifest.get("upstream_labels_hidden") is not True
        or manifest.get("model_scores_hidden") is not True
        or manifest.get("prior_targets_hidden") is not True
        or manifest.get("neighboring_jies_hidden") is not True
        or manifest.get("translations_hidden") is not True
        or manifest.get("knowledge_bases_hidden") is not True
        or manifest.get("confirmation_read") is not False
        or int(manifest.get("counts", {}).get("tasks", -1))
        != EXPECTED_CONFLICTS
        or manifest.get("sealed_component_membership_sha256")
        != _sha256(sealed_path)
    ):
        raise ValueError("conflict decision task binding differs")
    expected = {
        str(row["task_id"]): row for row in manifest["selected"]
    }
    sealed = {
        str(row["task_id"]): row for row in _read_jsonl(sealed_path)
    }
    if (
        len(expected) != EXPECTED_CONFLICTS
        or set(sealed) != set(expected)
    ):
        raise ValueError("conflict decision task inventory differs")

    raw_paths = sorted(raw_root.glob("*.json"))
    if len(raw_paths) != EXPECTED_CONFLICTS:
        raise ValueError("conflict decision raw-output count differs")
    frozen = []
    raw_hashes = {}
    seen = set()
    for path in raw_paths:
        payload = _read(path)
        task_id = str(payload.get("task_id", ""))
        selected = expected.get(task_id)
        membership = sealed.get(task_id)
        if selected is None or membership is None or task_id in seen:
            raise ValueError(f"unexpected conflict decision task: {task_id}")
        task_path = task_root / selected["task"]
        task = _read(task_path)
        expected_task_keys = {
            "schema_version",
            "status",
            "phase",
            "task_id",
            "juan",
            "jie_index",
            "review_scope",
            "jie",
            "included_candidates",
            "request",
            "allowed_decisions",
            "component_binding_sha256",
        }
        if (
            _sha256(task_path) != selected["task_sha256"]
            or set(task) != expected_task_keys
            or task.get("schema_version") != 1
            or task.get("status") != TASK_STATUS
            or task.get("phase")
            != "revision-13-candidate-closed-conflict-audit"
            or task.get("task_id") != task_id
            or task.get("review_scope") != "current-numbered-jie-only"
            or task.get("allowed_decisions") != sorted(DECISIONS)
            or task.get("component_binding_sha256")
            != membership.get("component_binding_sha256")
            or payload.get("schema_version") != 1
            or payload.get("phase")
            != "revision-13-candidate-closed-conflict-audit"
            or payload.get("adjudicator") != "copilot_teacher"
            or payload.get("task_sha256") != selected["task_sha256"]
            or payload.get("decision") not in DECISIONS
        ):
            raise ValueError(f"conflict decision provenance differs: {task_id}")
        rationale = payload.get("rationale")
        targets = payload.get("targets")
        if (
            not isinstance(rationale, str)
            or not rationale.strip()
            or not isinstance(targets, list)
            or (payload["decision"] == "targets" and not targets)
            or (payload["decision"] != "targets" and targets)
        ):
            raise ValueError(f"invalid conflict decision: {task_id}")

        included = task["included_candidates"]
        included_paragraphs = {
            int(candidate["para_id"]) for candidate in included
        }
        normalized = []
        geometries = set()
        for target in targets:
            if set(target) != {"para_id", "start", "end", "surface"}:
                raise ValueError("invalid conflict target keys")
            para_id = int(target["para_id"])
            start = int(target["start"])
            end = int(target["end"])
            geometry = (para_id, start, end)
            paragraph = _task_paragraph(task, para_id)
            overlaps_included = any(
                para_id == int(candidate["para_id"])
                and start < int(candidate["end"])
                and int(candidate["start"]) < end
                for candidate in included
            )
            if (
                para_id not in included_paragraphs
                or geometry in geometries
                or not 0 <= start < end <= len(paragraph)
                or paragraph[start:end] != str(target["surface"])
                or not overlaps_included
            ):
                raise ValueError(f"invalid conflict target: {task_id}")
            geometries.add(geometry)
            normalized.append({
                "para_id": para_id,
                "start": start,
                "end": end,
                "surface": str(target["surface"]),
            })
        ordered = sorted(
            normalized,
            key=lambda row: (row["para_id"], row["start"], row["end"]),
        )
        if any(
            left["para_id"] == right["para_id"]
            and left["start"] < right["end"]
            and right["start"] < left["end"]
            for index, left in enumerate(ordered)
            for right in ordered[index + 1:]
        ):
            raise ValueError(f"overlapping conflict targets: {task_id}")
        frozen.append({
            "task_id": task_id,
            "juan": int(selected["juan"]),
            "jie_index": int(selected["jie_index"]),
            "component_candidate_ids": sorted(
                str(value) for value in membership["candidate_ids"]
            ),
            "included_candidates": sorted(
                membership["candidates"],
                key=lambda row: str(row["candidate_id"]),
            ),
            "decision": str(payload["decision"]),
            "targets": ordered,
            "rationale": rationale.strip(),
            "decision_source": "user_authorized_copilot_teacher",
        })
        seen.add(task_id)
        raw_hashes[path.name] = _sha256(path)
    if seen != set(expected):
        raise ValueError("conflict decision coverage differs")

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
                for row in sorted(frozen, key=lambda row: row["task_id"])
            ),
            encoding="utf-8",
        )
        counts = Counter(row["decision"] for row in frozen)
        result = {
            "schema_version": 1,
            "status": FROZEN_CONFLICT_STATUS,
            "revision": 13,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "git_commit": _git_commit_clean(),
            "task_manifest_sha256": _sha256(manifest_path),
            "bindings": dict(manifest["bindings"]),
            "counts": {
                "tasks": len(frozen),
                "targets": counts["targets"],
                "none": counts["none"],
                "uncertain": counts["uncertain"],
                "exact_targets": sum(
                    len(row["targets"]) for row in frozen
                ),
            },
            "raw_outputs": raw_hashes,
            "decisions_sha256": _sha256(decisions_path),
            "claim_limit": (
                "Fit-only conflict re-adjudication by a user-authorized "
                "Copilot teacher; not formal-grade evidence and not a "
                "production-quality claim."
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
        description="Build or freeze Revision-13 target-conflict audit."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--grouped-data", type=Path, required=True)
    build.add_argument("--audit", type=Path, required=True)
    build.add_argument("--targets", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--tasks", type=Path, required=True)
    freeze.add_argument("--raw", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_tasks(
            args.grouped_data, args.audit, args.targets, args.output
        )
    else:
        result = freeze_decisions(args.tasks, args.raw, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
