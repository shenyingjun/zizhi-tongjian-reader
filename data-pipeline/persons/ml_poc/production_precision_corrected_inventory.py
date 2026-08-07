from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_boundary_target_audit import FROZEN_TARGET_STATUS
from production_precision_grouped_data import GROUPED_DATA_STATUS
from production_precision_grouped_verifier import _read, _read_jsonl, _sha256
from production_precision_hard_label_freeze import FROZEN_STATUS
from production_precision_negative_audit_freeze import (
    EXPECTED_SAFE,
    FROZEN_STATUS as SAFE_STATUS,
)
from production_precision_target_conflict_audit import (
    EXPECTED_CONFLICTS,
    FROZEN_CONFLICT_STATUS,
    derive_conflict_components,
)
from production_train import _make_read_only


CORRECTED_STATUS = "ml_production_precision_corrected_fit_inventory"
EXPECTED_AUDITED = 303
EXPECTED_TARGETS = 171
EXPECTED_REAL_ROWS = 2696


def _geometry(row: dict) -> tuple[int, int, int]:
    return int(row["para_id"]), int(row["start"]), int(row["end"])


def _full_geometry(row: dict) -> tuple[str, int, int, int]:
    return str(row["id"]), *_geometry(row)


def _overlaps(left: tuple[int, int, int], right: tuple[int, int, int]) -> bool:
    return (
        left[0] == right[0]
        and left[1] < right[2]
        and right[1] < left[2]
    )


def _source_paragraph(example: dict, para_id: int) -> str:
    matches = [
        row for row in example["segments"] if int(row["para_id"]) == para_id
    ]
    if len(matches) != 1:
        raise ValueError("corrected inventory paragraph differs")
    segment = matches[0]
    return str(example["text"])[
        int(segment["assembled_start"]):int(segment["assembled_end"])
    ]


def _validate_source(examples: dict[str, dict], row: dict) -> None:
    example = examples.get(str(row["id"]))
    if example is None:
        raise ValueError("corrected inventory example missing")
    paragraph = _source_paragraph(example, int(row["para_id"]))
    start = int(row["start"])
    end = int(row["end"])
    if (
        not 0 <= start < end <= len(paragraph)
        or paragraph[start:end] != str(row["surface"])
    ):
        raise ValueError("corrected inventory source geometry differs")


def _reference(row: dict, source: str) -> dict:
    return {
        "id": str(row["id"]),
        "juan": int(row["juan"]),
        "jie_index": int(row["jie_index"]),
        "para_id": int(row["para_id"]),
        "start": int(row["start"]),
        "end": int(row["end"]),
        "surface": str(row["surface"]),
        "source": source,
    }


def _candidate(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "juan": int(row["juan"]),
        "jie_index": int(row["jie_index"]),
        "fold": int(row["fold"]),
        "para_id": int(row["para_id"]),
        "start": int(row["start"]),
        "end": int(row["end"]),
        "surface": str(row["surface"]),
    }


def _build_corrected_inventory(
    examples: list[dict],
    existence_rows: list[dict],
    rank_pairs: list[dict],
    audit_rows: list[dict],
    target_rows: list[dict],
    conflict_rows: list[dict],
    easy_rows: list[dict],
) -> dict:
    examples_by_id = {str(row["id"]): row for row in examples}
    if len(examples_by_id) != len(examples):
        raise ValueError("duplicate corrected inventory example")

    real_by_geometry = {_full_geometry(row): row for row in existence_rows}
    if len(real_by_geometry) != len(existence_rows):
        raise ValueError("duplicate corrected real-row geometry")

    baseline_references: dict[tuple[str, int, int, int], dict] = {}
    for row in existence_rows:
        for reference in row["overlapping_references"]:
            normalized = {
                "id": str(row["id"]),
                "juan": int(row["juan"]),
                "jie_index": int(row["jie_index"]),
                **reference,
            }
            _validate_source(examples_by_id, normalized)
            key = _full_geometry(normalized)
            prior = baseline_references.get(key)
            if prior is not None and prior["surface"] != normalized["surface"]:
                raise ValueError("baseline reference surface differs")
            baseline_references[key] = _reference(normalized, "frozen_fit")

    audit_by_candidate = {}
    for row in audit_rows:
        key = (
            f"juan-{int(row['juan']):03d}-jie-{int(row['jie_index']):04d}",
            int(row["para_id"]),
            int(row["start"]),
            int(row["end"]),
        )
        real = real_by_geometry.get(key)
        if real is None or str(real["surface"]) != str(row["surface"]):
            raise ValueError("audited candidate is not a frozen real row")
        candidate_id = str(row["candidate_id"])
        if candidate_id in audit_by_candidate:
            raise ValueError("duplicate audited candidate")
        audit_by_candidate[candidate_id] = row

    targets_by_candidate = {}
    for row in target_rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id in targets_by_candidate:
            raise ValueError("duplicate corrected target decision")
        if row.get("decision") != "targets" or not row.get("targets"):
            raise ValueError("uncertain or empty corrected target decision")
        targets_by_candidate[candidate_id] = row
    wrong_ids = {
        candidate_id
        for candidate_id, row in audit_by_candidate.items()
        if row["audited_label"] == "wrong_boundary"
    }
    if set(targets_by_candidate) != wrong_ids:
        raise ValueError("corrected target coverage differs")

    expected_components = derive_conflict_components(audit_rows, target_rows)
    expected_memberships = {
        frozenset(str(value) for value in row["candidate_ids"])
        for row in expected_components
    }
    if len(expected_memberships) != len(expected_components):
        raise ValueError("duplicate corrected conflict component")
    actual_memberships = []
    conflict_targets_by_candidate: dict[str, list[dict]] = {}
    conflict_component_by_candidate = {}
    for conflict in conflict_rows:
        if conflict.get("decision") == "uncertain":
            raise ValueError("uncertain corrected conflict decision")
        if conflict.get("decision") not in {"targets", "none"}:
            raise ValueError("invalid corrected conflict decision")
        targets = conflict.get("targets")
        if (
            not isinstance(targets, list)
            or (conflict["decision"] == "targets" and not targets)
            or (conflict["decision"] == "none" and targets)
        ):
            raise ValueError("invalid corrected conflict targets")
        candidate_ids = [
            str(value) for value in conflict.get(
                "component_candidate_ids", []
            )
        ]
        membership = frozenset(candidate_ids)
        if (
            not membership
            or len(membership) != len(candidate_ids)
            or any(
                candidate_id in conflict_component_by_candidate
                for candidate_id in membership
            )
        ):
            raise ValueError("duplicate corrected conflict membership")
        included = conflict.get("included_candidates")
        if not isinstance(included, list):
            raise ValueError("missing corrected conflict candidates")
        included_by_id = {
            str(row["candidate_id"]): row for row in included
        }
        if len(included_by_id) != len(included) or set(included_by_id) != set(
            membership
        ):
            raise ValueError("corrected conflict candidate ownership differs")
        for candidate_id in membership:
            audit = audit_by_candidate.get(candidate_id)
            included_candidate = included_by_id[candidate_id]
            if audit is None or (
                int(included_candidate["juan"]),
                int(included_candidate["jie_index"]),
                int(included_candidate["para_id"]),
                int(included_candidate["start"]),
                int(included_candidate["end"]),
                str(included_candidate["surface"]),
            ) != (
                int(audit["juan"]),
                int(audit["jie_index"]),
                int(audit["para_id"]),
                int(audit["start"]),
                int(audit["end"]),
                str(audit["surface"]),
            ):
                raise ValueError("corrected conflict candidate differs")
            if (
                int(audit["juan"]) != int(conflict["juan"])
                or int(audit["jie_index"])
                != int(conflict["jie_index"])
            ):
                raise ValueError("corrected conflict component scope differs")

        normalized_targets = []
        target_geometries = set()
        for target in targets:
            normalized = {
                "id": (
                    f"juan-{int(conflict['juan']):03d}-jie-"
                    f"{int(conflict['jie_index']):04d}"
                ),
                "juan": int(conflict["juan"]),
                "jie_index": int(conflict["jie_index"]),
                "para_id": int(target["para_id"]),
                "start": int(target["start"]),
                "end": int(target["end"]),
                "surface": str(target["surface"]),
            }
            _validate_source(examples_by_id, normalized)
            key = _full_geometry(normalized)
            if key in target_geometries:
                raise ValueError("duplicate corrected conflict target")
            target_geometries.add(key)
            normalized_targets.append(
                _reference(normalized, "revision_13_conflict_target")
            )
        for index, left in enumerate(normalized_targets):
            if any(
                _overlaps(_geometry(left), _geometry(right))
                for right in normalized_targets[index + 1:]
            ):
                raise ValueError("overlapping corrected conflict targets")
            owners = {
                candidate_id
                for candidate_id, audit in audit_by_candidate.items()
                if (
                    int(audit["juan"]) == int(conflict["juan"])
                    and int(audit["jie_index"])
                    == int(conflict["jie_index"])
                    and _overlaps(_geometry(left), _geometry(audit))
                )
            }
            if not owners:
                raise ValueError("corrected conflict target missing ownership")
            if not owners.issubset(membership):
                raise ValueError("corrected conflict target external overlap")
        for candidate_id in membership:
            audit = audit_by_candidate[candidate_id]
            overlapping = [
                target for target in normalized_targets
                if _overlaps(_geometry(target), _geometry(audit))
            ]
            conflict_targets_by_candidate[candidate_id] = overlapping
            conflict_component_by_candidate[candidate_id] = str(
                conflict["task_id"]
            )
        actual_memberships.append(membership)
    if (
        len(actual_memberships) != len(set(actual_memberships))
        or set(actual_memberships) != expected_memberships
    ):
        raise ValueError("corrected conflict component coverage differs")

    removed_keys = set()
    corrections = []
    additions: dict[tuple[str, int, int, int], dict] = {}
    for candidate_id, audit in sorted(audit_by_candidate.items()):
        example_id = (
            f"juan-{int(audit['juan']):03d}-jie-"
            f"{int(audit['jie_index']):04d}"
        )
        candidate = {
            "id": example_id,
            "juan": int(audit["juan"]),
            "jie_index": int(audit["jie_index"]),
            "para_id": int(audit["para_id"]),
            "start": int(audit["start"]),
            "end": int(audit["end"]),
            "surface": str(audit["surface"]),
        }
        _validate_source(examples_by_id, candidate)
        added = []
        revised_label = str(audit["audited_label"])
        if candidate_id in conflict_targets_by_candidate:
            added = conflict_targets_by_candidate[candidate_id]
            exact = [
                reference for reference in added
                if _full_geometry(reference) == _full_geometry(candidate)
            ]
            if exact:
                if len(exact) != 1:
                    raise ValueError("duplicate exact conflict target")
                revised_label = "exact_person"
            elif added:
                revised_label = "wrong_boundary"
            else:
                revised_label = "not_person"
        elif revised_label == "exact_person":
            added = [_reference(candidate, "revision_12_exact_person")]
        elif revised_label == "wrong_boundary":
            target_decision = targets_by_candidate[candidate_id]
            wrong = target_decision["wrong_candidate"]
            if (
                int(wrong["para_id"]),
                int(wrong["start"]),
                int(wrong["end"]),
                str(wrong["surface"]),
            ) != (
                int(candidate["para_id"]),
                int(candidate["start"]),
                int(candidate["end"]),
                str(candidate["surface"]),
            ):
                raise ValueError("corrected target candidate differs")
            for target in target_decision["targets"]:
                normalized = {
                    "id": example_id,
                    "juan": int(audit["juan"]),
                    "jie_index": int(audit["jie_index"]),
                    **target,
                }
                _validate_source(examples_by_id, normalized)
                added.append(_reference(normalized, "revision_13_exact_target"))
        elif revised_label != "not_person":
            raise ValueError("unexpected corrected audit label")

        triggers = [("candidate", candidate)]
        triggers.extend(
            (f"added_reference:{index}", reference)
            for index, reference in enumerate(added)
        )
        removed_with_triggers = []
        for reference in baseline_references.values():
            matching = [
                name
                for name, trigger in triggers
                if reference["id"] == example_id
                and _overlaps(_geometry(reference), _geometry(trigger))
            ]
            if matching:
                removed_with_triggers.append({
                    "reference": reference,
                    "triggers": matching,
                })
        if revised_label == "not_person" and removed_with_triggers:
            raise ValueError("semantic negative removes a frozen reference")
        removed_keys.update(
            _full_geometry(row["reference"]) for row in removed_with_triggers
        )

        for reference in added:
            key = _full_geometry(reference)
            prior = additions.get(key)
            if prior is not None and prior["surface"] != reference["surface"]:
                raise ValueError("corrected addition surface differs")
            additions[key] = reference
        corrections.append({
            "candidate_id": candidate_id,
            "audited_label": revised_label,
            "conflict_task_id": conflict_component_by_candidate.get(
                candidate_id
            ),
            "candidate": candidate,
            "removed_references": sorted(
                removed_with_triggers,
                key=lambda row: _full_geometry(row["reference"]),
            ),
            "added_references": sorted(added, key=_full_geometry),
        })

    corrected = {
        key: row
        for key, row in baseline_references.items()
        if key not in removed_keys
    }
    corrected.update(additions)
    references = sorted(corrected.values(), key=_full_geometry)
    by_example_para: dict[tuple[str, int], list[dict]] = {}
    for reference in references:
        by_example_para.setdefault(
            (reference["id"], int(reference["para_id"])), []
        ).append(reference)
    conflicts = []
    for values in by_example_para.values():
        values.sort(key=_full_geometry)
        for index, left in enumerate(values):
            for right in values[index + 1:]:
                if int(right["start"]) >= int(left["end"]):
                    break
                if _overlaps(_geometry(left), _geometry(right)):
                    conflicts.append(
                        (_full_geometry(left), _full_geometry(right))
                    )
    if conflicts:
        raise ValueError(f"conflicting corrected exact targets: {conflicts}")

    for correction in corrections:
        candidate = correction["candidate"]
        final_overlaps = {
            _full_geometry(reference)
            for reference in references
            if reference["id"] == candidate["id"]
            and _overlaps(_geometry(reference), _geometry(candidate))
        }
        label = correction["audited_label"]
        if label == "exact_person":
            expected = {_full_geometry(candidate)}
        elif label == "wrong_boundary":
            expected = {
                _full_geometry(reference)
                for reference in correction["added_references"]
            }
        else:
            expected = set()
        if final_overlaps != expected:
            raise ValueError(
                "corrected candidate final references differ: "
                f"{correction['candidate_id']}"
            )

    corrected_existence = []
    for row in existence_rows:
        overlaps = [
            reference
            for reference in references
            if reference["id"] == str(row["id"])
            and _overlaps(_geometry(row), _geometry(reference))
        ]
        corrected_existence.append({
            **row,
            "label": int(bool(overlaps)),
            "overlapping_references": [
                {
                    "para_id": int(reference["para_id"]),
                    "start": int(reference["start"]),
                    "end": int(reference["end"]),
                    "surface": str(reference["surface"]),
                }
                for reference in overlaps
            ],
        })

    exact_keys = set(corrected)
    retained_pairs = []
    dropped_pairs = []
    for pair in rank_pairs:
        positive_key = _full_geometry(pair["positive"])
        negative_key = _full_geometry(pair["negative"])
        if positive_key not in exact_keys or negative_key in exact_keys:
            dropped_pairs.append(pair)
        else:
            retained_pairs.append(pair)
    added_pairs = []
    for correction in corrections:
        if correction["audited_label"] != "wrong_boundary":
            continue
        wrong = correction["candidate"]
        real = real_by_geometry[_full_geometry(wrong)]
        for reference in correction["added_references"]:
            added_pairs.append({
                "positive": _candidate({
                    **reference,
                    "fold": int(real["fold"]),
                }),
                "negative": {
                    **_candidate(real),
                    "policy_membership": ["audited_wrong_boundary"],
                },
            })
    pair_by_geometry = {}
    for pair in [*retained_pairs, *added_pairs]:
        key = (
            _full_geometry(pair["positive"]),
            _full_geometry(pair["negative"]),
        )
        pair_by_geometry[key] = pair
    corrected_pairs = [
        pair_by_geometry[key] for key in sorted(pair_by_geometry)
    ]

    for row in easy_rows:
        _validate_source(examples_by_id, row)
        if any(
            reference["id"] == str(row["id"])
            and _overlaps(_geometry(row), _geometry(reference))
            for reference in references
        ):
            raise ValueError("easy negative overlaps a corrected reference")

    return {
        "references": references,
        "existence": sorted(corrected_existence, key=_full_geometry),
        "rank_pairs": corrected_pairs,
        "easy_negatives": sorted(easy_rows, key=_full_geometry),
        "corrections": corrections,
        "counts": {
            "baseline_references": len(baseline_references),
            "removed_references": len(removed_keys),
            "added_reference_geometries": len(additions),
            "corrected_references": len(references),
            "real_rows": len(corrected_existence),
            "real_exact_person": sum(
                any(
                    _full_geometry(row) == _full_geometry(reference)
                    for reference in references
                )
                for row in corrected_existence
            ),
            "real_boundary_alternative": sum(
                bool(row["label"])
                and not any(
                    _full_geometry(row) == _full_geometry(reference)
                    for reference in references
                )
                for row in corrected_existence
            ),
            "semantic_negatives": sum(
                not bool(row["label"]) for row in corrected_existence
            ),
            "easy_negatives": len(easy_rows),
            "retained_rank_pairs": len(retained_pairs),
            "dropped_rank_pairs": len(dropped_pairs),
            "added_rank_pairs": len(added_pairs),
            "corrected_rank_pairs": len(corrected_pairs),
            "revised_audit_labels": dict(sorted(Counter(
                row["audited_label"] for row in corrections
            ).items())),
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def build_corrected_inventory(
    grouped_root: Path,
    audit_root: Path,
    targets_root: Path,
    conflicts_root: Path,
    safe_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"corrected inventory exists: {output_dir}")
    grouped_manifest_path = grouped_root / "manifest.json"
    audit_manifest_path = audit_root / "manifest.json"
    targets_manifest_path = targets_root / "manifest.json"
    conflicts_manifest_path = conflicts_root / "manifest.json"
    safe_manifest_path = safe_root / "manifest.json"
    grouped_manifest = _read(grouped_manifest_path)
    audit_manifest = _read(audit_manifest_path)
    targets_manifest = _read(targets_manifest_path)
    conflicts_manifest = _read(conflicts_manifest_path)
    safe_manifest = _read(safe_manifest_path)
    examples_path = grouped_root / "examples.jsonl"
    existence_path = grouped_root / "existence.jsonl"
    rank_pairs_path = grouped_root / "rank-pairs.jsonl"
    audit_path = audit_root / "decisions.jsonl"
    targets_path = targets_root / "targets.jsonl"
    conflicts_path = conflicts_root / "decisions.jsonl"
    safe_path = safe_root / "safe-negatives.jsonl"
    grouped_hash = _sha256(grouped_manifest_path)
    if (
        grouped_manifest.get("status") != GROUPED_DATA_STATUS
        or grouped_manifest.get("confirmation_read") is not False
        or grouped_manifest.get("outputs", {}).get("examples_sha256")
        != _sha256(examples_path)
        or grouped_manifest.get("outputs", {}).get("existence_sha256")
        != _sha256(existence_path)
        or grouped_manifest.get("outputs", {}).get("rank_pairs_sha256")
        != _sha256(rank_pairs_path)
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
        or targets_manifest.get("bindings", {}).get("audit_manifest_sha256")
        != _sha256(audit_manifest_path)
        or conflicts_manifest.get("status") != FROZEN_CONFLICT_STATUS
        or conflicts_manifest.get("confirmation_read") is not False
        or conflicts_manifest.get("counts", {}).get("uncertain") != 0
        or conflicts_manifest.get("counts", {}).get("tasks")
        != EXPECTED_CONFLICTS
        or conflicts_manifest.get("decisions_sha256")
        != _sha256(conflicts_path)
        or conflicts_manifest.get("bindings", {}).get(
            "grouped_manifest_sha256"
        ) != grouped_hash
        or conflicts_manifest.get("bindings", {}).get(
            "audit_manifest_sha256"
        ) != _sha256(audit_manifest_path)
        or conflicts_manifest.get("bindings", {}).get(
            "targets_manifest_sha256"
        ) != _sha256(targets_manifest_path)
        or safe_manifest.get("status") != SAFE_STATUS
        or safe_manifest.get("confirmation_read") is not False
        or safe_manifest.get("bindings", {}).get("grouped_manifest_sha256")
        != grouped_hash
        or safe_manifest.get("outputs", {}).get("safe_negatives_sha256")
        != _sha256(safe_path)
    ):
        raise ValueError("corrected inventory source binding differs")

    audit_rows = _read_jsonl(audit_path)
    target_rows = _read_jsonl(targets_path)
    conflict_rows = _read_jsonl(conflicts_path)
    easy_rows = _read_jsonl(safe_path)
    if (
        len(audit_rows) != EXPECTED_AUDITED
        or sum(len(row["targets"]) for row in target_rows) != EXPECTED_TARGETS
        or len(conflict_rows) != EXPECTED_CONFLICTS
        or len(easy_rows) != EXPECTED_SAFE
    ):
        raise ValueError("corrected inventory audited counts differ")

    inventory = _build_corrected_inventory(
        _read_jsonl(examples_path),
        _read_jsonl(existence_path),
        _read_jsonl(rank_pairs_path),
        audit_rows,
        target_rows,
        conflict_rows,
        easy_rows,
    )
    counts = inventory["counts"]
    if (
        counts["baseline_references"]
        != int(grouped_manifest["counts"]["fit_references"])
        or counts["real_rows"] != EXPECTED_REAL_ROWS
        or sum(counts["revised_audit_labels"].values())
        != EXPECTED_AUDITED
        or counts["easy_negatives"] != EXPECTED_SAFE
    ):
        raise ValueError("corrected inventory derived counts differ")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        output_names = {
            "references": "references.jsonl",
            "existence": "existence.jsonl",
            "rank_pairs": "rank-pairs.jsonl",
            "easy_negatives": "easy-negatives.jsonl",
            "corrections": "corrections.jsonl",
        }
        for key, name in output_names.items():
            _write_jsonl(staging / name, inventory[key])
        manifest = {
            "schema_version": 1,
            "status": CORRECTED_STATUS,
            "revision": 13,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "git_commit": _git_commit_clean(),
            "bindings": {
                "grouped_manifest_sha256": grouped_hash,
                "audit_manifest_sha256": _sha256(audit_manifest_path),
                "targets_manifest_sha256": _sha256(targets_manifest_path),
                "conflicts_manifest_sha256": _sha256(
                    conflicts_manifest_path
                ),
                "safe_manifest_sha256": _sha256(safe_manifest_path),
                "examples_sha256": _sha256(examples_path),
            },
            "counts": counts,
            "outputs": {
                f"{key}_sha256": _sha256(staging / name)
                for key, name in output_names.items()
            },
            "claim_limit": (
                "Fit-only corrected inventory using a user-authorized "
                "Copilot teacher; not formal-grade evidence and not a "
                "production-quality claim."
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
        description="Build the Revision-13 corrected fit inventory."
    )
    parser.add_argument("--grouped-data", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--conflicts", type=Path, required=True)
    parser.add_argument("--safe-negatives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_corrected_inventory(
        args.grouped_data,
        args.audit,
        args.targets,
        args.conflicts,
        args.safe_negatives,
        args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
