from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_corrected_error_audit import (
    ARTIFACT_STATUS as ERROR_TASK_STATUS,
)
from production_precision_corrected_error_reconcile import STATUS as RECONCILED_STATUS
from production_precision_corrected_inventory import CORRECTED_STATUS
from production_precision_grouped_data import GROUPED_DATA_STATUS
from production_precision_grouped_verifier import _read, _read_jsonl, _sha256
from production_precision_same_jie_attention import _folds
from production_train import _make_read_only


REVISION = 14
STATUS = "ml_production_precision_revision14_fit_inventory"
EXPECTED_OLD_REFERENCES = 2566
EXPECTED_REFERENCES = 2560
EXPECTED_DECISIONS = 97
EXPECTED_EXISTENCE = 2726
EXPECTED_EXACT_EXISTENCE = 2661
EXPECTED_SEMANTIC = 65
EXPECTED_SAFE = 3126
EXPECTED_EASY = 3115
EXPECTED_MINED_BOUNDARY = 11
EXPECTED_LATTICE = 12974
EXPECTED_BOUNDARY = 7219
EXPECTED_RANK_PAIRS = 7227
OUTPUT_FILES = {
    "references": "references.jsonl",
    "existence": "existence.jsonl",
    "rank_pairs": "rank-pairs.jsonl",
    "mandatory_pair_counts": "mandatory-pair-counts.jsonl",
    "easy_negatives": "easy-negatives.jsonl",
    "mined_boundaries": "mined-boundaries.jsonl",
    "corrections": "corrections.jsonl",
    "geometry_additions": "geometry-additions.jsonl",
    "geometry_removals": "geometry-removals.jsonl",
    "candidate_lattice": "candidate-lattice.jsonl",
}


def _key(row: dict) -> tuple[str, int, int, int]:
    return (
        str(row["id"]),
        int(row["para_id"]),
        int(row["start"]),
        int(row["end"]),
    )


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


def _ordered(row: dict) -> tuple[int, int, int, int, int, str]:
    return (
        int(row["juan"]),
        int(row["jie_index"]),
        int(row["para_id"]),
        int(row["start"]),
        int(row["end"]),
        str(row["id"]),
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _source_paragraph(example: dict, para_id: int) -> str:
    segments = [
        row for row in example["segments"] if int(row["para_id"]) == para_id
    ]
    if len(segments) != 1:
        raise ValueError("Revision-14 inventory paragraph differs")
    segment = segments[0]
    return str(example["text"])[
        int(segment["assembled_start"]):int(segment["assembled_end"])
    ]


def _validate_source(examples: dict[str, dict], row: dict) -> None:
    example = examples.get(str(row["id"]))
    if (
        example is None
        or int(example["juan"]) != int(row["juan"])
        or int(example["jie_index"]) != int(row["jie_index"])
    ):
        raise ValueError("Revision-14 inventory example differs")
    paragraph = _source_paragraph(example, int(row["para_id"]))
    start = int(row["start"])
    end = int(row["end"])
    if (
        not 0 <= start < end <= len(paragraph)
        or paragraph[start:end] != str(row["surface"])
    ):
        raise ValueError("Revision-14 inventory source geometry differs")


def _candidate(row: dict, fold_by_juan: dict[int, int]) -> dict:
    return {
        "id": str(row["id"]),
        "juan": int(row["juan"]),
        "jie_index": int(row["jie_index"]),
        "fold": int(fold_by_juan[int(row["juan"])]),
        "para_id": int(row["para_id"]),
        "start": int(row["start"]),
        "end": int(row["end"]),
        "surface": str(row["surface"]),
        "policy_membership": sorted(set(row.get("policy_membership", []))),
    }


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


def _apply_overlay(
    old_references: dict[tuple[str, int, int, int], dict],
    decisions: list[dict],
    selection: dict[str, dict],
    targets: dict[str, dict],
) -> tuple[dict, list[dict]]:
    removals = set()
    additions = {}
    corrections = []
    for decision in decisions:
        task_id = str(decision["task_id"])
        selected = selection[task_id]
        candidate_key = _key(selected)
        proposed = []
        if decision["final_label"] == "exact_person":
            proposed = [_reference(selected, "revision_14_error_exact")]
        elif decision["final_label"] == "wrong_boundary":
            target_row = targets.get(task_id)
            if target_row is None:
                raise ValueError("Revision-14 boundary target is missing")
            for target in target_row["targets"]:
                proposed.append(_reference(
                    {**selected, **target},
                    "revision_14_error_target",
                ))
        elif decision["final_label"] != "not_person":
            raise ValueError("Revision-14 unresolved audit label")
        proposed_keys = {_key(row) for row in proposed}
        if len(proposed_keys) != len(proposed):
            raise ValueError("Revision-14 duplicate proposed geometry")
        removed = []
        for key, reference in old_references.items():
            triggers = []
            if _overlaps(key, candidate_key):
                triggers.append("candidate")
            for index, proposed_key in enumerate(proposed_keys):
                if _overlaps(key, proposed_key):
                    triggers.append(f"added_reference:{index}")
            if triggers:
                removals.add(key)
                removed.append({"reference": reference, "triggers": triggers})
        for row in proposed:
            key = _key(row)
            prior = additions.get(key)
            if prior is not None and prior != row:
                raise ValueError("Revision-14 addition geometry differs")
            additions[key] = row
        corrections.append({
            "task_id": task_id,
            "candidate_id": str(decision["candidate_id"]),
            "initial_label": str(decision["initial_label"]),
            "final_label": str(decision["final_label"]),
            "resolution": str(decision["resolution"]),
            "candidate": _reference(selected, "revision_14_audited_candidate"),
            "removed_references": sorted(
                removed,
                key=lambda row: _ordered(row["reference"]),
            ),
            "added_references": sorted(proposed, key=_ordered),
        })
    final = {
        key: row for key, row in old_references.items() if key not in removals
    }
    final.update(additions)
    return final, sorted(corrections, key=lambda row: row["task_id"])


def build_inventory(
    grouped_root: Path,
    old_root: Path,
    error_task_root: Path,
    reconciled_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Revision-14 inventory exists: {output_dir}")
    grouped_manifest_path = grouped_root / "manifest.json"
    old_manifest_path = old_root / "manifest.json"
    error_manifest_path = error_task_root / "manifest.json"
    reconciled_manifest_path = reconciled_root / "manifest.json"
    selection_path = error_task_root / "sealed-selection" / "selection.jsonl"
    decisions_path = reconciled_root / "decisions.jsonl"
    targets_path = reconciled_root / "targets.jsonl"
    examples_path = grouped_root / "examples.jsonl"
    grouped_manifest = _read(grouped_manifest_path)
    old_manifest = _read(old_manifest_path)
    error_manifest = _read(error_manifest_path)
    reconciled_manifest = _read(reconciled_manifest_path)
    if (
        grouped_manifest.get("status") != GROUPED_DATA_STATUS
        or grouped_manifest.get("confirmation_read") is not False
        or grouped_manifest.get("outputs", {}).get("examples_sha256")
        != _sha256(examples_path)
        or old_manifest.get("status") != CORRECTED_STATUS
        or old_manifest.get("confirmation_read") is not False
        or old_manifest.get("bindings", {}).get("grouped_manifest_sha256")
        != _sha256(grouped_manifest_path)
        or error_manifest.get("status") != ERROR_TASK_STATUS
        or error_manifest.get("outputs", {}).get("selection_sha256")
        != _sha256(selection_path)
        or reconciled_manifest.get("status") != RECONCILED_STATUS
        or reconciled_manifest.get("confirmation_read") is not False
        or reconciled_manifest.get("bindings", {}).get(
            "frozen_manifest_sha256"
        ) is None
        or reconciled_manifest.get("outputs", {}).get("decisions_sha256")
        != _sha256(decisions_path)
        or reconciled_manifest.get("outputs", {}).get("targets_sha256")
        != _sha256(targets_path)
    ):
        raise ValueError("Revision-14 inventory source binding differs")
    old_paths = {key: old_root / name for key, name in OUTPUT_FILES.items() if (
        key not in {
            "geometry_additions",
            "geometry_removals",
            "candidate_lattice",
        }
    )}
    for key, path in old_paths.items():
        if old_manifest.get("outputs", {}).get(f"{key}_sha256") != _sha256(path):
            raise ValueError(f"Revision-14 old inventory binding differs: {key}")

    example_rows = _read_jsonl(examples_path)
    examples = {str(row["id"]): row for row in example_rows}
    fold_by_juan = _folds([int(row["juan"]) for row in example_rows])
    selection_rows = _read_jsonl(selection_path)
    selection = {str(row["task_id"]): row for row in selection_rows}
    decisions = _read_jsonl(decisions_path)
    targets = {
        str(row["source_task_id"]): row for row in _read_jsonl(targets_path)
    }
    if (
        len(selection) != EXPECTED_DECISIONS
        or len(selection_rows) != EXPECTED_DECISIONS
        or len(decisions) != EXPECTED_DECISIONS
        or {str(row["task_id"]) for row in decisions} != set(selection)
        or len(targets) != 3
        or set(targets) != {
            str(row["task_id"]) for row in decisions
            if row["final_label"] == "wrong_boundary"
        }
    ):
        raise ValueError("Revision-14 overlay inventory differs")
    for row in selection_rows:
        _validate_source(examples, row)
    for task_id, row in targets.items():
        selected = selection[task_id]
        for target in row["targets"]:
            normalized = {**selected, **target}
            _validate_source(examples, normalized)
            if not _overlaps(_key(selected), _key(normalized)):
                raise ValueError("Revision-14 target does not overlap candidate")

    old_reference_rows = _read_jsonl(old_paths["references"])
    old_references = {_key(row): row for row in old_reference_rows}
    if (
        len(old_reference_rows) != EXPECTED_OLD_REFERENCES
        or len(old_references) != len(old_reference_rows)
    ):
        raise ValueError("Revision-14 old reference inventory differs")
    references, corrections = _apply_overlay(
        old_references, decisions, selection, targets
    )
    reversed_references, reversed_corrections = _apply_overlay(
        old_references, list(reversed(decisions)), selection, targets
    )
    if (
        references != reversed_references
        or {
            row["task_id"]: row for row in corrections
        } != {
            row["task_id"]: row for row in reversed_corrections
        }
        or len(references) != EXPECTED_REFERENCES
    ):
        raise ValueError("Revision-14 correction order differs")
    reference_keys = sorted(references)
    conflicts = [
        (left, right)
        for index, left in enumerate(reference_keys)
        for right in reference_keys[index + 1:]
        if _overlaps(left, right)
    ]
    if conflicts:
        raise ValueError(f"Revision-14 exact-reference conflicts: {conflicts}")

    by_task = {str(row["task_id"]): row for row in decisions}
    for task_id, selected in selection.items():
        key = _key(selected)
        overlaps = [ref for ref in references if _overlaps(key, ref)]
        label = by_task[task_id]["final_label"]
        if label == "exact_person" and key not in references:
            raise ValueError("Revision-14 exact candidate is not a reference")
        if label == "not_person" and overlaps:
            raise ValueError("Revision-14 not-person overlaps a reference")
        if label == "wrong_boundary" and (
            key in references or not overlaps
        ):
            raise ValueError("Revision-14 boundary reconciliation differs")

    def overlapping_references(key: tuple[str, int, int, int]) -> list[dict]:
        return [
            {
                "para_id": int(references[ref]["para_id"]),
                "start": int(references[ref]["start"]),
                "end": int(references[ref]["end"]),
                "surface": str(references[ref]["surface"]),
            }
            for ref in reference_keys
            if _overlaps(key, ref)
        ]

    old_existence = _read_jsonl(old_paths["existence"])
    existence_by_key = {_key(row): row for row in old_existence}
    if len(existence_by_key) != len(old_existence):
        raise ValueError("Revision-14 old existence duplicates")
    for selected in selection_rows:
        existence_by_key.setdefault(_key(selected), selected)
    existence = []
    for key, row in existence_by_key.items():
        candidate = _candidate(row, fold_by_juan)
        overlaps = overlapping_references(key)
        existence.append({
            **candidate,
            "label": int(bool(overlaps)),
            "overlapping_references": overlaps,
        })
    existence.sort(key=_ordered)

    old_easy = _read_jsonl(old_paths["easy_negatives"])
    old_mined = _read_jsonl(old_paths["mined_boundaries"])
    safe_by_key = {_key(row): row for row in [*old_easy, *old_mined]}
    if len(safe_by_key) != EXPECTED_SAFE:
        raise ValueError("Revision-14 safe inventory differs")
    easy_negatives = []
    mined_boundaries = []
    for key, row in safe_by_key.items():
        overlaps = overlapping_references(key)
        normalized = {
            **row,
            "fold": int(fold_by_juan[int(row["juan"])]),
            "occurrence_label": int(bool(overlaps)),
            "exact_class": (
                "boundary_alternative" if overlaps else "not_person"
            ),
            "rank_role": "negative" if overlaps else "none",
        }
        if overlaps:
            normalized["overlapping_references"] = overlaps
            mined_boundaries.append(normalized)
        else:
            normalized.pop("overlapping_references", None)
            easy_negatives.append(normalized)
    easy_negatives.sort(key=_ordered)
    mined_boundaries.sort(key=_ordered)

    lattice = {}
    def add_lattice(row: dict) -> None:
        candidate = _candidate(row, fold_by_juan)
        _validate_source(examples, candidate)
        key = _key(candidate)
        prior = lattice.get(key)
        if prior is not None and prior["surface"] != candidate["surface"]:
            raise ValueError("Revision-14 candidate lattice surface differs")
        if prior is None:
            lattice[key] = candidate
        else:
            prior["policy_membership"] = sorted(set(
                prior["policy_membership"] + candidate["policy_membership"]
            ))

    for row in references.values():
        add_lattice(row)
    for row in existence:
        add_lattice(row)
    for row in [*easy_negatives, *mined_boundaries]:
        add_lattice(row)
    for pair in _read_jsonl(old_paths["rank_pairs"]):
        add_lattice(pair["positive"])
        add_lattice(pair["negative"])
    if len(lattice) != EXPECTED_LATTICE:
        raise ValueError("Revision-14 candidate lattice count differs")

    boundary = {
        key: row for key, row in lattice.items()
        if key not in references and overlapping_references(key)
    }
    rank_pairs = []
    counts_by_positive = Counter()
    for negative_key, negative in boundary.items():
        for positive_key in reference_keys:
            if not _overlaps(negative_key, positive_key):
                continue
            positive = _candidate(references[positive_key], fold_by_juan)
            rank_pairs.append({"positive": positive, "negative": negative})
            counts_by_positive[positive_key] += 1
    rank_pairs.sort(key=lambda pair: (*_ordered(pair["positive"]), *_ordered(
        pair["negative"]
    )))
    mandatory_pair_counts = [
        {
            **_candidate(references[key], fold_by_juan),
            "boundary_pairs": int(count),
            "total_mandatory_pairs": int(count),
        }
        for key, count in sorted(counts_by_positive.items())
    ]

    if (
        len(existence) != EXPECTED_EXISTENCE
        or sum(row["label"] == 1 for row in existence)
        != EXPECTED_EXACT_EXISTENCE
        or sum(row["label"] == 0 for row in existence) != EXPECTED_SEMANTIC
        or len(easy_negatives) != EXPECTED_EASY
        or len(mined_boundaries) != EXPECTED_MINED_BOUNDARY
        or len(boundary) != EXPECTED_BOUNDARY
        or len(rank_pairs) != EXPECTED_RANK_PAIRS
    ):
        raise ValueError("Revision-14 derived inventory counts differ")

    old_keys = set(old_references)
    additions = sorted(
        [references[key] for key in set(references) - old_keys], key=_ordered
    )
    removals = sorted(
        [old_references[key] for key in old_keys - set(references)], key=_ordered
    )
    if len(additions) != 4 or len(removals) != 10:
        raise ValueError("Revision-14 raw geometry delta differs")
    all_corrections = [
        *_read_jsonl(old_paths["corrections"]),
        *corrections,
    ]
    outputs = {
        "references": sorted(references.values(), key=_ordered),
        "existence": existence,
        "rank_pairs": rank_pairs,
        "mandatory_pair_counts": mandatory_pair_counts,
        "easy_negatives": easy_negatives,
        "mined_boundaries": mined_boundaries,
        "corrections": all_corrections,
        "geometry_additions": additions,
        "geometry_removals": removals,
        "candidate_lattice": sorted(lattice.values(), key=_ordered),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        for key, name in OUTPUT_FILES.items():
            _write_jsonl(staging / name, outputs[key])
        manifest = {
            "schema_version": 1,
            "status": STATUS,
            "revision": REVISION,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "git_commit": _git_commit_clean(),
            "bindings": {
                "grouped_manifest_sha256": _sha256(grouped_manifest_path),
                "old_inventory_manifest_sha256": _sha256(old_manifest_path),
                "error_task_manifest_sha256": _sha256(error_manifest_path),
                "reconciled_manifest_sha256": _sha256(
                    reconciled_manifest_path
                ),
                "selection_sha256": _sha256(selection_path),
                "decisions_sha256": _sha256(decisions_path),
                "targets_sha256": _sha256(targets_path),
                "examples_sha256": _sha256(examples_path),
            },
            "counts": {
                "old_references": EXPECTED_OLD_REFERENCES,
                "references": len(references),
                "raw_additions": len(additions),
                "removals": len(removals),
                "net_growth": len(additions) - len(removals),
                "geometry_replacements": 3,
                "existence": len(existence),
                "existence_positive": EXPECTED_EXACT_EXISTENCE,
                "semantic_negatives": EXPECTED_SEMANTIC,
                "easy_negatives": len(easy_negatives),
                "mined_boundaries": len(mined_boundaries),
                "candidate_lattice": len(lattice),
                "boundary_alternatives": len(boundary),
                "rank_pairs": len(rank_pairs),
                "revision14_labels": dict(sorted(Counter(
                    row["final_label"] for row in decisions
                ).items())),
                "conflict_components": 0,
            },
            "outputs": {
                f"{key}_sha256": _sha256(staging / name)
                for key, name in OUTPUT_FILES.items()
            },
            "next_action": "fold_local_structural_mining",
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the reconciled Revision-14 fit inventory."
    )
    parser.add_argument("--grouped-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--error-task-root", type=Path, required=True)
    parser.add_argument("--reconciled-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_inventory(
        args.grouped_root,
        args.old_root,
        args.error_task_root,
        args.reconciled_root,
        args.output_dir,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
