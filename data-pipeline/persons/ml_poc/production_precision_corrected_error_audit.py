from __future__ import annotations

import argparse
import hashlib
import json
import math
import secrets
import tempfile
from pathlib import Path

import numpy as np

from p3_compact import _git_commit_clean
from production_precision_corrected_encoder_finetune import (
    EXPECTED_INVENTORY,
    FINETUNE_STATUS_BLOCKED,
)
from production_precision_corrected_inventory import CORRECTED_STATUS
from production_precision_grouped_data import GROUPED_DATA_STATUS
from production_precision_grouped_verifier import _read, _read_jsonl, _sha256
from production_span_verifier import _assembled_bounds
from production_train import _make_read_only


REVISION = 14
TASK_STATUS = "ml_production_precision_corrected_error_audit_task"
ARTIFACT_STATUS = "ml_production_precision_corrected_error_audit_tasks"
SELECTION_STATUS = "ml_production_precision_corrected_error_audit_selection"
EXPECTED_MISSED_EXACT = 78
EXPECTED_SEMANTIC_FALSE_POSITIVES = 19
EXPECTED_TASKS = 97
THRESHOLD = np.float32(0.50)
LABELS = ("exact_person", "not_person", "uncertain", "wrong_boundary")
SALT_BYTES = 32


def _geometry(row: dict) -> tuple[str, int, int, int]:
    return (
        str(row["id"]),
        int(row["para_id"]),
        int(row["start"]),
        int(row["end"]),
    )


def _opaque_id(salt: bytes, domain: str, row: dict) -> str:
    payload = "\x1f".join(
        (
            domain,
            str(row["id"]),
            str(int(row["para_id"])),
            str(int(row["start"])),
            str(int(row["end"])),
        )
    ).encode("utf-8")
    return hashlib.sha256(salt + b"\0" + payload).hexdigest()[:24]


def _random_order_key(salt: bytes, row: dict) -> str:
    return _opaque_id(salt, "revision-14-order", row)


def _fold_holdouts(manifest: dict) -> dict[int, set[int]]:
    result = {}
    for row in manifest.get("folds", []):
        fold = int(row["fold"])
        if fold in result:
            raise ValueError("corrected error audit duplicate fold")
        train = {int(value) for value in row["train_juans"]}
        heldout = {int(value) for value in row["heldout_juans"]}
        if train & heldout:
            raise ValueError("corrected error audit fold train/holdout collision")
        result[fold] = heldout
    if set(result) != set(range(7)):
        raise ValueError("corrected error audit fold inventory differs")
    return result


def _select_error_rows(
    scores: list[dict],
    reference_geometries: set[tuple[str, int, int, int]],
    semantic_geometries: set[tuple[str, int, int, int]],
    heldout_by_fold: dict[int, set[int]],
    *,
    enforce_production_counts: bool = True,
) -> list[dict]:
    seen = set()
    selected = []
    for row in scores:
        geometry = _geometry(row)
        if geometry in seen:
            raise ValueError("corrected error audit duplicate OOF geometry")
        seen.add(geometry)
        probability = float(row["oof_exact_probability"])
        if not math.isfinite(probability):
            raise ValueError("corrected error audit non-finite OOF score")
        score = np.float32(probability)
        fold = int(row["fold"])
        if fold not in heldout_by_fold or int(row["juan"]) not in heldout_by_fold[fold]:
            raise ValueError("corrected error audit score is not held out by juan")

        side = None
        if geometry in reference_geometries:
            if (
                row.get("class_label") != "exact_person"
                or row.get("stratum") != "exact_person"
                or row.get("inventory_source") != "corrected_reference"
            ):
                raise ValueError("corrected error audit reference score class differs")
            if score < THRESHOLD:
                side = "missed_exact"
        elif geometry in semantic_geometries:
            if (
                row.get("class_label") != "not_person"
                or row.get("stratum") != "real_not_person"
                or row.get("inventory_source") != "corrected_semantic"
            ):
                raise ValueError("corrected error audit semantic score class differs")
            if score >= THRESHOLD:
                side = "semantic_false_positive"
        if side is not None:
            selected.append({
                **row,
                "selection_side": side,
                "oof_exact_probability_float32": float(score),
            })

    if enforce_production_counts:
        counts = {
            side: sum(row["selection_side"] == side for row in selected)
            for side in ("missed_exact", "semantic_false_positive")
        }
        if len(seen) != EXPECTED_INVENTORY:
            raise ValueError("corrected error audit OOF inventory differs")
        if (
            counts["missed_exact"] != EXPECTED_MISSED_EXACT
            or counts["semantic_false_positive"]
            != EXPECTED_SEMANTIC_FALSE_POSITIVES
            or len(selected) != EXPECTED_TASKS
        ):
            raise ValueError("corrected error audit selected inventory differs")
    return selected


def _validate_task(task: dict) -> None:
    if set(task) != {
        "schema_version",
        "status",
        "phase",
        "task_id",
        "review_scope",
        "protocol",
        "jie",
        "candidate",
        "allowed_labels",
    }:
        raise ValueError("corrected error audit task fields differ")
    if set(task["protocol"]) != {"decision", "evidence", "independence"}:
        raise ValueError("corrected error audit protocol fields differ")
    if set(task["jie"]) != {"text", "segments"}:
        raise ValueError("corrected error audit jie fields differ")
    if any(
        set(segment) != {"para_id", "assembled_start", "assembled_end"}
        for segment in task["jie"]["segments"]
    ):
        raise ValueError("corrected error audit segment fields differ")
    if set(task["candidate"]) != {
        "candidate_id",
        "para_id",
        "start",
        "end",
        "surface",
    }:
        raise ValueError("corrected error audit candidate fields differ")
    if tuple(task["allowed_labels"]) != LABELS:
        raise ValueError("corrected error audit labels differ")


def build_tasks(
    corrected_root: Path,
    encoder_root: Path,
    grouped_root: Path,
    output_dir: Path,
    *,
    salt: bytes | None = None,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"corrected error audit output exists: {output_dir}")
    salt = secrets.token_bytes(SALT_BYTES) if salt is None else salt
    if len(salt) != SALT_BYTES:
        raise ValueError("corrected error audit salt must contain 32 bytes")

    corrected_manifest_path = corrected_root / "manifest.json"
    encoder_manifest_path = encoder_root / "manifest.json"
    grouped_manifest_path = grouped_root / "manifest.json"
    corrected_manifest = _read(corrected_manifest_path)
    encoder_manifest = _read(encoder_manifest_path)
    grouped_manifest = _read(grouped_manifest_path)
    examples_path = grouped_root / "examples.jsonl"
    scores_path = encoder_root / "oof-scores.jsonl"
    references_path = corrected_root / "references.jsonl"
    existence_path = corrected_root / "existence.jsonl"
    script_path = Path(__file__).resolve()

    if (
        corrected_manifest.get("status") != CORRECTED_STATUS
        or encoder_manifest.get("status") != FINETUNE_STATUS_BLOCKED
        or grouped_manifest.get("status") != GROUPED_DATA_STATUS
        or corrected_manifest.get("confirmation_read") is not False
        or encoder_manifest.get("confirmation_read") is not False
        or grouped_manifest.get("confirmation_read") is not False
    ):
        raise ValueError("corrected error audit source status differs")
    if (
        encoder_manifest.get("bindings", {}).get("corrected_manifest_sha256")
        != _sha256(corrected_manifest_path)
        or encoder_manifest.get("bindings", {}).get("grouped_manifest_sha256")
        != _sha256(grouped_manifest_path)
        or encoder_manifest.get("bindings", {}).get("grouped_examples_sha256")
        != _sha256(examples_path)
        or encoder_manifest.get("bindings", {}).get("oof_scores_sha256")
        != _sha256(scores_path)
        or corrected_manifest.get("outputs", {}).get("references_sha256")
        != _sha256(references_path)
        or corrected_manifest.get("outputs", {}).get("existence_sha256")
        != _sha256(existence_path)
    ):
        raise ValueError("corrected error audit source binding differs")

    examples_rows = _read_jsonl(examples_path)
    examples = {str(row["id"]): row for row in examples_rows}
    if len(examples) != len(examples_rows):
        raise ValueError("corrected error audit duplicate example")
    references = {_geometry(row) for row in _read_jsonl(references_path)}
    semantic = {
        _geometry(row)
        for row in _read_jsonl(existence_path)
        if int(row["label"]) == 0
    }
    if references & semantic:
        raise ValueError("corrected error audit source classes overlap")
    selected = _select_error_rows(
        _read_jsonl(scores_path),
        references,
        semantic,
        _fold_holdouts(encoder_manifest),
    )

    prepared = []
    for row in selected:
        example = examples.get(str(row["id"]))
        if example is None:
            raise ValueError("corrected error audit example missing")
        if (
            int(example["juan"]) != int(row["juan"])
            or int(example["jie_index"]) != int(row["jie_index"])
        ):
            raise ValueError("corrected error audit jie binding differs")
        _assembled_bounds(example, row)
        candidate_id = _opaque_id(salt, "revision-14-candidate", row)
        task_id = _opaque_id(salt, "revision-14-task", row)
        prepared.append((row, example, candidate_id, task_id))
    if (
        len({item[2] for item in prepared}) != len(prepared)
        or len({item[3] for item in prepared}) != len(prepared)
    ):
        raise ValueError("corrected error audit opaque ID collision")
    prepared.sort(key=lambda item: _random_order_key(salt, item[0]))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        tasks_dir = staging / "reviewer-tasks"
        tasks_dir.mkdir()
        selection_dir = staging / "sealed-selection"
        selection_dir.mkdir()
        selection_rows = []
        task_hashes = []
        for order, (row, example, candidate_id, task_id) in enumerate(prepared):
            task = {
                "schema_version": 1,
                "status": TASK_STATUS,
                "phase": "revision-14-blind-corrected-oof-error-audit",
                "task_id": task_id,
                "review_scope": "current-numbered-jie-only",
                "protocol": {
                    "decision": (
                        "Judge only whether the marked source occurrence is an exact "
                        "person span, a wrong-boundary person span, or not a person."
                    ),
                    "evidence": "Use only the complete numbered jie in this task.",
                    "independence": (
                        "Return one first judgment without seeking sibling tasks."
                    ),
                },
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
                "candidate": {
                    "candidate_id": candidate_id,
                    "para_id": int(row["para_id"]),
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "surface": str(row["surface"]),
                },
                "allowed_labels": list(LABELS),
            }
            _validate_task(task)
            path = tasks_dir / f"task_{task_id}.json"
            path.write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            task_hashes.append({
                "task_id": task_id,
                "task_sha256": _sha256(path),
            })
            selection_rows.append({
                "random_order": order,
                "task_id": task_id,
                "candidate_id": candidate_id,
                "id": str(row["id"]),
                "juan": int(row["juan"]),
                "jie_index": int(row["jie_index"]),
                "para_id": int(row["para_id"]),
                "start": int(row["start"]),
                "end": int(row["end"]),
                "surface": str(row["surface"]),
                "fold": int(row["fold"]),
                "selection_side": str(row["selection_side"]),
                "oof_exact_probability_source": float(
                    row["oof_exact_probability"]
                ),
                "oof_exact_probability_float32": float(
                    row["oof_exact_probability_float32"]
                ),
            })

        selection_path = selection_dir / "selection.jsonl"
        selection_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in selection_rows
            ),
            encoding="utf-8",
        )
        salt_path = selection_dir / "salt.txt"
        salt_path.write_text(salt.hex() + "\n", encoding="ascii")
        tasks_path = staging / "task-hashes.jsonl"
        tasks_path.write_text(
            "".join(
                json.dumps(row, separators=(",", ":")) + "\n"
                for row in sorted(task_hashes, key=lambda value: value["task_id"])
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "status": ARTIFACT_STATUS,
            "revision": REVISION,
            "formal_grade": False,
            "fit_only": True,
            "eligible_for_production": False,
            "confirmation_read": False,
            "one_candidate_per_task": True,
            "reviewer_progress_disclosure": False,
            "git_commit": _git_commit_clean(),
            "bindings": {
                "selector_sha256": _sha256(script_path),
                "corrected_manifest_sha256": _sha256(corrected_manifest_path),
                "encoder_manifest_sha256": _sha256(encoder_manifest_path),
                "grouped_manifest_sha256": _sha256(grouped_manifest_path),
                "references_sha256": _sha256(references_path),
                "existence_sha256": _sha256(existence_path),
                "examples_sha256": _sha256(examples_path),
                "oof_scores_sha256": _sha256(scores_path),
            },
            "selector": {
                "score_type": "numpy.float32",
                "threshold": float(THRESHOLD),
                "missed_exact_comparison": "score < threshold",
                "semantic_false_positive_comparison": "score >= threshold",
            },
            "counts": {
                "missed_exact": EXPECTED_MISSED_EXACT,
                "semantic_false_positive": EXPECTED_SEMANTIC_FALSE_POSITIVES,
                "tasks": EXPECTED_TASKS,
            },
            "outputs": {
                "task_hashes_sha256": _sha256(tasks_path),
                "selection_sha256": _sha256(selection_path),
                "salt_sha256": _sha256(salt_path),
            },
            "claim_limit": (
                "Adaptive fit-only diagnostic labels; not fresh generalization "
                "evidence and not a production-quality claim."
            ),
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
        description="Build blind Revision-14 corrected-OOF error-audit tasks."
    )
    parser.add_argument("--corrected-root", type=Path, required=True)
    parser.add_argument("--encoder-root", type=Path, required=True)
    parser.add_argument("--grouped-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_tasks(
        args.corrected_root,
        args.encoder_root,
        args.grouped_root,
        args.output_dir,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
