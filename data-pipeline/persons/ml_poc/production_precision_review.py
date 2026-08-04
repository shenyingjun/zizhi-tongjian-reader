from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import stat
import tempfile
from pathlib import Path

from production_review import _read, _sha256, _validate_teacher


STATUS_PRE_AUDIT = "ml_production_precision_reference_review_pre_audit"
STATUS_READY = "ml_production_precision_reference_review"
NEGATIVE_AUDIT_SEED = 20260809


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _freeze(staging: Path, output_dir: Path) -> None:
    for path in staging.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IREAD)
    staging.replace(output_dir)


def _dataset_rows(root: Path) -> dict[tuple[int, int], dict]:
    rows = {}
    for split in ("train.jsonl", "development.jsonl"):
        path = root / split
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (int(row["juan"]), int(row["jie_index"]))
            if key in rows:
                raise ValueError(f"duplicate source dataset pair: {key}")
            rows[key] = row
    return rows


def _label_geometries(row: dict) -> set[tuple[int, int, int]]:
    geometries = set()
    open_start = None
    spans = []
    for index, label in enumerate(row["labels"]):
        if label == "B-PER":
            if open_start is not None:
                spans.append((open_start, index))
            open_start = index
        elif label != "I-PER" and open_start is not None:
            spans.append((open_start, index))
            open_start = None
    if open_start is not None:
        spans.append((open_start, len(row["labels"])))
    for start, end in spans:
        matches = [
            segment for segment in row["segments"]
            if int(segment["assembled_start"]) <= start < end
            <= int(segment["assembled_end"])
        ]
        if len(matches) != 1:
            raise ValueError(f"label span crosses paragraph segment: {(start, end)}")
        segment = matches[0]
        offset = int(segment["assembled_start"])
        geometries.add((int(segment["para_id"]), start - offset, end - offset))
    return geometries


def prepare(
    partition_root: Path,
    round1_review: Path,
    round1_dataset: Path,
    round2_review: Path,
    round2_dataset: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"precision review output exists: {output_dir}")
    partition_manifest_path = partition_root / "manifest.json"
    partition_private_path = partition_root / "private.json"
    partition_manifest = _read(partition_manifest_path)
    partition_private = _read(partition_private_path)
    if (
        partition_manifest.get("status") != "ml_production_precision_partition"
        or partition_manifest.get("outputs", {}).get("private_sha256")
        != _sha256(partition_private_path)
    ):
        raise ValueError("precision partition binding differs")

    sources = {
        1: (round1_review, round1_dataset),
        2: (round2_review, round2_dataset),
    }
    source_data = {}
    for round_number, (review_root, dataset_root) in sources.items():
        manifest_path = review_root / "manifest.json"
        manifest = _read(manifest_path)
        dataset_manifest = _read(dataset_root / "manifest.json")
        if (
            manifest.get("status")
            != "ml_production_focused_review_with_reduced_audit"
            or dataset_manifest.get("inputs", {}).get("review_manifest_sha256")
            != _sha256(manifest_path)
        ):
            raise ValueError(f"Round {round_number} review binding differs")
        source_data[round_number] = (
            review_root,
            {str(row["task_id"]): row for row in manifest["selected"]},
            _dataset_rows(dataset_root),
            _sha256(manifest_path),
            _sha256(dataset_root / "manifest.json"),
        )

    held_out = [
        row for row in partition_private["rows"]
        if row["partition"] in {"calibration", "confirmation"}
    ]
    if len(held_out) != 91:
        raise ValueError("precision held-out inventory differs")

    selected = []
    private_rows = []
    negative_task_ids = []
    candidate_count = 0
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        (staging / "tasks").mkdir()
        (staging / "review").mkdir()
        (staging / "negative-audit-tasks").mkdir()
        (staging / "private").mkdir()

        for partition_row in held_out:
            round_number = int(partition_row["round"])
            task_id = str(partition_row["task_id"])
            key = (int(partition_row["juan"]), int(partition_row["jie_index"]))
            review_root, source_selected, dataset_rows, _, _ = source_data[round_number]
            source_selection = source_selected.get(task_id)
            dataset_row = dataset_rows.get(key)
            if source_selection is None or dataset_row is None:
                raise ValueError(f"missing precision review source: {task_id}")
            source_task_path = review_root / source_selection["task"]
            source_review_path = review_root / source_selection["review"]
            if (
                _sha256(source_task_path) != source_selection["task_sha256"]
                or _sha256(source_review_path) != source_selection["review_sha256"]
            ):
                raise PermissionError(f"source review hash differs: {task_id}")
            task = _read(source_task_path)
            source_review_payload = _read(source_review_path)

            candidates_by_geometry = {}
            has_ab_candidate = False
            for source_candidate in source_review_payload["candidates"]:
                geometry = (
                    int(source_candidate["para_id"]),
                    int(source_candidate["start"]),
                    int(source_candidate["end"]),
                )
                channels = [str(value) for value in source_candidate["channels"]]
                has_ab_candidate |= any(
                    value in {"copilot_independent_a", "copilot_independent_b"}
                    for value in channels
                )
                candidate = {
                    key: value for key, value in source_candidate.items()
                    if key not in {"third_teacher", "confidence", "review_reason"}
                }
                candidate["confidence"] = "low"
                candidate["review_reason"] = (
                    "Formal-grade held-out reference requires an explicit human decision."
                )
                candidates_by_geometry[geometry] = candidate

            jie = task["jies"][0]
            task_text = str(jie["text"])
            paragraphs = {
                int(segment["para_id"]): task_text[
                    int(segment["assembled_start"]) : int(segment["assembled_end"])
                ]
                for segment in jie["segments"]
            }
            for para_id, start, end in _label_geometries(dataset_row):
                geometry = (para_id, start, end)
                if geometry not in candidates_by_geometry:
                    candidates_by_geometry[geometry] = {
                        "id": f"copilot:{task_id}:{para_id}:{start}:{end}",
                        "para_id": para_id,
                        "start": start,
                        "end": end,
                        "surface": paragraphs[para_id][start:end],
                        "channels": ["frozen_reference"],
                        "confidence": "low",
                        "review_reason": (
                            "Formal-grade held-out reference requires an explicit "
                            "human decision."
                        ),
                    }

            candidates = [
                candidates_by_geometry[key] for key in sorted(candidates_by_geometry)
            ]
            candidate_count += len(candidates)
            task_relative = f"tasks/task_{task_id}.json"
            review_relative = f"review/task_{task_id}.json"
            task_path = staging / task_relative
            review_path = staging / review_relative
            shutil.copy2(source_task_path, task_path)
            review_payload = {
                "schema_version": 1,
                "phase": "precision-reference-review",
                "training_only": True,
                "candidate_model_blind": True,
                "task_id": task_id,
                "juan": key[0],
                "jie_index": key[1],
                "human_review_scope": "all_candidates_and_recall_audit_additions",
                "initial_annotations": [],
                "initial_decisions": {},
                "candidates": candidates,
            }
            _write(review_path, review_payload)
            selected.append({
                "task_id": task_id,
                "juan": key[0],
                "jie_index": key[1],
                "task": task_relative,
                "task_sha256": _sha256(task_path),
                "review": review_relative,
                "review_sha256": _sha256(review_path),
                "review_candidates": len(candidates),
            })
            private_rows.append(dict(partition_row))
            if not has_ab_candidate:
                negative_task_ids.append(task_id)
                shutil.copy2(
                    source_task_path,
                    staging / "negative-audit-tasks" / f"task_{task_id}.json",
                )

        audit_count = math.ceil(len(negative_task_ids) * 0.25)
        rng = random.Random(NEGATIVE_AUDIT_SEED)
        chosen_negative = sorted(rng.sample(negative_task_ids, audit_count))
        for task_id in set(negative_task_ids) - set(chosen_negative):
            path = staging / "negative-audit-tasks" / f"task_{task_id}.json"
            path.chmod(stat.S_IWRITE)
            path.unlink()

        private = {
            "schema_version": 1,
            "status": "ml_production_precision_reference_private",
            "negative_audit_seed": NEGATIVE_AUDIT_SEED,
            "negative_audit_task_ids": chosen_negative,
            "rows": private_rows,
        }
        private_path = staging / "private" / "selection.json"
        _write(private_path, private)
        manifest = {
            "schema_version": 1,
            "status": STATUS_PRE_AUDIT,
            "candidate_model_blind": True,
            "model_predictions_used": False,
            "expected_tasks": len(selected),
            "partition_manifest_sha256": _sha256(partition_manifest_path),
            "partition_private_sha256": _sha256(partition_private_path),
            "source_review_manifests": {
                str(number): values[3] for number, values in source_data.items()
            },
            "source_dataset_manifests": {
                str(number): values[4] for number, values in source_data.items()
            },
            "private_selection_sha256": _sha256(private_path),
            "selected": selected,
            "counts": {
                "tasks": len(selected),
                "candidate_union": candidate_count,
                "consensus_negative_jies": len(negative_task_ids),
                "negative_jie_third_pass": len(chosen_negative),
                "negative_audit_review": 0,
            },
        }
        _write(staging / "manifest.json", manifest)
        _freeze(staging, output_dir)
    return manifest


def finalize(preaudit_root: Path, audit_root: Path, output_dir: Path) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"precision review output exists: {output_dir}")
    source_manifest_path = preaudit_root / "manifest.json"
    source_manifest = _read(source_manifest_path)
    private_path = preaudit_root / "private" / "selection.json"
    private = _read(private_path)
    if (
        source_manifest.get("status") != STATUS_PRE_AUDIT
        or source_manifest.get("private_selection_sha256") != _sha256(private_path)
        or private.get("status") != "ml_production_precision_reference_private"
    ):
        raise ValueError("precision pre-audit binding differs")
    task_ids = set(private["negative_audit_task_ids"])
    expected_names = {f"task_{task_id}.json" for task_id in task_ids}
    audit_paths = list(audit_root.glob("*.json"))
    if {path.name for path in audit_paths} != expected_names:
        raise ValueError("negative-audit output inventory differs")
    selected = {str(row["task_id"]): row for row in source_manifest["selected"]}

    validated = {}
    for task_id in sorted(task_ids):
        task_path = preaudit_root / selected[task_id]["task"]
        audit_path = audit_root / f"task_{task_id}.json"
        validated[task_id] = (
            audit_path,
            _validate_teacher(
                _read(task_path),
                task_id,
                _sha256(task_path),
                _read(audit_path),
                expected_pass="C-negative-recall-audit",
                expected_channel="copilot_independent_c",
                expected_phase="negative-audit",
            ),
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    additions = 0
    inventory = {}
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        for name in ("tasks", "review", "private"):
            shutil.copytree(preaudit_root / name, staging / name)
        for task_id, (audit_path, candidates) in validated.items():
            row = selected[task_id]
            review_path = staging / row["review"]
            review = _read(review_path)
            existing = {
                (
                    int(candidate["para_id"]),
                    int(candidate["start"]),
                    int(candidate["end"]),
                )
                for candidate in review["candidates"]
            }
            added = 0
            for geometry, source in sorted(candidates.items()):
                if geometry in existing:
                    continue
                para_id, start, end = geometry
                review["candidates"].append({
                    "id": f"copilot:{task_id}:{para_id}:{start}:{end}",
                    "para_id": para_id,
                    "start": start,
                    "end": end,
                    "surface": source["surface"],
                    "channels": ["copilot_independent_c"],
                    "confidence": "low",
                    "review_reason": (
                        "Independent held-out recall audit found this candidate."
                    ),
                    "pass_confidence": {
                        "a": None,
                        "b": None,
                        "c": source["confidence"],
                    },
                })
                added += 1
            review["candidates"].sort(
                key=lambda value: (
                    int(value["para_id"]),
                    int(value["start"]),
                    int(value["end"]),
                )
            )
            review["negative_audit_sha256"] = _sha256(audit_path)
            review_path.chmod(stat.S_IWRITE)
            _write(review_path, review)
            row["review_sha256"] = _sha256(review_path)
            row["review_candidates"] = len(review["candidates"])
            additions += added
            inventory[task_id] = {
                "sha256": _sha256(audit_path),
                "candidates": added,
            }

        manifest = {
            **source_manifest,
            "status": STATUS_READY,
            "source_review_manifest_sha256": _sha256(source_manifest_path),
            "negative_audit_inventory": inventory,
            "third_teacher_inventory": {},
            "counts": {
                **source_manifest["counts"],
                "candidate_union": (
                    int(source_manifest["counts"]["candidate_union"]) + additions
                ),
                "negative_audit_review": additions,
            },
        }
        _write(staging / "manifest.json", manifest)
        _freeze(staging, output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--partition", type=Path, required=True)
    prepare_parser.add_argument("--round1-review", type=Path, required=True)
    prepare_parser.add_argument("--round1-dataset", type=Path, required=True)
    prepare_parser.add_argument("--round2-review", type=Path, required=True)
    prepare_parser.add_argument("--round2-dataset", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--preaudit", type=Path, required=True)
    finalize_parser.add_argument("--audit", type=Path, required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare(
            args.partition,
            args.round1_review,
            args.round1_dataset,
            args.round2_review,
            args.round2_dataset,
            args.output,
        )
    else:
        manifest = finalize(args.preaudit, args.audit, args.output)
    print(json.dumps(manifest["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
