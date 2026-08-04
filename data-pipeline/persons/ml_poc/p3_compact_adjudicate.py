from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from p1_windows import labels_to_spans


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _geometry(row: dict) -> tuple[int, int, int]:
    return (
        int(row["para_id"]),
        int(row["start"]),
        int(row["end"]),
    )


def disagreement_rows(prediction: dict) -> list[dict]:
    reference = {
        _geometry(row): row for row in prediction["reference_spans"]
    }
    model = {
        _geometry(row): row for row in prediction["prediction_spans"]
    }
    disagreements = sorted(set(reference) ^ set(model))
    return [
        {
            "para_id": geometry[0],
            "start": geometry[1],
            "end": geometry[2],
            "surface": str((reference.get(geometry) or model[geometry])["surface"]),
        }
        for geometry in disagreements
    ]


def _reference_rows(example: dict) -> list[dict]:
    spans = labels_to_spans(
        example,
        example["labels"],
        [
            char != "\n" and is_target
            for char, is_target in zip(
                example["text"],
                example.get("target_mask", [True] * len(example["text"])),
            )
        ],
    )
    return [span.__dict__ for span in spans]


def consensus_annotations(
    annotations: list[dict],
    candidates: list[dict],
) -> list[dict]:
    disagreement_geometry = {_geometry(row) for row in candidates}
    return [
        row for row in annotations
        if _geometry(row) not in disagreement_geometry
    ]


def prepare_adjudication(
    tasks_dir: Path,
    state_dir: Path,
    frozen_dir: Path,
    evaluation_dir: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"adjudication output exists: {output_dir}")
    git_commit = _git_commit_clean()
    source_manifest_path = tasks_dir / "manifest.json"
    source_manifest = _read(source_manifest_path)
    freeze_report_path = frozen_dir / "freeze_report.json"
    freeze_report = _read(freeze_report_path)
    reference_path = frozen_dir / "reference.jsonl"
    evaluation_report_path = evaluation_dir / "report.json"
    evaluation_report = _read(evaluation_report_path)
    predictions_path = evaluation_dir / "predictions.json"
    predictions = _read(predictions_path)
    reference_by_id = {
        str(row["id"]): row
        for row in (
            json.loads(line)
            for line in reference_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line
        )
    }
    reference_ids = {
        str(row_id)
        for row_id in reference_by_id
    }
    if (
        freeze_report.get("status")
        != "frozen_candidate_blind_compact_p3"
        or freeze_report.get("source_manifest_sha256")
        != _sha256(source_manifest_path)
        or freeze_report.get("reference_sha256") != _sha256(reference_path)
        or evaluation_report.get("status")
        != "formal_compact_p3_evaluated"
        or evaluation_report.get("inputs", {}).get("manifest_sha256")
        != _sha256(source_manifest_path)
        or evaluation_report.get("inputs", {}).get("reference_sha256")
        != _sha256(reference_path)
        or evaluation_report.get("inputs", {}).get("freeze_report_sha256")
        != _sha256(freeze_report_path)
        or evaluation_report.get("outputs", {}).get("predictions_sha256")
        != _sha256(predictions_path)
    ):
        raise ValueError("adjudication inputs are not the bound compact P3")
    prediction_by_id = {str(row["id"]): row for row in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("evaluation predictions duplicate an ID")
    if set(prediction_by_id) != reference_ids:
        raise ValueError("evaluation prediction IDs differ from reference")
    for example_id, prediction in prediction_by_id.items():
        if {
            (
                int(row["para_id"]),
                int(row["start"]),
                int(row["end"]),
                str(row["surface"]),
            )
            for row in prediction["reference_spans"]
        } != {
            (
                int(row["para_id"]),
                int(row["start"]),
                int(row["end"]),
                str(row["surface"]),
            )
            for row in _reference_rows(reference_by_id[example_id])
        }:
            raise ValueError(
                f"prediction reference differs from frozen BIO: {example_id}"
            )
    source_selection = {
        int(row["juan"]): row for row in source_manifest["selected"]
    }
    packs = {}
    candidate_total = 0
    for example_id, prediction in prediction_by_id.items():
        candidates = disagreement_rows(prediction)
        if not candidates:
            continue
        parts = example_id.split("-")
        juan = int(parts[1])
        for candidate in candidates:
            identity = (
                f"{example_id}:{candidate['para_id']}:"
                f"{candidate['start']}:{candidate['end']}"
            )
            candidate["id"] = (
                "adjudication:"
                + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
            )
            candidate["channels"] = ["source_hidden"]
        pack = packs.setdefault(juan, {
            "schema_version": 1,
            "phase": "recall",
            "juan": juan,
            "source_hidden": True,
            "initial_annotations": "model-v1 consensus only",
            "candidates": [],
            "note_evidence": [],
        })
        pack["candidates"].extend(candidates)
        candidate_total += len(candidates)
    expected_candidates = (
        int(evaluation_report["audit_summary"]["delta"][
            "model_non_reference_geometries"
        ])
        + int(evaluation_report["audit_summary"]["delta"][
            "model_reference_misses"
        ])
    )
    if candidate_total != expected_candidates:
        raise ValueError(
            "adjudication candidate count differs from evaluation audit"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        task_output = staging / "tasks"
        recall_output = staging / "recall"
        state_output = staging / "state"
        task_output.mkdir()
        recall_output.mkdir()
        state_output.mkdir()
        manifest = {
            "schema_version": 1,
            "status": "post_sealed_source_hidden_adjudication",
            "formal_p3": False,
            "source_hidden": True,
            "original_v1_immutable": True,
            "git_commit": git_commit,
            "inputs": {
                "source_manifest_sha256": _sha256(source_manifest_path),
                "freeze_report_sha256": _sha256(freeze_report_path),
                "reference_sha256": _sha256(reference_path),
                "evaluation_report_sha256": _sha256(
                    evaluation_report_path
                ),
                "predictions_sha256": _sha256(predictions_path),
            },
            "candidate_geometries": candidate_total,
            "selected": [],
        }
        for juan in source_manifest["selected"]:
            juan_number = int(juan["juan"])
            pack = packs.get(juan_number)
            if pack is None:
                continue
            source_task_path = tasks_dir / str(
                source_selection[juan_number]["task"]
            )
            frozen_input = freeze_report.get(
                "frozen_inputs", {}
            ).get(str(juan_number), {})
            source_state_path = state_dir / f"juan_{juan_number:03d}.json"
            if (
                _sha256(source_task_path)
                != source_selection[juan_number].get("task_sha256")
                or _sha256(source_task_path)
                != frozen_input.get("task_sha256")
                or _sha256(source_state_path)
                != frozen_input.get("state_sha256")
            ):
                raise ValueError(
                    f"sealed task/state hash differs: {juan_number}"
                )
            task_path = task_output / source_task_path.name
            task_path.write_bytes(source_task_path.read_bytes())
            pack["candidates"].sort(
                key=lambda row: (
                    row["para_id"], row["start"], row["end"], row["id"]
                )
            )
            pack_path = recall_output / f"recall_juan_{juan_number:03d}.json"
            _write(pack_path, pack)
            source_state = _read(source_state_path)
            if not source_state.get("blind", {}).get("complete"):
                raise ValueError(f"source state is not locked: {juan_number}")
            annotations = source_state["blind"]["annotations"]
            initial_annotations = consensus_annotations(
                annotations, pack["candidates"]
            )
            _write(state_output / source_state_path.name, {
                "schema_version": 1,
                "juan": juan_number,
                "blind": {
                    "complete": True,
                    "annotations": annotations,
                },
                "recall": {
                    "complete": False,
                    "annotations": initial_annotations,
                    "decisions": {},
                    "note_decisions": {},
                },
                "role_audit": {
                    "complete": False,
                    "initialized": False,
                    "annotations": [],
                    "decisions": {},
                },
                "assisted": {
                    "complete": False,
                    "initialized": False,
                    "annotations": [],
                    "decisions": {},
                },
            })
            manifest["selected"].append({
                "juan": juan_number,
                "role": "post_sealed_adjudication",
                "mode": "adjudication",
                "task": task_path.name,
                "task_sha256": _sha256(task_path),
                "pack_sha256": _sha256(pack_path),
                "candidates": len(pack["candidates"]),
            })
        _write(task_output / "manifest.json", manifest)
        for directory in (task_output, recall_output):
            for path in directory.iterdir():
                path.chmod(stat.S_IREAD)
                if path.stat().st_mode & (
                    stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
                ):
                    raise PermissionError(
                        f"failed to freeze adjudication input: {path}"
                    )
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare source-hidden compact P3 disagreement review."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_adjudication(
        args.tasks,
        args.state,
        args.frozen,
        args.evaluation,
        args.output,
    )
    print(json.dumps({
        "status": manifest["status"],
        "tasks": len(manifest["selected"]),
        "candidate_geometries": manifest["candidate_geometries"],
        "source_hidden": manifest["source_hidden"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
