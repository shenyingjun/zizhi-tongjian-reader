from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from core import Span, score_spans
from p1_dataset import build_examples
from report import geometry_delta


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spans(rows: list[dict]) -> list[Span]:
    return [
        Span(
            int(row["para_id"]),
            int(row["start"]),
            int(row["end"]),
            str(row["surface"]),
        )
        for row in rows
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def validate_copilot_pack(
    pack: dict,
    juan: int,
    *,
    task: dict | None = None,
    task_sha256: str | None = None,
) -> None:
    if pack.get("juan") != juan or pack.get("phase") != "assisted":
        raise ValueError(f"assisted pack identity differs from juan {juan}")
    teacher = pack.get("candidate_teacher", {})
    if teacher.get("version") != "copilot_v1":
        raise ValueError(f"assisted juan {juan} is not a Copilot v1 pack")
    contract = pack.get("provenance_contract", {})
    required_contract = {
        "v1_used": False,
        "rules_used": False,
        "identity_fields_present": False,
        "full_juan_context_visible": True,
        "target_jie_authorization_only": True,
        "hu_notes_used": True,
        "translation_prose_transient": True,
        "human_review_required": True,
        "juan_76_labels_used": False,
    }
    for field, expected in required_contract.items():
        if contract.get(field) is not expected:
            raise ValueError(
                f"assisted juan {juan} has invalid provenance field {field}"
            )
    hashes = teacher.get("input_sha256", {})
    required_hashes = {
        "prompt",
        "spec",
        "boundary_guide",
        "task",
        "ml_seed",
        "teacher_evidence",
        "note_source",
        "translation_mapping",
        "translation_source",
    }
    demonstration_hashes = teacher.get("demonstration_sha256", {})
    values = [hashes.get(field) for field in required_hashes]
    values.extend(demonstration_hashes.get(name) for name in (
        "train.jsonl", "dev.jsonl", "pilot_holdout.jsonl"
    ))
    if not all(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value)
        for value in values
    ):
        raise ValueError(f"assisted juan {juan} has incomplete input hashes")
    scope = teacher.get("target_scope", {})
    if (
        scope.get("unit") != "numbered_jie"
        or scope.get("cross_jie_authorization") is not False
        or not scope.get("jie_indexes")
    ):
        raise ValueError(f"assisted juan {juan} has invalid target scope")
    if task is None or task_sha256 is None:
        return
    if hashes.get("task") != task_sha256:
        raise ValueError(f"assisted juan {juan} pack task hash differs")
    task_jies = {int(row["jie_index"]) for row in task["jies"]}
    if {int(value) for value in scope["jie_indexes"]} != task_jies:
        raise ValueError(f"assisted juan {juan} target scope differs from task")
    text_by_para = {}
    for jie in task["jies"]:
        text = str(jie["text"])
        for segment in jie["segments"]:
            start = int(segment["assembled_start"])
            end = int(segment["assembled_end"])
            text_by_para[int(segment["para_id"])] = text[start:end]
    geometries = set()
    for candidate in pack.get("candidates", []):
        para_id = int(candidate["para_id"])
        start = int(candidate["start"])
        end = int(candidate["end"])
        paragraph = text_by_para.get(para_id)
        if (
            paragraph is None
            or not 0 <= start < end <= len(paragraph)
            or paragraph[start:end] != candidate.get("surface")
        ):
            raise ValueError(
                f"assisted juan {juan} candidate is outside its task"
            )
        geometry = (para_id, start, end)
        if geometry in geometries:
            raise ValueError(
                f"assisted juan {juan} has duplicate candidate geometry"
            )
        geometries.add(geometry)


def candidate_metrics(reference: list[Span], candidates: list[Span]) -> dict:
    exact = score_spans(reference, candidates)
    overlap = score_spans(reference, candidates, overlap=True)
    return {
        "reference_spans": exact.reference,
        "candidate_spans": exact.predicted,
        "exact": {
            "true_positive": exact.true_positive,
            "precision": exact.precision,
            "recall": exact.recall,
            "f1": exact.f1,
        },
        "overlap_diagnostic": {
            "true_positive": overlap.true_positive,
            "precision": overlap.precision,
            "recall": overlap.recall,
            "f1": overlap.f1,
        },
        "geometry_delta": geometry_delta(candidates, reference),
    }


def aggregate_candidate_metrics(per_juan: dict[str, dict]) -> dict:
    reference = sum(row["reference_spans"] for row in per_juan.values())
    candidates = sum(row["candidate_spans"] for row in per_juan.values())
    exact_tp = sum(
        row["exact"]["true_positive"] for row in per_juan.values()
    )
    overlap_tp = sum(
        row["overlap_diagnostic"]["true_positive"]
        for row in per_juan.values()
    )

    def scores(true_positive: int) -> dict:
        precision = true_positive / candidates if candidates else 0.0
        recall = true_positive / reference if reference else 0.0
        total = precision + recall
        return {
            "true_positive": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / total if total else 0.0,
        }

    geometry_fields = (
        "raw_additions",
        "removals",
        "net_growth",
        "geometry_replacements",
        "pure_additions",
        "pure_removals",
    )
    geometry = {
        field: sum(
            row["geometry_delta"][field] for row in per_juan.values()
        )
        for field in geometry_fields
    }
    examples = []
    for juan, row in per_juan.items():
        for example in row["geometry_delta"]["replacement_examples"]:
            examples.append({"juan": int(juan), **example})
    geometry["replacement_examples"] = examples[:50]
    return {
        "reference_spans": reference,
        "candidate_spans": candidates,
        "exact": scores(exact_tp),
        "overlap_diagnostic": scores(overlap_tp),
        "geometry_delta": geometry,
    }


def freeze_round(
    tasks_dir: Path,
    assisted_dir: Path,
    state_dir: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(
            f"frozen round output already exists: {output_dir}"
        )
    manifest_path = tasks_dir / "manifest.json"
    manifest = _read(manifest_path)
    if manifest.get("v1_used") is not False:
        raise ValueError("round manifest must explicitly exclude v1")
    train_examples = []
    dev_examples = []
    inputs = {}
    per_juan = {}
    for selection in manifest["selected"]:
        juan = int(selection["juan"])
        mode = selection["mode"]
        task_path = tasks_dir / f"blind_juan_{juan:03d}.json"
        state_path = state_dir / f"juan_{juan:03d}.json"
        task = _read(task_path)
        state = _read(state_path)
        inputs[str(juan)] = {
            "task_sha256": _sha256(task_path),
            "state_sha256": _sha256(state_path),
            "mode": mode,
        }
        if mode == "blind_anchor":
            if not state["blind"]["complete"]:
                raise ValueError(f"blind anchor {juan} is incomplete")
            dev_examples.extend(build_examples(
                juan,
                task,
                {
                    "role_audit": {
                        "complete": True,
                        "annotations": state["blind"]["annotations"],
                    },
                },
                label_provenance="human_blind_anchor",
            ))
            continue
        if not state["assisted"]["complete"]:
            raise ValueError(f"assisted juan {juan} is incomplete")
        pack_path = assisted_dir / f"assisted_juan_{juan:03d}.json"
        pack_sha256 = _sha256(pack_path)
        if state["assisted"].get("pack_sha256") != pack_sha256:
            raise ValueError(
                f"assisted juan {juan} is not bound to the current pack"
            )
        pack = _read(pack_path)
        validate_copilot_pack(
            pack,
            juan,
            task=task,
            task_sha256=inputs[str(juan)]["task_sha256"],
        )
        annotations = state["assisted"]["annotations"]
        train_examples.extend(build_examples(
            juan,
            task,
            {
                "role_audit": {
                    "complete": True,
                    "annotations": annotations,
                },
            },
            label_provenance="human_assisted_copilot",
        ))
        reference = _spans(annotations)
        candidates = _spans(pack["candidates"])
        per_juan[str(juan)] = candidate_metrics(reference, candidates)
        inputs[str(juan)]["pack_sha256"] = pack_sha256

    if not dev_examples or not train_examples:
        raise ValueError("round requires blind dev and assisted training examples")
    output_dir.mkdir(parents=True)
    _write_jsonl(output_dir / "train_assisted.jsonl", train_examples)
    _write_jsonl(output_dir / "dev_blind_anchor.jsonl", dev_examples)
    report = {
        "schema_version": 1,
        "round_status": "frozen",
        "v1_used": False,
        "blind_anchor_is_training_data": False,
        "assisted_labels_are_dev_data": False,
        "candidate_accuracy": aggregate_candidate_metrics(per_juan),
        "candidate_accuracy_by_juan": per_juan,
        "splits": {
            "assisted_train": {
                "examples": len(train_examples),
                "spans": sum(row["span_count"] for row in train_examples),
            },
            "blind_anchor_dev": {
                "examples": len(dev_examples),
                "spans": sum(row["span_count"] for row in dev_examples),
            },
        },
        "frozen_inputs": inputs,
        "source_manifest_sha256": _sha256(manifest_path),
        "promotion_rule": (
            "promote only if blind-anchor exact F1 improves and no "
            "declared challenge stratum regresses"
        ),
    }
    (output_dir / "round_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze one human-corrected assisted-learning round."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--assisted", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = freeze_round(
        args.tasks, args.assisted, args.state, args.output
    )
    print(json.dumps(report["splits"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
