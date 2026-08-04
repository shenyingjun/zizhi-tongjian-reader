from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tempfile
from pathlib import Path

from p1_audit import audit_split
from p1_dataset import build_examples
from p3_compact import EXPECTED_MODEL_SHA256, _git_commit_clean
from p3_compact_adjudicate import disagreement_rows
from p3_compact_evaluate import (
    _aggregate_audits,
    _counts,
    _read_jsonl,
    _role_metrics,
    _rule_span,
    _selection_roles,
    _span,
    bootstrap_probability,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _geometry(rows: list[dict]) -> set[tuple[int, int, int, str]]:
    return {
        (
            int(row["para_id"]),
            int(row["start"]),
            int(row["end"]),
            str(row["surface"]),
        )
        for row in rows
    }


def _normalized_annotations(
    rows: list[dict],
) -> list[tuple[int, int, int, str]]:
    normalized = [
        (
            int(row["para_id"]),
            int(row["start"]),
            int(row["end"]),
            str(row["surface"]),
        )
        for row in rows
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError("annotations contain duplicate geometry")
    return sorted(normalized)


def _expected_candidates(
    juan: int,
    original_examples: dict[str, dict],
    prediction_by_id: dict[str, dict],
) -> list[dict]:
    expected = []
    for example_id, example in original_examples.items():
        if int(example["juan"]) != juan:
            continue
        for candidate in disagreement_rows(prediction_by_id[example_id]):
            identity = (
                f"{example_id}:{candidate['para_id']}:"
                f"{candidate['start']}:{candidate['end']}"
            )
            candidate["id"] = (
                "adjudication:"
                + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
            )
            candidate["channels"] = ["source_hidden"]
            expected.append(candidate)
    return sorted(expected, key=lambda row: (
        row["para_id"], row["start"], row["end"], row["id"]
    ))


def finalize_adjudication(
    adjudication_dir: Path,
    original_tasks_dir: Path,
    frozen_dir: Path,
    evaluation_dir: Path,
    rules_dir: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"adjudicated output exists: {output_dir}")
    git_commit = _git_commit_clean()
    tasks_dir = adjudication_dir / "tasks"
    recall_dir = adjudication_dir / "recall"
    state_dir = adjudication_dir / "state"
    adjudication_manifest_path = tasks_dir / "manifest.json"
    adjudication_manifest = _read(adjudication_manifest_path)
    original_manifest_path = original_tasks_dir / "manifest.json"
    original_manifest = _read(original_manifest_path)
    freeze_report_path = frozen_dir / "freeze_report.json"
    freeze_report = _read(freeze_report_path)
    original_reference_path = frozen_dir / "reference.jsonl"
    evaluation_report_path = evaluation_dir / "report.json"
    evaluation_report = _read(evaluation_report_path)
    predictions_path = evaluation_dir / "predictions.json"
    predictions = _read(predictions_path)
    if (
        adjudication_manifest.get("status")
        != "post_sealed_source_hidden_adjudication"
        or adjudication_manifest.get("original_v1_immutable") is not True
        or adjudication_manifest.get("inputs", {}).get(
            "source_manifest_sha256"
        ) != _sha256(original_manifest_path)
        or adjudication_manifest.get("inputs", {}).get(
            "freeze_report_sha256"
        ) != _sha256(freeze_report_path)
        or adjudication_manifest.get("inputs", {}).get("reference_sha256")
        != _sha256(original_reference_path)
        or adjudication_manifest.get("inputs", {}).get(
            "evaluation_report_sha256"
        ) != _sha256(evaluation_report_path)
        or adjudication_manifest.get("inputs", {}).get(
            "predictions_sha256"
        ) != _sha256(predictions_path)
        or evaluation_report.get("outputs", {}).get("predictions_sha256")
        != _sha256(predictions_path)
    ):
        raise ValueError("adjudication finalization input binding differs")
    roles = _selection_roles(original_manifest)
    prediction_by_id = {str(row["id"]): row for row in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("predictions duplicate an example ID")
    original_examples = {
        str(row["id"]): row for row in _read_jsonl(original_reference_path)
    }
    if len(original_examples) != 20:
        raise ValueError("original compact reference must have 20 unique IDs")
    selections = adjudication_manifest.get("selected", [])
    selected_juans = [int(row["juan"]) for row in selections]
    if len(selected_juans) != 12 or len(set(selected_juans)) != 12:
        raise ValueError("adjudication must contain 12 unique task juans")
    expected_state_names = {
        f"juan_{juan:03d}.json" for juan in selected_juans
    }
    expected_task_names = {
        str(row["task"]) for row in selections
    }
    expected_pack_names = {
        f"recall_juan_{int(row['juan']):03d}.json"
        for row in selections
    }
    actual_state_names = {
        path.name for path in state_dir.iterdir() if path.is_file()
    }
    actual_task_names = {
        path.name for path in tasks_dir.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    actual_pack_names = {
        path.name for path in recall_dir.iterdir() if path.is_file()
    }
    if actual_state_names != expected_state_names:
        raise ValueError("adjudication state files differ from manifest")
    if actual_task_names != expected_task_names:
        raise ValueError("adjudication task files differ from manifest")
    if actual_pack_names != expected_pack_names:
        raise ValueError("adjudication pack files differ from manifest")
    original_selection = {
        int(row["juan"]): row for row in original_manifest["selected"]
    }
    expected_by_juan = {}
    for juan in {int(row["juan"]) for row in original_examples.values()}:
        candidates = _expected_candidates(
            juan, original_examples, prediction_by_id
        )
        if candidates:
            expected_by_juan[juan] = candidates
    if (
        set(expected_by_juan) != set(selected_juans)
        or sum(len(rows) for rows in expected_by_juan.values()) != 43
        or adjudication_manifest.get("candidate_geometries") != 43
    ):
        raise ValueError(
            "adjudication manifest does not cover every sealed disagreement"
        )

    examples = []
    state_hashes = {}
    decision_counts = {"accept": 0, "reject": 0}
    changed_juans = set()
    for selection in selections:
        juan = int(selection["juan"])
        task_path = tasks_dir / str(selection["task"])
        original_task_path = original_tasks_dir / str(selection["task"])
        pack_path = recall_dir / f"recall_juan_{juan:03d}.json"
        state_path = state_dir / f"juan_{juan:03d}.json"
        if (
            _sha256(task_path) != selection.get("task_sha256")
            or _sha256(pack_path) != selection.get("pack_sha256")
            or _sha256(task_path)
            != original_selection.get(juan, {}).get("task_sha256")
            or _sha256(original_task_path)
            != original_selection.get(juan, {}).get("task_sha256")
            or _sha256(original_task_path) != _sha256(task_path)
        ):
            raise ValueError(f"adjudication task/pack hash differs: {juan}")
        task = _read(task_path)
        pack = _read(pack_path)
        state = _read(state_path)
        expected_candidates = expected_by_juan[juan]
        actual_candidates = sorted(
            pack.get("candidates", []),
            key=lambda row: (
                row["para_id"], row["start"], row["end"], row["id"]
            ),
        )
        if actual_candidates != expected_candidates:
            raise ValueError(
                f"adjudication pack differs from sealed disagreement: {juan}"
            )
        recall = state.get("recall", {})
        if not recall.get("complete"):
            raise ValueError(f"adjudication task is not locked: {juan}")
        candidate_ids = {str(row["id"]) for row in pack["candidates"]}
        decisions = recall.get("decisions", {})
        if set(decisions) != candidate_ids or any(
            value not in {"accept", "reject"} for value in decisions.values()
        ):
            raise ValueError(f"adjudication decisions differ: {juan}")
        annotation_geometry = {
            (
                int(row["para_id"]),
                int(row["start"]),
                int(row["end"]),
            )
            for row in recall["annotations"]
        }
        for candidate in pack["candidates"]:
            decision = decisions[candidate["id"]]
            decision_counts[decision] += 1
            geometry = (
                int(candidate["para_id"]),
                int(candidate["start"]),
                int(candidate["end"]),
            )
            if (decision == "accept") != (geometry in annotation_geometry):
                raise ValueError(
                    f"adjudication annotation/decision differs: {juan}"
                )
        candidate_geometry = {
            (
                int(row["para_id"]),
                int(row["start"]),
                int(row["end"]),
            )
            for row in pack["candidates"]
        }
        frozen_annotations = []
        for example in original_examples.values():
            if int(example["juan"]) != juan:
                continue
            frozen_annotations.extend(build_examples_to_spans(example))
        if _normalized_annotations(
            state["blind"]["annotations"]
        ) != _normalized_annotations(
            frozen_annotations
        ):
            raise ValueError(
                f"adjudication blind copy differs from frozen BIO: {juan}"
            )
        expected_annotations = [
            {
                "para_id": int(row["para_id"]),
                "start": int(row["start"]),
                "end": int(row["end"]),
                "surface": str(row["surface"]),
            }
            for row in frozen_annotations
            if (
                int(row["para_id"]),
                int(row["start"]),
                int(row["end"]),
            ) not in candidate_geometry
        ]
        expected_annotations.extend({
            "para_id": int(candidate["para_id"]),
            "start": int(candidate["start"]),
            "end": int(candidate["end"]),
            "surface": str(candidate["surface"]),
        } for candidate in pack["candidates"]
            if decisions[candidate["id"]] == "accept")
        if _normalized_annotations(
            recall["annotations"]
        ) != _normalized_annotations(
            expected_annotations
        ):
            raise ValueError(
                f"adjudicated annotations are not consensus plus accepts: {juan}"
            )
        built = build_examples(
            juan,
            task,
            {
                "role_audit": {
                    "complete": True,
                    "annotations": recall["annotations"],
                },
            },
            label_provenance="human_post_sealed_adjudicated_compact_p3",
        )
        for example in built:
            key = (juan, int(example["jie_index"]))
            example["evaluation_role"] = roles[key]
            original = original_examples[str(example["id"])]
            if example["labels"] != original["labels"]:
                changed_juans.add(juan)
        examples.extend(built)
        state_hashes[str(juan)] = _sha256(state_path)
    unchanged_examples = [
        row for row in original_examples.values()
        if int(row["juan"]) not in set(selected_juans)
    ]
    if len(unchanged_examples) != 8:
        raise ValueError("adjudication must preserve exactly 8 unchanged jies")
    examples.extend(unchanged_examples)
    examples.sort(key=lambda row: (int(row["juan"]), int(row["jie_index"])))
    output_ids = [str(row["id"]) for row in examples]
    if (
        len(examples) != 20
        or len(set(output_ids)) != 20
        or set(output_ids) != set(original_examples)
    ):
        raise ValueError("adjudicated examples differ from original IDs")

    rule_documents = {}
    rule_hashes = {}
    per_jie = []
    per_jie_audits = []
    for example in examples:
        example_id = str(example["id"])
        prediction = prediction_by_id[example_id]
        juan = int(example["juan"])
        if juan not in rule_documents:
            rule_path = rules_dir / f"juan_{juan:03d}.json"
            rule_documents[juan] = _read(rule_path)
            rule_hashes[str(juan)] = _sha256(rule_path)
            if rule_hashes[str(juan)] != (
                evaluation_report.get("inputs", {})
                .get("rule_sha256_by_juan", {})
                .get(str(juan))
            ):
                raise ValueError(f"sealed rule hash differs: {juan}")
        text = str(example["text"])
        paragraphs = {}
        para_ids = set()
        for segment in example["segments"]:
            para_id = int(segment["para_id"])
            start = int(segment["assembled_start"])
            end = int(segment["assembled_end"])
            paragraphs[para_id] = text[start:end]
            para_ids.add(para_id)
        reference = {
            _span(row)
            for row in build_examples_to_spans(example)
        }
        model = {
            _span(row) for row in prediction["prediction_spans"]
        }
        rules = {
            _rule_span(row, paragraphs)
            for row in rule_documents[juan].get("occurrences", [])
            if row.get("field", "main") == "main"
            and int(row["para_id"]) in para_ids
        }
        normalized_rules = [{
            "para_id": span.para_id,
            "start": span.start,
            "end": span.end,
            "surface": span.surface,
            "field": "main",
        } for span in sorted(rules)]
        reference_rows = [span.__dict__ for span in sorted(reference)]
        adjudicated_prediction = {
            "id": example_id,
            "reference_spans": reference_rows,
            "prediction_spans": prediction["prediction_spans"],
        }
        per_jie.append({
            "id": example_id,
            "role": str(example["evaluation_role"]),
            "characters": len(text),
            "model": _counts(reference, model),
            "rules": _counts(reference, rules),
        })
        per_jie_audits.append({
            "id": example_id,
            "role": str(example["evaluation_role"]),
            "audit": audit_split(
                [example],
                [adjudicated_prediction],
                normalized_rules,
                paragraphs,
            ),
        })
    role_metrics = _role_metrics(per_jie)
    probability_rows = [
        row for row in per_jie if row["role"] == "probability_random"
    ]
    report = {
        "schema_version": 1,
        "status": "post_sealed_adjudicated_compact_p3_v2",
        "formal_p3": False,
        "original_v1_preserved": True,
        "claim_limit": (
            "post-sealed adjudicated sensitivity analysis; not an "
            "independent promotion metric"
        ),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "git_commit": git_commit,
        "jies": len(examples),
        "characters": sum(len(row["text"]) for row in examples),
        "spans": sum(int(row["span_count"]) for row in examples),
        "changed_juans": sorted(changed_juans),
        "decisions": decision_counts,
        "probability_metrics": role_metrics["probability_random"],
        "challenge_metrics": {
            role: role_metrics[role]
            for role in (
                "role_appellation_challenge",
                "foreign_title_challenge",
            )
        },
        "bootstrap": bootstrap_probability(probability_rows),
        "audit_summary": _aggregate_audits(per_jie_audits),
        "inputs": {
            "adjudication_manifest_sha256": _sha256(
                adjudication_manifest_path
            ),
            "original_manifest_sha256": _sha256(original_manifest_path),
            "original_reference_sha256": _sha256(
                original_reference_path
            ),
            "evaluation_report_sha256": _sha256(
                evaluation_report_path
            ),
            "predictions_sha256": _sha256(predictions_path),
            "state_sha256_by_juan": state_hashes,
            "rule_sha256_by_juan": rule_hashes,
        },
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        reference_v2_path = staging / "reference_v2.jsonl"
        _write_jsonl(reference_v2_path, examples)
        report["reference_v2_sha256"] = _sha256(reference_v2_path)
        _write(staging / "report.json", report)
        _write(staging / "per_jie_metrics.json", per_jie)
        _write(staging / "geometry_audit.json", {
            "schema_version": 1,
            "identity": (
                "(juan, jie_index, para_id, start, end); matches performed "
                "within jie"
            ),
            "per_jie": per_jie_audits,
        })
        for path in staging.iterdir():
            path.chmod(stat.S_IREAD)
            if path.stat().st_mode & (
                stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
            ):
                raise PermissionError(f"failed to freeze v2 output: {path}")
        staging.replace(output_dir)
    return report


def build_examples_to_spans(example: dict) -> list[dict]:
    from p1_windows import labels_to_spans

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze and score compact P3 adjudicated v2."
    )
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--original-tasks", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = finalize_adjudication(
        args.adjudication,
        args.original_tasks,
        args.frozen,
        args.evaluation,
        args.rules,
        args.output,
    )
    print(json.dumps({
        "status": report["status"],
        "spans": report["spans"],
        "changed_juans": report["changed_juans"],
        "decisions": report["decisions"],
        "probability_metrics": report["probability_metrics"],
        "challenge_metrics": report["challenge_metrics"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
