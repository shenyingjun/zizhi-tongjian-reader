from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tempfile
from collections import Counter
from pathlib import Path

from core import Span, score_spans
from p1_dataset import build_examples
from p1_train import evaluate
from p2_round import _spans
from p3_active_finalize import _aggregate_deltas, _inventory, _validate_decisions
from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from report import geometry_delta


EXPECTED_REVIEW_MANIFEST_SHA256 = (
    "14796c5c3e02ae1955c680b578c3e7bc8e3e65f0c90e02aaa1a211f2bd53244a"
)
PROVENANCE = "copilot_double_pass_blind_diagnostic"
ROLE_COUNTS = {
    "probability_random": 12,
    "role_appellation_challenge": 4,
    "foreign_title_challenge": 4,
}


def _snapshot(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
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


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _publish(staging: Path, output_dir: Path) -> None:
    for path in staging.iterdir():
        if path.is_file():
            path.chmod(stat.S_IREAD)
    staging.replace(output_dir)


def _roles(source_manifest: dict) -> dict[tuple[int, int], str]:
    roles = {}
    counts = Counter()
    for row in source_manifest.get("private_selected_jies", []):
        key = int(row["juan"]), int(row["jie_index"])
        role = str(row["role"])
        if key in roles or role not in ROLE_COUNTS:
            raise ValueError("locked diagnostic role inventory is invalid")
        roles[key] = role
        counts[role] += 1
    if dict(counts) != ROLE_COUNTS:
        raise ValueError(f"locked diagnostic role counts differ: {dict(counts)}")
    return roles


def freeze_reference(
    review_dir: Path,
    state_dir: Path,
    source_manifest_path: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"locked diagnostic freeze exists: {output_dir}")
    tasks_dir = review_dir / "tasks"
    assisted_dir = review_dir / "assisted"
    review_manifest_path = tasks_dir / "manifest.json"
    review_manifest, review_manifest_sha256 = _snapshot(review_manifest_path)
    source_manifest, source_manifest_sha256 = _snapshot(source_manifest_path)
    selections = review_manifest.get("selected", [])
    juans = [int(row["juan"]) for row in selections]
    if (
        review_manifest_sha256 != EXPECTED_REVIEW_MANIFEST_SHA256
        or review_manifest.get("status")
        != "candidate_blind_copilot_double_pass_focused_review"
        or review_manifest.get("formal_evaluation") is not False
        or review_manifest.get("eligible_for_promotion") is not False
        or review_manifest.get("candidate_model_blind") is not True
        or review_manifest.get("reference_locked") is not False
        or review_manifest.get("provenance") != PROVENANCE
        or review_manifest.get("source_manifest_sha256")
        != source_manifest_sha256
        or source_manifest.get("model_predictions_generated") is not False
        or source_manifest.get("rules_generated") is not False
        or len(juans) != 19
        or len(set(juans)) != 19
        or any(row.get("mode") != "diagnostic_assisted" for row in selections)
    ):
        raise ValueError("locked diagnostic manifest binding differs")
    roles = _roles(source_manifest)
    expected_tasks = {str(row["task"]) for row in selections} | {"manifest.json"}
    expected_packs = {f"assisted_juan_{juan:03d}.json" for juan in juans}
    expected_states = {f"juan_{juan:03d}.json" for juan in juans}
    if (
        _inventory(tasks_dir) != expected_tasks
        or _inventory(assisted_dir) != expected_packs
        or _inventory(state_dir) != expected_states
    ):
        raise ValueError("locked diagnostic review inventory differs")

    examples = []
    attribution = {}
    inputs = {}
    decisions = Counter()
    review_decisions = Counter()
    delta_by_juan = {}
    seen_jies = set()
    for selection in selections:
        juan = int(selection["juan"])
        task_path = tasks_dir / str(selection["task"])
        pack_path = assisted_dir / f"assisted_juan_{juan:03d}.json"
        state_path = state_dir / f"juan_{juan:03d}.json"
        task, task_sha256 = _snapshot(task_path)
        pack, pack_sha256 = _snapshot(pack_path)
        state, state_sha256 = _snapshot(state_path)
        if (
            selection.get("task") != f"blind_juan_{juan:03d}.json"
            or task_sha256 != selection.get("task_sha256")
            or pack_sha256 != selection.get("pack_sha256")
            or task.get("juan") != juan
            or pack.get("juan") != juan
            or pack.get("diagnostic_only") is not True
            or pack.get("candidate_model_blind") is not True
        ):
            raise ValueError(f"locked diagnostic input binding differs: {juan}")
        assisted = state.get("assisted", {})
        if (
            assisted.get("complete") is not True
            or assisted.get("pack_sha256") != pack_sha256
        ):
            raise ValueError(f"locked diagnostic state is incomplete: {juan}")
        decisions.update(_validate_decisions(juan, task, pack, assisted))
        candidate_by_geometry = {
            (
                int(row["para_id"]),
                int(row["start"]),
                int(row["end"]),
            ): row
            for row in pack["candidates"]
        }
        for candidate in pack["candidates"]:
            if candidate.get("confidence") == "low":
                review_decisions[assisted["decisions"][candidate["id"]]] += 1
        built = build_examples(
            juan,
            task,
            {
                "role_audit": {
                    "complete": True,
                    "annotations": assisted["annotations"],
                },
            },
            label_provenance=PROVENANCE,
        )
        for example in built:
            key = juan, int(example["jie_index"])
            if key not in roles or key in seen_jies:
                raise ValueError(f"locked diagnostic jie differs: {key}")
            seen_jies.add(key)
            example["evaluation_role"] = roles[key]
            span_sources = []
            para_ids = {
                int(segment["para_id"]) for segment in example["segments"]
            }
            for annotation in assisted["annotations"]:
                if int(annotation["para_id"]) not in para_ids:
                    continue
                geometry = (
                    int(annotation["para_id"]),
                    int(annotation["start"]),
                    int(annotation["end"]),
                )
                candidate = candidate_by_geometry.get(geometry)
                source = (
                    "manual_addition"
                    if candidate is None
                    else (
                        "focused_review_accepted"
                        if candidate.get("confidence") == "low"
                        else "auto_consensus"
                    )
                )
                span_sources.append({
                    "para_id": geometry[0],
                    "start": geometry[1],
                    "end": geometry[2],
                    "surface": str(annotation["surface"]),
                    "source": source,
                })
            attribution[str(example["id"])] = sorted(
                span_sources,
                key=lambda row: (row["para_id"], row["start"], row["end"]),
            )
        examples.extend(built)
        delta_by_juan[str(juan)] = geometry_delta(
            _spans(pack.get("initial_annotations", [])),
            _spans(assisted["annotations"]),
        )
        inputs[str(juan)] = {
            "task_sha256": task_sha256,
            "pack_sha256": pack_sha256,
            "state_sha256": state_sha256,
        }
    if seen_jies != set(roles):
        raise ValueError("not every locked diagnostic jie was frozen")
    examples.sort(key=lambda row: (int(row["juan"]), int(row["jie_index"])))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        reference_path = staging / "reference.jsonl"
        attribution_path = staging / "reference_attribution.json"
        _write_jsonl(reference_path, examples)
        _write_json(attribution_path, attribution)
        report = {
            "schema_version": 1,
            "status": "frozen_copilot_double_pass_blind_diagnostic",
            "formal_evaluation": False,
            "eligible_for_promotion": False,
            "candidate_model_blind": True,
            "reference_locked": True,
            "provenance": PROVENANCE,
            "git_commit": _git_commit_clean(),
            "jies": len(examples),
            "characters": sum(len(row["text"]) for row in examples),
            "spans": sum(int(row["span_count"]) for row in examples),
            "roles": dict(Counter(row["evaluation_role"] for row in examples)),
            "candidate_decisions": dict(decisions),
            "focused_review_decisions": dict(review_decisions),
            "initial_to_final_geometry": _aggregate_deltas(delta_by_juan),
            "initial_to_final_geometry_by_juan": delta_by_juan,
            "frozen_inputs": inputs,
            "source_manifest_sha256": source_manifest_sha256,
            "review_manifest_sha256": review_manifest_sha256,
            "frozen_candidates": source_manifest["frozen_candidates"],
            "outputs": {
                "reference_sha256": _sha256(reference_path),
                "reference_attribution_sha256": _sha256(attribution_path),
            },
        }
        _write_json(staging / "freeze_report.json", report)
        _publish(staging, output_dir)
    return report


def _span(row: dict) -> Span:
    return Span(
        int(row["para_id"]),
        int(row["start"]),
        int(row["end"]),
        str(row["surface"]),
    )


def _metrics(rows: list[tuple[list[Span], list[Span]]]) -> dict:
    reference = predicted = true_positive = 0
    for expected, actual in rows:
        score = score_spans(expected, actual)
        reference += score.reference
        predicted += score.predicted
        true_positive += score.true_positive
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / reference if reference else 0.0
    return {
        "reference_spans": reference,
        "prediction_spans": predicted,
        "true_positive": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        ),
    }


def compare_models(
    frozen_dir: Path,
    model_roots: dict[str, Path],
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"locked diagnostic comparison exists: {output_dir}")
    freeze_report_path = frozen_dir / "freeze_report.json"
    reference_path = frozen_dir / "reference.jsonl"
    attribution_path = frozen_dir / "reference_attribution.json"
    freeze_report, freeze_report_sha256 = _snapshot(freeze_report_path)
    if (
        freeze_report.get("status")
        != "frozen_copilot_double_pass_blind_diagnostic"
        or freeze_report.get("formal_evaluation") is not False
        or freeze_report.get("eligible_for_promotion") is not False
        or freeze_report.get("reference_locked") is not True
        or freeze_report.get("outputs", {}).get("reference_sha256")
        != _sha256(reference_path)
        or freeze_report.get("outputs", {}).get("reference_attribution_sha256")
        != _sha256(attribution_path)
    ):
        raise ValueError("locked diagnostic freeze binding differs")
    expected = {
        str(row["label"]): row for row in freeze_report["frozen_candidates"]
    }
    if set(model_roots) != set(expected):
        raise ValueError("comparison model labels differ from frozen candidates")

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    examples = _read_jsonl(reference_path)
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions_by_model = {}
    model_inputs = {}
    for label, root in model_roots.items():
        candidate = expected[label]
        artifact = _model_artifact(root / "model")
        report_sha256 = _sha256(root / "report.json")
        if (
            artifact["combined_sha256"] != candidate["artifact_sha256"]
            or report_sha256 != candidate["report_sha256"]
        ):
            raise ValueError(f"frozen candidate artifact differs: {label}")
        tokenizer = AutoTokenizer.from_pretrained(root / "model", use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(root / "model")
        model.to(device)
        _, predictions = evaluate(
            model,
            tokenizer,
            examples,
            device,
            max_length=512,
            stride=128,
            batch_size=8,
        )
        predictions_by_model[label] = {
            str(row["id"]): row for row in predictions
        }
        if len(predictions_by_model[label]) != len(examples):
            raise ValueError(f"prediction inventory differs: {label}")
        model_inputs[label] = {
            "artifact_sha256": artifact["combined_sha256"],
            "report_sha256": report_sha256,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    before_label = "round3"
    after_label = "round6_seed_20260727"
    labels = [before_label, after_label]
    metric_rows = {label: [] for label in labels}
    role_rows = {
        label: {role: [] for role in ROLE_COUNTS} for label in labels
    }
    attribution_hits = {
        label: Counter() for label in labels
    }
    attribution_totals = Counter()
    changes = []
    totals = Counter()
    replacement_total = 0
    for example in examples:
        example_id = str(example["id"])
        reference = [_span(row) for row in
                     predictions_by_model[before_label][example_id]["reference_spans"]]
        predicted = {
            label: [
                _span(row) for row in
                predictions_by_model[label][example_id]["prediction_spans"]
            ]
            for label in labels
        }
        role = str(example["evaluation_role"])
        for label in labels:
            pair = reference, predicted[label]
            metric_rows[label].append(pair)
            role_rows[label][role].append(pair)
        source_by_span = {
            _span(row): str(row["source"])
            for row in attribution[example_id]
        }
        for span, source in source_by_span.items():
            attribution_totals[source] += 1
            for label in labels:
                if span in set(predicted[label]):
                    attribution_hits[label][source] += 1

        reference_set = set(reference)
        before = set(predicted[before_label])
        after = set(predicted[after_label])
        additions = sorted(after - before)
        removals = sorted(before - after)
        recovered = sorted((after - before) & reference_set)
        regressed = sorted((before - after) & reference_set)
        added_fp = sorted((after - before) - reference_set)
        removed_fp = sorted((before - after) - reference_set)
        delta = geometry_delta(list(before), list(after))
        replacement_total += int(delta["geometry_replacements"])
        totals.update({
            "raw_additions": len(additions),
            "removals": len(removals),
            "net_growth": len(additions) - len(removals),
            "reference_recoveries": len(recovered),
            "reference_regressions": len(regressed),
            "added_false_positives": len(added_fp),
            "removed_false_positives": len(removed_fp),
        })
        if additions or removals:
            changes.append({
                "id": example_id,
                "role": role,
                "additions": [row.__dict__ for row in additions],
                "removals": [row.__dict__ for row in removals],
                "reference_recoveries": [row.__dict__ for row in recovered],
                "reference_regressions": [row.__dict__ for row in regressed],
                "added_false_positives": [row.__dict__ for row in added_fp],
                "removed_false_positives": [row.__dict__ for row in removed_fp],
                "geometry_replacements": delta["replacement_examples"],
            })
    totals["geometry_replacements"] = replacement_total
    report = {
        "schema_version": 1,
        "status": "locked_copilot_assisted_diagnostic_compared",
        "formal_evaluation": False,
        "eligible_for_promotion": False,
        "provenance": PROVENANCE,
        "claim_limit": (
            "locked candidate-model-blind Copilot-assisted diagnostic only; "
            "not formal promotion evidence"
        ),
        "device": str(device),
        "git_commit": _git_commit_clean(),
        "inputs": {
            "freeze_report_sha256": freeze_report_sha256,
            "reference_sha256": _sha256(reference_path),
            "models": model_inputs,
        },
        "metrics": {
            label: {
                "all": _metrics(metric_rows[label]),
                "by_role": {
                    role: _metrics(role_rows[label][role])
                    for role in ROLE_COUNTS
                },
            }
            for label in labels
        },
        "reference_recall_by_label_source": {
            label: {
                source: {
                    "reference_spans": total,
                    "true_positive": attribution_hits[label][source],
                    "recall": (
                        attribution_hits[label][source] / total if total else 0.0
                    ),
                }
                for source, total in sorted(attribution_totals.items())
            }
            for label in labels
        },
        "comparison": {
            "before": before_label,
            "after": after_label,
            **dict(totals),
            "changed_jies": len(changes),
        },
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        predictions_path = staging / "predictions.json"
        changes_path = staging / "geometry_changes.json"
        _write_json(predictions_path, predictions_by_model)
        _write_json(changes_path, changes)
        report["outputs"] = {
            "predictions_sha256": _sha256(predictions_path),
            "geometry_changes_sha256": _sha256(changes_path),
        }
        _write_json(staging / "report.json", report)
        _publish(staging, output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze and compare the locked assisted P3 diagnostic."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--review", type=Path, required=True)
    freeze_parser.add_argument("--state", type=Path, required=True)
    freeze_parser.add_argument("--source-manifest", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--frozen", type=Path, required=True)
    compare_parser.add_argument("--round3", type=Path, required=True)
    compare_parser.add_argument("--round6", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        report = freeze_reference(
            args.review, args.state, args.source_manifest, args.output
        )
        result = {
            key: report[key]
            for key in (
                "status", "jies", "characters", "spans", "roles",
                "focused_review_decisions", "initial_to_final_geometry",
            )
        }
    else:
        report = compare_models(
            args.frozen,
            {
                "round3": args.round3,
                "round6_seed_20260727": args.round6,
            },
            args.output,
        )
        result = {
            "status": report["status"],
            "metrics": report["metrics"],
            "comparison": report["comparison"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
