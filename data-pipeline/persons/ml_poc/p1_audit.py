from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from core import Span, score_spans


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _span(row: dict) -> Span:
    return Span(
        int(row["para_id"]),
        int(row["start"]),
        int(row["end"]),
        str(row["surface"]),
    )


def _overlaps(left: Span, right: Span) -> bool:
    return (
        left.para_id == right.para_id
        and left.start < right.end
        and right.start < left.end
    )


def _metric(reference: set[Span], prediction: set[Span]) -> dict:
    exact = score_spans(reference, prediction)
    overlap = score_spans(reference, prediction, overlap=True)
    return {
        "reference_spans": exact.reference,
        "prediction_spans": exact.predicted,
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
    }


def _geometry_partition(
    before: set[Span],
    after: set[Span],
) -> tuple[list[tuple[Span, Span]], set[Span], set[Span]]:
    removals = sorted(before - after)
    additions = sorted(after - before)
    edges = [
        [
            addition_index
            for addition_index, addition in enumerate(additions)
            if _overlaps(removal, addition)
        ]
        for removal in removals
    ]
    assigned: dict[int, int] = {}

    def augment(removal_index: int, visited: set[int]) -> bool:
        for addition_index in edges[removal_index]:
            if addition_index in visited:
                continue
            visited.add(addition_index)
            previous = assigned.get(addition_index)
            if previous is None or augment(previous, visited):
                assigned[addition_index] = removal_index
                return True
        return False

    for removal_index in range(len(removals)):
        augment(removal_index, set())
    replacements = [
        (removals[removal_index], additions[addition_index])
        for addition_index, removal_index in sorted(assigned.items())
    ]
    paired_removals = {before_span for before_span, _ in replacements}
    paired_additions = {after_span for _, after_span in replacements}
    return (
        replacements,
        set(additions) - paired_additions,
        set(removals) - paired_removals,
    )


def _paragraphs(blind_task: dict) -> dict[int, str]:
    paragraphs = {}
    for jie in blind_task["jies"]:
        text = str(jie["text"])
        for segment in jie["segments"]:
            para_id = int(segment["para_id"])
            start = int(segment["assembled_start"])
            end = int(segment["assembled_end"])
            paragraph = text[start:end]
            previous = paragraphs.setdefault(para_id, paragraph)
            if previous != paragraph:
                raise ValueError(f"conflicting text for paragraph {para_id}")
    return paragraphs


def _case(
    span: Span,
    paragraphs: dict[int, str],
    overlaps: list[Span],
) -> dict:
    paragraph = paragraphs[span.para_id]
    if paragraph[span.start:span.end] != span.surface:
        raise ValueError(f"surface mismatch: {span}")
    context_start = max(0, span.start - 20)
    context_end = min(len(paragraph), span.end + 20)
    return {
        **span.__dict__,
        "context": paragraph[context_start:context_end],
        "context_span": [
            span.start - context_start,
            span.end - context_start,
        ],
        "overlaps": [row.__dict__ for row in sorted(overlaps)],
    }


def audit_split(
    examples: list[dict],
    predictions: list[dict],
    rule_rows: list[dict],
    paragraphs: dict[int, str],
) -> dict:
    example_ids = {str(row["id"]) for row in examples}
    prediction_ids = {str(row["id"]) for row in predictions}
    if prediction_ids != example_ids:
        raise ValueError("prediction IDs do not match dataset split")

    para_ids = {
        int(segment["para_id"])
        for example in examples
        for segment in example["segments"]
    }
    reference = {
        _span(row)
        for prediction in predictions
        for row in prediction["reference_spans"]
    }
    model = {
        _span(row)
        for prediction in predictions
        for row in prediction["prediction_spans"]
    }
    rules = {
        _span(row)
        for row in rule_rows
        if row.get("field", "main") == "main"
        and int(row["para_id"]) in para_ids
    }

    rule_omissions = reference - rules
    recovered_omissions = rule_omissions & model
    rule_true_positives = reference & rules
    rule_regressions = rule_true_positives - model
    model_non_reference = model - reference
    model_misses = reference - model
    replacements, model_false_positives, pure_misses = _geometry_partition(
        reference, model
    )
    model_geometry = {addition for _, addition in replacements}

    def cases(spans: set[Span], overlap_pool: set[Span]) -> list[dict]:
        return [
            _case(
                row,
                paragraphs,
                [target for target in overlap_pool if _overlaps(row, target)],
            )
            for row in sorted(spans)
        ]

    return {
        "metrics": {
            "model": _metric(reference, model),
            "rules": _metric(reference, rules),
        },
        "gate": {
            "rule_omissions": len(rule_omissions),
            "recovered_rule_omissions": len(recovered_omissions),
            "rule_omission_recovery_rate": (
                len(recovered_omissions) / len(rule_omissions)
                if rule_omissions else 1.0
            ),
            "rule_true_positives": len(rule_true_positives),
            "rule_true_positive_regressions": len(rule_regressions),
        },
        "delta": {
            "model_prediction_spans": len(model),
            "rule_prediction_spans": len(rules),
            "raw_model_additions_vs_rules": len(model - rules),
            "model_removals_vs_rules": len(rules - model),
            "net_growth_vs_rules": len(model) - len(rules),
            "model_non_reference_geometries": len(model_non_reference),
            "geometry_replacements": len(replacements),
            "pure_false_positives": len(model_false_positives),
            "model_reference_misses": len(model_misses),
            "pure_misses": len(pure_misses),
        },
        "groups": {
            "pure_false_positive_surfaces": dict(sorted(Counter(
                row.surface for row in model_false_positives
            ).items(), key=lambda item: (-item[1], item[0]))),
            "pure_miss_surfaces": dict(sorted(Counter(
                row.surface for row in pure_misses
            ).items(), key=lambda item: (-item[1], item[0]))),
        },
        "cases": {
            "recovered_rule_omissions": cases(recovered_omissions, rules),
            "rule_true_positive_regressions": cases(rule_regressions, model),
            "model_geometry_replacements": cases(model_geometry, reference),
            "model_pure_false_positives": cases(model_false_positives, reference),
            "model_pure_misses": cases(pure_misses, model),
        },
    }


def build_audit(
    dataset_dir: Path,
    predictions_dir: Path,
    rules_dir: Path,
    blind_dir: Path,
) -> dict:
    split_specs = {
        "challenge_dev": ("dev.jsonl", "dev_predictions.json", 27),
        "random_pilot_holdout": (
            "pilot_holdout.jsonl",
            "holdout_predictions.json",
            52,
        ),
    }
    splits = {}
    for name, (dataset_name, prediction_name, juan) in split_specs.items():
        blind_task = _read_json(blind_dir / f"blind_juan_{juan:03d}.json")
        rules = _read_json(rules_dir / f"juan_{juan:03d}.json")
        splits[name] = audit_split(
            _read_jsonl(dataset_dir / dataset_name),
            _read_json(predictions_dir / prediction_name),
            rules["occurrences"],
            _paragraphs(blind_task),
        )
    return {
        "schema_version": 1,
        "claim_limit": "pilot evidence only; holdout is not a sealed test",
        "splits": splits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the P1 challenger against human references and rules."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--blind-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_audit(
        args.dataset,
        args.predictions,
        args.rules,
        args.blind_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
