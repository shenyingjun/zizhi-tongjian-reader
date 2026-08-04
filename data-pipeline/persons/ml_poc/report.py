from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from core import Span, score_spans
from pilot import REPO, RULES, V1, _load


def _annotation_spans(rows: list[dict]) -> list[Span]:
    return [
        Span(
            int(row["para_id"]),
            int(row["start"]),
            int(row["end"]),
            str(row["surface"]),
        )
        for row in rows
    ]


def _rule_spans(juan: int) -> list[Span]:
    return [
        Span(
            int(row["para_id"]),
            int(row["start"]),
            int(row["end"]),
            str(row["surface"]),
        )
        for row in _load(RULES / f"juan_{juan:03d}.json")["occurrences"]
        if row.get("field", "main") == "main"
    ]


def _v1_spans(juan: int) -> list[Span]:
    return [
        Span(
            int(row["pid"]),
            int(row["start"]),
            int(row["end"]),
            str(row["surface"]),
        )
        for row in _load(V1 / f"juan_{juan:03d}.json")["mentions"]
        if row.get("source", "main") == "main"
    ]


def _metric_payload(reference: list[Span], predictions: list[Span]) -> dict:
    exact = score_spans(reference, predictions)
    overlap = score_spans(reference, predictions, overlap=True)
    return {
        "prediction_spans": len(predictions),
        "reference_spans": len(reference),
        "exact": {
            "true_positive": exact.true_positive,
            "precision": round(exact.precision, 6),
            "recall": round(exact.recall, 6),
            "f1": round(exact.f1, 6),
        },
        "overlap_diagnostic": {
            "true_positive": overlap.true_positive,
            "precision": round(overlap.precision, 6),
            "recall": round(overlap.recall, 6),
            "f1": round(overlap.f1, 6),
        },
    }


def _overlap(left: Span, right: Span) -> bool:
    return (
        left.para_id == right.para_id
        and left.start < right.end
        and right.start < left.end
    )


def geometry_delta(before: list[Span], after: list[Span]) -> dict:
    before_set = set(before)
    after_set = set(after)
    additions = sorted(after_set - before_set)
    removals = sorted(before_set - after_set)
    edges = [
        [index for index, addition in enumerate(additions)
         if _overlap(removal, addition)]
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
        {
            "before": removals[removal_index].__dict__,
            "after": additions[addition_index].__dict__,
        }
        for addition_index, removal_index in sorted(assigned.items())
    ]
    paired_removal_indexes = set(assigned.values())
    paired_addition_indexes = set(assigned)
    return {
        "raw_additions": len(additions),
        "removals": len(removals),
        "net_growth": len(additions) - len(removals),
        "geometry_replacements": len(replacements),
        "pure_additions": len(additions) - len(paired_addition_indexes),
        "pure_removals": len(removals) - len(paired_removal_indexes),
        "replacement_examples": replacements[:50],
    }


def build_report(juan: int, state_path: Path, recall_path: Path) -> dict:
    state = _load(state_path)
    if not state["blind"]["complete"] or not state["recall"]["complete"]:
        raise ValueError("both annotation phases must be complete")
    blind = _annotation_spans(state["blind"]["annotations"])
    recall_reference = _annotation_spans(state["recall"]["annotations"])
    role_audit = state.get("role_audit")
    if role_audit and role_audit.get("initialized"):
        if not role_audit.get("complete"):
            raise ValueError("role audit is initialized but incomplete")
        reference = _annotation_spans(role_audit["annotations"])
    else:
        reference = recall_reference
    recall = _load(recall_path)
    decisions = state["recall"]["decisions"]
    if len(decisions) != len(recall["candidates"]):
        raise ValueError("recall candidate decisions are incomplete")

    by_decision = Counter(decisions.values())
    by_channel_and_decision = Counter()
    for candidate in recall["candidates"]:
        decision = decisions[candidate["id"]]
        channel_key = "+".join(candidate["channels"])
        by_channel_and_decision[(channel_key, decision)] += 1

    return {
        "schema_version": 1,
        "juan": juan,
        "reference": (
            "human blind pass corrected by candidate-union recall pass"
            " and specific-role audit"
            if role_audit and role_audit.get("complete")
            else "human blind pass corrected by candidate-union recall pass"
        ),
        "boundary_policy": (
            "explicit name core; role-only references require a same-jie anchor"
        ),
        "annotation": {
            "blind_spans": len(blind),
            "recall_spans": len(recall_reference),
            "final_spans": len(reference),
            "candidate_decisions": dict(sorted(by_decision.items())),
            "note_evidence_reviewed": len(recall["note_evidence"]),
            "blind_to_final_delta": geometry_delta(blind, reference),
            "recall_to_role_audit_delta": (
                geometry_delta(recall_reference, reference)
                if role_audit and role_audit.get("complete")
                else None
            ),
            "role_audit_decisions": (
                dict(sorted(Counter(role_audit["decisions"].values()).items()))
                if role_audit and role_audit.get("complete")
                else {}
            ),
        },
        "systems": {
            "rules": _metric_payload(reference, _rule_spans(juan)),
            "v1_diagnostic": _metric_payload(reference, _v1_spans(juan)),
        },
        "candidate_channels": [
            {
                "channels": channels,
                "decision": decision,
                "count": count,
            }
            for (channels, decision), count
            in sorted(by_channel_and_decision.items())
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a reproducible P0 annotation and baseline report."
    )
    parser.add_argument("--juan", type=int, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--recall", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.juan, args.state, args.recall)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
