from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tempfile
from collections import Counter
from pathlib import Path

from core import Span
from p3_compact import _git_commit_clean
from report import geometry_delta


EXPECTED_ROUND2 = {
    "report.json": "e5bf3104adf0016331eb5c9a061844f7afa4ead8895f95313579c1261f182e50",
    "dev_predictions.json": "b0d9f913db3cb372f526bedec420433a2c7efa7e5ffc884d6073efd5b7f57040",
    "evaluation_predictions.json": "6a647a41c0fdec30671db4bb357500a78ba103b2e08eb61d8d3eae761376bc0a",
}
CHALLENGE_TERMS = ("可汗", "叶护", "特勒", "皇后", "太后", "公主")


def _snapshot(path: Path) -> tuple[bytes, str]:
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def _span_map(rows: list[dict], source: str) -> dict[tuple[int, int, int], str]:
    result = {}
    for row in rows:
        geometry = (
            int(row["para_id"]), int(row["start"]), int(row["end"]),
        )
        if geometry in result:
            raise ValueError(f"duplicate geometry in {source}: {geometry}")
        result[geometry] = str(row["surface"])
    return result


def _geometry_spans(
    geometries: set[tuple[int, int, int]],
) -> list[Span]:
    return [Span(para_id, start, end, "") for para_id, start, end in geometries]


def _metrics(rows: list[dict]) -> dict:
    reference_count = prediction_count = true_positive = 0
    for row in rows:
        identity = str(row["id"])
        reference = set(_span_map(
            row["reference_spans"], f"{identity} reference"
        ))
        prediction = set(_span_map(
            row["prediction_spans"], f"{identity} prediction"
        ))
        reference_count += len(reference)
        prediction_count += len(prediction)
        true_positive += len(reference & prediction)
    precision = true_positive / prediction_count if prediction_count else 0.0
    recall = true_positive / reference_count if reference_count else 0.0
    return {
        "reference_spans": reference_count,
        "prediction_spans": prediction_count,
        "exact": {
            "true_positive": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            ),
        },
    }


def _require_metrics(actual: dict, reported: dict, source: str) -> None:
    for field in ("reference_spans", "prediction_spans"):
        if actual[field] != reported.get(field):
            raise ValueError(f"{source} reported {field} differs")
    for field in ("true_positive", "precision", "recall", "f1"):
        if abs(actual["exact"][field] - reported.get("exact", {}).get(
            field, float("inf")
        )) > 1e-12:
            raise ValueError(f"{source} reported exact {field} differs")


def _aggregate_deltas(rows: list[tuple[str, dict]]) -> dict:
    fields = (
        "raw_additions",
        "removals",
        "net_growth",
        "geometry_replacements",
        "pure_additions",
        "pure_removals",
    )
    result = {
        field: sum(delta[field] for _, delta in rows)
        for field in fields
    }
    result["replacement_examples"] = [
        {"id": identity, **example}
        for identity, delta in rows
        for example in delta["replacement_examples"]
    ][:100]
    return result


def compare_predictions(old_rows: list[dict], new_rows: list[dict]) -> dict:
    if any(type(row.get("id")) is not str for row in old_rows + new_rows):
        raise ValueError("comparison prediction IDs must be strings")
    old_by_id = {row["id"]: row for row in old_rows}
    new_by_id = {row["id"]: row for row in new_rows}
    if (
        len(old_by_id) != len(old_rows)
        or len(new_by_id) != len(new_rows)
        or set(old_by_id) != set(new_by_id)
    ):
        raise ValueError("comparison prediction identities differ")
    attribution = Counter()
    length = {}
    challenge = {}
    deltas = []
    changed = []
    for identity in sorted(old_by_id):
        old_row = old_by_id[identity]
        new_row = new_by_id[identity]
        reference_map = _span_map(
            old_row["reference_spans"], f"{identity} old reference"
        )
        if reference_map != _span_map(
            new_row["reference_spans"], f"{identity} new reference"
        ):
            raise ValueError(f"comparison references differ: {identity}")
        reference = set(reference_map)
        old = set(_span_map(
            old_row["prediction_spans"], f"{identity} old prediction"
        ))
        new = set(_span_map(
            new_row["prediction_spans"], f"{identity} new prediction"
        ))
        added = new - old
        removed = old - new
        counts = {
            "new_true_positives": len(added & reference),
            "lost_true_positives": len(removed & reference),
            "added_false_positives": len(added - reference),
            "removed_false_positives": len(removed - reference),
        }
        attribution.update(counts)
        delta = geometry_delta(
            _geometry_spans(old), _geometry_spans(new)
        )
        deltas.append((identity, delta))
        if added or removed:
            changed.append({"id": identity, **counts, "geometry": delta})
        for geometry, surface in reference_map.items():
            _, start, end = geometry
            key = str(end - start)
            bucket = length.setdefault(key, {
                "reference": 0, "round2_hits": 0, "round3_hits": 0,
            })
            bucket["reference"] += 1
            bucket["round2_hits"] += geometry in old
            bucket["round3_hits"] += geometry in new
            for term in CHALLENGE_TERMS:
                if term in surface:
                    term_bucket = challenge.setdefault(term, {
                        "reference": 0, "round2_hits": 0, "round3_hits": 0,
                    })
                    term_bucket["reference"] += 1
                    term_bucket["round2_hits"] += geometry in old
                    term_bucket["round3_hits"] += geometry in new
    return {
        "examples": len(old_rows),
        "attribution": dict(attribution),
        "prediction_geometry": _aggregate_deltas(deltas),
        "reference_length_recall": length,
        "reference_term_recall": challenge,
        "changed_examples": changed,
    }


def compare_rounds(round2_dir: Path, round3_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Round 3 comparison exists: {output_dir}")
    git_commit = _git_commit_clean()
    snapshots = {}
    hashes = {"round2": {}, "round3": {}}
    for label, directory in (("round2", round2_dir), ("round3", round3_dir)):
        for name in (
            "report.json",
            "dev_predictions.json",
            "evaluation_predictions.json",
        ):
            raw, digest = _snapshot(directory / name)
            snapshots[(label, name)] = raw
            hashes[label][name] = digest
    if hashes["round2"] != EXPECTED_ROUND2:
        raise ValueError("Round 2 comparison artifact differs")
    old_report = json.loads(snapshots[("round2", "report.json")])
    new_report = json.loads(snapshots[("round3", "report.json")])
    control = new_report.get("round3_control", {})
    if (
        control.get("formal_evaluation") is not False
        or control.get("eligible_for_promotion_without_fresh_sealed_set")
        is not False
        or any(
            hashes["round3"][name] != control.get("run_artifacts", {}).get(name)
            for name in ("dev_predictions.json", "evaluation_predictions.json")
        )
        or old_report.get("evaluation", {}).get("name")
        != "locked_blind_anchor_diagnostic"
        or new_report.get("evaluation", {}).get("name")
        != "locked_blind_anchor_diagnostic"
    ):
        raise ValueError("Round 3 comparison provenance differs")

    split_files = {
        "dev": "dev_predictions.json",
        "evaluation": "evaluation_predictions.json",
    }
    comparisons = {}
    metrics = {}
    for split, name in split_files.items():
        old_rows = json.loads(snapshots[("round2", name)])
        new_rows = json.loads(snapshots[("round3", name)])
        comparisons[split] = compare_predictions(
            old_rows,
            new_rows,
        )
        metrics[split] = {
            "round2": _metrics(old_rows),
            "round3": _metrics(new_rows),
        }
    metric_delta = {}
    for split, report_key in (
        ("dev", "dev_challenge"),
        ("evaluation", "evaluation"),
    ):
        old = old_report[report_key]["exact"]
        new = new_report[report_key]["exact"]
        _require_metrics(metrics[split]["round2"], old_report[report_key], (
            f"Round 2 {split}"
        ))
        _require_metrics(metrics[split]["round3"], new_report[report_key], (
            f"Round 3 {split}"
        ))
        metric_delta[split] = {
            "round2": old,
            "round3": new,
            "precision_delta": new["precision"] - old["precision"],
            "recall_delta": new["recall"] - old["recall"],
            "f1_delta": new["f1"] - old["f1"],
        }
    report = {
        "schema_version": 1,
        "status": "round3_controlled_diagnostic_comparison",
        "formal_evaluation": False,
        "eligible_for_promotion": False,
        "git_commit": git_commit,
        "inputs": hashes,
        "metric_delta": metric_delta,
        "splits": comparisons,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        path = staging / "comparison.json"
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        path.chmod(stat.S_IREAD)
        staging.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare controlled Round 2 and Round 3 predictions."
    )
    parser.add_argument("--round2", type=Path, required=True)
    parser.add_argument("--round3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_rounds(args.round2, args.round3, args.output)
    print(json.dumps({
        "metric_delta": report["metric_delta"],
        "dev_attribution": report["splits"]["dev"]["attribution"],
        "evaluation_attribution": report["splits"]["evaluation"]["attribution"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
