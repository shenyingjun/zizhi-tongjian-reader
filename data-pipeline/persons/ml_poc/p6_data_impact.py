from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tempfile
from collections import Counter
from pathlib import Path

from p3_compact import _git_commit_clean
from p3_round3_compare import _metrics, compare_predictions
from p3_round3_training import _read_jsonl


EXPECTED_INPUTS = {
    "round4_train": "78995e7a26f2712a6a189283d9d64a1b82a6bc6945d5d9e43f03688571fa2c87",
    "round4_manifest": "84c244e7719e891f9a53a618ffb5bca7de1756a596104ab29d974dc837496a7b",
    "round5_train": "c1642bcfa0e67a4972b651248890d52ccc9e653b9628109eae821ef783923e69",
    "round5_report": "00ae224fd3a84f9564eba66337d51fcb676b9b36221b7d46658b47f759ae01a4",
    "round6_train": "6d46c47b37f72c06549b89655000c793fd0eb5b168cb8e392feba170dc7ec769",
    "round6_manifest": "7788c8d3341e56a17eaa89fd582df6fcc930d9f5c723d1bd5a0c78b61f3852a1",
    "dev": "5fb15836f52baa989b1e65c17c19f2090efb9ca96b5ce075a484484de411a09f",
    "evaluation": "5a6ce12b0f07579ab12b3d93eef6da34f336f4e20a0d173a1d78758fab9e804c",
    "round4_report": "5381f1a24601a17927d448a8bd10f62dd8a78d0a7e89469a6bc02fbe16fa5757",
    "round4_history": "e5583e58bf74596d501f1bda4c42a9040a59e1a3f13c1dcb0e18e7b5c8aa9e3c",
    "round4_dev_predictions": "37d2dae3335e0f275a19f52c3b21ce0a91757c96d6df69860741707a47ddbd0c",
    "round4_evaluation_predictions": "1e28c0510ef641740fbf6e05015f34dc902a1186e64e83c3e186876b36f1f3b3",
    "round6_report": "e34278caa2f1e68888784cd6aa62a44f3c8fa4981992e478b7c3b6f38ba1c90e",
    "round6_history": "87d5953c5a4a911281c5f015204aa7092334a602ccb8bac10ce2f7e9ed926111",
    "round6_dev_predictions": "0ae1eadc2098026439635ad7da71ddd97d1c1c9232221e4b6b75417211a0e53f",
    "round6_evaluation_predictions": "fcf2ebea6b7665074b50fbba85bc1723c9a56feb9cf80ceeab29485fffad6835",
}
PROBE_TERMS = (
    "管", "蔡", "燕", "盖", "师", "秉", "夔", "夔之", "秉之",
    "明", "明将军", "诸葛", "诸葛氏", "高贵", "高贵鄕公",
    "将", "将军", "曹", "曹氏", "明公", "贵人", "后",
)
FOLLOWING_TERMS = ("之", "氏", "公", "将军")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _bio_spans(row: dict) -> list[tuple[int, int, str]]:
    text = str(row["text"])
    labels = list(row["labels"])
    spans = []
    index = 0
    while index < len(labels):
        if labels[index] != "B-PER":
            index += 1
            continue
        end = index + 1
        while end < len(labels) and labels[end] == "I-PER":
            end += 1
        spans.append((index, end, text[index:end]))
        index = end
    return spans


def _context(text: str, start: int, end: int, radius: int = 12) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:start] + "[" + text[start:end] + "]" + text[end:right]


def _occurrence_status(
    start: int,
    end: int,
    spans: list[tuple[int, int, str]],
) -> str:
    geometries = {(span_start, span_end) for span_start, span_end, _ in spans}
    if (start, end) in geometries:
        return "exact_gold"
    if any(span_start <= start and end <= span_end for span_start, span_end, _ in spans):
        return "inside_larger_gold"
    if any(start < span_end and span_start < end for span_start, span_end, _ in spans):
        return "overlaps_gold"
    return "untagged"


def _probe_occurrences(rows: list[dict]) -> dict:
    result = {}
    for term in PROBE_TERMS:
        counts = Counter()
        examples: dict[str, list[dict]] = {}
        for row in rows:
            text = str(row["text"])
            spans = _bio_spans(row)
            start = text.find(term)
            while start >= 0:
                end = start + len(term)
                status = _occurrence_status(start, end, spans)
                counts[status] += 1
                bucket = examples.setdefault(status, [])
                if len(bucket) < 3:
                    bucket.append({
                        "id": row["id"],
                        "context": _context(text, start, end),
                    })
                start = text.find(term, start + 1)
        result[term] = {"counts": dict(counts), "examples": examples}
    return result


def _profile(rows: list[dict]) -> dict:
    lengths = Counter()
    following = Counter()
    ending = Counter()
    exact_surfaces = Counter()
    single_character_following = Counter()
    spans = []
    for row in rows:
        text = str(row["text"])
        for start, end, surface in _bio_spans(row):
            spans.append((row["id"], start, end, surface))
            lengths[str(end - start)] += 1
            exact_surfaces[surface] += 1
            for term in FOLLOWING_TERMS:
                if text.startswith(term, end):
                    following[term] += 1
                if surface.endswith(term):
                    ending[term] += 1
            if end - start == 1:
                for term in FOLLOWING_TERMS:
                    if text.startswith(term, end):
                        single_character_following[term] += 1
    single_count = lengths.get("1", 0)
    total = len(spans)
    return {
        "examples": len(rows),
        "characters": sum(len(str(row["text"])) for row in rows),
        "spans": total,
        "span_length_counts": dict(sorted(lengths.items(), key=lambda item: int(item[0]))),
        "single_character_spans": single_count,
        "single_character_span_rate": single_count / total if total else 0.0,
        "gold_followed_by": dict(following),
        "gold_ending_in": dict(ending),
        "single_character_gold_followed_by": dict(single_character_following),
        "probe_exact_gold_surfaces": {
            term: exact_surfaces[term]
            for term in PROBE_TERMS
            if exact_surfaces[term]
        },
        "probe_occurrences": _probe_occurrences(rows),
    }


def _prediction_map(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        identity = str(row["id"])
        if identity in result:
            raise ValueError(f"duplicate prediction identity: {identity}")
        result[identity] = row
    return result


def _span_map(rows: list[dict]) -> dict[tuple[int, int, int], str]:
    return {
        (int(span["para_id"]), int(span["start"]), int(span["end"])): str(
            span["surface"]
        )
        for span in rows
    }


def _assembled_geometry(
    dataset_row: dict,
    geometry: tuple[int, int, int],
) -> tuple[int, int]:
    para_id, start, end = geometry
    segments = [
        segment for segment in dataset_row["segments"]
        if int(segment["para_id"]) == para_id
    ]
    if len(segments) != 1:
        raise ValueError(
            f"{dataset_row['id']} lacks unique paragraph {para_id}"
        )
    segment = segments[0]
    assembled_start = int(segment["assembled_start"]) + start
    assembled_end = int(segment["assembled_start"]) + end
    if assembled_end > int(segment["assembled_end"]):
        raise ValueError(f"span exceeds paragraph in {dataset_row['id']}")
    return assembled_start, assembled_end


def _surfaced_changes(
    old_rows: list[dict],
    new_rows: list[dict],
    dataset_rows: list[dict],
) -> list[dict]:
    old_by_id = _prediction_map(old_rows)
    new_by_id = _prediction_map(new_rows)
    dataset_by_id = {str(row["id"]): row for row in dataset_rows}
    changes = []
    for identity in sorted(old_by_id):
        old = old_by_id[identity]
        new = new_by_id[identity]
        reference = _span_map(old["reference_spans"])
        if reference != _span_map(new["reference_spans"]):
            raise ValueError(f"reference geometry differs: {identity}")
        old_predictions = _span_map(old["prediction_spans"])
        new_predictions = _span_map(new["prediction_spans"])
        for action, geometries, surfaces in (
            ("added", set(new_predictions) - set(old_predictions), new_predictions),
            ("removed", set(old_predictions) - set(new_predictions), old_predictions),
        ):
            for geometry in sorted(geometries):
                dataset_row = dataset_by_id[identity]
                start, end = _assembled_geometry(dataset_row, geometry)
                surface = surfaces[geometry]
                if str(dataset_row["text"])[start:end] != surface:
                    raise ValueError(
                        f"surface/text mismatch in {identity}: {surface}"
                    )
                changes.append({
                    "id": identity,
                    "action": action,
                    "surface": surface,
                    "reference_status": (
                        "true_positive" if geometry in reference
                        else "false_positive"
                    ),
                    "geometry": {
                        "para_id": geometry[0],
                        "start": geometry[1],
                        "end": geometry[2],
                    },
                    "context": _context(str(dataset_row["text"]), start, end),
                })
    return changes


def _round_comparison(
    old_rows: list[dict],
    new_rows: list[dict],
    dataset_rows: list[dict],
) -> dict:
    comparison = compare_predictions(old_rows, new_rows)
    return {
        "metrics": {"round4": _metrics(old_rows), "round6": _metrics(new_rows)},
        "attribution": comparison["attribution"],
        "prediction_geometry": comparison["prediction_geometry"],
        "changed_examples": comparison["changed_examples"],
        "surfaced_changes": _surfaced_changes(
            old_rows, new_rows, dataset_rows
        ),
    }


def _validate_provenance(snapshots: dict[str, bytes], hashes: dict[str, str]) -> None:
    round4_manifest = json.loads(snapshots["round4_manifest"])
    round5_report = json.loads(snapshots["round5_report"])
    round6_manifest = json.loads(snapshots["round6_manifest"])
    reports = {
        "round4": json.loads(snapshots["round4_report"]),
        "round6": json.loads(snapshots["round6_report"]),
    }
    if (
        round4_manifest.get("status") != "round4_controlled_training_dataset"
        or round4_manifest.get("formal_evaluation") is not False
        or round4_manifest.get("outputs", {}).get("train.jsonl")
        != hashes["round4_train"]
        or round5_report.get("status")
        != "frozen_copilot_double_pass_diagnostic"
        or round5_report.get("formal_evaluation") is not False
        or round5_report.get("eligible_for_training") is not True
        or round5_report.get("outputs", {}).get(
            "train_assisted_round5_sha256"
        ) != hashes["round5_train"]
        or round6_manifest.get("status") != "round6_controlled_training_dataset"
        or round6_manifest.get("formal_evaluation") is not False
        or round6_manifest.get("outputs", {}).get("train.jsonl")
        != hashes["round6_train"]
        or round6_manifest.get("inputs", {}).get("base", {}).get(
            "manifest.json"
        ) != hashes["round4_manifest"]
        or round6_manifest.get("inputs", {}).get("assisted", {}).get("train")
        != hashes["round5_train"]
        or round6_manifest.get("inputs", {}).get("assisted", {}).get("report")
        != hashes["round5_report"]
    ):
        raise ValueError("dataset/freeze provenance differs")
    for label in ("round4", "round6"):
        control = reports[label].get(f"{label}_control", {})
        artifact_prefix = f"{label}_"
        if (
            control.get("formal_evaluation") is not False
            or control.get("eligible_for_promotion") is not False
            or control.get("dataset_manifest_sha256")
            != hashes[f"{label}_manifest"]
            or control.get("run_artifacts", {}).get("history.json")
            != hashes[f"{label}_history"]
            or control.get("run_artifacts", {}).get("dev_predictions.json")
            != hashes[f"{artifact_prefix}dev_predictions"]
            or control.get("run_artifacts", {}).get(
                "evaluation_predictions.json"
            ) != hashes[f"{artifact_prefix}evaluation_predictions"]
            or control.get("full_control", {}).get("seed") != 20260727
            or control.get("full_control", {}).get("context_mode")
            != "target_only"
        ):
            raise ValueError(f"{label} model provenance differs")
    if (
        reports["round4"]["round4_control"]["full_control"]
        != reports["round6"]["round6_control"]["full_control"]
        or reports["round4"]["round4_control"]["base_model"]
        != reports["round6"]["round6_control"]["base_model"]
        or reports["round4"]["round4_control"]["base_model_revision"]
        != reports["round6"]["round6_control"]["base_model_revision"]
    ):
        raise ValueError("Round 4/6 training controls differ")


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _synthesize_findings(profiles: dict, comparisons: dict) -> dict:
    existing = profiles["round4_existing"]
    added = profiles["round5_added"]
    evaluation = comparisons["evaluation"]
    old_metrics = evaluation["metrics"]["round4"]
    new_metrics = evaluation["metrics"]["round6"]
    attribution = evaluation["attribution"]
    changed_recall_terms = ("管", "蔡", "燕", "盖", "师", "秉")
    added_occurrences = {
        term: added["probe_occurrences"][term]["counts"]
        for term in changed_recall_terms
    }
    direct_exact_support = {
        term: added["probe_occurrences"][term]["counts"].get("exact_gold", 0)
        for term in changed_recall_terms
    }
    existing_single = existing["single_character_spans"]
    added_single = added["single_character_spans"]
    existing_before_zhi = existing["single_character_gold_followed_by"].get(
        "之", 0
    )
    added_before_zhi = added["single_character_gold_followed_by"].get("之", 0)
    old_exact = old_metrics["exact"]
    new_exact = new_metrics["exact"]
    return {
        "component_scale": {
            "character_ratio_added_to_existing": (
                added["characters"] / existing["characters"]
            ),
            "span_ratio_added_to_existing": added["spans"] / existing["spans"],
        },
        "single_character_supervision": {
            "round4_rate": existing["single_character_span_rate"],
            "round5_rate": added["single_character_span_rate"],
            "rate_delta": (
                added["single_character_span_rate"]
                - existing["single_character_span_rate"]
            ),
            "conclusion": (
                "Round 5 does not dilute person spans by length; its "
                "single-character span rate is slightly higher."
            ),
        },
        "zhi_boundary_supervision": {
            "round4_single_gold_before_zhi": existing_before_zhi,
            "round5_single_gold_before_zhi": added_before_zhi,
            "round4_rate_among_single_gold": _rate(
                existing_before_zhi, existing_single
            ),
            "round5_rate_among_single_gold": _rate(
                added_before_zhi, added_single
            ),
            "conclusion": (
                "Round 5 supplies rather than removes examples where a "
                "single-character person ends before 之; the 秉→秉之 error is "
                "not a direct consequence of missing this boundary pattern."
            ),
        },
        "changed_recall_term_evidence": {
            "round5_occurrence_status": added_occurrences,
            "round5_exact_gold_support": direct_exact_support,
            "conclusion": (
                "Round 5 adds untagged occurrences for 管, 燕, and 师, but no "
                "exact positive occurrence for the changed recall terms. 蔡 and "
                "盖 regress without any Round 5 occurrence, so surface-level "
                "label conflict cannot explain the complete regression set."
            ),
        },
        "incremental_model_shift": {
            "prediction_count_delta": (
                new_metrics["prediction_spans"]
                - old_metrics["prediction_spans"]
            ),
            "true_positive_delta": (
                new_exact["true_positive"] - old_exact["true_positive"]
            ),
            "precision_delta": (
                new_exact["precision"] - old_exact["precision"]
            ),
            "recall_delta": new_exact["recall"] - old_exact["recall"],
            "f1_delta": new_exact["f1"] - old_exact["f1"],
            **attribution,
            "conclusion": (
                "Round 5 produces a more conservative model: false positives "
                "fall, but true-positive losses exceed recoveries and exact F1 "
                "still declines on the reused Juan 76 diagnostic."
            ),
        },
        "root_cause_assessment": (
            "No single policy or mislabeled surface explains the changes. The "
            "evidence supports an optimization/generalization shift amplified "
            "by weak exact support for rare one-character historical names; "
            "this remains a hypothesis until replicated across seeds."
        ),
    }


def diagnose_data_impact(
    round4_dataset: Path,
    round5_assisted: Path,
    round6_dataset: Path,
    round4_model: Path,
    round6_model: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"data-impact diagnosis exists: {output_dir}")
    paths = {
        "round4_train": round4_dataset / "train.jsonl",
        "round4_manifest": round4_dataset / "manifest.json",
        "round5_train": round5_assisted / "train_assisted_round5.jsonl",
        "round5_report": round5_assisted / "report.json",
        "round6_train": round6_dataset / "train.jsonl",
        "round6_manifest": round6_dataset / "manifest.json",
        "dev": round6_dataset / "dev.jsonl",
        "evaluation": round6_dataset / "evaluation.jsonl",
        "round4_report": round4_model / "report.json",
        "round4_history": round4_model / "history.json",
        "round4_dev_predictions": round4_model / "dev_predictions.json",
        "round4_evaluation_predictions": (
            round4_model / "evaluation_predictions.json"
        ),
        "round6_report": round6_model / "report.json",
        "round6_history": round6_model / "history.json",
        "round6_dev_predictions": round6_model / "dev_predictions.json",
        "round6_evaluation_predictions": (
            round6_model / "evaluation_predictions.json"
        ),
    }
    snapshots = {name: path.read_bytes() for name, path in paths.items()}
    hashes = {name: _sha256(raw) for name, raw in snapshots.items()}
    if hashes != EXPECTED_INPUTS:
        raise ValueError("Round 4/5/6 diagnostic inputs differ")
    _validate_provenance(snapshots, hashes)

    round4_rows = _read_jsonl(snapshots["round4_train"], "round4 train")
    round5_rows = _read_jsonl(snapshots["round5_train"], "round5 assisted")
    round6_rows = _read_jsonl(snapshots["round6_train"], "round6 train")
    expected_round6 = sorted(
        round4_rows + round5_rows,
        key=lambda row: (int(row["juan"]), int(row["jie_index"])),
    )
    if round6_rows != expected_round6:
        raise ValueError("Round 6 train is not the exact Round 4 + Round 5 union")
    if {row["id"] for row in round4_rows} & {row["id"] for row in round5_rows}:
        raise ValueError("Round 4 and Round 5 training identities overlap")

    dev_rows = _read_jsonl(snapshots["dev"], "dev")
    evaluation_rows = _read_jsonl(snapshots["evaluation"], "evaluation")
    prediction_rows = {
        name: json.loads(raw)
        for name, raw in snapshots.items()
        if name.endswith("_predictions")
    }
    comparisons = {
        "dev": _round_comparison(
            prediction_rows["round4_dev_predictions"],
            prediction_rows["round6_dev_predictions"],
            dev_rows,
        ),
        "evaluation": _round_comparison(
            prediction_rows["round4_evaluation_predictions"],
            prediction_rows["round6_evaluation_predictions"],
            evaluation_rows,
        ),
    }
    profiles = {
        "round4_existing": _profile(round4_rows),
        "round5_added": _profile(round5_rows),
    }
    report = {
        "schema_version": 1,
        "status": "round6_training_data_impact_diagnostic",
        "formal_evaluation": False,
        "eligible_for_promotion": False,
        "git_commit": _git_commit_clean(),
        "inputs": hashes,
        "dataset_integrity": {
            "round6_is_exact_round4_plus_round5_union": True,
            "component_identity_overlap": 0,
        },
        "supervision_profiles": profiles,
        "incremental_round4_to_round6": comparisons,
        "findings": _synthesize_findings(profiles, comparisons),
        "interpretation_limits": [
            "The fixed-seed Round 4 to Round 6 comparison isolates the added "
            "Round 5 dataset at the experiment level, not individual examples.",
            "Feature counts are correlations and cannot identify which training "
            "example caused a prediction change.",
            "Juan 27 and Juan 76 are reused diagnostics, not formal evaluation.",
        ],
        "decision": {
            "round7_ready": False,
            "active_model": "round3",
            "reason": (
                "Do not select a data correction or weighting rule from a single "
                "cumulative retrain and reused diagnostics."
            ),
            "required_next_experiment": (
                "Replicate Round 4 and Round 6 training with three matched seeds "
                "before changing labels or weighting. Compare exact geometry per "
                "seed to separate stable data effects from optimization variance."
            ),
        },
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        output = staging / "report.json"
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose the incremental impact of Round 5 training data."
    )
    parser.add_argument("--round4-dataset", type=Path, required=True)
    parser.add_argument("--round5-assisted", type=Path, required=True)
    parser.add_argument("--round6-dataset", type=Path, required=True)
    parser.add_argument("--round4-model", type=Path, required=True)
    parser.add_argument("--round6-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose_data_impact(
        args.round4_dataset,
        args.round5_assisted,
        args.round6_dataset,
        args.round4_model,
        args.round6_model,
        args.output,
    )
    print(json.dumps({
        "round5_profile": {
            key: value
            for key, value in report["supervision_profiles"]["round5_added"].items()
            if key != "probe_occurrences"
        },
        "dev": report["incremental_round4_to_round6"]["dev"]["attribution"],
        "evaluation": report["incremental_round4_to_round6"][
            "evaluation"
        ]["attribution"],
        "decision": report["decision"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
