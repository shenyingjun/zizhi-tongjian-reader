from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean


EXPECTED_REFERENCE_REPORT_SHA256 = (
    "25f8788b1d49c22d133a4eca0851e746cb905c15f0d30b836bda27213e15e9be"
)
EXPECTED_REFERENCE_SHA256 = (
    "30c5bc7e839b28f331cc04db38a13a69a355ef96e3bfc5d112a28c0c0db57264"
)
EXPECTED_COMPARISON_REPORT_SHA256 = (
    "592062f91932c7c2ec293a22f32838e415f45596185c6a723c8be3c9c34a0ec8"
)
EXPECTED_MODEL_PREDICTIONS_SHA256 = (
    "f96cbac995cdad20af4e4f9035c9862f3790f1de60856b8d2194d334852022d3"
)
EXPECTED_RULES_MANIFEST_SHA256 = (
    "b297a3ca6fe768ecafd4184e07fc7dde00e9d2e9e9850ecf241dce4571958a0f"
)
EXPECTED_RULES_SHA256 = (
    "b50e6dfaa1f204fba405ea2c548fcdb4fba0edaca38b9ae4c171c1cb7c5f9c06"
)
EXPECTED_RULE_DOCUMENT_ROOT_SHA256 = (
    "8411b2a52843a6b303ce320b5843d560b08822e1855de1f994d12e00527ab3f6"
)
EXPECTED_TRANSLATION_MANIFEST_SHA256 = (
    "eb4a4005cf1605b56843eda907368ceb54f7cfcbd00b50e91ff27db91e9bd412"
)
BOOTSTRAP_SEED = 20260805
BOOTSTRAP_REPLICATES = 20_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _geometry(row: dict) -> tuple[int, int, int, str]:
    return (
        int(row["para_id"]),
        int(row["start"]),
        int(row["end"]),
        str(row["surface"]),
    )


def _metric(reference: int, predicted: int, true_positive: int) -> dict:
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


def _counts(reference: set, prediction: set) -> tuple[int, int, int]:
    return len(reference), len(prediction), len(reference & prediction)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_intervals(
    counts_by_jie: list[dict[str, tuple[int, int, int]]],
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict:
    if not counts_by_jie or replicates < 2:
        raise ValueError("bootstrap requires jies and at least two replicates")
    systems = tuple(counts_by_jie[0])
    if any(set(row) != set(systems) for row in counts_by_jie):
        raise ValueError("bootstrap system inventory differs")
    rng = random.Random(seed)
    samples = {
        system: {"precision": [], "recall": [], "f1": []}
        for system in systems
    }
    for _ in range(replicates):
        indices = [
            rng.randrange(len(counts_by_jie))
            for _ in range(len(counts_by_jie))
        ]
        for system in systems:
            reference = predicted = true_positive = 0
            for index in indices:
                counts = counts_by_jie[index][system]
                reference += counts[0]
                predicted += counts[1]
                true_positive += counts[2]
            metric = _metric(reference, predicted, true_positive)
            for name in samples[system]:
                samples[system][name].append(metric[name])
    return {
        system: {
            "jie_bootstrap_90pct": {
                name: {
                    "lower": _quantile(values, 0.05),
                    "upper": _quantile(values, 0.95),
                }
                for name, values in metrics.items()
            },
            "precision_one_sided_95pct_lower": _quantile(
                metrics["precision"], 0.05
            ),
        }
        for system, metrics in samples.items()
    }


def compute_adoption_statistics(
    reference_dir: Path,
    comparison_dir: Path,
    rules_dir: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"adoption statistics exist: {output_dir}")
    reference_report_path = reference_dir / "report.json"
    reference_path = reference_dir / "promotion_reference.jsonl"
    comparison_report_path = comparison_dir / "report.json"
    model_predictions_path = comparison_dir / "predictions.json"
    rules_manifest_path = rules_dir / "manifest.json"
    reference_report_raw = reference_report_path.read_bytes()
    reference_raw = reference_path.read_bytes()
    comparison_report_raw = comparison_report_path.read_bytes()
    model_predictions_raw = model_predictions_path.read_bytes()
    rules_manifest_raw = rules_manifest_path.read_bytes()
    reference_report = json.loads(reference_report_raw)
    comparison_report = json.loads(comparison_report_raw)
    rules_manifest = json.loads(rules_manifest_raw)
    if (
        _digest(reference_report_raw) != EXPECTED_REFERENCE_REPORT_SHA256
        or _digest(reference_raw) != EXPECTED_REFERENCE_SHA256
        or _digest(comparison_report_raw)
        != EXPECTED_COMPARISON_REPORT_SHA256
        or _digest(model_predictions_raw)
        != EXPECTED_MODEL_PREDICTIONS_SHA256
        or _digest(rules_manifest_raw) != EXPECTED_RULES_MANIFEST_SHA256
        or reference_report.get("formal_evaluation") is not True
        or reference_report.get("candidate_model_blind") is not True
        or comparison_report.get("decision") != "promote_round7_recipe"
        or rules_manifest.get("preset") != "PRESET_RECALL"
        or rules_manifest.get("scope") != "numbered-jie"
        or rules_manifest.get("rules_sha256") != EXPECTED_RULES_SHA256
        or rules_manifest.get("translation_evidence", {}).get(
            "manifest_sha256"
        ) != EXPECTED_TRANSLATION_MANIFEST_SHA256
        or rules_manifest.get("juans") != reference_report.get("juans")
    ):
        raise ValueError("adoption statistics input binding differs")

    examples = [
        json.loads(line) for line in reference_raw.decode("utf-8").splitlines()
        if line
    ]
    model_bundle = json.loads(model_predictions_raw)
    model_rows = {
        str(row["id"]): row
        for row in model_bundle["round7"]["ensemble"]
    }
    if len(examples) != 37 or set(model_rows) != {
        str(row["id"]) for row in examples
    }:
        raise ValueError("adoption evaluation identity inventory differs")

    expected_rule_names = {
        f"juan_{juan:03d}.json" for juan in reference_report["juans"]
    }
    if {
        path.name for path in rules_dir.iterdir() if path.is_file()
    } != expected_rule_names | {"manifest.json"}:
        raise ValueError("production rule document inventory differs")
    rule_snapshots = {
        name: (rules_dir / name).read_bytes()
        for name in sorted(expected_rule_names)
    }
    rule_hashes = {
        name: hashlib.sha256(raw).hexdigest()
        for name, raw in rule_snapshots.items()
    }
    rule_document_root_sha256 = hashlib.sha256(json.dumps(
        rule_hashes, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if rule_document_root_sha256 != EXPECTED_RULE_DOCUMENT_ROOT_SHA256:
        raise ValueError("production rule document root hash differs")
    rules_documents = {}
    for juan in reference_report["juans"]:
        document = json.loads(rule_snapshots[f"juan_{juan:03d}.json"])
        if (
            document.get("juan") != juan
            or document.get("rules_sha256") != EXPECTED_RULES_SHA256
            or document.get("translation_evidence_manifest_sha256")
            != EXPECTED_TRANSLATION_MANIFEST_SHA256
            or len(document.get("occurrences", []))
            != rules_manifest["occurrences_by_juan"][str(juan)]
        ):
            raise ValueError(f"production rule document binding differs: {juan}")
        rules_documents[juan] = document
    counts_by_jie = []
    point_counts = {
        "round7_model": [0, 0, 0],
        "production_rules_translation_assisted": [0, 0, 0],
    }
    rules_predictions = []
    per_jie = []
    for example in examples:
        identity = str(example["id"])
        juan = int(example["juan"])
        paragraph_ids = {
            int(segment["para_id"]) for segment in example["segments"]
        }
        model_row = model_rows[identity]
        reference = {
            _geometry(row) for row in model_row["reference_spans"]
        }
        model = {
            _geometry(row) for row in model_row["prediction_spans"]
        }
        rule_rows_by_geometry = {}
        for row in rules_documents[juan]["occurrences"]:
            if int(row["para_id"]) in paragraph_ids:
                rule_rows_by_geometry.setdefault(_geometry(row), row)
        rules = set(rule_rows_by_geometry)
        row_counts = {
            "round7_model": _counts(reference, model),
            "production_rules_translation_assisted": _counts(
                reference, rules
            ),
        }
        counts_by_jie.append(row_counts)
        for system, values in row_counts.items():
            for index, value in enumerate(values):
                point_counts[system][index] += value
        per_jie.append({
            "id": identity,
            "round7_model": _metric(*row_counts["round7_model"]),
            "production_rules_translation_assisted": _metric(
                *row_counts["production_rules_translation_assisted"]
            ),
        })
        rules_predictions.append({
            "id": identity,
            "reference_spans": model_row["reference_spans"],
            "prediction_spans": [
                {
                    key: rule_rows_by_geometry[geometry][key]
                    for key in ("para_id", "start", "end", "surface")
                }
                for geometry in sorted(rules)
            ],
        })

    metrics = {
        system: _metric(*values) for system, values in point_counts.items()
    }
    intervals = bootstrap_intervals(counts_by_jie)
    model_f1 = intervals["round7_model"]["jie_bootstrap_90pct"]["f1"]
    rules_f1 = intervals[
        "production_rules_translation_assisted"
    ]["jie_bootstrap_90pct"]["f1"]
    gates = {
        "model_exact_f1_at_least_rules": (
            metrics["round7_model"]["f1"]
            >= metrics["production_rules_translation_assisted"]["f1"]
        ),
        "f1_90pct_intervals_non_overlapping_in_model_favor": (
            model_f1["lower"] > rules_f1["upper"]
        ),
        "model_precision_one_sided_95pct_lower_at_least_0_98": (
            intervals["round7_model"][
                "precision_one_sided_95pct_lower"
            ] >= 0.98
        ),
        "predeclared_challenge_strata_available": False,
        "no_challenge_stratum_down_more_than_5_points": False,
    }
    adoption_authorized = all(gates.values())
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        rules_predictions_path = staging / "rules_predictions.json"
        per_jie_path = staging / "per_jie_metrics.json"
        rules_predictions_path.write_text(
            json.dumps(rules_predictions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        per_jie_path.write_text(
            json.dumps(per_jie, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report = {
            "schema_version": 1,
            "status": "round11_spec_aligned_adoption_statistics",
            "formal_probability_evidence": True,
            "production_adoption_authorized": adoption_authorized,
            "decision": (
                "production_adoption_authorized"
                if adoption_authorized
                else "production_adoption_not_authorized"
            ),
            "sampling_scope": (
                "one randomly sampled eligible jie in each of all 37 remaining "
                "wholly unused eligible juans"
            ),
            "claim_limit": (
                "Probability-sample evidence only. No predeclared challenge cohort "
                "was included, so the SPEC challenge-stratum gate is unmeasured."
            ),
            "bootstrap": {
                "unit": "jie",
                "method": "percentile_resampling_with_replacement",
                "seed": BOOTSTRAP_SEED,
                "replicates": BOOTSTRAP_REPLICATES,
            },
            "metrics": metrics,
            "intervals": intervals,
            "adoption_gates": gates,
            "inputs": {
                "reference_report_sha256": EXPECTED_REFERENCE_REPORT_SHA256,
                "reference_sha256": EXPECTED_REFERENCE_SHA256,
                "comparison_report_sha256": (
                    EXPECTED_COMPARISON_REPORT_SHA256
                ),
                "model_predictions_sha256": (
                    EXPECTED_MODEL_PREDICTIONS_SHA256
                ),
                "rules_manifest_sha256": EXPECTED_RULES_MANIFEST_SHA256,
                "rule_document_root_sha256": (
                    EXPECTED_RULE_DOCUMENT_ROOT_SHA256
                ),
                "rules_sha256": EXPECTED_RULES_SHA256,
                "translation_evidence_manifest_sha256": (
                    EXPECTED_TRANSLATION_MANIFEST_SHA256
                ),
            },
            "git_commit": _git_commit_clean(),
            "outputs": {
                "rules_predictions_sha256": _sha256(
                    rules_predictions_path
                ),
                "per_jie_metrics_sha256": _sha256(per_jie_path),
            },
        }
        (staging / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in staging.iterdir():
            path.chmod(0o444)
        staging.chmod(0o555)
        staging.replace(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute SPEC-aligned probability-sample adoption statistics."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compute_adoption_statistics(
        args.reference, args.comparison, args.rules, args.output
    )
    print(json.dumps({
        "decision": report["decision"],
        "metrics": report["metrics"],
        "intervals": report["intervals"],
        "adoption_gates": report["adoption_gates"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
