from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from p1_windows import labels_to_spans
from production_train import SEEDS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _geometry(span: dict) -> tuple[int, int, int, str]:
    if not isinstance(span, dict):
        return (
            int(span.para_id),
            int(span.start),
            int(span.end),
            str(span.surface),
        )
    return (
        int(span["para_id"]),
        int(span["start"]),
        int(span["end"]),
        str(span["surface"]),
    )


def _span(geometry: tuple[int, int, int, str]) -> dict:
    return dict(zip(("para_id", "start", "end", "surface"), geometry))


def _metric(reference: set, prediction: set) -> dict:
    true_positive = len(reference & prediction)
    precision = true_positive / len(prediction) if prediction else 0.0
    recall = true_positive / len(reference) if reference else 0.0
    return {
        "reference_spans": len(reference),
        "prediction_spans": len(prediction),
        "true_positive": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        ),
    }


def _aggregate(rows: list[dict]) -> dict:
    reference = sum(row["reference_spans"] for row in rows)
    prediction = sum(row["prediction_spans"] for row in rows)
    true_positive = sum(row["true_positive"] for row in rows)
    return _metric(
        set(range(reference)),
        set(range(true_positive))
        | set(range(reference, reference + prediction - true_positive)),
    )


def _vote(seed_predictions: list[set], threshold: int) -> set:
    counts = Counter(
        geometry for prediction in seed_predictions for geometry in prediction
    )
    return {geometry for geometry, count in counts.items() if count >= threshold}


def select_ensemble(
    dataset_dir: Path,
    private_roles_path: Path,
    run_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"production selection output exists: {output_dir}")
    manifest_path = dataset_dir / "manifest.json"
    development_path = dataset_dir / "development.jsonl"
    manifest = _load_json(manifest_path)
    development = _load_jsonl(development_path)
    roles = _load_json(private_roles_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "ml_production_round1_frozen_dataset"
        or manifest.get("outputs", {}).get("development_sha256")
        != _sha256(development_path)
        or manifest.get("inputs", {}).get("private_roles_sha256")
        != _sha256(private_roles_path)
        or not isinstance(roles, dict)
        or roles.get("status") != "ml_production_private_task_roles"
        or len(development) != 40
    ):
        raise ValueError("production selection dataset binding differs")
    role_by_key = {
        (int(row["juan"]), int(row["jie_index"])): str(row["stratum"])
        for row in roles["selected_jies"]
        if row["split"] == "development"
    }
    example_by_id = {str(row["id"]): row for row in development}
    if (
        len(role_by_key) != 40
        or len(example_by_id) != 40
        or {
            (int(row["juan"]), int(row["jie_index"])) for row in development
        } != set(role_by_key)
    ):
        raise ValueError("development role inventory differs")

    seed_rows = {}
    seed_inputs = {}
    diagnostics = {}
    for seed in SEEDS:
        run_dir = run_root / f"ml-production-round1-seed-{seed}-v1"
        report_path = run_dir / "report.json"
        predictions_path = run_dir / "dev_predictions.json"
        report = _load_json(report_path)
        predictions = _load_json(predictions_path)
        control = report.get("production_control", {})
        if (
            not isinstance(predictions, list)
            or control.get("seed") != seed
            or control.get("dataset_manifest_sha256") != _sha256(manifest_path)
            or control.get("formal_evaluation") is not False
            or control.get("eligible_for_development_selection") is not True
            or report.get("evaluation", {}).get("name")
            != "development_duplicate_not_independent"
            or _model_artifact(run_dir / "model")
            != control.get("model_artifact")
        ):
            raise ValueError(f"production seed {seed} binding differs")
        indexed = {str(row["id"]): row for row in predictions}
        if len(indexed) != 40 or set(indexed) != set(example_by_id):
            raise ValueError(f"production seed {seed} prediction inventory differs")
        seed_rows[seed] = indexed
        seed_inputs[str(seed)] = {
            "report_sha256": _sha256(report_path),
            "development_predictions_sha256": _sha256(predictions_path),
            "model_artifact_sha256": control["model_artifact"]["combined_sha256"],
        }
        diagnostics[str(seed)] = report["dev_challenge"]["exact"]

    per_jie = []
    predictions_by_threshold = {2: [], 3: []}
    for identity, example in example_by_id.items():
        reference = {
            _geometry(span) for span in labels_to_spans(
                example,
                example["labels"],
                [character != "\n" for character in example["text"]],
            )
        }
        if any(
            {_geometry(span) for span in seed_rows[seed][identity][
                "reference_spans"
            ]} != reference
            for seed in SEEDS
        ):
            raise ValueError(
                f"seed reference differs from frozen labels: {identity}"
            )
        seed_predictions = [
            {
                _geometry(span)
                for span in seed_rows[seed][identity]["prediction_spans"]
            }
            for seed in SEEDS
        ]
        key = int(example["juan"]), int(example["jie_index"])
        row = {
            "id": identity,
            "juan": key[0],
            "jie_index": key[1],
            "stratum": role_by_key[key],
            "reference_spans": len(reference),
            "operating_points": {},
        }
        for threshold in (2, 3):
            prediction = _vote(seed_predictions, threshold)
            row["operating_points"][f"{threshold}_of_3"] = _metric(
                reference, prediction
            )
            predictions_by_threshold[threshold].append({
                "id": identity,
                "reference_spans": [
                    _span(geometry) for geometry in sorted(reference)
                ],
                "prediction_spans": [
                    _span(geometry) for geometry in sorted(prediction)
                ],
            })
        per_jie.append(row)

    operating_points = {}
    for threshold in (2, 3):
        name = f"{threshold}_of_3"
        overall = _aggregate([
            row["operating_points"][name] for row in per_jie
        ])
        strata = {}
        for stratum in sorted(set(role_by_key.values())):
            metric = _aggregate([
                row["operating_points"][name]
                for row in per_jie if row["stratum"] == stratum
            ])
            strata[stratum] = {
                **metric,
                "jie_count": sum(row["stratum"] == stratum for row in per_jie),
                "statistical_gate": metric["reference_spans"] >= 50,
            }
        operating_points[name] = {
            "vote_threshold": threshold,
            "overall": overall,
            "strata": strata,
            "precision_at_least_0_99": overall["precision"] >= 0.99,
            "recall_at_least_0_95": overall["recall"] >= 0.95,
            "eligible_for_rules_comparison": (
                overall["precision"] >= 0.99 and overall["recall"] >= 0.95
            ),
        }
    eligible = [
        name for name, row in operating_points.items()
        if row["eligible_for_rules_comparison"]
    ]
    if eligible:
        raise RuntimeError(
            "an operating point passed precision/recall; rules comparison is required"
        )
    report = {
        "schema_version": 1,
        "status": "ml_production_round1_development_selection",
        "formal_evaluation": False,
        "eligible_for_promotion": False,
        "development_comparison_consumed": True,
        "individual_seed_diagnostics": diagnostics,
        "operating_points": operating_points,
        "rules_comparison": {
            "status": "not_run",
            "reason": (
                "short_circuited_after_no_operating_point_passed_the_frozen_"
                "precision_and_recall_gates"
            ),
        },
        "selected_operating_point": None,
        "decision": "start_new_training_data_round",
        "fresh_formal_evaluation_created": False,
        "claim_limit": (
            "Fresh development selection only. No operating point met the "
            "predeclared gates, and no formal evaluation was created or consumed."
        ),
        "inputs": {
            "dataset_manifest_sha256": _sha256(manifest_path),
            "development_sha256": _sha256(development_path),
            "private_roles_sha256": _sha256(private_roles_path),
            "seeds": seed_inputs,
        },
        "git_commit": _git_commit_clean(),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        (staging / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "per_jie_metrics.json").write_text(
            json.dumps(per_jie, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for threshold, rows in predictions_by_threshold.items():
            (staging / f"predictions_{threshold}_of_3.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        for path in staging.iterdir():
            path.chmod(0o444)
        staging.chmod(0o555)
        staging.replace(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the production development ensemble gate."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--private-roles", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = select_ensemble(
        args.dataset, args.private_roles, args.run_root, args.output
    )
    print(json.dumps({
        "operating_points": report["operating_points"],
        "decision": report["decision"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
