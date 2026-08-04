from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from p6_locked_assisted_finalize import _read_jsonl
from p6_seed_ensemble import (
    _predict_examples,
    _prediction_metrics,
    vote_predictions,
)
from p6_seed_replication_train import (
    CUBLAS_WORKSPACE_CONFIG,
    REPLICATION_SEEDS,
    _configure_determinism,
)
from p10_independent_dev_compare import RECIPES, _compare


EXPECTED_REFERENCE_REPORT_SHA256 = (
    "25f8788b1d49c22d133a4eca0851e746cb905c15f0d30b836bda27213e15e9be"
)
EXPECTED_REFERENCE_SHA256 = (
    "30c5bc7e839b28f331cc04db38a13a69a355ef96e3bfc5d112a28c0c0db57264"
)
EXPECTED_DEV_SELECTION_SHA256 = (
    "076dda3f24654c5f4a034224c18f8a48c327d6d1c39d6773c5889baf493c561e"
)
ENSEMBLE_THRESHOLD = 2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_promotion(
    reference_dir: Path,
    development_selection_report: Path,
    artifacts_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"promotion comparison exists: {output_dir}")
    report_path = reference_dir / "report.json"
    reference_path = reference_dir / "promotion_reference.jsonl"
    reference_report = json.loads(report_path.read_text(encoding="utf-8"))
    development_selection = json.loads(
        development_selection_report.read_text(encoding="utf-8")
    )
    if (
        _sha256(report_path) != EXPECTED_REFERENCE_REPORT_SHA256
        or _sha256(reference_path) != EXPECTED_REFERENCE_SHA256
        or reference_report.get("status")
        != "frozen_round11_fresh_promotion_reference"
        or reference_report.get("formal_evaluation") is not True
        or reference_report.get("eligible_for_training") is not False
        or reference_report.get("eligible_for_promotion_metric") is not True
        or reference_report.get("candidate_model_blind") is not True
        or reference_report.get("model_predictions_used_in_labeling") is not False
        or reference_report.get("outputs", {}).get(
            "promotion_reference_sha256"
        ) != EXPECTED_REFERENCE_SHA256
        or _sha256(development_selection_report)
        != EXPECTED_DEV_SELECTION_SHA256
        or development_selection.get("status")
        != "round10_independent_dev_recipe_selection"
        or development_selection.get("selected_recipe") != "round7"
        or development_selection.get(
            "fresh_promotion_evaluation_consumed"
        ) is not False
    ):
        raise ValueError("promotion reference or dev selection binding differs")
    examples = _read_jsonl(reference_path)
    if len(examples) != 37:
        raise ValueError("promotion reference inventory differs")
    _configure_determinism()

    predictions = {}
    metrics = {}
    inputs = {}
    for recipe in ("round6", "round7"):
        by_seed = {}
        inputs[recipe] = {}
        for seed in REPLICATION_SEEDS:
            name, expected_report, expected_model = RECIPES[recipe][seed]
            root = artifacts_root / name
            model_report = json.loads(
                (root / "report.json").read_text(encoding="utf-8")
            )
            artifact = _model_artifact(root / "model")["combined_sha256"]
            if (
                _sha256(root / "report.json") != expected_report
                or artifact != expected_model
                or model_report.get("config", {}).get("seed") != seed
            ):
                raise ValueError(f"{recipe} seed {seed} binding differs")
            by_seed[seed] = _predict_examples(root, examples)
            inputs[recipe][str(seed)] = {
                "artifact_name": name,
                "report_sha256": expected_report,
                "model_artifact_sha256": expected_model,
            }
        ensemble = vote_predictions(by_seed, ENSEMBLE_THRESHOLD)
        predictions[recipe] = {"by_seed": by_seed, "ensemble": ensemble}
        metrics[recipe] = {
            "ensemble": _prediction_metrics(ensemble),
            "by_seed": {
                str(seed): _prediction_metrics(by_seed[seed])
                for seed in REPLICATION_SEEDS
            },
        }
    comparison, changes = _compare(
        predictions["round6"]["ensemble"],
        predictions["round7"]["ensemble"],
    )
    promoted = (
        metrics["round7"]["ensemble"]["f1"]
        > metrics["round6"]["ensemble"]["f1"]
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        predictions_path = staging / "predictions.json"
        changes_path = staging / "geometry_changes.json"
        predictions_path.write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        changes_path.write_text(
            json.dumps(changes, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report = {
            "schema_version": 1,
            "status": "round11_fresh_promotion_comparison",
            "formal_evaluation": True,
            "eligible_for_training": False,
            "candidate_model_blind_reference": True,
            "baseline": "round6_2_of_3_exact_geometry_ensemble",
            "challenger": "round7_2_of_3_exact_geometry_ensemble",
            "promotion_gate": "challenger_exact_f1_strictly_exceeds_baseline",
            "gate_passed": promoted,
            "decision": (
                "promote_round7_recipe"
                if promoted else "retain_round6_recipe"
            ),
            "metrics": metrics,
            "comparison": comparison,
            "determinism": {
                "torch_deterministic_algorithms": True,
                "warn_only": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
            },
            "inputs": {
                "reference_report_sha256": EXPECTED_REFERENCE_REPORT_SHA256,
                "reference_sha256": EXPECTED_REFERENCE_SHA256,
                "development_selection_report_sha256": (
                    EXPECTED_DEV_SELECTION_SHA256
                ),
                "models": inputs,
            },
            "git_commit": _git_commit_clean(),
            "outputs": {
                "predictions_sha256": _sha256(predictions_path),
                "geometry_changes_sha256": _sha256(changes_path),
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
        description="Run the fresh Round 11 formal promotion comparison."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--development-selection-report", type=Path, required=True
    )
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_promotion(
        args.reference,
        args.development_selection_report,
        args.artifacts_root,
        args.output,
    )
    print(json.dumps({
        "gate_passed": report["gate_passed"],
        "decision": report["decision"],
        "metrics": {
            name: row["ensemble"] for name, row in report["metrics"].items()
        },
        "comparison": report["comparison"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
