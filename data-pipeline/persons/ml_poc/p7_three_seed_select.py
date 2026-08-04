from __future__ import annotations

import argparse
import hashlib
import json
import stat
import statistics
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from p6_seed_replication_train import REPLICATION_SEEDS


EXPECTED_BASELINE_SHA256 = (
    "9cd19f24d31941ccbdc95981eed152b3f33b2b271779cf3b3654931329024179"
)
EXPECTED_REPORTS = {
    20260727: "0a75d04c04972bc2895992b240b914a1e2d0c485b9a520bdbd7d5fecd8e4d684",
    20260728: "e605c9d6f1e44cd51ae5151054da2480bfe944dfaed1e81113bf8bd71ec1b62c",
    20260729: "2ad035b62a733462e4817da2a1e4e53b971347416bc2c8c57ca5b9169420473a",
}
EXPECTED_DATASET_SHA256 = (
    "09c2724e346b8df5b0ace423016155167790bc7324bf8876f1bfa5942e958eb5"
)


def _snapshot(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _summary(values: list[float]) -> dict:
    return {
        "mean_f1": statistics.mean(values),
        "population_stdev_f1": statistics.pstdev(values),
        "min_f1": min(values),
        "max_f1": max(values),
    }


def passes_stability_gate(round6: dict, round7: dict) -> bool:
    return (
        round7["mean_f1"] > round6["mean_f1"]
        and round7["min_f1"] > round6["min_f1"]
    )


def select_round7(
    baseline_report_path: Path,
    run_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Round 7 selection exists: {output_dir}")
    baseline, baseline_sha256 = _snapshot(baseline_report_path)
    if (
        baseline_sha256 != EXPECTED_BASELINE_SHA256
        or baseline.get("status") != "round4_round6_three_seed_diagnostic"
        or baseline.get("formal_evaluation") is not False
        or baseline.get("eligible_for_promotion") is not False
    ):
        raise ValueError("Round 6 three-seed baseline binding differs")

    run_inputs = {}
    dev_values = []
    evaluation_values = []
    per_seed = {}
    for seed in REPLICATION_SEEDS:
        report_path = run_root / f"ml-poc-round7-seed-{seed}-v1" / "report.json"
        report, report_sha256 = _snapshot(report_path)
        control = report.get("seed_replication_control", {})
        if (
            report_sha256 != EXPECTED_REPORTS[seed]
            or report.get("config", {}).get("seed") != seed
            or control.get("dataset_kind") != "round7"
            or control.get("dataset_manifest_sha256")
            != EXPECTED_DATASET_SHA256
            or control.get("formal_evaluation") is not False
            or control.get("eligible_for_promotion") is not False
        ):
            raise ValueError(f"Round 7 seed {seed} provenance differs")
        dev_f1 = float(report["dev_challenge"]["exact"]["f1"])
        evaluation_f1 = float(report["evaluation"]["exact"]["f1"])
        dev_values.append(dev_f1)
        evaluation_values.append(evaluation_f1)
        run_inputs[str(seed)] = report_sha256
        per_seed[str(seed)] = {
            "selected_epoch": int(report["config"]["selected_epoch"]),
            "dev_exact_f1": dev_f1,
            "reused_evaluation_exact_f1_diagnostic_only": evaluation_f1,
            "model_artifact_sha256": control["model_artifact"][
                "combined_sha256"
            ],
        }

    round6_dev = baseline["aggregate"]["dev"]["round6"]
    round6_evaluation = baseline["aggregate"]["evaluation"]["round6"]
    round7_dev = _summary(dev_values)
    round7_evaluation = _summary(evaluation_values)
    passed = passes_stability_gate(round6_dev, round7_dev)
    report = {
        "schema_version": 1,
        "status": "round7_three_seed_stability_selection",
        "formal_evaluation": False,
        "eligible_for_promotion": False,
        "selection_inputs": "Juan 27 dev mean and worst-seed exact F1 only",
        "reused_juan76_role": "diagnostic_only_not_used_for_selection",
        "gate": {
            "mean_dev_must_exceed_round6": True,
            "worst_seed_dev_must_exceed_round6": True,
            "passed": passed,
        },
        "decision": (
            "retain_existing_active_and_challenger_models"
            if not passed
            else "round7_recipe_may_enter_fresh_promotion_evaluation"
        ),
        "fresh_promotion_evaluation_consumed": False,
        "seeds": list(REPLICATION_SEEDS),
        "per_seed": per_seed,
        "aggregate": {
            "dev": {
                "round6": round6_dev,
                "round7": round7_dev,
                "mean_delta": round7_dev["mean_f1"] - round6_dev["mean_f1"],
                "worst_seed_delta": round7_dev["min_f1"] - round6_dev["min_f1"],
            },
            "reused_evaluation_diagnostic_only": {
                "round6": round6_evaluation,
                "round7": round7_evaluation,
                "mean_delta": (
                    round7_evaluation["mean_f1"]
                    - round6_evaluation["mean_f1"]
                ),
            },
        },
        "inputs": {
            "round6_comparison_report_sha256": baseline_sha256,
            "round7_seed_report_sha256": run_inputs,
        },
        "git_commit": _git_commit_clean(),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        report_path = staging / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report_path.chmod(stat.S_IREAD)
        staging.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the Round 7 three-seed stability gate."
    )
    parser.add_argument("--round6-comparison", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = select_round7(
        args.round6_comparison, args.run_root, args.output
    )
    print(json.dumps({
        "gate": report["gate"],
        "decision": report["decision"],
        "aggregate": report["aggregate"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
