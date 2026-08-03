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
from p7_three_seed_select import passes_stability_gate


EXPECTED_BASELINE_SHA256 = (
    "9cd19f24d31941ccbdc95981eed152b3f33b2b271779cf3b3654931329024179"
)
EXPECTED_REPORTS = {
    20260727: "6eb0947e9522dddadd74248487470ca6350b0107aa58240377f2036b1d3477fe",
    20260728: "e8bbc587d5e607ac4b19c66c5e906c6051d4f5798fbb074a6d4b814cf208f09e",
    20260729: "044754c96803c58dd5b8205039d5f8ab12bf47922995db20297d81b5837a0d1c",
}
EXPERIMENT = "round8-lr2e-5"


def _snapshot(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def select_round8(
    baseline_path: Path,
    run_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Round 8 selection exists: {output_dir}")
    baseline, baseline_sha256 = _snapshot(baseline_path)
    if (
        baseline_sha256 != EXPECTED_BASELINE_SHA256
        or baseline.get("status") != "round4_round6_three_seed_diagnostic"
    ):
        raise ValueError("Round 6 baseline binding differs")
    dev_values = []
    evaluation_values = []
    inputs = {}
    per_seed = {}
    for seed in REPLICATION_SEEDS:
        path = run_root / f"ml-poc-round8-lr2e-5-seed-{seed}-v1" / "report.json"
        report, digest = _snapshot(path)
        control = report.get("seed_replication_control", {})
        if (
            digest != EXPECTED_REPORTS[seed]
            or report.get("config", {}).get("seed") != seed
            or report.get("config", {}).get("learning_rate") != 2e-5
            or control.get("experiment") != EXPERIMENT
            or control.get("dataset_kind") != "round7"
            or control.get("formal_evaluation") is not False
            or control.get("eligible_for_promotion") is not False
        ):
            raise ValueError(f"Round 8 seed {seed} binding differs")
        dev = float(report["dev_challenge"]["exact"]["f1"])
        evaluation = float(report["evaluation"]["exact"]["f1"])
        dev_values.append(dev)
        evaluation_values.append(evaluation)
        inputs[str(seed)] = digest
        per_seed[str(seed)] = {
            "selected_epoch": report["config"]["selected_epoch"],
            "dev_exact_f1": dev,
            "reused_evaluation_exact_f1_diagnostic_only": evaluation,
            "model_artifact_sha256": control["model_artifact"]["combined_sha256"],
        }
    round6 = baseline["aggregate"]["dev"]["round6"]
    round8 = {
        "mean_f1": statistics.mean(dev_values),
        "population_stdev_f1": statistics.pstdev(dev_values),
        "min_f1": min(dev_values),
        "max_f1": max(dev_values),
    }
    passed = passes_stability_gate(round6, round8)
    report = {
        "schema_version": 1,
        "status": "round8_lower_lr_stability_selection",
        "formal_evaluation": False,
        "eligible_for_promotion": False,
        "experiment": EXPERIMENT,
        "selection_inputs": "Juan 27 dev mean and worst-seed exact F1 only",
        "reused_juan76_role": "diagnostic_only_not_used_for_selection",
        "gate": {
            "mean_dev_must_exceed_round6": True,
            "worst_seed_dev_must_exceed_round6": True,
            "passed": passed,
        },
        "decision": (
            "round8_recipe_may_enter_fresh_promotion_evaluation"
            if passed
            else "continue_once_to_predeclared_lr_midpoint"
        ),
        "fresh_promotion_evaluation_consumed": False,
        "per_seed": per_seed,
        "aggregate": {
            "round6": round6,
            "round8": round8,
            "mean_delta": round8["mean_f1"] - round6["mean_f1"],
            "worst_seed_delta": round8["min_f1"] - round6["min_f1"],
            "reused_evaluation_mean_diagnostic_only": statistics.mean(
                evaluation_values
            ),
        },
        "inputs": {
            "round6_comparison_report_sha256": baseline_sha256,
            "round8_seed_report_sha256": inputs,
        },
        "git_commit": _git_commit_clean(),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        path = staging / "report.json"
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        path.chmod(stat.S_IREAD)
        staging.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply the Round 8 stability gate.")
    parser.add_argument("--round6-comparison", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = select_round8(args.round6_comparison, args.run_root, args.output)
    print(json.dumps({
        "gate": report["gate"],
        "decision": report["decision"],
        "aggregate": report["aggregate"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
