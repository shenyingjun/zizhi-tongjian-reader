from __future__ import annotations

import argparse
import hashlib
import json
import stat
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

from p3_compact import _git_commit_clean
from p3_round3_compare import _metrics, compare_predictions
from p6_seed_replication_compare import _surfaced_changes
from p6_seed_replication_train import REPLICATION_SEEDS


EXPECTED_DEV_SHA256 = (
    "5fb15836f52baa989b1e65c17c19f2090efb9ca96b5ce075a484484de411a09f"
)
EXPECTED_PREDICTIONS = {
    20260727: {
        "round6": "0ae1eadc2098026439635ad7da71ddd97d1c1c9232221e4b6b75417211a0e53f",
        "round7": "bd6c60a2e9af0bc19634193aad44bc66a888bee80afcadec93dd6c7cac118d60",
    },
    20260728: {
        "round6": "176689f983993b39247c2d39735edcd47569d4419bfdd8936ab6a40767a28094",
        "round7": "464f45d4fdb5f7862595c20aced3e504313a9cc7d56deef5682d2c8b1d1b4cb8",
    },
    20260729: {
        "round6": "c79092d5df34beb58190a0fcd1868082faae42d8d55ada7b61ae53d9fe401fd4",
        "round7": "9d47871f8838e4a136381fbafa712b1c3ab644ee9f478005761e83137a303728",
    },
}


def _snapshot(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _effect(change: dict) -> str:
    if change["action"] == "added":
        return (
            "recovery"
            if change["reference_status"] == "true_positive"
            else "added_false_positive"
        )
    return (
        "regression"
        if change["reference_status"] == "true_positive"
        else "removed_false_positive"
    )


def _stability(changes_by_seed: dict[int, list[dict]]) -> dict:
    seeds_by_change: dict[tuple, list[int]] = defaultdict(list)
    rows_by_change = {}
    for seed, rows in changes_by_seed.items():
        for row in rows:
            key = (
                row["id"],
                row["action"],
                row["para_id"],
                row["start"],
                row["end"],
                row["surface"],
                row["reference_status"],
            )
            seeds_by_change[key].append(seed)
            rows_by_change[key] = row
    result = {"three_seeds": [], "two_seeds": [], "one_seed": []}
    for key in sorted(seeds_by_change):
        seeds = sorted(seeds_by_change[key])
        item = {
            **rows_by_change[key],
            "effect": _effect(rows_by_change[key]),
            "seeds": seeds,
        }
        result[{3: "three_seeds", 2: "two_seeds", 1: "one_seed"}[len(seeds)]].append(
            item
        )
    result["counts"] = {
        name: len(rows)
        for name, rows in result.items()
        if isinstance(rows, list)
    }
    result["harmful_counts"] = {
        name: sum(
            row["effect"] in {"regression", "added_false_positive"}
            for row in rows
        )
        for name, rows in result.items()
        if isinstance(rows, list)
    }
    return result


def attribute_round7(
    run_root: Path,
    dev_path: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Round 7 attribution exists: {output_dir}")
    dev_raw = dev_path.read_bytes()
    dev_sha256 = hashlib.sha256(dev_raw).hexdigest()
    if dev_sha256 != EXPECTED_DEV_SHA256:
        raise ValueError("Juan 27 dev binding differs")

    paths = {
        20260727: {
            "round6": run_root / "ml-poc-round6-controlled-model-v1"
            / "dev_predictions.json",
            "round7": run_root / "ml-poc-round7-seed-20260727-v1"
            / "dev_predictions.json",
        },
        20260728: {
            "round6": run_root / "ml-poc-round6-attribution-seed-20260728-v1"
            / "dev_predictions.json",
            "round7": run_root / "ml-poc-round7-seed-20260728-v1"
            / "dev_predictions.json",
        },
        20260729: {
            "round6": run_root / "ml-poc-round6-attribution-seed-20260729-v1"
            / "dev_predictions.json",
            "round7": run_root / "ml-poc-round7-seed-20260729-v1"
            / "dev_predictions.json",
        },
    }
    per_seed = {}
    changes_by_seed = {}
    input_hashes = {}
    f1_deltas = []
    for seed in REPLICATION_SEEDS:
        old_rows, old_sha256 = _snapshot(paths[seed]["round6"])
        new_rows, new_sha256 = _snapshot(paths[seed]["round7"])
        hashes = {"round6": old_sha256, "round7": new_sha256}
        if hashes != EXPECTED_PREDICTIONS[seed]:
            raise ValueError(f"seed {seed} prediction binding differs")
        old_metrics = _metrics(old_rows)
        new_metrics = _metrics(new_rows)
        comparison = compare_predictions(old_rows, new_rows)
        changes = _surfaced_changes(old_rows, new_rows)
        for row in changes:
            row["effect"] = _effect(row)
        delta = (
            new_metrics["exact"]["f1"] - old_metrics["exact"]["f1"]
        )
        f1_deltas.append(delta)
        per_seed[str(seed)] = {
            "round6_metrics": old_metrics,
            "round7_metrics": new_metrics,
            "f1_delta": delta,
            "attribution": comparison["attribution"],
            "prediction_geometry": comparison["prediction_geometry"],
            "changes": changes,
        }
        changes_by_seed[seed] = changes
        input_hashes[str(seed)] = hashes

    stability = _stability(changes_by_seed)
    stable_harmful = [
        row
        for bucket in ("three_seeds", "two_seeds")
        for row in stability[bucket]
        if row["effect"] in {"regression", "added_false_positive"}
    ]
    report = {
        "schema_version": 1,
        "status": "round7_dev_failure_attribution",
        "formal_evaluation": False,
        "selection_split": "Juan 27 dev only",
        "juan76_used": False,
        "seeds": list(REPLICATION_SEEDS),
        "aggregate": {
            "mean_f1_delta": statistics.mean(f1_deltas),
            "population_stdev_f1_delta": statistics.pstdev(f1_deltas),
            "min_f1_delta": min(f1_deltas),
            "max_f1_delta": max(f1_deltas),
            "positive_seeds": sum(value > 0 for value in f1_deltas),
            "negative_seeds": sum(value < 0 for value in f1_deltas),
        },
        "per_seed": per_seed,
        "change_stability": stability,
        "stable_harmful_changes": stable_harmful,
        "diagnosis": {
            "stable_harmful_change_count": len(stable_harmful),
            "seed_specific_harmful_change_count": stability[
                "harmful_counts"
            ]["one_seed"],
            "next_experiment": (
                "Build targeted Round 7 ablations around the training geometries "
                "matching stable harmful boundary/length families; do not remove "
                "labels solely because of one-seed changes."
            ),
        },
        "inputs": {
            "dev_sha256": dev_sha256,
            "prediction_sha256": input_hashes,
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
        description="Attribute Round 7 dev instability across three seeds."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = attribute_round7(args.run_root, args.dev, args.output)
    print(json.dumps({
        "aggregate": report["aggregate"],
        "change_stability_counts": report["change_stability"]["counts"],
        "harmful_counts": report["change_stability"]["harmful_counts"],
        "stable_harmful_changes": report["stable_harmful_changes"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
