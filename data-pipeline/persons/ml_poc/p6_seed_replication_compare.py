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
from p3_round3_compare import _metrics, _require_metrics, compare_predictions
from p6_seed_replication_train import (
    CUBLAS_WORKSPACE_CONFIG,
    DATASETS,
    REPLICATION_SEEDS,
)


EXPECTED_RUNS = {
    20260727: {
        "round4": {
            "report.json": "c79d3162146254b964ab1463ccabe3c597e8b314d643d8d793579b07de4ebd2d",
            "history.json": "e5583e58bf74596d501f1bda4c42a9040a59e1a3f13c1dcb0e18e7b5c8aa9e3c",
            "dev_predictions.json": "37d2dae3335e0f275a19f52c3b21ce0a91757c96d6df69860741707a47ddbd0c",
            "evaluation_predictions.json": "1e28c0510ef641740fbf6e05015f34dc902a1186e64e83c3e186876b36f1f3b3",
        },
        "round6": {
            "report.json": "072090a323146cfa8f83da3e86098dc3d042f9b4ab28aaa2f78f86b9956e1afa",
            "history.json": "87d5953c5a4a911281c5f015204aa7092334a602ccb8bac10ce2f7e9ed926111",
            "dev_predictions.json": "0ae1eadc2098026439635ad7da71ddd97d1c1c9232221e4b6b75417211a0e53f",
            "evaluation_predictions.json": "fcf2ebea6b7665074b50fbba85bc1723c9a56feb9cf80ceeab29485fffad6835",
        },
    },
    20260728: {
        "round4": {
            "report.json": "4c3c48025845024e2f78cac214f858c9eed8217cfe4e1e2665f1801aef31f9cc",
            "history.json": "61113cc87849bd6883bc4d57e01a7b6294d656ad38af1191efbbe8d5c4f5f0ca",
            "dev_predictions.json": "4269b8aaedde9beb9d5bc5cb2e3eec65d2127667afcf16084935def5d41dda11",
            "evaluation_predictions.json": "619579635290ccf35a45cd977740dd30b3101471c27730629c1fd3509b2e9156",
        },
        "round6": {
            "report.json": "ebbbe49fabaa174868c99fe9e6e9c82386695d66d6b22726502b2fa2057837da",
            "history.json": "12278c1aeaec3e6ca447efc938525c0ec0a8c2d6d65b528662b0f1b1469a4aff",
            "dev_predictions.json": "176689f983993b39247c2d39735edcd47569d4419bfdd8936ab6a40767a28094",
            "evaluation_predictions.json": "c3479da527a8c444f173bd74d11eb9107fa51765eb996463a3763b05859abd5f",
        },
    },
    20260729: {
        "round4": {
            "report.json": "b97a0fa177ac52c7ba1278057131456659eac7f20d07287ea616e0c3a0253eda",
            "history.json": "664cc1c0177dce52e0c9ae6d672dbbe232b30cc2774a539d36bb963a3bd5878a",
            "dev_predictions.json": "b6071230ec5ed4b1009817d80da4d7411e8041322fb94ca49f201ae30effe112",
            "evaluation_predictions.json": "9b2ece4111effde604ca522846dc70826f1872c1f708ccfeb4e74d04d6f5825b",
        },
        "round6": {
            "report.json": "a21c58ace15b633ad46c9b9e57522ac69ee42639f7ce1ee92029e87d2ea4925b",
            "history.json": "410ec6f09d712d8d1b9c3e56aec03d1db6c3bf0e068b6f5d07db27fca519d1d3",
            "dev_predictions.json": "c79092d5df34beb58190a0fcd1868082faae42d8d55ada7b61ae53d9fe401fd4",
            "evaluation_predictions.json": "cc6d84fa2ebc4dfaf30732e2c1332549eb8e421a95c77aa7a18cd93d9251fa64",
        },
    },
}
EXPECTED_DETERMINISM = {
    "torch_deterministic_algorithms": True,
    "warn_only": False,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _span_map(rows: list[dict]) -> dict[tuple[int, int, int], str]:
    result = {}
    for span in rows:
        geometry = (
            int(span["para_id"]), int(span["start"]), int(span["end"]),
        )
        if geometry in result:
            raise ValueError(f"duplicate prediction geometry: {geometry}")
        result[geometry] = str(span["surface"])
    return result


def _surfaced_changes(old_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    old_by_id = {str(row["id"]): row for row in old_rows}
    new_by_id = {str(row["id"]): row for row in new_rows}
    if (
        len(old_by_id) != len(old_rows)
        or len(new_by_id) != len(new_rows)
        or set(old_by_id) != set(new_by_id)
    ):
        raise ValueError("prediction identities differ")
    changes = []
    for identity in sorted(old_by_id):
        reference = _span_map(old_by_id[identity]["reference_spans"])
        if reference != _span_map(new_by_id[identity]["reference_spans"]):
            raise ValueError(f"reference differs: {identity}")
        old = _span_map(old_by_id[identity]["prediction_spans"])
        new = _span_map(new_by_id[identity]["prediction_spans"])
        for action, geometries, surfaces in (
            ("added", set(new) - set(old), new),
            ("removed", set(old) - set(new), old),
        ):
            for geometry in sorted(geometries):
                changes.append({
                    "id": identity,
                    "action": action,
                    "surface": surfaces[geometry],
                    "reference_status": (
                        "true_positive" if geometry in reference
                        else "false_positive"
                    ),
                    "para_id": geometry[0],
                    "start": geometry[1],
                    "end": geometry[2],
                })
    return changes


def _summarize_metrics(pairs: dict) -> dict:
    result = {}
    for split in ("dev", "evaluation"):
        result[split] = {}
        for label in ("round4", "round6"):
            values = [
                pair[split]["metrics"][label]["exact"]["f1"]
                for pair in pairs.values()
            ]
            result[split][label] = {
                "mean_f1": statistics.mean(values),
                "population_stdev_f1": statistics.pstdev(values),
                "min_f1": min(values),
                "max_f1": max(values),
            }
        deltas = [
            pair[split]["f1_delta"] for pair in pairs.values()
        ]
        result[split]["paired_f1_delta"] = {
            "mean": statistics.mean(deltas),
            "population_stdev": statistics.pstdev(deltas),
            "min": min(deltas),
            "max": max(deltas),
            "positive_seeds": sum(delta > 0 for delta in deltas),
            "negative_seeds": sum(delta < 0 for delta in deltas),
            "zero_seeds": sum(delta == 0 for delta in deltas),
        }
    return result


def _change_stability(pairs: dict, split: str) -> dict:
    seeds_by_change = defaultdict(list)
    rows_by_key = {}
    for seed, pair in pairs.items():
        for row in pair[split]["surfaced_changes"]:
            key = (
                row["id"], row["action"], row["para_id"], row["start"],
                row["end"], row["surface"], row["reference_status"],
            )
            seeds_by_change[key].append(seed)
            rows_by_key[key] = row
    groups = {"all_three_seeds": [], "two_seeds": [], "one_seed": []}
    for key in sorted(seeds_by_change):
        seeds = sorted(seeds_by_change[key])
        item = {**rows_by_key[key], "seeds": seeds}
        bucket = {
            3: "all_three_seeds",
            2: "two_seeds",
            1: "one_seed",
        }[len(seeds)]
        groups[bucket].append(item)
    return {
        **groups,
        "counts": {name: len(rows) for name, rows in groups.items()},
    }


def _validate_run(
    seed: int,
    label: str,
    snapshots: dict[str, bytes],
    hashes: dict[str, str],
) -> dict:
    if hashes != EXPECTED_RUNS[seed][label]:
        raise ValueError(f"{label} seed {seed} hashes differ")
    report = json.loads(snapshots["report.json"])
    control = report.get("seed_replication_control", {})
    if (
        control.get("dataset_kind") != label
        or control.get("dataset_manifest_sha256")
        != DATASETS[label]["manifest_sha256"]
        or control.get("full_control", {}).get("seed") != seed
        or control.get("determinism") != EXPECTED_DETERMINISM
        or control.get("formal_evaluation") is not False
        or control.get("eligible_for_promotion") is not False
        or any(
            control.get("run_artifacts", {}).get(name) != hashes[name]
            for name in (
                "history.json", "dev_predictions.json",
                "evaluation_predictions.json",
            )
        )
    ):
        raise ValueError(f"{label} seed {seed} provenance differs")
    return report


def compare_seed_replications(input_root: Path, output_dir: Path) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"seed comparison exists: {output_dir}")
    run_inputs = {}
    pairs = {}
    controls = {}
    for seed in REPLICATION_SEEDS:
        run_inputs[str(seed)] = {}
        rows = {}
        reports = {}
        for label in ("round4", "round6"):
            directory = (
                input_root
                / f"ml-poc-{label}-deterministic-seed-{seed}-v1"
            )
            snapshots = {
                name: (directory / name).read_bytes()
                for name in (
                    "report.json", "history.json", "dev_predictions.json",
                    "evaluation_predictions.json",
                )
            }
            hashes = {name: _sha256(raw) for name, raw in snapshots.items()}
            reports[label] = _validate_run(seed, label, snapshots, hashes)
            run_inputs[str(seed)][label] = hashes
            rows[label] = {
                split: json.loads(snapshots[f"{split}_predictions.json"])
                for split in ("dev", "evaluation")
            }
        pair = {}
        for split in ("dev", "evaluation"):
            metrics = {
                label: _metrics(rows[label][split])
                for label in ("round4", "round6")
            }
            report_key = "dev_challenge" if split == "dev" else "evaluation"
            for label in ("round4", "round6"):
                _require_metrics(
                    metrics[label], reports[label][report_key],
                    f"{label} seed {seed} {split}",
                )
            comparison = compare_predictions(
                rows["round4"][split], rows["round6"][split]
            )
            old_f1 = metrics["round4"]["exact"]["f1"]
            new_f1 = metrics["round6"]["exact"]["f1"]
            pair[split] = {
                "metrics": metrics,
                "f1_delta": new_f1 - old_f1,
                "attribution": comparison["attribution"],
                "prediction_geometry": comparison["prediction_geometry"],
                "surfaced_changes": _surfaced_changes(
                    rows["round4"][split], rows["round6"][split]
                ),
            }
        pairs[seed] = pair
        pair_controls = {
            label: {
                key: value
                for key, value in reports[label][
                    "seed_replication_control"
                ]["full_control"].items()
                if key != "seed"
            }
            for label in ("round4", "round6")
        }
        if pair_controls["round4"] != pair_controls["round6"]:
            raise ValueError(f"paired controls differ for seed {seed}")
        controls[str(seed)] = pair_controls["round4"]
    if len({json.dumps(value, sort_keys=True) for value in controls.values()}) != 1:
        raise ValueError("non-seed controls differ across replications")

    aggregate = _summarize_metrics(pairs)
    evaluation_delta = aggregate["evaluation"]["paired_f1_delta"]
    direction = (
        "round6_degradation"
        if evaluation_delta["negative_seeds"] == len(REPLICATION_SEEDS)
        else "round6_improvement"
        if evaluation_delta["positive_seeds"] == len(REPLICATION_SEEDS)
        else "mixed"
    )
    report = {
        "schema_version": 1,
        "status": "round4_round6_three_seed_diagnostic",
        "formal_evaluation": False,
        "eligible_for_promotion": False,
        "git_commit": _git_commit_clean(),
        "seeds": list(REPLICATION_SEEDS),
        "inputs": run_inputs,
        "paired_runs": pairs,
        "aggregate": aggregate,
        "change_stability": {
            split: _change_stability(pairs, split)
            for split in ("dev", "evaluation")
        },
        "decision": {
            "stable_round6_degradation": (
                evaluation_delta["negative_seeds"] == len(REPLICATION_SEEDS)
            ),
            "stable_round6_improvement": (
                evaluation_delta["positive_seeds"] == len(REPLICATION_SEEDS)
            ),
            "observed_direction": direction,
            "active_model": "round3",
            "round7_ready": False,
            "conclusion": (
                "The original Round 4 to Round 6 decline is seed-sensitive, "
                "not a stable consequence of adding Round 5 data. Two of three "
                "paired seeds improve reused Juan 76 F1, while one declines."
            ),
            "claim_limit": (
                "Juan 27 and Juan 76 are reused diagnostics; these replications "
                "cannot promote Round 6 or replace a fresh sealed evaluation."
            ),
        },
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
        staging.replace(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare three deterministic Round 4/6 seed pairs."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_seed_replications(args.input_root, args.output)
    print(json.dumps({
        "aggregate": report["aggregate"],
        "change_stability_counts": {
            split: value["counts"]
            for split, value in report["change_stability"].items()
        },
        "decision": report["decision"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
