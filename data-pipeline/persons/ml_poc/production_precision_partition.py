from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

from p3_compact import _git_commit_clean


PARTITION_SEED = 20260807
PARTITIONS = ("fit", "calibration", "confirmation")
FRACTIONS = {"fit": 5 / 7, "calibration": 1 / 7, "confirmation": 1 / 7}
STRATA = (
    "uniform_random",
    "role_appellation",
    "foreign_title",
    "boundary_anaphora",
)
METRICS = ("examples", "spans", *STRATA)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def assign_juans(groups: dict[int, dict[str, int]]) -> dict[int, str]:
    totals = {
        metric: sum(row[metric] for row in groups.values())
        for metric in METRICS
    }
    targets = {
        partition: {
            metric: totals[metric] * FRACTIONS[partition]
            for metric in METRICS
        }
        for partition in PARTITIONS
    }
    current = {
        partition: dict.fromkeys(METRICS, 0)
        for partition in PARTITIONS
    }

    def order_hash(juan: int) -> str:
        return hashlib.sha256(
            f"{PARTITION_SEED}:{juan}".encode("ascii")
        ).hexdigest()

    order = sorted(groups, key=lambda juan: (
        -groups[juan]["examples"],
        -groups[juan]["spans"],
        -max(groups[juan][stratum] for stratum in STRATA),
        order_hash(juan),
    ))

    def objective(trial: dict[str, dict[str, int]]) -> float:
        return sum(
            (
                (trial[partition][metric] - targets[partition][metric])
                / max(1, targets[partition][metric])
            ) ** 2
            for partition in PARTITIONS
            for metric in METRICS
        )

    assignments = {}
    for juan in order:
        choices = []
        for rank, partition in enumerate(PARTITIONS):
            trial = {
                name: dict(values) for name, values in current.items()
            }
            for metric in METRICS:
                trial[partition][metric] += groups[juan][metric]
            choices.append((objective(trial), rank, partition))
        _score, _rank, selected = min(choices)
        assignments[juan] = selected
        for metric in METRICS:
            current[selected][metric] += groups[juan][metric]
    return assignments


def freeze_partition(
    cumulative_dir: Path,
    round1_dir: Path,
    round1_roles_path: Path,
    round2_dir: Path,
    round2_roles_path: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"precision partition exists: {output_dir}")
    cumulative_manifest_path = cumulative_dir / "manifest.json"
    cumulative_train_path = cumulative_dir / "train.jsonl"
    source_dirs = {1: round1_dir, 2: round2_dir}
    role_paths = {1: round1_roles_path, 2: round2_roles_path}
    input_paths = {
        cumulative_manifest_path,
        cumulative_train_path,
        *(root / "manifest.json" for root in source_dirs.values()),
        *role_paths.values(),
    }
    snapshots = {path: path.read_bytes() for path in input_paths}
    cumulative = json.loads(snapshots[cumulative_manifest_path])
    source_manifests = {
        number: json.loads(snapshots[root / "manifest.json"])
        for number, root in source_dirs.items()
    }
    role_manifests = {
        number: json.loads(snapshots[path]) for number, path in role_paths.items()
    }
    if (
        cumulative.get("status")
        != "ml_production_round2_cumulative_frozen_dataset"
        or cumulative.get("outputs", {}).get("train_sha256")
        != _digest(snapshots[cumulative_train_path])
        or cumulative.get("inputs", {}).get("round1_manifest_sha256")
        != _digest(snapshots[round1_dir / "manifest.json"])
        or cumulative.get("inputs", {}).get("round2_manifest_sha256")
        != _digest(snapshots[round2_dir / "manifest.json"])
    ):
        raise ValueError("cumulative precision dataset binding differs")
    role_by_key = {}
    provenance_by_key = {}
    for number in (1, 2):
        source = source_manifests[number]
        roles = role_manifests[number]
        if (
            source.get("status")
            != f"ml_production_round{number}_frozen_dataset"
            or source.get("inputs", {}).get("private_roles_sha256")
            != _digest(snapshots[role_paths[number]])
            or roles.get("status") != "ml_production_private_task_roles"
        ):
            raise ValueError(f"Round {number} partition provenance differs")
        for row in roles["selected_jies"]:
            if row["split"] != "train":
                continue
            key = int(row["juan"]), int(row["jie_index"])
            if key in role_by_key:
                raise ValueError(f"duplicate production training jie: {key}")
            role_by_key[key] = str(row["stratum"])
            provenance_by_key[key] = {
                "round": number,
                "task_id": str(row["task_id"]),
                "stratum": str(row["stratum"]),
            }
    rows = [
        json.loads(line)
        for line in snapshots[cumulative_train_path].decode("utf-8").splitlines()
        if line
    ]
    row_by_key = {
        (int(row["juan"]), int(row["jie_index"])): row for row in rows
    }
    if (
        len(rows) != 280
        or len(row_by_key) != 280
        or set(row_by_key) != set(role_by_key)
        or Counter(role_by_key.values())
        != {
            "uniform_random": 194,
            "role_appellation": 40,
            "foreign_title": 6,
            "boundary_anaphora": 40,
        }
    ):
        raise ValueError("cumulative training role inventory differs")
    groups: dict[int, dict[str, int]] = {}
    for key, row in row_by_key.items():
        juan = key[0]
        group = groups.setdefault(juan, dict.fromkeys(METRICS, 0))
        group["examples"] += 1
        group["spans"] += int(row["span_count"])
        group[role_by_key[key]] += 1
    assignments = assign_juans(groups)
    partition_rows = {name: [] for name in PARTITIONS}
    partition_metrics = {
        name: dict.fromkeys(METRICS, 0) for name in PARTITIONS
    }
    private_rows = []
    for key, row in sorted(row_by_key.items()):
        partition = assignments[key[0]]
        partition_rows[partition].append(row)
        partition_metrics[partition]["examples"] += 1
        partition_metrics[partition]["spans"] += int(row["span_count"])
        partition_metrics[partition][role_by_key[key]] += 1
        private_rows.append({
            "juan": key[0],
            "jie_index": key[1],
            "partition": partition,
            **provenance_by_key[key],
        })
    totals = {
        metric: sum(values[metric] for values in partition_metrics.values())
        for metric in METRICS
    }
    for partition in ("calibration", "confirmation"):
        metrics = partition_metrics[partition]
        if (
            any(metrics[stratum] < 1 for stratum in STRATA)
            or not 0.8 <= (
                metrics["examples"] / (totals["examples"] / 7)
            ) <= 1.2
            or not 0.8 <= (
                metrics["spans"] / (totals["spans"] / 7)
            ) <= 1.2
        ):
            raise ValueError(f"{partition} partition is infeasible")

    git_commit = _git_commit_clean()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        for partition, selected_rows in partition_rows.items():
            _write_jsonl(staging / f"{partition}.jsonl", selected_rows)
        private_path = staging / "private.json"
        private_path.write_text(
            json.dumps({
                "schema_version": 1,
                "status": "ml_production_precision_private_partition",
                "seed": PARTITION_SEED,
                "rows": private_rows,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "status": "ml_production_precision_partition",
            "candidate_model_blind": True,
            "formal_evaluation": False,
            "seed": PARTITION_SEED,
            "algorithm": (
                "juan_grouped_normalized_squared_target_deviation_5_1_1"
            ),
            "partitions": partition_metrics,
            "juans": {
                name: sorted(
                    juan for juan, partition in assignments.items()
                    if partition == name
                )
                for name in PARTITIONS
            },
            "inputs": {
                "cumulative_manifest_sha256": _digest(
                    snapshots[cumulative_manifest_path]
                ),
                "round1_manifest_sha256": _digest(
                    snapshots[round1_dir / "manifest.json"]
                ),
                "round1_roles_sha256": _digest(snapshots[round1_roles_path]),
                "round2_manifest_sha256": _digest(
                    snapshots[round2_dir / "manifest.json"]
                ),
                "round2_roles_sha256": _digest(snapshots[round2_roles_path]),
            },
            "outputs": {
                **{
                    f"{partition}_sha256": _sha256(
                        staging / f"{partition}.jsonl"
                    )
                    for partition in PARTITIONS
                },
                "private_sha256": _sha256(private_path),
            },
            "git_commit": git_commit,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in staging.iterdir():
            path.chmod(0o444)
        staging.chmod(0o555)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the revision-2 fit/calibration/confirmation partition."
    )
    parser.add_argument("--cumulative", type=Path, required=True)
    parser.add_argument("--round1", type=Path, required=True)
    parser.add_argument("--round1-roles", type=Path, required=True)
    parser.add_argument("--round2", type=Path, required=True)
    parser.add_argument("--round2-roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = freeze_partition(
        args.cumulative,
        args.round1,
        args.round1_roles,
        args.round2,
        args.round2_roles,
        args.output,
    )
    print(json.dumps(report["partitions"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
