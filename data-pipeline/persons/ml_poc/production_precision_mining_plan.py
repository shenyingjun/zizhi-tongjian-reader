from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean


# Revision-4 section 5.3.1 replaces the section 5.1 ordering seed with this value
# for the fit-only mining fold split. Everything else about the normalized squared
# target-deviation objective is unchanged; the split now targets one fifth of every
# vector component per fold and resolves placement ties by fold number 1..5.
MINING_ORDER_SEED = 20260813
FOLDS = 5
FOLD_NUMBERS = tuple(range(1, FOLDS + 1))

PARTITION_STATUS = "ml_production_precision_partition"
PRIVATE_STATUS = "ml_production_precision_private_partition"

STRATA = (
    "uniform_random",
    "role_appellation",
    "foreign_title",
    "boundary_anaphora",
)
METRICS = ("examples", "spans", *STRATA)

EXPECTED_FIT_EXAMPLES = 189
EXPECTED_FIT_JUANS = 28


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load(raw: bytes) -> dict:
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _rows_from(raw: bytes) -> list[dict]:
    return [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
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


def order_juans(groups: dict[int, dict[str, int]], seed: int) -> list[int]:
    """Section 5.1 ordering with the revision-4 seed override.

    Descending examples, descending reference spans, descending maximum stratum
    count, then ascending SHA-256 of ``<seed>:<juan>``.
    """
    def order_hash(juan: int) -> str:
        return hashlib.sha256(f"{seed}:{juan}".encode("ascii")).hexdigest()

    return sorted(groups, key=lambda juan: (
        -groups[juan]["examples"],
        -groups[juan]["spans"],
        -max(groups[juan][stratum] for stratum in STRATA),
        order_hash(juan),
    ))


def assign_folds(
    groups: dict[int, dict[str, int]],
    *,
    seed: int = MINING_ORDER_SEED,
    folds: int = FOLDS,
) -> dict[int, int]:
    """Greedily group juans into ``folds`` folds minimizing normalized squared
    target deviation across every fold and vector component.

    Each fold targets one ``folds``-th of the global vector. Placement ties are
    resolved by ascending fold number ``1..folds``.
    """
    fold_numbers = tuple(range(1, folds + 1))
    totals = {
        metric: sum(row[metric] for row in groups.values())
        for metric in METRICS
    }
    target = {
        metric: totals[metric] / folds
        for metric in METRICS
    }
    current = {
        fold: dict.fromkeys(METRICS, 0)
        for fold in fold_numbers
    }

    def objective(trial: dict[int, dict[str, int]]) -> float:
        return sum(
            (
                (trial[fold][metric] - target[metric])
                / max(1, target[metric])
            ) ** 2
            for fold in fold_numbers
            for metric in METRICS
        )

    assignments: dict[int, int] = {}
    for juan in order_juans(groups, seed):
        choices = []
        for fold in fold_numbers:
            trial = {name: dict(values) for name, values in current.items()}
            for metric in METRICS:
                trial[fold][metric] += groups[juan][metric]
            choices.append((objective(trial), fold))
        _score, selected = min(choices)
        assignments[juan] = selected
        for metric in METRICS:
            current[selected][metric] += groups[juan][metric]
    return assignments


def plan_folds(
    fit_rows: list[dict],
    stratum_by_key: dict[tuple[int, int], str],
    *,
    seed: int = MINING_ORDER_SEED,
    folds: int = FOLDS,
) -> dict:
    """Pure, deterministic fold plan over the fit rows.

    Returns fold assignments, per-fold holdout/train row inventories, and per-fold
    metrics. Raises if the split is not a complete disjoint cover of every jie and
    juan, or if any fold is empty.
    """
    row_by_key: dict[tuple[int, int], dict] = {}
    for row in fit_rows:
        key = int(row["juan"]), int(row["jie_index"])
        if key in row_by_key:
            raise ValueError(f"duplicate fit jie: {key}")
        row_by_key[key] = row
    if set(row_by_key) != set(stratum_by_key):
        raise ValueError("fit strata inventory differs from fit rows")

    groups: dict[int, dict[str, int]] = {}
    for key, row in row_by_key.items():
        juan = key[0]
        group = groups.setdefault(juan, dict.fromkeys(METRICS, 0))
        group["examples"] += 1
        group["spans"] += int(row["span_count"])
        group[stratum_by_key[key]] += 1

    fold_by_juan = assign_folds(groups, seed=seed, folds=folds)
    if set(fold_by_juan) != set(groups):
        raise ValueError("fold assignment did not cover every juan")
    if set(fold_by_juan.values()) != set(range(1, folds + 1)):
        raise ValueError("fold assignment left a fold empty")

    holdout_rows: dict[int, list[dict]] = {fold: [] for fold in range(1, folds + 1)}
    train_rows: dict[int, list[dict]] = {fold: [] for fold in range(1, folds + 1)}
    holdout_juans: dict[int, set[int]] = {fold: set() for fold in range(1, folds + 1)}
    fold_metrics = {
        fold: dict.fromkeys(METRICS, 0) for fold in range(1, folds + 1)
    }
    for key in sorted(row_by_key):
        row = row_by_key[key]
        juan = key[0]
        fold = fold_by_juan[juan]
        holdout_rows[fold].append(row)
        holdout_juans[fold].add(juan)
        fold_metrics[fold]["examples"] += 1
        fold_metrics[fold]["spans"] += int(row["span_count"])
        fold_metrics[fold][stratum_by_key[key]] += 1
        for other in range(1, folds + 1):
            if other != fold:
                train_rows[other].append(row)

    seen_holdout: set[tuple[int, int]] = set()
    for fold in range(1, folds + 1):
        fold_keys = {(int(r["juan"]), int(r["jie_index"])) for r in holdout_rows[fold]}
        train_keys = {(int(r["juan"]), int(r["jie_index"])) for r in train_rows[fold]}
        train_juans = {int(r["juan"]) for r in train_rows[fold]}
        if holdout_juans[fold] & train_juans:
            raise ValueError(f"fold {fold} train/holdout juan overlap")
        if fold_keys & train_keys:
            raise ValueError(f"fold {fold} train/holdout jie overlap")
        if fold_keys & seen_holdout:
            raise ValueError(f"fold {fold} repeats a holdout jie")
        seen_holdout |= fold_keys
    if seen_holdout != set(row_by_key):
        raise ValueError("holdouts do not cover every jie exactly once")

    return {
        "fold_by_juan": fold_by_juan,
        "holdout_rows": holdout_rows,
        "train_rows": train_rows,
        "holdout_juans": {fold: sorted(v) for fold, v in holdout_juans.items()},
        "fold_metrics": fold_metrics,
    }


def freeze_plan(partition_root: Path, output_dir: Path) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"mining plan output exists: {output_dir}")
    partition_manifest_path = partition_root / "manifest.json"
    fit_path = partition_root / "fit.jsonl"
    private_path = partition_root / "private.json"
    snapshots = {
        path: path.read_bytes()
        for path in (partition_manifest_path, fit_path, private_path)
    }
    partition_manifest = _load(snapshots[partition_manifest_path])
    private_manifest = _load(snapshots[private_path])
    fit_rows = _rows_from(snapshots[fit_path])
    outputs = partition_manifest.get("outputs", {})
    if (
        partition_manifest.get("status") != PARTITION_STATUS
        or outputs.get("fit_sha256") != _digest(snapshots[fit_path])
        or outputs.get("private_sha256") != _digest(snapshots[private_path])
        or partition_manifest.get("partitions", {}).get("fit", {}).get("examples")
        != EXPECTED_FIT_EXAMPLES
        or len(fit_rows) != EXPECTED_FIT_EXAMPLES
        or private_manifest.get("status") != PRIVATE_STATUS
    ):
        raise ValueError("mining plan partition binding differs")

    stratum_by_key: dict[tuple[int, int], str] = {}
    for row in private_manifest.get("rows", []):
        if row.get("partition") != "fit":
            continue
        key = int(row["juan"]), int(row["jie_index"])
        if key in stratum_by_key:
            raise ValueError(f"duplicate fit private row: {key}")
        stratum_by_key[key] = str(row["stratum"])

    plan = plan_folds(fit_rows, stratum_by_key)
    fold_by_juan = plan["fold_by_juan"]
    if len(fold_by_juan) != EXPECTED_FIT_JUANS:
        raise ValueError("fit juan count differs from the frozen partition")

    git_commit = _git_commit_clean()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        folds_dir = staging / "folds"
        folds_dir.mkdir()
        fold_outputs = {}
        fold_summaries = {}
        for fold in FOLD_NUMBERS:
            fold_dir = folds_dir / f"fold-{fold}"
            fold_dir.mkdir()
            train_file = fold_dir / "train.jsonl"
            holdout_file = fold_dir / "holdout.jsonl"
            _write_jsonl(train_file, plan["train_rows"][fold])
            _write_jsonl(holdout_file, plan["holdout_rows"][fold])
            fold_outputs[str(fold)] = {
                "train_sha256": _sha256(train_file),
                "holdout_sha256": _sha256(holdout_file),
            }
            fold_summaries[str(fold)] = {
                "holdout_juans": plan["holdout_juans"][fold],
                "holdout": plan["fold_metrics"][fold],
                "holdout_examples": len(plan["holdout_rows"][fold]),
                "train_examples": len(plan["train_rows"][fold]),
                **fold_outputs[str(fold)],
            }
        manifest = {
            "schema_version": 1,
            "status": "ml_production_precision_mining_plan",
            "mining_only": True,
            "eligible_for_deployment": False,
            "eligible_for_production": False,
            "eligible_for_production_precision_claim": False,
            "formal_grade": False,
            "formal_evaluation": False,
            "order_seed": MINING_ORDER_SEED,
            "folds": FOLDS,
            "algorithm": (
                "juan_grouped_normalized_squared_target_deviation_five_fold"
            ),
            "fold_by_juan": {
                str(juan): fold for juan, fold in sorted(fold_by_juan.items())
            },
            "fold_summaries": fold_summaries,
            "inputs": {
                "partition_manifest_sha256": _digest(
                    snapshots[partition_manifest_path]
                ),
                "fit_sha256": _digest(snapshots[fit_path]),
                "private_sha256": _digest(snapshots[private_path]),
            },
            "outputs": fold_outputs,
            "claim_limit": (
                "Mining out-of-fold generator split only. These folds and their "
                "models are verifier-training artifacts and can never be "
                "deployment or production candidates."
            ),
            "git_commit": git_commit,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in sorted(
            staging.rglob("*"), key=lambda item: len(item.parts), reverse=True
        ):
            path.chmod(0o555 if path.is_dir() else 0o444)
        staging.chmod(0o555)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the revision-4 fit-only five-fold mining split from the "
            "frozen precision partition."
        )
    )
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_plan(args.partition, args.output)
    print(json.dumps({
        "fold_by_juan": manifest["fold_by_juan"],
        "fold_summaries": {
            fold: {
                "holdout_examples": summary["holdout_examples"],
                "train_examples": summary["train_examples"],
                "holdout_juans": summary["holdout_juans"],
            }
            for fold, summary in manifest["fold_summaries"].items()
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
