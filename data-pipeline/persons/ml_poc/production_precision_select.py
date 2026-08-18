from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections import Counter
from pathlib import Path

from p3_compact import _git_commit_clean
from production_train import SEEDS, _make_read_only


CONFIDENCE_THRESHOLDS = (
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
    0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99,
)
ONE_SIDED_95_Z = 1.6448536269514722


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _geometry(span: dict) -> tuple[int, int, int, str]:
    return (
        int(span["para_id"]),
        int(span["start"]),
        int(span["end"]),
        str(span["surface"]),
    )


def _wilson_lower(successes: int, trials: int) -> float:
    if trials <= 0 or not 0 <= successes <= trials:
        return 0.0
    proportion = successes / trials
    z2 = ONE_SIDED_95_Z**2
    return (
        proportion
        + z2 / (2 * trials)
        - ONE_SIDED_95_Z
        * math.sqrt(
            proportion * (1 - proportion) / trials
            + z2 / (4 * trials**2)
        )
    ) / (1 + z2 / trials)


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
        "wilson_precision_lower_one_sided_95": _wilson_lower(
            true_positive, len(prediction)
        ),
    }


def select_controller(
    prediction_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"precision selection output exists: {output_dir}")
    by_seed = {}
    inputs = {}
    reference_manifest_sha = None
    for seed in SEEDS:
        root = prediction_root / (
            f"ml-production-precision-calibration-seed-{seed}-v1"
        )
        manifest_path = root / "manifest.json"
        predictions_path = root / "predictions.json"
        manifest = _read(manifest_path)
        rows = _read(predictions_path)
        if (
            not isinstance(manifest, dict)
            or not isinstance(rows, list)
            or manifest.get("status")
            != "ml_production_precision_confidence_predictions"
            or manifest.get("split") != "calibration"
            or manifest.get("seed") != seed
            or manifest.get("examples") != 45
            or manifest.get("predictions_sha256") != _sha256(predictions_path)
            or manifest.get("eligible_for_production_precision_claim") is not False
        ):
            raise ValueError(f"calibration prediction binding differs: {seed}")
        current_reference_sha = manifest.get("reference_manifest_sha256")
        if (
            not isinstance(current_reference_sha, str)
            or reference_manifest_sha not in {None, current_reference_sha}
        ):
            raise ValueError("calibration references differ across seeds")
        reference_manifest_sha = current_reference_sha
        indexed = {str(row["id"]): row for row in rows}
        if len(indexed) != 45:
            raise ValueError(f"calibration prediction inventory differs: {seed}")
        by_seed[seed] = indexed
        inputs[str(seed)] = {
            "manifest_sha256": _sha256(manifest_path),
            "predictions_sha256": _sha256(predictions_path),
            "model_report_sha256": manifest["model_report_sha256"],
        }

    identities = set(by_seed[SEEDS[0]])
    if any(set(by_seed[seed]) != identities for seed in SEEDS[1:]):
        raise ValueError("calibration jie inventory differs across seeds")
    reference = set()
    seed_predictions = {seed: {} for seed in SEEDS}
    for identity in sorted(identities):
        references = [
            {_geometry(span) for span in by_seed[seed][identity]["reference_spans"]}
            for seed in SEEDS
        ]
        if any(value != references[0] for value in references[1:]):
            raise ValueError(f"calibration reference differs: {identity}")
        reference.update((identity, geometry) for geometry in references[0])
        for seed in SEEDS:
            for span in by_seed[seed][identity]["prediction_spans"]:
                geometry = _geometry(span)
                confidence = float(span["confidence"])
                if not 0.0 < confidence <= 1.0:
                    raise ValueError(f"invalid span confidence: {identity}")
                seed_predictions[seed][(identity, geometry)] = confidence

    table = []
    for vote_threshold in (2, 3):
        counts = Counter(
            key for predictions in seed_predictions.values() for key in predictions
        )
        for confidence_threshold in CONFIDENCE_THRESHOLDS:
            prediction = set()
            for key, count in counts.items():
                if count < vote_threshold:
                    continue
                confidences = sorted(
                    (
                        seed_predictions[seed].get(key, 0.0)
                        for seed in SEEDS
                    ),
                    reverse=True,
                )
                if confidences[vote_threshold - 1] >= confidence_threshold:
                    prediction.add(key)
            metrics = _metric(reference, prediction)
            eligible = (
                metrics["prediction_spans"] >= 300
                and metrics["precision"] >= 0.99
                and metrics["recall"] >= 0.95
                and metrics["wilson_precision_lower_one_sided_95"] >= 0.98
            )
            table.append({
                "vote_threshold": vote_threshold,
                "confidence_threshold": confidence_threshold,
                **metrics,
                "eligible": eligible,
            })
    eligible = [row for row in table if row["eligible"]]
    selected = max(
        eligible,
        key=lambda row: (
            row["recall"],
            row["precision"],
            row["confidence_threshold"],
            row["vote_threshold"],
        ),
        default=None,
    )
    git_commit = _git_commit_clean()
    manifest = {
        "schema_version": 1,
        "status": (
            "ml_production_precision_controller_selected_ai_assisted"
            if selected is not None
            else "ml_production_precision_controller_blocked_ai_assisted"
        ),
        "formal_grade": False,
        "formal_evaluation": False,
        "eligible_for_production_precision_claim": False,
        "confirmation_read": False,
        "git_commit": git_commit,
        "reference_manifest_sha256": reference_manifest_sha,
        "inputs": inputs,
        "grid": {
            "vote_thresholds": [2, 3],
            "confidence_thresholds": list(CONFIDENCE_THRESHOLDS),
            "points": len(table),
        },
        "gates": {
            "minimum_predictions": 300,
            "minimum_precision": 0.99,
            "minimum_recall": 0.95,
            "minimum_wilson_precision_lower_one_sided_95": 0.98,
        },
        "selected": selected,
        "table": table,
        "claim_limit": (
            "AI-assisted diagnostic selection only; cannot authorize production "
            "promotion or formal evaluation."
        ),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = select_controller(args.prediction_root, args.output)
    print(json.dumps({
        "status": manifest["status"],
        "selected": manifest["selected"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
