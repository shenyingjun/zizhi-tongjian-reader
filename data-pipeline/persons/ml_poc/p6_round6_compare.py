from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from p3_round3_compare import _metrics, _require_metrics, compare_predictions


EXPECTED_ROUND3 = {
    "history.json": "a6a6cb9346550bb333fee0c4a674c626590e77450273db2dacd4e81a4810829c",
    "report.json": "b65bccba68bb37dea23da7a57b84336b906a32769145f0d33b9f679d57d73353",
    "dev_predictions.json": "6e4158e169fe239214a193b292e7e30292a299cc113f3d437f854352712ecdd4",
    "evaluation_predictions.json": "f295091dfa549074f707852cd02c3f5cf9539d279c2bf83b7324d015decb99f6",
}
EXPECTED_ROUND6 = {
    "history.json": "87d5953c5a4a911281c5f015204aa7092334a602ccb8bac10ce2f7e9ed926111",
    "report.json": "e34278caa2f1e68888784cd6aa62a44f3c8fa4981992e478b7c3b6f38ba1c90e",
    "dev_predictions.json": "0ae1eadc2098026439635ad7da71ddd97d1c1c9232221e4b6b75417211a0e53f",
    "evaluation_predictions.json": "fcf2ebea6b7665074b50fbba85bc1723c9a56feb9cf80ceeab29485fffad6835",
}


def _snapshot(path: Path) -> tuple[bytes, str]:
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def _rename_hit_buckets(comparison: dict) -> None:
    for section in ("reference_length_recall", "reference_term_recall"):
        for bucket in comparison[section].values():
            old_hits = bucket.pop("round2_hits")
            new_hits = bucket.pop("round3_hits")
            bucket["round3_hits"] = old_hits
            bucket["round6_hits"] = new_hits


def compare_round6(
    round3_dir: Path,
    round6_dir: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Round 6 comparison exists: {output_dir}")
    git_commit = _git_commit_clean()
    snapshots = {}
    hashes = {"round3": {}, "round6": {}}
    for label, directory in (("round3", round3_dir), ("round6", round6_dir)):
        for name in (
            "report.json", "history.json", "dev_predictions.json",
            "evaluation_predictions.json",
        ):
            raw, digest = _snapshot(directory / name)
            snapshots[(label, name)] = raw
            hashes[label][name] = digest
    if hashes["round3"] != EXPECTED_ROUND3 or hashes["round6"] != EXPECTED_ROUND6:
        raise ValueError("Round 3/6 comparison artifacts differ")
    reports = {
        label: json.loads(snapshots[(label, "report.json")])
        for label in ("round3", "round6")
    }
    if (
        reports["round3"].get("round3_control", {}).get("formal_evaluation")
        is not False
        or reports["round3"].get("round3_control", {}).get(
            "eligible_for_promotion_without_fresh_sealed_set"
        ) is not False
        or any(
            hashes["round3"][name]
            != reports["round3"]["round3_control"]["run_artifacts"].get(name)
            for name in (
                "history.json", "dev_predictions.json",
                "evaluation_predictions.json",
            )
        )
        or reports["round6"].get("round6_control", {}).get("formal_evaluation")
        is not False
        or reports["round6"].get("round6_control", {}).get(
            "eligible_for_promotion"
        ) is not False
        or any(
            hashes["round6"][name]
            != reports["round6"]["round6_control"]["run_artifacts"].get(name)
            for name in (
                "history.json", "dev_predictions.json",
                "evaluation_predictions.json",
            )
        )
    ):
        raise ValueError("Round 3/6 provenance differs")

    comparisons = {}
    metric_delta = {}
    for split, name, report_key in (
        ("dev", "dev_predictions.json", "dev_challenge"),
        ("evaluation", "evaluation_predictions.json", "evaluation"),
    ):
        rows = {
            label: json.loads(snapshots[(label, name)])
            for label in ("round3", "round6")
        }
        comparison = compare_predictions(rows["round3"], rows["round6"])
        _rename_hit_buckets(comparison)
        comparisons[split] = comparison
        metrics = {label: _metrics(value) for label, value in rows.items()}
        for label in ("round3", "round6"):
            _require_metrics(
                metrics[label],
                reports[label][report_key],
                f"{label} {split}",
            )
        old = reports["round3"][report_key]["exact"]
        new = reports["round6"][report_key]["exact"]
        metric_delta[split] = {
            "round3": old,
            "round6": new,
            "precision_delta": new["precision"] - old["precision"],
            "recall_delta": new["recall"] - old["recall"],
            "f1_delta": new["f1"] - old["f1"],
        }
    report = {
        "schema_version": 1,
        "status": "round6_controlled_diagnostic_comparison",
        "formal_evaluation": False,
        "eligible_for_promotion": False,
        "git_commit": git_commit,
        "inputs": hashes,
        "metric_delta": metric_delta,
        "splits": comparisons,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        path = staging / "comparison.json"
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        path.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare controlled Round 3 and Round 6 predictions."
    )
    parser.add_argument("--round3", type=Path, required=True)
    parser.add_argument("--round6", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_round6(args.round3, args.round6, args.output)
    print(json.dumps({
        "metric_delta": report["metric_delta"],
        "dev_attribution": report["splits"]["dev"]["attribution"],
        "evaluation_attribution": report["splits"]["evaluation"]["attribution"],
        "evaluation_geometry": report["splits"]["evaluation"][
            "prediction_geometry"
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
