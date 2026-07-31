from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tempfile
from collections import Counter
from pathlib import Path

from p1_train import evaluate
from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from p6_locked_assisted_finalize import (
    PROVENANCE,
    ROLE_COUNTS,
    _metrics,
    _read_jsonl,
    _span,
)
from p6_locked_assisted_tasks import evaluation_claim_metadata
from p6_seed_replication_compare import EXPECTED_RUNS, _validate_run
from p6_seed_replication_train import REPLICATION_SEEDS
from report import geometry_delta


EXPECTED_MODEL_ARTIFACTS = {
    20260727: "d62b795488292638dac2425f5f7b6798f39810d38746f508271ce174abe8d536",
    20260728: "3a0b200027891c5f3f9ed89460d960c59af620d33dd043bb6cd18f845342911e",
    20260729: "5c875b98d454dbfcf5bc0a7c0e2f508d6d83c96e17e94a265e3d6653cb74fe71",
}
ROUND3_ARTIFACT_SHA256 = (
    "fb611bee98cdc672e19163ca84aef49612815cab285889c7458cff07ba098cbd"
)
ROUND3_REPORT_SHA256 = (
    "b65bccba68bb37dea23da7a57b84336b906a32769145f0d33b9f679d57d73353"
)
VOTE_THRESHOLDS = (2, 3)
EXPECTED_LOCKED_P3_FREEZE_REPORT_SHA256 = (
    "cc75535319184c2b0d2c931b51eae95701675400d438b3e5b2e411562d035ac7"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _publish(staging: Path, output_dir: Path) -> None:
    for path in staging.iterdir():
        if path.is_file():
            path.chmod(stat.S_IREAD)
    staging.replace(output_dir)


def _geometry(row: dict) -> tuple[int, int, int]:
    return int(row["para_id"]), int(row["start"]), int(row["end"])


def vote_predictions(
    predictions_by_seed: dict[int, list[dict]],
    threshold: int,
) -> list[dict]:
    if set(predictions_by_seed) != set(REPLICATION_SEEDS):
        raise ValueError("ensemble seed inventory differs")
    if threshold not in VOTE_THRESHOLDS:
        raise ValueError(f"unsupported ensemble threshold: {threshold}")
    indexed = {
        seed: {str(row["id"]): row for row in rows}
        for seed, rows in predictions_by_seed.items()
    }
    identities = set(next(iter(indexed.values())))
    if any(
        len(rows) != len(predictions_by_seed[seed])
        or set(rows) != identities
        for seed, rows in indexed.items()
    ):
        raise ValueError("ensemble prediction identity inventory differs")
    voted = []
    for identity in sorted(identities):
        references = [
            indexed[seed][identity]["reference_spans"]
            for seed in REPLICATION_SEEDS
        ]
        reference_geometry = {
            _geometry(row): str(row["surface"]) for row in references[0]
        }
        if any(
            {_geometry(row): str(row["surface"]) for row in rows}
            != reference_geometry
            for rows in references[1:]
        ):
            raise ValueError(f"ensemble reference differs: {identity}")
        votes = Counter()
        surfaces = {}
        for seed in REPLICATION_SEEDS:
            seen = set()
            for row in indexed[seed][identity]["prediction_spans"]:
                geometry = _geometry(row)
                if geometry in seen:
                    raise ValueError(
                        f"duplicate seed prediction: {seed} {identity} {geometry}"
                    )
                seen.add(geometry)
                surface = str(row["surface"])
                if geometry in surfaces and surfaces[geometry] != surface:
                    raise ValueError(
                        f"ensemble prediction surface differs: {identity}"
                    )
                surfaces[geometry] = surface
                votes[geometry] += 1
        voted.append({
            "id": identity,
            "reference_spans": references[0],
            "prediction_spans": [
                {
                    "para_id": geometry[0],
                    "start": geometry[1],
                    "end": geometry[2],
                    "surface": surfaces[geometry],
                }
                for geometry in sorted(votes)
                if votes[geometry] >= threshold
            ],
        })
    return voted


def _prediction_metrics(rows: list[dict]) -> dict:
    return _metrics([
        (
            [_span(span) for span in row["reference_spans"]],
            [_span(span) for span in row["prediction_spans"]],
        )
        for row in rows
    ])


def _run_root(input_root: Path, seed: int) -> Path:
    return input_root / f"ml-poc-round6-deterministic-seed-{seed}-v1"


def select_ensemble(input_root: Path, output_dir: Path) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"ensemble selection exists: {output_dir}")
    dev_by_seed = {}
    inputs = {}
    individual_metrics = {}
    for seed in REPLICATION_SEEDS:
        root = _run_root(input_root, seed)
        snapshots = {
            name: (root / name).read_bytes()
            for name in (
                "report.json",
                "history.json",
                "dev_predictions.json",
                "evaluation_predictions.json",
            )
        }
        hashes = {
            name: hashlib.sha256(raw).hexdigest()
            for name, raw in snapshots.items()
        }
        _validate_run(seed, "round6", snapshots, hashes)
        artifact = _model_artifact(root / "model")
        if artifact["combined_sha256"] != EXPECTED_MODEL_ARTIFACTS[seed]:
            raise ValueError(f"Round 6 model artifact differs: {seed}")
        dev_by_seed[seed] = json.loads(snapshots["dev_predictions.json"])
        individual_metrics[str(seed)] = _prediction_metrics(dev_by_seed[seed])
        inputs[str(seed)] = {
            "report_sha256": hashes["report.json"],
            "dev_predictions_sha256": hashes["dev_predictions.json"],
            "model_artifact_sha256": artifact["combined_sha256"],
        }
    candidate_predictions = {
        threshold: vote_predictions(dev_by_seed, threshold)
        for threshold in VOTE_THRESHOLDS
    }
    candidate_metrics = {
        threshold: _prediction_metrics(rows)
        for threshold, rows in candidate_predictions.items()
    }
    selected_threshold = max(
        VOTE_THRESHOLDS,
        key=lambda threshold: (
            candidate_metrics[threshold]["f1"],
            candidate_metrics[threshold]["precision"],
            threshold,
        ),
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        selected_predictions_path = staging / "selected_dev_predictions.json"
        _write(selected_predictions_path, candidate_predictions[selected_threshold])
        config = {
            "schema_version": 1,
            "status": "round6_seed_ensemble_selected_on_juan27_dev",
            "selection_split": "juan_27_dev_only",
            "vote_unit": "exact_span_geometry",
            "seed_count": len(REPLICATION_SEEDS),
            "threshold": selected_threshold,
            "tie_break": "highest_f1_then_precision_then_stricter_threshold",
            "locked_p3_predictions_generated": False,
            "inputs": inputs,
            "selected_dev_predictions_sha256": _sha256(
                selected_predictions_path
            ),
            "git_commit": _git_commit_clean(),
        }
        _write(staging / "config.json", config)
        report = {
            "schema_version": 1,
            "status": "round6_seed_ensemble_dev_selection_report",
            **evaluation_claim_metadata(),
            "promotion_eligibility_reason": (
                "ensemble_is_research_challenger_selected_on_reused_dev"
            ),
            "selection_split": "juan_27_dev_only",
            "individual_seed_metrics": individual_metrics,
            "candidate_metrics": {
                str(threshold): candidate_metrics[threshold]
                for threshold in VOTE_THRESHOLDS
            },
            "selected_threshold": selected_threshold,
            "config_sha256": _sha256(staging / "config.json"),
        }
        _write(staging / "report.json", report)
        _publish(staging, output_dir)
    return report


def _predict_examples(model_root: Path, examples: list[dict]) -> list[dict]:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_root / "model", use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_root / "model")
    model.to(device)
    _, predictions = evaluate(
        model,
        tokenizer,
        examples,
        device,
        max_length=512,
        stride=128,
        batch_size=8,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions


def _comparison(
    examples: list[dict],
    before_rows: list[dict],
    after_rows: list[dict],
) -> tuple[dict, list[dict]]:
    before_by_id = {str(row["id"]): row for row in before_rows}
    after_by_id = {str(row["id"]): row for row in after_rows}
    if set(before_by_id) != set(after_by_id):
        raise ValueError("ensemble comparison identities differ")
    totals = Counter()
    changes = []
    replacement_total = 0
    role_by_id = {
        str(row["id"]): str(row["evaluation_role"]) for row in examples
    }
    for identity in sorted(before_by_id):
        reference = {
            _span(row) for row in before_by_id[identity]["reference_spans"]
        }
        if reference != {
            _span(row) for row in after_by_id[identity]["reference_spans"]
        }:
            raise ValueError(f"ensemble comparison reference differs: {identity}")
        before = {
            _span(row) for row in before_by_id[identity]["prediction_spans"]
        }
        after = {
            _span(row) for row in after_by_id[identity]["prediction_spans"]
        }
        additions = sorted(after - before)
        removals = sorted(before - after)
        recovered = sorted((after - before) & reference)
        regressed = sorted((before - after) & reference)
        added_fp = sorted((after - before) - reference)
        removed_fp = sorted((before - after) - reference)
        delta = geometry_delta(list(before), list(after))
        replacement_total += int(delta["geometry_replacements"])
        totals.update({
            "raw_additions": len(additions),
            "removals": len(removals),
            "net_growth": len(additions) - len(removals),
            "reference_recoveries": len(recovered),
            "reference_regressions": len(regressed),
            "added_false_positives": len(added_fp),
            "removed_false_positives": len(removed_fp),
        })
        if additions or removals:
            changes.append({
                "id": identity,
                "role": role_by_id[identity],
                "additions": [row.__dict__ for row in additions],
                "removals": [row.__dict__ for row in removals],
                "reference_recoveries": [row.__dict__ for row in recovered],
                "reference_regressions": [row.__dict__ for row in regressed],
                "added_false_positives": [row.__dict__ for row in added_fp],
                "removed_false_positives": [row.__dict__ for row in removed_fp],
                "geometry_replacements": delta["replacement_examples"],
            })
    totals["geometry_replacements"] = replacement_total
    totals["changed_jies"] = len(changes)
    return dict(totals), changes


def evaluate_ensemble(
    selection_dir: Path,
    frozen_dir: Path,
    round3_root: Path,
    seed_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"ensemble diagnostic exists: {output_dir}")
    config_path = selection_dir / "config.json"
    selected_dev_predictions_path = (
        selection_dir / "selected_dev_predictions.json"
    )
    selection_report = _read(selection_dir / "report.json")
    config = _read(config_path)
    if (
        not isinstance(config, dict)
        or config.get("status")
        != "round6_seed_ensemble_selected_on_juan27_dev"
        or config.get("locked_p3_predictions_generated") is not False
        or selection_report.get("config_sha256") != _sha256(config_path)
    ):
        raise ValueError("ensemble selection was not locked before P3 inference")

    dev_by_seed = {}
    model_inputs = {}
    for seed in REPLICATION_SEEDS:
        root = _run_root(seed_root, seed)
        snapshots = {
            name: (root / name).read_bytes()
            for name in (
                "report.json",
                "history.json",
                "dev_predictions.json",
                "evaluation_predictions.json",
            )
        }
        hashes = {
            name: hashlib.sha256(raw).hexdigest()
            for name, raw in snapshots.items()
        }
        _validate_run(seed, "round6", snapshots, hashes)
        artifact = _model_artifact(root / "model")["combined_sha256"]
        expected_input = config["inputs"][str(seed)]
        if (
            artifact != EXPECTED_MODEL_ARTIFACTS[seed]
            or artifact != expected_input["model_artifact_sha256"]
            or hashes["report.json"] != expected_input["report_sha256"]
            or hashes["dev_predictions.json"]
            != expected_input["dev_predictions_sha256"]
        ):
            raise ValueError(f"locked ensemble model or dev input differs: {seed}")
        dev_by_seed[seed] = json.loads(snapshots["dev_predictions.json"])
        model_inputs[str(seed)] = {
            "artifact_sha256": artifact,
            "report_sha256": hashes["report.json"],
        }
    candidate_predictions = {
        threshold: vote_predictions(dev_by_seed, threshold)
        for threshold in VOTE_THRESHOLDS
    }
    candidate_metrics = {
        threshold: _prediction_metrics(rows)
        for threshold, rows in candidate_predictions.items()
    }
    selected_threshold = max(
        VOTE_THRESHOLDS,
        key=lambda threshold: (
            candidate_metrics[threshold]["f1"],
            candidate_metrics[threshold]["precision"],
            threshold,
        ),
    )
    if (
        config.get("threshold") != selected_threshold
        or selection_report.get("selected_threshold") != selected_threshold
        or selection_report.get("candidate_metrics") != {
            str(threshold): candidate_metrics[threshold]
            for threshold in VOTE_THRESHOLDS
        }
        or config.get("selected_dev_predictions_sha256")
        != _sha256(selected_dev_predictions_path)
        or _read(selected_dev_predictions_path)
        != candidate_predictions[selected_threshold]
    ):
        raise ValueError("ensemble dev selection does not reproduce")

    freeze_report_path = frozen_dir / "freeze_report.json"
    reference_path = frozen_dir / "reference.jsonl"
    freeze_report = _read(freeze_report_path)
    if (
        not isinstance(freeze_report, dict)
        or _sha256(freeze_report_path)
        != EXPECTED_LOCKED_P3_FREEZE_REPORT_SHA256
        or freeze_report.get("status")
        != "frozen_copilot_double_pass_blind_diagnostic"
        or freeze_report.get("reference_locked") is not True
        or freeze_report.get("candidate_model_blind") is not True
        or freeze_report.get("provenance") != PROVENANCE
        or freeze_report.get("jies") != sum(ROLE_COUNTS.values())
        or freeze_report.get("roles") != ROLE_COUNTS
        or freeze_report.get("outputs", {}).get("reference_sha256")
        != _sha256(reference_path)
    ):
        raise ValueError("locked P3 reference binding differs")
    examples = _read_jsonl(reference_path)
    if (
        _model_artifact(round3_root / "model")["combined_sha256"]
        != ROUND3_ARTIFACT_SHA256
        or _sha256(round3_root / "report.json") != ROUND3_REPORT_SHA256
    ):
        raise ValueError("Round 3 artifact differs")

    round3_predictions = _predict_examples(round3_root, examples)
    seed_predictions = {}
    for seed in REPLICATION_SEEDS:
        root = _run_root(seed_root, seed)
        seed_predictions[seed] = _predict_examples(root, examples)
    ensemble_predictions = vote_predictions(
        seed_predictions, int(config["threshold"])
    )
    metrics = {
        "round3": {
            "all": _prediction_metrics(round3_predictions),
            "by_role": {},
        },
        "round6_seed_ensemble": {
            "all": _prediction_metrics(ensemble_predictions),
            "by_role": {},
        },
    }
    for role in ROLE_COUNTS:
        ids = {
            str(row["id"]) for row in examples
            if row["evaluation_role"] == role
        }
        for label, rows in (
            ("round3", round3_predictions),
            ("round6_seed_ensemble", ensemble_predictions),
        ):
            metrics[label]["by_role"][role] = _prediction_metrics([
                row for row in rows if str(row["id"]) in ids
            ])
    comparison, changes = _comparison(
        examples, round3_predictions, ensemble_predictions
    )
    report = {
        "schema_version": 1,
        "status": "round6_seed_ensemble_locked_p3_diagnostic",
        **evaluation_claim_metadata(),
        "promotion_eligibility_reason": (
            "ensemble_is_research_challenger_and_locked_p3_has_no_promotion_gate"
        ),
        "copilot_assistance_is_disqualifying": False,
        "provenance": PROVENANCE,
        "selection": {
            "config_sha256": _sha256(config_path),
            "threshold": config["threshold"],
            "selection_split": config["selection_split"],
            "locked_before_p3_inference": True,
        },
        "inputs": {
            "freeze_report_sha256": _sha256(freeze_report_path),
            "reference_sha256": _sha256(reference_path),
            "round3_artifact_sha256": ROUND3_ARTIFACT_SHA256,
            "round6_models": model_inputs,
        },
        "metrics": metrics,
        "comparison": {
            "before": "round3",
            "after": "round6_seed_ensemble",
            **comparison,
        },
        "git_commit": _git_commit_clean(),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        predictions_path = staging / "predictions.json"
        changes_path = staging / "geometry_changes.json"
        _write(predictions_path, {
            "round3": round3_predictions,
            "round6_seed_ensemble": ensemble_predictions,
            "round6_by_seed": seed_predictions,
        })
        _write(changes_path, changes)
        report["outputs"] = {
            "predictions_sha256": _sha256(predictions_path),
            "geometry_changes_sha256": _sha256(changes_path),
        }
        _write(staging / "report.json", report)
        _publish(staging, output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select and evaluate an exact-span Round 6 seed ensemble."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--input-root", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--selection", type=Path, required=True)
    evaluate_parser.add_argument("--frozen", type=Path, required=True)
    evaluate_parser.add_argument("--round3", type=Path, required=True)
    evaluate_parser.add_argument("--seed-root", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        report = select_ensemble(args.input_root, args.output)
        result = {
            "selected_threshold": report["selected_threshold"],
            "candidate_metrics": report["candidate_metrics"],
        }
    else:
        report = evaluate_ensemble(
            args.selection,
            args.frozen,
            args.round3,
            args.seed_root,
            args.output,
        )
        result = {
            "metrics": report["metrics"],
            "comparison": report["comparison"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
