from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from p6_locked_assisted_finalize import _read_jsonl
from p6_seed_ensemble import _predict_examples, vote_predictions
from p6_seed_replication_train import (
    CUBLAS_WORKSPACE_CONFIG,
    REPLICATION_SEEDS,
    _configure_determinism,
)
from p10_independent_dev_compare import RECIPES, _compare
from p12_adoption_statistics import _counts, _metric


EXPECTED_REFERENCE_REPORT_SHA256 = (
    "8e2a52fcca448eccf7e3b89987b7d6f50823e2f701fa9b4e53f2da5de652ac56"
)
EXPECTED_REFERENCE_SHA256 = (
    "e32b17a7be53b01807cd6c85c3d98f720e96565cc119e7a82270d5d6284d298b"
)
EXPECTED_TASK_MANIFEST_SHA256 = (
    "5b5c6b46375ef09ae7447b2af800d3f37eff2905ca28942c532bc14359e7e937"
)
EXPECTED_RULES_MANIFEST_SHA256 = (
    "0aed4700e1da5acc910e0661efcd8160aa49c09ab8542c021072c317db971b34"
)
EXPECTED_RULE_DOCUMENT_ROOT_SHA256 = (
    "73330a92e1425734891abe4b04d0a6995bd4f244d6dc7d0427e45447be1d585f"
)
EXPECTED_RULES_SHA256 = (
    "b50e6dfaa1f204fba405ea2c548fcdb4fba0edaca38b9ae4c171c1cb7c5f9c06"
)
EXPECTED_TRANSLATION_MANIFEST_SHA256 = (
    "eb4a4005cf1605b56843eda907368ceb54f7cfcbd00b50e91ff27db91e9bd412"
)
ENSEMBLE_THRESHOLD = 2
SYSTEMS = (
    "round6_model",
    "round7_model",
    "production_rules_translation_assisted",
)
STRATA = (
    "role_appellation_challenge",
    "foreign_title_challenge",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _geometry(row: dict) -> tuple[int, int, int, str]:
    return (
        int(row["para_id"]),
        int(row["start"]),
        int(row["end"]),
        str(row["surface"]),
    )


def _span(geometry: tuple[int, int, int, str]) -> dict:
    return {
        "para_id": geometry[0],
        "start": geometry[1],
        "end": geometry[2],
        "surface": geometry[3],
    }


def _aggregate(rows: list[dict], system: str) -> dict:
    reference = predicted = true_positive = 0
    for row in rows:
        counts = row["counts"][system]
        reference += counts[0]
        predicted += counts[1]
        true_positive += counts[2]
    return _metric(reference, predicted, true_positive)


def evaluate_compact_challenge(
    reference_dir: Path,
    task_manifest_path: Path,
    rules_dir: Path,
    artifacts_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"compact challenge evaluation exists: {output_dir}")
    reference_report_path = reference_dir / "report.json"
    reference_path = reference_dir / "challenge_reference.jsonl"
    rules_manifest_path = rules_dir / "manifest.json"
    reference_report = json.loads(
        reference_report_path.read_text(encoding="utf-8")
    )
    task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    rules_manifest = json.loads(
        rules_manifest_path.read_text(encoding="utf-8")
    )
    if (
        _sha256(reference_report_path) != EXPECTED_REFERENCE_REPORT_SHA256
        or _sha256(reference_path) != EXPECTED_REFERENCE_SHA256
        or _sha256(task_manifest_path) != EXPECTED_TASK_MANIFEST_SHA256
        or _sha256(rules_manifest_path) != EXPECTED_RULES_MANIFEST_SHA256
        or reference_report.get("status")
        != "frozen_round13_compact_challenge_reference"
        or reference_report.get("supplementary_challenge_evidence_only")
        is not True
        or reference_report.get("eligible_for_training") is not False
        or reference_report.get("eligible_to_reverse_failed_precision_gate")
        is not False
        or reference_report.get("candidate_model_blind") is not True
        or task_manifest.get("status")
        != "round13_compact_challenge_tasks_before_labeling"
        or rules_manifest.get("preset") != "PRESET_RECALL"
        or rules_manifest.get("scope") != "numbered-jie"
        or rules_manifest.get("rules_sha256") != EXPECTED_RULES_SHA256
        or rules_manifest.get("translation_evidence", {}).get(
            "manifest_sha256"
        ) != EXPECTED_TRANSLATION_MANIFEST_SHA256
        or rules_manifest.get("juans") != reference_report.get("juans")
    ):
        raise ValueError("compact challenge evaluation input binding differs")
    examples = _read_jsonl(reference_path)
    if len(examples) != 8:
        raise ValueError("compact challenge reference inventory differs")
    role_by_identity = {
        (int(row["juan"]), int(row["jie_index"])): str(row["role"])
        for row in task_manifest["private_selected_jies"]
    }
    if (
        len(role_by_identity) != 8
        or {role: list(role_by_identity.values()).count(role) for role in STRATA}
        != {role: 4 for role in STRATA}
        or {
            (int(row["juan"]), int(row["jie_index"])) for row in examples
        } != set(role_by_identity)
    ):
        raise ValueError("compact challenge stratum inventory differs")

    expected_rule_names = {
        f"juan_{juan:03d}.json" for juan in reference_report["juans"]
    }
    if {
        path.name for path in rules_dir.iterdir() if path.is_file()
    } != expected_rule_names | {"manifest.json"}:
        raise ValueError("compact challenge rule inventory differs")
    rule_snapshots = {
        name: (rules_dir / name).read_bytes()
        for name in sorted(expected_rule_names)
    }
    rule_hashes = {
        name: hashlib.sha256(raw).hexdigest()
        for name, raw in rule_snapshots.items()
    }
    rule_root = hashlib.sha256(json.dumps(
        rule_hashes, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if rule_root != EXPECTED_RULE_DOCUMENT_ROOT_SHA256:
        raise ValueError("compact challenge rule document root differs")
    rules_documents = {}
    for juan in reference_report["juans"]:
        document = json.loads(rule_snapshots[f"juan_{juan:03d}.json"])
        if (
            document.get("juan") != juan
            or document.get("rules_sha256") != EXPECTED_RULES_SHA256
            or document.get("translation_evidence_manifest_sha256")
            != EXPECTED_TRANSLATION_MANIFEST_SHA256
            or len(document.get("occurrences", []))
            != rules_manifest["occurrences_by_juan"][str(juan)]
        ):
            raise ValueError(f"compact challenge rule binding differs: {juan}")
        rules_documents[juan] = document

    _configure_determinism()
    predictions = {}
    model_inputs = {}
    for recipe in ("round6", "round7"):
        by_seed = {}
        model_inputs[recipe] = {}
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
            model_inputs[recipe][str(seed)] = {
                "artifact_name": name,
                "report_sha256": expected_report,
                "model_artifact_sha256": expected_model,
            }
        predictions[recipe] = {
            "by_seed": by_seed,
            "ensemble": vote_predictions(by_seed, ENSEMBLE_THRESHOLD),
        }

    ensemble_rows = {
        recipe: {
            str(row["id"]): row for row in predictions[recipe]["ensemble"]
        }
        for recipe in ("round6", "round7")
    }
    expected_ids = {str(row["id"]) for row in examples}
    if any(set(rows) != expected_ids for rows in ensemble_rows.values()):
        raise ValueError("compact challenge prediction identity differs")

    evaluation_rows = []
    audit_rows = []
    rules_prediction_rows = []
    for example in examples:
        identity = str(example["id"])
        juan = int(example["juan"])
        jie_index = int(example["jie_index"])
        role = role_by_identity[(juan, jie_index)]
        reference = {
            _geometry(row)
            for row in ensemble_rows["round7"][identity]["reference_spans"]
        }
        model_sets = {
            f"{recipe}_model": {
                _geometry(row)
                for row in ensemble_rows[recipe][identity]["prediction_spans"]
            }
            for recipe in ("round6", "round7")
        }
        paragraph_ids = {
            int(segment["para_id"]) for segment in example["segments"]
        }
        rule_by_geometry = {}
        for row in rules_documents[juan]["occurrences"]:
            if int(row["para_id"]) in paragraph_ids:
                rule_by_geometry.setdefault(_geometry(row), row)
        system_sets = {
            **model_sets,
            "production_rules_translation_assisted": set(rule_by_geometry),
        }
        counts = {
            system: _counts(reference, prediction)
            for system, prediction in system_sets.items()
        }
        evaluation_rows.append({
            "id": identity,
            "juan": juan,
            "jie_index": jie_index,
            "stratum": role,
            "counts": counts,
        })
        audit_rows.append({
            "id": identity,
            "juan": juan,
            "jie_index": jie_index,
            "stratum": role,
            "systems": {
                system: {
                    "false_positives": [
                        _span(span) for span in sorted(prediction - reference)
                    ],
                    "false_negatives": [
                        _span(span) for span in sorted(reference - prediction)
                    ],
                }
                for system, prediction in system_sets.items()
            },
        })
        rules_prediction_rows.append({
            "id": identity,
            "reference_spans": [
                _span(span) for span in sorted(reference)
            ],
            "prediction_spans": [
                {
                    **_span(span),
                    "rule_provenance": rule_by_geometry[span],
                }
                for span in sorted(rule_by_geometry)
            ],
        })

    metrics = {
        "overall": {
            system: _aggregate(evaluation_rows, system) for system in SYSTEMS
        },
        "by_stratum": {
            stratum: {
                system: _aggregate(
                    [
                        row for row in evaluation_rows
                        if row["stratum"] == stratum
                    ],
                    system,
                )
                for system in SYSTEMS
            }
            for stratum in STRATA
        },
    }
    challenge_deltas = {
        stratum: (
            metrics["by_stratum"][stratum]["round7_model"]["f1"]
            - metrics["by_stratum"][stratum][
                "production_rules_translation_assisted"
            ]["f1"]
        )
        for stratum in STRATA
    }
    challenge_gate_diagnostic = all(
        delta >= -0.05 for delta in challenge_deltas.values()
    )
    comparison, changes = _compare(
        predictions["round6"]["ensemble"],
        predictions["round7"]["ensemble"],
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        predictions_path = staging / "predictions.json"
        audit_path = staging / "error_audit.json"
        changes_path = staging / "round6_round7_geometry_changes.json"
        predictions_path.write_text(
            json.dumps({
                **predictions,
                "production_rules_translation_assisted": rules_prediction_rows,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        audit_path.write_text(
            json.dumps(audit_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        changes_path.write_text(
            json.dumps(changes, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report = {
            "schema_version": 1,
            "status": "round13_compact_challenge_evaluation",
            "supplementary_challenge_evidence_only": True,
            "formal_probability_metric": False,
            "eligible_for_training": False,
            "eligible_to_reverse_failed_precision_gate": False,
            "candidate_model_blind_reference": True,
            "systems": list(SYSTEMS),
            "metrics": metrics,
            "round6_round7_comparison": comparison,
            "round7_minus_rules_f1_by_stratum": challenge_deltas,
            "challenge_no_more_than_5_point_decline_diagnostic": (
                challenge_gate_diagnostic
            ),
            "production_adoption_authorized": False,
            "production_adoption_blockers": [
                "round11_one_sided_95pct_precision_lower_below_0.98",
                "round13_is_supplementary_challenge_only",
            ],
            "determinism": {
                "torch_deterministic_algorithms": True,
                "warn_only": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
                "ensemble_threshold": ENSEMBLE_THRESHOLD,
            },
            "inputs": {
                "reference_report_sha256": EXPECTED_REFERENCE_REPORT_SHA256,
                "reference_sha256": EXPECTED_REFERENCE_SHA256,
                "task_manifest_sha256": EXPECTED_TASK_MANIFEST_SHA256,
                "rules_manifest_sha256": EXPECTED_RULES_MANIFEST_SHA256,
                "rule_document_root_sha256": rule_root,
                "models": model_inputs,
            },
            "git_commit": _git_commit_clean(),
            "outputs": {
                "predictions_sha256": _sha256(predictions_path),
                "error_audit_sha256": _sha256(audit_path),
                "round6_round7_geometry_changes_sha256": _sha256(changes_path),
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
        description="Evaluate fixed models and rules on Round 13 challenges."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_compact_challenge(
        args.reference,
        args.task_manifest,
        args.rules,
        args.artifacts_root,
        args.output,
    )
    print(json.dumps({
        "metrics": report["metrics"],
        "round6_round7_comparison": report["round6_round7_comparison"],
        "round7_minus_rules_f1_by_stratum": (
            report["round7_minus_rules_f1_by_stratum"]
        ),
        "challenge_diagnostic": (
            report["challenge_no_more_than_5_point_decline_diagnostic"]
        ),
        "production_adoption_authorized": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
