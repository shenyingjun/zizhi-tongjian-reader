from __future__ import annotations

import argparse
import hashlib
import json
import random
import stat
import tempfile
from collections import Counter
from pathlib import Path

from core import Span, score_spans
from p1_audit import audit_split
from p1_dataset import build_examples
from p1_train import evaluate
from p3_compact import (
    EXPECTED_MODEL_SHA256,
    FOREIGN_JIES,
    RANDOM_JIES,
    ROLE_JIES,
    _git_commit_clean,
)


BOOTSTRAP_SEED = 20260729
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_ROLES = {
    "probability_random": RANDOM_JIES,
    "role_appellation_challenge": ROLE_JIES,
    "foreign_title_challenge": FOREIGN_JIES,
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_artifact(model_dir: Path) -> dict:
    files = {}
    for path in sorted(model_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"model artifact may not contain symlinks: {path}")
        if path.is_file():
            files[path.relative_to(model_dir).as_posix()] = _sha256(path)
    if "model.safetensors" not in files:
        raise FileNotFoundError("model artifact has no model.safetensors")
    canonical = json.dumps(
        files, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "files": files,
        "combined_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _publish_read_only(staging: Path, output_dir: Path) -> None:
    for path in staging.iterdir():
        if path.is_file():
            path.chmod(stat.S_IREAD)
            if path.stat().st_mode & (
                stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
            ):
                raise PermissionError(f"failed to freeze output: {path}")
    staging.replace(output_dir)


def _task_paragraphs(task: dict) -> dict[int, str]:
    paragraphs = {}
    for jie in task["jies"]:
        text = str(jie["text"])
        for segment in jie["segments"]:
            para_id = int(segment["para_id"])
            start = int(segment["assembled_start"])
            end = int(segment["assembled_end"])
            paragraph = text[start:end]
            if para_id in paragraphs and paragraphs[para_id] != paragraph:
                raise ValueError(f"conflicting paragraph text: {para_id}")
            paragraphs[para_id] = paragraph
    return paragraphs


def _validated_annotations(task: dict, rows: object) -> list[dict]:
    if not isinstance(rows, list):
        raise ValueError("blind annotations must be a list")
    paragraphs = _task_paragraphs(task)
    validated = []
    geometries = set()
    by_para: dict[int, list[tuple[int, int]]] = {}
    for raw in rows:
        para_id = int(raw["para_id"])
        start = int(raw["start"])
        end = int(raw["end"])
        paragraph = paragraphs.get(para_id)
        if paragraph is None or not 0 <= start < end <= len(paragraph):
            raise ValueError("blind annotation is outside sampled text")
        surface = str(raw["surface"])
        if paragraph[start:end] != surface:
            raise ValueError("blind annotation surface differs")
        geometry = (para_id, start, end)
        if geometry in geometries:
            raise ValueError("duplicate blind annotation geometry")
        geometries.add(geometry)
        by_para.setdefault(para_id, []).append((start, end))
        validated.append({
            "para_id": para_id,
            "start": start,
            "end": end,
            "surface": surface,
        })
    for spans in by_para.values():
        spans.sort()
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            raise ValueError("blind annotations overlap")
    return sorted(
        validated,
        key=lambda row: (row["para_id"], row["start"], row["end"]),
    )


def _selection_roles(manifest: dict) -> dict[tuple[int, int], str]:
    roles = {}
    counts = Counter()
    for row in manifest.get("private_selected_jies", []):
        key = (int(row["juan"]), int(row["jie_index"]))
        role = str(row["role"])
        if key in roles:
            raise ValueError("duplicate compact jie selection")
        if role not in EXPECTED_ROLES:
            raise ValueError(f"unknown compact selection role: {role}")
        roles[key] = role
        counts[role] += 1
    if dict(counts) != EXPECTED_ROLES:
        raise ValueError(f"compact role counts differ: {dict(counts)}")
    return roles


def freeze_reference(
    tasks_dir: Path,
    state_dir: Path,
    model_dir: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"compact reference exists: {output_dir}")
    git_commit = _git_commit_clean()
    manifest_path = tasks_dir / "manifest.json"
    manifest = _read(manifest_path)
    model_artifact = _model_artifact(model_dir)
    if (
        manifest.get("formal_p3") is not True
        or manifest.get("candidate_blind") is not True
        or manifest.get("model_predictions_generated") is not False
        or manifest.get("selected_model", {}).get("sha256")
        != EXPECTED_MODEL_SHA256
        or model_artifact["files"]["model.safetensors"]
        != EXPECTED_MODEL_SHA256
    ):
        raise ValueError("compact manifest is not an untouched formal P3 set")
    roles = _selection_roles(manifest)
    selections = manifest.get("selected", [])
    juans = [int(row["juan"]) for row in selections]
    if len(juans) != len(set(juans)):
        raise ValueError("compact public tasks duplicate a juan")
    expected_state_names = {f"juan_{juan:03d}.json" for juan in juans}
    actual_state_names = {
        path.name for path in state_dir.glob("juan_*.json") if path.is_file()
    }
    if actual_state_names != expected_state_names:
        raise ValueError("compact state files differ from selected tasks")

    examples = []
    frozen_inputs = {}
    seen_jies = set()
    total_spans = 0
    for selection in selections:
        juan = int(selection["juan"])
        if (
            selection.get("mode") != "sealed_blind"
            or selection.get("role") != "compact_sealed"
        ):
            raise ValueError(f"task {juan} is not compact sealed-blind")
        task_path = tasks_dir / str(selection["task"])
        state_path = state_dir / f"juan_{juan:03d}.json"
        task_sha256 = _sha256(task_path)
        if task_sha256 != selection.get("task_sha256"):
            raise ValueError(f"task hash differs: {juan}")
        task = _read(task_path)
        state = _read(state_path)
        blind = state.get("blind", {})
        if not blind.get("complete"):
            raise ValueError(f"compact task is not locked: {juan}")
        annotations = _validated_annotations(
            task, blind.get("annotations")
        )
        built = build_examples(
            juan,
            task,
            {
                "role_audit": {
                    "complete": True,
                    "annotations": annotations,
                },
            },
            label_provenance="human_candidate_blind_compact_p3",
        )
        for example in built:
            key = (juan, int(example["jie_index"]))
            if key not in roles or key in seen_jies:
                raise ValueError("task jie differs from private selection")
            seen_jies.add(key)
            example["evaluation_role"] = roles[key]
        examples.extend(built)
        total_spans += len(annotations)
        frozen_inputs[str(juan)] = {
            "task_sha256": task_sha256,
            "state_sha256": _sha256(state_path),
            "sampled_jies": len(built),
            "spans": len(annotations),
        }
    if seen_jies != set(roles):
        raise ValueError("not every selected compact jie was frozen")
    if len(examples) != sum(EXPECTED_ROLES.values()):
        raise ValueError("compact reference does not contain 20 jies")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        reference_path = staging / "reference.jsonl"
        _write_jsonl(reference_path, examples)
        report = {
            "schema_version": 1,
            "status": "frozen_candidate_blind_compact_p3",
            "formal_p3": True,
            "label_provenance": "human_candidate_blind_compact_p3",
            "jies": len(examples),
            "characters": sum(len(row["text"]) for row in examples),
            "spans": total_spans,
            "roles": dict(Counter(
                str(row["evaluation_role"]) for row in examples
            )),
            "source_manifest_sha256": _sha256(manifest_path),
            "reference_sha256": _sha256(reference_path),
            "selected_model_sha256": EXPECTED_MODEL_SHA256,
            "model_artifact": model_artifact,
            "selection_git_commit": manifest["git_commit"],
            "freeze_git_commit": git_commit,
            "frozen_inputs": frozen_inputs,
        }
        _write_json(staging / "freeze_report.json", report)
        _publish_read_only(staging, output_dir)
    return report


def _span(row: dict) -> Span:
    return Span(
        int(row["para_id"]),
        int(row["start"]),
        int(row["end"]),
        str(row["surface"]),
    )


def _counts(reference: set[Span], prediction: set[Span]) -> dict:
    exact = score_spans(reference, prediction)
    return {
        "reference_spans": exact.reference,
        "prediction_spans": exact.predicted,
        "true_positive": exact.true_positive,
    }


def _scores(counts: dict) -> dict:
    reference = int(counts["reference_spans"])
    prediction = int(counts["prediction_spans"])
    true_positive = int(counts["true_positive"])
    precision = true_positive / prediction if prediction else 0.0
    recall = true_positive / reference if reference else 0.0
    total = precision + recall
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / total if total else 0.0,
    }


def _sum_counts(rows: list[dict], system: str) -> dict:
    return {
        field: sum(int(row[system][field]) for row in rows)
        for field in ("reference_spans", "prediction_spans", "true_positive")
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_probability(
    rows: list[dict],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict:
    if len(rows) != RANDOM_JIES:
        raise ValueError("probability bootstrap requires 12 random jies")
    rng = random.Random(seed)
    values = {
        system: {metric: [] for metric in ("precision", "recall", "f1")}
        for system in ("model", "rules")
    }
    deltas = {metric: [] for metric in ("precision", "recall", "f1")}
    for _ in range(replicates):
        sample = [rng.choice(rows) for _ in rows]
        scores = {
            system: _scores(_sum_counts(sample, system))
            for system in ("model", "rules")
        }
        for system in ("model", "rules"):
            for metric in values[system]:
                values[system][metric].append(scores[system][metric])
        for metric in deltas:
            deltas[metric].append(
                scores["model"][metric] - scores["rules"][metric]
            )

    def intervals(samples: dict[str, list[float]]) -> dict:
        return {
            metric: {
                "ci90": [
                    _percentile(metric_values, 0.05),
                    _percentile(metric_values, 0.95),
                ],
                "ci95": [
                    _percentile(metric_values, 0.025),
                    _percentile(metric_values, 0.975),
                ],
                "one_sided_95_lower": _percentile(
                    metric_values, 0.05
                ),
            }
            for metric, metric_values in samples.items()
        }

    return {
        "unit": "jie",
        "method": "paired nonparametric percentile bootstrap",
        "seed": seed,
        "replicates": replicates,
        "model": intervals(values["model"]),
        "rules": intervals(values["rules"]),
        "model_minus_rules": intervals(deltas),
    }


def _role_metrics(rows: list[dict]) -> dict:
    result = {}
    for role in EXPECTED_ROLES:
        selected = [row for row in rows if row["role"] == role]
        result[role] = {
            "jies": len(selected),
            "characters": sum(int(row["characters"]) for row in selected),
            "model": _scores(_sum_counts(selected, "model")),
            "rules": _scores(_sum_counts(selected, "rules")),
        }
    return result


def evaluate_compact(
    tasks_dir: Path,
    frozen_dir: Path,
    model_dir: Path,
    rules_dir: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"compact evaluation exists: {output_dir}")
    git_commit = _git_commit_clean()
    manifest_path = tasks_dir / "manifest.json"
    manifest = _read(manifest_path)
    freeze_report_path = frozen_dir / "freeze_report.json"
    freeze_report = _read(freeze_report_path)
    reference_path = frozen_dir / "reference.jsonl"
    if (
        freeze_report.get("status")
        != "frozen_candidate_blind_compact_p3"
        or freeze_report.get("formal_p3") is not True
        or freeze_report.get("source_manifest_sha256")
        != _sha256(manifest_path)
        or freeze_report.get("reference_sha256") != _sha256(reference_path)
    ):
        raise ValueError("compact frozen reference binding differs")
    model_path = model_dir / "model.safetensors"
    if _sha256(model_path) != EXPECTED_MODEL_SHA256:
        raise ValueError("evaluation model is not frozen Round 2")
    if freeze_report.get("model_artifact") != _model_artifact(model_dir):
        raise ValueError("complete inference model artifact differs from freeze")
    examples = _read_jsonl(reference_path)
    roles = _selection_roles(manifest)
    if {
        (int(row["juan"]), int(row["jie_index"])): row["evaluation_role"]
        for row in examples
    } != roles:
        raise ValueError("frozen reference roles differ from selection")

    import torch
    import transformers
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to(device)
    model_metrics, predictions = evaluate(
        model,
        tokenizer,
        examples,
        device,
        max_length=512,
        stride=128,
        batch_size=8,
    )
    prediction_by_id = {str(row["id"]): row for row in predictions}
    if len(prediction_by_id) != len(examples):
        raise ValueError("model predictions duplicate an example ID")

    rule_documents = {}
    rule_hashes = {}
    all_rule_rows = []
    per_jie = []
    paragraphs = {}
    for example in examples:
        example_id = str(example["id"])
        juan = int(example["juan"])
        if juan not in rule_documents:
            rule_path = rules_dir / f"juan_{juan:03d}.json"
            rule_documents[juan] = _read(rule_path)
            rule_hashes[str(juan)] = _sha256(rule_path)
            all_rule_rows.extend(
                rule_documents[juan].get("occurrences", [])
            )
        para_ids = {
            int(segment["para_id"]) for segment in example["segments"]
        }
        reference = {
            _span(row)
            for row in prediction_by_id[example_id]["reference_spans"]
        }
        model_spans = {
            _span(row)
            for row in prediction_by_id[example_id]["prediction_spans"]
        }
        rules = {
            _span(row)
            for row in rule_documents[juan].get("occurrences", [])
            if row.get("field", "main") == "main"
            and int(row["para_id"]) in para_ids
        }
        text = str(example["text"])
        for segment in example["segments"]:
            start = int(segment["assembled_start"])
            end = int(segment["assembled_end"])
            paragraphs[int(segment["para_id"])] = text[start:end]
        per_jie.append({
            "id": example_id,
            "role": str(example["evaluation_role"]),
            "characters": len(text),
            "model": _counts(reference, model_spans),
            "rules": _counts(reference, rules),
        })

    probability = [
        row for row in per_jie if row["role"] == "probability_random"
    ]
    role_metrics = _role_metrics(per_jie)
    bootstrap = bootstrap_probability(probability)
    probability_metrics = role_metrics["probability_random"]
    challenge_pass = all(
        role_metrics[role]["model"]["f1"]
        >= role_metrics[role]["rules"]["f1"] - 0.05
        for role in (
            "role_appellation_challenge",
            "foreign_title_challenge",
        )
    )
    model_f1_ci = bootstrap["model"]["f1"]["ci90"]
    rule_f1_ci = bootstrap["rules"]["f1"]["ci90"]
    nonoverlapping_f1 = (
        model_f1_ci[0] > rule_f1_ci[1]
        or rule_f1_ci[0] > model_f1_ci[1]
    )
    adoption_gate = {
        "model_f1_at_least_rules": (
            probability_metrics["model"]["f1"]
            >= probability_metrics["rules"]["f1"]
        ),
        "f1_ci90_nonoverlapping": nonoverlapping_f1,
        "no_challenge_stratum_down_over_5_points": challenge_pass,
        "model_precision_one_sided_95_lower_at_least_0_98": (
            bootstrap["model"]["precision"]["one_sided_95_lower"]
            >= 0.98
        ),
    }
    adoption_gate["all_pass"] = all(adoption_gate.values())
    audit = audit_split(
        examples,
        predictions,
        all_rule_rows,
        paragraphs,
    )
    if audit["metrics"]["model"] != model_metrics:
        raise ValueError("independent model metric aggregation differs")

    report = {
        "schema_version": 1,
        "status": "formal_compact_p3_evaluated",
        "formal_p3": True,
        "claim_scope": (
            "previously unused numbered jies of 20-600 Unicode codepoints; "
            "probability estimates use only the 12 uniform random jies"
        ),
        "low_power_warning": (
            "twelve probability units produce wide intervals; challenge "
            "strata are descriptive and not population estimates"
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "model_predictions_generated_after_reference_freeze": True,
        "evaluation_git_commit": git_commit,
        "inputs": {
            "manifest_sha256": _sha256(manifest_path),
            "freeze_report_sha256": _sha256(freeze_report_path),
            "reference_sha256": _sha256(reference_path),
            "rule_sha256_by_juan": rule_hashes,
        },
        "probability_metrics": probability_metrics,
        "challenge_metrics": {
            role: role_metrics[role]
            for role in (
                "role_appellation_challenge",
                "foreign_title_challenge",
            )
        },
        "bootstrap": bootstrap,
        "adoption_gate": adoption_gate,
        "audit_summary": {
            "gate": audit["gate"],
            "delta": audit["delta"],
            "groups": audit["groups"],
        },
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        _write_json(staging / "report.json", report)
        _write_json(staging / "predictions.json", predictions)
        _write_json(staging / "per_jie_metrics.json", per_jie)
        _write_json(staging / "geometry_audit.json", audit)
        _publish_read_only(staging, output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze and evaluate the compact sealed P3 reference."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--tasks", type=Path, required=True)
    freeze_parser.add_argument("--state", type=Path, required=True)
    freeze_parser.add_argument("--model", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--tasks", type=Path, required=True)
    evaluate_parser.add_argument("--frozen", type=Path, required=True)
    evaluate_parser.add_argument("--model", type=Path, required=True)
    evaluate_parser.add_argument("--rules", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        report = freeze_reference(
            args.tasks, args.state, args.model, args.output
        )
        print(json.dumps({
            key: report[key]
            for key in ("status", "jies", "characters", "spans", "roles")
        }, ensure_ascii=False, indent=2))
    else:
        report = evaluate_compact(
            args.tasks,
            args.frozen,
            args.model,
            args.rules,
            args.output,
        )
        print(json.dumps({
            "status": report["status"],
            "probability_metrics": report["probability_metrics"],
            "challenge_metrics": report["challenge_metrics"],
            "adoption_gate": report["adoption_gate"],
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
