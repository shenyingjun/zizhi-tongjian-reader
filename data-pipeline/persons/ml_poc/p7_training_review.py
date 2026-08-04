from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from collections import Counter
from pathlib import Path

from p1_train import evaluate
from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from p4_assisted_review import build_review_pack
from p6_locked_assisted_review import _validate_teacher


EXPECTED_TASK_MANIFEST_SHA256 = (
    "38ee21a4f57cffdbab962a61d0528beb0080387d67fb239c58b8e81ece4b633c"
)
EXPECTED_MODEL_ARTIFACT_SHA256 = (
    "fb611bee98cdc672e19163ca84aef49612815cab285889c7458cff07ba098cbd"
)
EXPECTED_MODEL_REPORT_SHA256 = (
    "b65bccba68bb37dea23da7a57b84336b906a32769145f0d33b9f679d57d73353"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _publish(staging: Path, output: Path) -> None:
    for path in staging.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IREAD)
    staging.replace(output)


def _validate_source(tasks_root: Path) -> tuple[dict, Path]:
    manifest_path = tasks_root / "manifest.json"
    manifest = _read(manifest_path)
    if (
        _sha256(manifest_path) != EXPECTED_TASK_MANIFEST_SHA256
        or manifest.get("status")
        != "round7_copilot_double_pass_training_tasks_before_labeling"
        or manifest.get("training_only") is not True
        or manifest.get("eligible_for_evaluation") is not False
        or manifest.get("model_predictions_used_for_selection") is not False
        or manifest.get("locked_p3_outputs_used_for_selection") is not False
        or manifest.get("model_predictions_generated") is not False
        or len(manifest.get("selected_jies", [])) != 40
        or len(manifest.get("selected", [])) != 40
    ):
        raise ValueError("Round 7 source task binding differs")
    return manifest, manifest_path


def _validate_model(model_root: Path) -> dict:
    artifact = _model_artifact(model_root / "model")
    if (
        artifact["combined_sha256"] != EXPECTED_MODEL_ARTIFACT_SHA256
        or _sha256(model_root / "report.json") != EXPECTED_MODEL_REPORT_SHA256
    ):
        raise ValueError("active Round 3 model binding differs")
    return artifact


def _teacher_inventory(
    tasks_root: Path,
    teachers_root: Path,
    manifest: dict,
) -> tuple[
    dict[str, dict[str, dict]],
    dict[str, dict[str, str]],
    str,
]:
    selections = manifest["selected"]
    expected_names = {
        f"assisted_juan_{int(row['juan']):03d}.json" for row in selections
    }
    teacher_payloads: dict[str, dict[str, dict]] = {}
    teacher_hashes: dict[str, dict[str, str]] = {}
    for pass_name, teacher_pass, source in (
        ("pass-a", "A-recall-first", "copilot_independent_a"),
        ("pass-b", "B-boundary-first", "copilot_independent_b"),
    ):
        paths = list((teachers_root / pass_name).glob("batch-*/*.json"))
        by_name = {path.name: path for path in paths}
        if len(paths) != len(by_name) or set(by_name) != expected_names:
            raise ValueError(f"Round 7 {pass_name} teacher inventory differs")
        teacher_payloads[pass_name] = {}
        for selection in selections:
            juan = int(selection["juan"])
            task_path = tasks_root / str(selection["task"])
            task = _read(task_path)
            teacher_path = by_name[f"assisted_juan_{juan:03d}.json"]
            teacher_bytes = teacher_path.read_bytes()
            teacher = json.loads(teacher_bytes)
            _validate_teacher(task, teacher, juan, teacher_pass, source)
            teacher_payloads[pass_name][teacher_path.name] = teacher
            teacher_hashes.setdefault(str(juan), {})[pass_name] = (
                hashlib.sha256(teacher_bytes).hexdigest()
            )
    inventory_sha256 = hashlib.sha256(
        json.dumps(
            teacher_hashes,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return teacher_payloads, teacher_hashes, inventory_sha256


def predict_round3(
    tasks_root: Path,
    teachers_root: Path,
    model_root: Path,
    output_path: Path,
) -> dict:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Round 7 prediction bundle exists: {output_path}")
    manifest, manifest_path = _validate_source(tasks_root)
    _, teacher_hashes, teacher_inventory_sha256 = _teacher_inventory(
        tasks_root, teachers_root, manifest
    )
    artifact = _validate_model(model_root)
    examples = []
    expected_ids = set()
    for selection in manifest["selected"]:
        juan = int(selection["juan"])
        task_path = tasks_root / str(selection["task"])
        if _sha256(task_path) != selection["task_sha256"]:
            raise ValueError(f"Round 7 task hash differs: {juan}")
        task = _read(task_path)
        for jie in task["jies"]:
            identity = f"juan-{juan:03d}-jie-{int(jie['jie_index']):04d}"
            if identity in expected_ids:
                raise ValueError(f"duplicate Round 7 prediction ID: {identity}")
            expected_ids.add(identity)
            examples.append({
                "id": identity,
                "juan": juan,
                "jie_index": int(jie["jie_index"]),
                "text": str(jie["text"]),
                "labels": ["O"] * len(str(jie["text"])),
                "segments": jie["segments"],
            })

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_root / "model", use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_root / "model")
    model.to(device)
    _, rows = evaluate(
        model,
        tokenizer,
        examples,
        device,
        max_length=512,
        stride=128,
        batch_size=8,
    )
    predictions = [{
        "id": str(row["id"]),
        "prediction_spans": row["prediction_spans"],
    } for row in rows]
    if (
        {row["id"] for row in predictions} != expected_ids
        or len(predictions) != len(expected_ids)
    ):
        raise ValueError("Round 7 model prediction inventory differs")
    bundle = {
        "schema_version": 1,
        "status": "round7_round3_omission_predictions",
        "training_only": True,
        "model_role": "post_teacher_omission_candidates_only",
        "teacher_outputs_completed_before_prediction": True,
        "teacher_sha256_by_juan": teacher_hashes,
        "teacher_inventory_sha256": teacher_inventory_sha256,
        "model_artifact_sha256": artifact["combined_sha256"],
        "model_report_sha256": _sha256(model_root / "report.json"),
        "source_manifest_sha256": _sha256(manifest_path),
        "git_commit": _git_commit_clean(),
        "device": str(device),
        "predictions": predictions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.name}-", dir=output_path.parent
    ) as temporary:
        staging = Path(temporary) / output_path.name
        _write(staging, bundle)
        staging.chmod(stat.S_IREAD)
        os.link(staging, output_path)
    return bundle


def prepare_review(
    tasks_root: Path,
    teachers_root: Path,
    predictions_path: Path,
    model_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Round 7 review exists: {output_dir}")
    manifest, manifest_path = _validate_source(tasks_root)
    _validate_model(model_root)
    selections = manifest["selected"]
    teacher_payloads, teacher_hashes, teacher_inventory_sha256 = _teacher_inventory(
        tasks_root, teachers_root, manifest
    )
    prediction_bundle = _read(predictions_path)
    if (
        prediction_bundle.get("status")
        != "round7_round3_omission_predictions"
        or prediction_bundle.get("training_only") is not True
        or prediction_bundle.get("teacher_outputs_completed_before_prediction")
        is not True
        or prediction_bundle.get("model_artifact_sha256")
        != EXPECTED_MODEL_ARTIFACT_SHA256
        or prediction_bundle.get("model_report_sha256")
        != EXPECTED_MODEL_REPORT_SHA256
        or prediction_bundle.get("source_manifest_sha256")
        != _sha256(manifest_path)
        or prediction_bundle.get("teacher_inventory_sha256")
        != teacher_inventory_sha256
        or prediction_bundle.get("teacher_sha256_by_juan") != teacher_hashes
    ):
        raise ValueError("Round 7 prediction provenance differs")
    prediction_by_juan = {}
    prediction_ids = set()
    for row in prediction_bundle["predictions"]:
        parts = str(row["id"]).split("-")
        juan = int(parts[1])
        jie_index = int(parts[3])
        key = juan, jie_index
        if key in prediction_ids:
            raise ValueError("duplicate Round 7 model prediction")
        prediction_ids.add(key)
        prediction_by_juan.setdefault(juan, []).extend(
            row["prediction_spans"]
        )
    selected_jies = {
        (int(row["juan"]), int(row["jie_index"]))
        for row in manifest["selected_jies"]
    }
    if prediction_ids != selected_jies:
        raise ValueError("Round 7 prediction jie inventory differs")

    totals = Counter()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        tasks_output = staging / "tasks"
        assisted_output = staging / "assisted"
        tasks_output.mkdir()
        assisted_output.mkdir()
        selected = []
        for selection in selections:
            juan = int(selection["juan"])
            task_path = tasks_root / str(selection["task"])
            if _sha256(task_path) != selection["task_sha256"]:
                raise ValueError(f"Round 7 task hash differs: {juan}")
            task = _read(task_path)
            filename = f"assisted_juan_{juan:03d}.json"
            pass_a = teacher_payloads["pass-a"][filename]
            pass_b = teacher_payloads["pass-b"][filename]
            pack, counts = build_review_pack(
                task,
                pass_a,
                pass_b,
                prediction_by_juan.get(juan, []),
                juan,
            )
            totals.update(counts)
            task_target = tasks_output / task_path.name
            pack_target = assisted_output / f"assisted_juan_{juan:03d}.json"
            task_target.write_bytes(task_path.read_bytes())
            _write(pack_target, pack)
            selected.append({
                "juan": juan,
                "role": "round7_training_only",
                "mode": "active_assisted",
                "task": task_target.name,
                "task_sha256": _sha256(task_target),
                "pack_sha256": _sha256(pack_target),
            })
        review_manifest = {
            "schema_version": 1,
            "status": "round7_copilot_assisted_training_review",
            "training_only": True,
            "formal_evaluation": False,
            "eligible_for_training_after_focused_review": True,
            "eligible_for_evaluation": False,
            "source_manifest_sha256": _sha256(manifest_path),
            "model_artifact_sha256": EXPECTED_MODEL_ARTIFACT_SHA256,
            "model_predictions_sha256": _sha256(predictions_path),
            "teacher_sha256_by_juan": teacher_hashes,
            "teacher_inventory_sha256": teacher_inventory_sha256,
            "git_commit": _git_commit_clean(),
            "counts": dict(totals),
            "selected": selected,
        }
        _write(tasks_output / "manifest.json", review_manifest)
        _publish(staging, output_dir)
    return review_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create Round 7 model omissions and focused review."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--tasks", type=Path, required=True)
    predict_parser.add_argument("--teachers", type=Path, required=True)
    predict_parser.add_argument("--model-root", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("--tasks", type=Path, required=True)
    review_parser.add_argument("--teachers", type=Path, required=True)
    review_parser.add_argument("--predictions", type=Path, required=True)
    review_parser.add_argument("--model-root", type=Path, required=True)
    review_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "predict":
        bundle = predict_round3(
            args.tasks, args.teachers, args.model_root, args.output
        )
        result = {
            "jies": len(bundle["predictions"]),
            "spans": sum(
                len(row["prediction_spans"])
                for row in bundle["predictions"]
            ),
        }
    else:
        manifest = prepare_review(
            args.tasks,
            args.teachers,
            args.predictions,
            args.model_root,
            args.output,
        )
        result = manifest["counts"]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
