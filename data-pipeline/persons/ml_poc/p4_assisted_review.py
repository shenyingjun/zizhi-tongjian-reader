from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import tempfile
from collections import Counter
from pathlib import Path

from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from p3_diagnostic import _validate_pack
from p4_fresh_sealed import (
    EXPECTED_MODEL_ARTIFACT_SHA256,
    EXPECTED_MODEL_REPORT_SHA256,
)


def _read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _geometry(row: dict) -> tuple[int, int, int]:
    return int(row["para_id"]), int(row["start"]), int(row["end"])


def _paragraphs(task: dict) -> dict[int, str]:
    result = {}
    for jie in task["jies"]:
        text = str(jie["text"])
        for segment in jie["segments"]:
            result[int(segment["para_id"])] = text[
                int(segment["assembled_start"]):
                int(segment["assembled_end"])
            ]
    return result


def _validate_prediction_rows(task: dict, rows: list[dict]) -> dict:
    paragraphs = _paragraphs(task)
    result = {}
    for row in rows:
        geometry = _geometry(row)
        para_id, start, end = geometry
        paragraph = paragraphs.get(para_id)
        if (
            geometry in result
            or paragraph is None
            or not 0 <= start < end <= len(paragraph)
            or paragraph[start:end] != row.get("surface")
        ):
            raise ValueError(f"invalid model prediction: {geometry}")
        result[geometry] = row
    return result


def build_review_pack(
    task: dict,
    pass_a: dict,
    pass_b: dict,
    model_rows: list[dict],
    juan: int,
) -> tuple[dict, Counter]:
    _validate_pack(task, pass_a, juan)
    _validate_pack(task, pass_b, juan)
    for pack, expected_pass, expected_channel in (
        (pass_a, "A-recall-first", "copilot_independent_a"),
        (pass_b, "B-boundary-first", "copilot_independent_b"),
    ):
        if (
            pack.get("teacher_pass") != expected_pass
            or any(
                row.get("channels") != [expected_channel]
                for row in pack["candidates"]
            )
        ):
            raise ValueError(f"teacher pass provenance differs for juan {juan}")
    a = {_geometry(row): row for row in pass_a["candidates"]}
    b = {_geometry(row): row for row in pass_b["candidates"]}
    model = _validate_prediction_rows(task, model_rows)
    consensus = set(a) & set(b)
    candidates = []
    initial_annotations = []
    initial_decisions = {}
    counts = Counter()
    for geometry in sorted(set(a) | set(b) | set(model)):
        sources = []
        if geometry in a:
            sources.append("copilot_independent_a")
        if geometry in b:
            sources.append("copilot_independent_b")
        if geometry in model:
            sources.append("round3_ml_omission_check")
        source = a.get(geometry) or b.get(geometry) or model[geometry]
        agreed = geometry in consensus
        explicit_low = any(
            row.get("confidence") == "low"
            for row in (a.get(geometry), b.get(geometry))
            if row is not None
        )
        confidence = "high" if agreed and not explicit_low else "low"
        if confidence == "high":
            reason = ""
            counts["auto_accepted_consensus"] += 1
        elif geometry in model and geometry not in a and geometry not in b:
            reason = (
                "Round 3 ML-only omission candidate absent from both "
                "independent Copilot passes."
            )
            counts["model_only_review"] += 1
        elif agreed:
            reason = "Both Copilot passes agree, but at least one marked it low."
            counts["explicit_low_review"] += 1
        else:
            reason = "The independent Copilot passes disagree on this geometry."
            counts["teacher_disagreement_review"] += 1
        candidate = {
            "id": f"copilot:{geometry[0]}:{geometry[1]}:{geometry[2]}",
            "para_id": geometry[0],
            "start": geometry[1],
            "end": geometry[2],
            "surface": str(source["surface"]),
            "channels": sources,
            "confidence": confidence,
            "review_reason": reason,
        }
        candidates.append(candidate)
        if agreed:
            annotation = {
                key: candidate[key]
                for key in ("para_id", "start", "end", "surface")
            }
            initial_annotations.append(annotation)
            if confidence == "high":
                initial_decisions[candidate["id"]] = "accept"
    counts["candidate_union"] = len(candidates)
    counts["initial_annotations"] = len(initial_annotations)
    counts["review_candidates"] = sum(
        row["confidence"] == "low" for row in candidates
    )
    return {
        "schema_version": 1,
        "phase": "assisted",
        "juan": juan,
        "diagnostic_only": True,
        "human_review_scope": "teacher_disagreement_low_and_model_only",
        "initial_annotations": initial_annotations,
        "initial_decisions": initial_decisions,
        "candidates": candidates,
        "note_evidence": [],
    }, counts


def prepare_assisted_review(
    tasks_dir: Path,
    teachers_dir: Path,
    predictions_path: Path,
    model_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"assisted review exists: {output_dir}")
    manifest_path = tasks_dir / "manifest.json"
    source_manifest = _read(manifest_path)
    selections = source_manifest["selected"]
    next_round = (
        source_manifest.get("status")
        == "copilot_double_pass_tasks_before_labeling"
    )
    selected_jie_key = "selected_jies" if next_round else "private_selected_jies"
    expected_jies = 60 if next_round else 20
    private_jies = {
        (int(row["juan"]), int(row["jie_index"]))
        for row in source_manifest.get(selected_jie_key, [])
    }
    if (
        (
            not next_round
            and source_manifest.get("status") != "fresh_sealed_before_annotation"
        )
        or (
            next_round
            and source_manifest.get("model_predictions_used_for_selection")
            is not False
        )
        or (
            not next_round
            and source_manifest.get("model_predictions_generated") is not False
        )
        or len(private_jies) != expected_jies
        or len(selections) != expected_jies
        or len({int(row["juan"]) for row in selections}) != len(selections)
        or (
            next_round
            and any(int(row.get("sampled_jies", 0)) != 1 for row in selections)
        )
    ):
        raise ValueError("source is not an untouched supported task set")
    report_path = model_root / "report.json"
    if (
        _sha256(report_path) != EXPECTED_MODEL_REPORT_SHA256
        or _model_artifact(model_root / "model")["combined_sha256"]
        != EXPECTED_MODEL_ARTIFACT_SHA256
    ):
        raise ValueError("Round 3 model binding differs")
    pass_a_list = list(teachers_dir.glob("pass-a/batch-*/*.json"))
    pass_b_list = list(teachers_dir.glob("pass-b/batch-*/*.json"))
    pass_a_paths = {path.name: path for path in pass_a_list}
    pass_b_paths = {path.name: path for path in pass_b_list}
    expected_names = {
        f"assisted_juan_{int(row['juan']):03d}.json" for row in selections
    }
    if (
        len(pass_a_paths) != len(pass_a_list)
        or len(pass_b_paths) != len(pass_b_list)
        or set(pass_a_paths) != expected_names
        or set(pass_b_paths) != expected_names
    ):
        raise ValueError("teacher pass inventory differs")
    prediction_bundle = _read(predictions_path)
    if (
        not isinstance(prediction_bundle, dict)
        or prediction_bundle.get("model_artifact_sha256")
        != EXPECTED_MODEL_ARTIFACT_SHA256
        or prediction_bundle.get("model_report_sha256")
        != EXPECTED_MODEL_REPORT_SHA256
        or prediction_bundle.get("source_manifest_sha256")
        != _sha256(manifest_path)
    ):
        raise ValueError("model prediction provenance differs")
    prediction_by_juan: dict[int, list[dict]] = {}
    prediction_jies = set()
    for row in prediction_bundle.get("predictions", []):
        parts = str(row["id"]).split("-")
        juan = int(parts[1])
        jie_index = int(parts[3])
        key = (juan, jie_index)
        if key in prediction_jies:
            raise ValueError("duplicate model prediction jie")
        prediction_jies.add(key)
        prediction_by_juan.setdefault(juan, []).extend(
            row["prediction_spans"]
        )
    if prediction_jies != private_jies:
        raise ValueError("model prediction inventory differs")

    totals = Counter()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        task_output = staging / "tasks"
        assisted_output = staging / "assisted"
        task_output.mkdir()
        assisted_output.mkdir()
        selected = []
        teacher_hashes = {}
        processed_jies = set()
        for selection in selections:
            juan = int(selection["juan"])
            task_source = tasks_dir / str(selection["task"])
            if _sha256(task_source) != selection["task_sha256"]:
                raise ValueError(f"task hash differs: {juan}")
            task = _read(task_source)
            task_jies = {
                (juan, int(jie["jie_index"])) for jie in task["jies"]
            }
            expected_task_jies = {
                key for key in private_jies if key[0] == juan
            }
            if (
                task_jies != expected_task_jies
                or len(task_jies) != int(selection["sampled_jies"])
            ):
                raise ValueError(f"task jie inventory differs: {juan}")
            processed_jies.update(task_jies)
            name = f"assisted_juan_{juan:03d}.json"
            pass_a = _read(pass_a_paths[name])
            pass_b = _read(pass_b_paths[name])
            pack, counts = build_review_pack(
                task, pass_a, pass_b, prediction_by_juan[juan], juan
            )
            totals.update(counts)
            task_target = task_output / task_source.name
            pack_target = assisted_output / name
            shutil.copyfile(task_source, task_target)
            pack_target.write_text(
                json.dumps(pack, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            teacher_hashes[str(juan)] = {
                "pass_a": _sha256(pass_a_paths[name]),
                "pass_b": _sha256(pass_b_paths[name]),
            }
            selected.append({
                "juan": juan,
                "role": "copilot_double_pass_diagnostic",
                "mode": "active_assisted",
                "task": task_target.name,
                "task_sha256": _sha256(task_target),
                "pack_sha256": _sha256(pack_target),
            })
        if processed_jies != private_jies:
            raise ValueError("selected tasks do not cover every private jie")
        manifest = {
            "schema_version": 1,
            "status": (
                "round5_copilot_assisted_diagnostic_review"
                if next_round
                else "round3_copilot_assisted_diagnostic_review"
            ),
            "formal_evaluation": False,
            "candidate_blind": False,
            "eligible_for_training_after_human_review": True,
            "source_manifest_sha256": _sha256(manifest_path),
            "model_artifact_sha256": EXPECTED_MODEL_ARTIFACT_SHA256,
            "model_predictions_sha256": _sha256(predictions_path),
            "teacher_sha256_by_juan": teacher_hashes,
            "git_commit": _git_commit_clean(),
            "counts": dict(totals),
            "selected": selected,
        }
        (task_output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for directory in (task_output, assisted_output):
            for path in directory.iterdir():
                path.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare focused review from two Copilot passes and ML."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--teachers", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_assisted_review(
        args.tasks,
        args.teachers,
        args.predictions,
        args.model_root,
        args.output,
    )
    print(json.dumps(manifest["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
