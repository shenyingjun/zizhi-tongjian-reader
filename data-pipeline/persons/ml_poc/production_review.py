from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import stat
import tempfile
from collections import Counter
from pathlib import Path


PASS_CONFIG = {
    "pass-a": ("A-recall-first", "copilot_independent_a"),
    "pass-b": ("B-boundary-first", "copilot_independent_b"),
}
CONSENSUS_AUDIT_RATE = 0.20
NEGATIVE_JIE_AUDIT_RATE = 0.20


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _geometry(row: dict) -> tuple[int, int, int]:
    return int(row["para_id"]), int(row["start"]), int(row["end"])


def _task_paragraphs(task: dict) -> dict[int, str]:
    jies = task.get("jies")
    if not isinstance(jies, list) or len(jies) != 1:
        raise ValueError("production review task must contain exactly one jie")
    jie = jies[0]
    text = str(jie["text"])
    paragraphs = {}
    for segment in jie["segments"]:
        start = int(segment["assembled_start"])
        end = int(segment["assembled_end"])
        para_id = int(segment["para_id"])
        if para_id in paragraphs or not 0 <= start <= end <= len(text):
            raise ValueError("invalid task segment geometry")
        paragraphs[para_id] = text[start:end]
    return paragraphs


def _validate_teacher(
    task: dict,
    task_id: str,
    task_sha256: str,
    payload: dict,
    *,
    expected_pass: str,
    expected_channel: str,
    expected_phase: str = "assisted",
) -> dict[tuple[int, int, int], dict]:
    jie = task["jies"][0]
    if (
        payload.get("schema_version") != 1
        or payload.get("phase") != expected_phase
        or payload.get("training_only") is not True
        or payload.get("candidate_model_blind") is not True
        or payload.get("task_id") != task_id
        or payload.get("task_sha256") != task_sha256
        or int(payload.get("juan")) != int(task["juan"])
        or int(payload.get("jie_index")) != int(jie["jie_index"])
        or payload.get("teacher_pass") != expected_pass
        or payload.get("channel") != expected_channel
    ):
        raise ValueError(f"teacher provenance differs: {task_id} {expected_pass}")
    paragraphs = _task_paragraphs(task)
    indexed = {}
    by_para: dict[int, list[tuple[int, int]]] = {}
    previous = None
    for row in payload.get("candidates", []):
        geometry = _geometry(row)
        para_id, start, end = geometry
        paragraph = paragraphs.get(para_id)
        confidence = row.get("confidence")
        reason = row.get("review_reason")
        if (
            geometry in indexed
            or paragraph is None
            or not 0 <= start < end <= len(paragraph)
            or paragraph[start:end] != row.get("surface")
            or row.get("id") != f"copilot:{para_id}:{start}:{end}"
            or confidence not in {"high", "medium", "low"}
            or (confidence == "high" and reason != "")
            or (confidence != "high" and not reason)
            or (previous is not None and geometry < previous)
        ):
            raise ValueError(f"invalid teacher candidate: {task_id} {geometry}")
        previous = geometry
        indexed[geometry] = row
        by_para.setdefault(para_id, []).append((start, end))
    for spans in by_para.values():
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            raise ValueError(f"overlapping teacher candidates: {task_id}")
    return indexed


def _review_candidate(
    task_id: str,
    geometry: tuple[int, int, int],
    a: dict | None,
    b: dict | None,
    *,
    audited: bool,
) -> tuple[dict, bool]:
    sources = []
    if a is not None:
        sources.append("copilot_independent_a")
    if b is not None:
        sources.append("copilot_independent_b")
    source = a or b
    assert source is not None
    agreed = a is not None and b is not None
    explicit_low = any(
        row["confidence"] == "low" for row in (a, b) if row is not None
    )
    if audited:
        reason = "Predeclared 20% audit of exact non-low A/B consensus."
    elif explicit_low:
        reason = "At least one independent pass marked this geometry low confidence."
    elif not agreed:
        reason = "Independent candidate-blind passes disagree on this geometry."
    else:
        reason = ""
    auto_accept = agreed and not explicit_low and not audited
    para_id, start, end = geometry
    return {
        "id": f"copilot:{task_id}:{para_id}:{start}:{end}",
        "para_id": para_id,
        "start": start,
        "end": end,
        "surface": source["surface"],
        "channels": sources,
        "confidence": "high" if auto_accept else "low",
        "review_reason": reason,
        "pass_confidence": {
            "a": a["confidence"] if a is not None else None,
            "b": b["confidence"] if b is not None else None,
        },
    }, auto_accept


def prepare_review(
    round_root: Path,
    teachers_root: Path,
    output_dir: Path,
    *,
    audit_seed: int,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"production review output exists: {output_dir}")
    source_manifest_path = round_root / "manifest.json"
    private_path = round_root / "private" / "selection.json"
    source = _read(source_manifest_path)
    if (
        source.get("schema_version") != 1
        or source.get("status") != "ml_production_round_tasks_before_labeling"
        or source.get("candidate_model_blind") is not True
        or source.get("model_predictions_generated") is not False
        or source.get("rules_loaded") is not False
        or source.get("v1_loaded") is not False
        or source.get("identity_data_loaded") is not False
        or source.get("private_selection_sha256") != _sha256(private_path)
        or len(source.get("tasks", [])) != 180
    ):
        raise ValueError("production round source binding differs")
    expected_names = {
        Path(str(row["task"])).name for row in source["tasks"]
    }
    if len(expected_names) != len(source["tasks"]):
        raise ValueError("production task filenames are not unique")
    teacher_paths = {}
    for pass_name in PASS_CONFIG:
        paths = list((teachers_root / pass_name).glob("*.json"))
        by_name = {path.name: path for path in paths}
        if len(paths) != len(by_name) or set(by_name) != expected_names:
            raise ValueError(f"{pass_name} teacher inventory differs")
        teacher_paths[pass_name] = by_name

    loaded = []
    consensus = []
    negative_tasks = []
    for row in source["tasks"]:
        task_id = str(row["task_id"])
        filename = Path(str(row["task"])).name
        task_path = round_root / str(row["task"])
        task_sha256 = _sha256(task_path)
        if task_sha256 != row["task_sha256"]:
            raise ValueError(f"production task hash differs: {task_id}")
        task = _read(task_path)
        passes = {}
        pass_hashes = {}
        for pass_name, (expected_pass, expected_channel) in PASS_CONFIG.items():
            path = teacher_paths[pass_name][filename]
            payload = _read(path)
            passes[pass_name] = _validate_teacher(
                task,
                task_id,
                task_sha256,
                payload,
                expected_pass=expected_pass,
                expected_channel=expected_channel,
            )
            pass_hashes[pass_name] = _sha256(path)
        exact_consensus = set(passes["pass-a"]) & set(passes["pass-b"])
        for geometry in exact_consensus:
            if all(
                passes[name][geometry]["confidence"] != "low"
                for name in PASS_CONFIG
            ):
                consensus.append((task_id, geometry))
        if not set(passes["pass-a"]) | set(passes["pass-b"]):
            negative_tasks.append(task_id)
        loaded.append((row, task_path, task, passes, pass_hashes))

    rng = random.Random(audit_seed)
    audit_count = math.ceil(len(consensus) * CONSENSUS_AUDIT_RATE)
    audited_consensus = set(rng.sample(sorted(consensus), audit_count))
    negative_count = math.ceil(len(negative_tasks) * NEGATIVE_JIE_AUDIT_RATE)
    audited_negative = set(rng.sample(sorted(negative_tasks), negative_count))

    totals = Counter()
    manifest = {
        "schema_version": 1,
        "status": "ml_production_focused_review",
        "candidate_model_blind": True,
        "model_predictions_used": False,
        "source_manifest_sha256": _sha256(source_manifest_path),
        "private_selection_sha256": _sha256(private_path),
        "consensus_audit_rate": CONSENSUS_AUDIT_RATE,
        "negative_jie_audit_rate": NEGATIVE_JIE_AUDIT_RATE,
        "selected": [],
        "counts": {},
    }
    private = {
        "schema_version": 1,
        "status": "ml_production_private_review_selection",
        "audit_seed": audit_seed,
        "audited_consensus": [
            {
                "task_id": task_id,
                "para_id": geometry[0],
                "start": geometry[1],
                "end": geometry[2],
            }
            for task_id, geometry in sorted(audited_consensus)
        ],
        "negative_audit_task_ids": sorted(audited_negative),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        task_dir = staging / "tasks"
        pack_dir = staging / "review"
        negative_dir = staging / "negative-audit-tasks"
        private_dir = staging / "private"
        for directory in (task_dir, pack_dir, negative_dir, private_dir):
            directory.mkdir()
        for row, task_path, task, passes, pass_hashes in loaded:
            task_id = str(row["task_id"])
            geometries = sorted(set(passes["pass-a"]) | set(passes["pass-b"]))
            candidates = []
            initial_annotations = []
            initial_decisions = {}
            for geometry in geometries:
                candidate, auto_accept = _review_candidate(
                    task_id,
                    geometry,
                    passes["pass-a"].get(geometry),
                    passes["pass-b"].get(geometry),
                    audited=(task_id, geometry) in audited_consensus,
                )
                candidates.append(candidate)
                totals["candidate_union"] += 1
                if len(candidate["channels"]) == 2:
                    totals["exact_consensus"] += 1
                else:
                    totals["teacher_disagreement_review"] += 1
                if candidate["review_reason"].startswith("Predeclared"):
                    totals["consensus_audit_review"] += 1
                if any(
                    value == "low"
                    for value in candidate["pass_confidence"].values()
                ):
                    totals["explicit_low_review"] += 1
                if auto_accept:
                    annotation = {
                        key: candidate[key]
                        for key in ("para_id", "start", "end", "surface")
                    }
                    initial_annotations.append(annotation)
                    initial_decisions[candidate["id"]] = "accept"
                    totals["auto_accepted_consensus"] += 1
            if task_id in audited_negative:
                totals["negative_jie_third_pass"] += 1
                shutil.copyfile(
                    task_path, negative_dir / task_path.name
                )
            pack = {
                "schema_version": 1,
                "phase": "assisted",
                "training_only": True,
                "candidate_model_blind": True,
                "task_id": task_id,
                "juan": int(task["juan"]),
                "jie_index": int(task["jies"][0]["jie_index"]),
                "human_review_scope": (
                    "teacher_disagreement_explicit_low_and_consensus_audit"
                ),
                "initial_annotations": initial_annotations,
                "initial_decisions": initial_decisions,
                "candidates": candidates,
            }
            task_target = task_dir / task_path.name
            pack_target = pack_dir / task_path.name
            shutil.copyfile(task_path, task_target)
            pack_target.write_text(
                json.dumps(pack, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest["selected"].append({
                "task_id": task_id,
                "task": str(Path("tasks") / task_target.name),
                "task_sha256": _sha256(task_target),
                "review": str(Path("review") / pack_target.name),
                "review_sha256": _sha256(pack_target),
                "teacher_sha256": pass_hashes,
                "review_candidates": sum(
                    candidate["confidence"] == "low"
                    for candidate in candidates
                ),
            })
        private_path_out = private_dir / "selection.json"
        private_path_out.write_text(
            json.dumps(private, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["private_selection_sha256"] = _sha256(private_path_out)
        manifest["counts"] = dict(totals)
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in staging.rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate blind teacher passes and freeze focused review."
    )
    parser.add_argument("--round", type=Path, required=True)
    parser.add_argument("--teachers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-seed", type=int, required=True)
    args = parser.parse_args()
    manifest = prepare_review(
        args.round,
        args.teachers,
        args.output,
        audit_seed=args.audit_seed,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
