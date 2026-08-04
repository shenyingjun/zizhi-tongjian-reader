from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tempfile
from collections import Counter
from pathlib import Path

from p3_compact import EXPECTED_MODEL_SHA256, _git_commit_clean
from p3_diagnostic import _validate_pack


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(path: Path) -> tuple[bytes, dict, str]:
    raw = path.read_bytes()
    return raw, json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _geometry(row: dict) -> tuple[int, int, int]:
    return (
        int(row["para_id"]),
        int(row["start"]),
        int(row["end"]),
    )


def _validate_ml_seed(task: dict, ml_seed: dict, juan: int) -> None:
    if (
        ml_seed.get("juan") != juan
        or ml_seed.get("phase") != "assisted"
        or ml_seed.get("candidate_model", {}).get("sha256")
        != EXPECTED_MODEL_SHA256
    ):
        raise ValueError(f"invalid ML seed identity for juan {juan}")
    text_by_para = {}
    for jie in task["jies"]:
        text = str(jie["text"])
        for segment in jie["segments"]:
            start = int(segment["assembled_start"])
            end = int(segment["assembled_end"])
            text_by_para[int(segment["para_id"])] = text[start:end]
    seen = set()
    by_para: dict[int, list[tuple[int, int]]] = {}
    for candidate in ml_seed.get("candidates", []):
        geometry = _geometry(candidate)
        para_id, start, end = geometry
        paragraph = text_by_para.get(para_id)
        if (
            geometry in seen
            or paragraph is None
            or not 0 <= start < end <= len(paragraph)
            or paragraph[start:end] != candidate.get("surface")
        ):
            raise ValueError(f"invalid ML seed geometry in juan {juan}: {geometry}")
        seen.add(geometry)
        by_para.setdefault(para_id, []).append((start, end))
    for spans in by_para.values():
        spans.sort()
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            raise ValueError(f"overlapping ML seed candidates in juan {juan}")


def build_review_pack(
    task: dict,
    teacher: dict,
    ml_seed: dict,
    juan: int,
) -> tuple[dict, Counter]:
    teacher_counts = _validate_pack(task, teacher, juan)
    _validate_ml_seed(task, ml_seed, juan)
    teacher_by_geometry = {
        _geometry(row): row for row in teacher["candidates"]
    }
    ml_by_geometry = {
        _geometry(row): row for row in ml_seed.get("candidates", [])
    }
    for geometry, candidate in teacher_by_geometry.items():
        if (
            candidate["confidence"] in {"high", "medium"}
            and geometry not in ml_by_geometry
        ):
            raise ValueError(
                f"non-low teacher/ML disagreement in juan {juan}: {geometry}"
            )
    union = []
    for geometry in sorted(set(teacher_by_geometry) | set(ml_by_geometry)):
        teacher_candidate = teacher_by_geometry.get(geometry)
        if teacher_candidate is not None:
            union.append(dict(teacher_candidate))
            continue
        ml_candidate = ml_by_geometry[geometry]
        union.append({
            "id": f"copilot:{geometry[0]}:{geometry[1]}:{geometry[2]}",
            "para_id": geometry[0],
            "start": geometry[1],
            "end": geometry[2],
            "surface": str(ml_candidate["surface"]),
            "channels": ["round2_ml_only"],
            "confidence": "low",
            "review_reason": (
                "Round 2 ML-only geometry omitted by the independent "
                "Copilot teacher; review as a possible false positive."
            ),
        })
    initial_annotations = [{
        "para_id": int(row["para_id"]),
        "start": int(row["start"]),
        "end": int(row["end"]),
        "surface": str(row["surface"]),
    } for row in teacher["candidates"]]
    initial_decisions = {
        row["id"]: "accept"
        for row in teacher["candidates"]
        if row["confidence"] != "low"
    }
    counts = Counter({
        "teacher_total": len(teacher_by_geometry),
        "ml_total": len(ml_by_geometry),
        "union_total": len(union),
        "teacher_only": len(set(teacher_by_geometry) - set(ml_by_geometry)),
        "ml_only": len(set(ml_by_geometry) - set(teacher_by_geometry)),
        "exact_agreement": len(set(teacher_by_geometry) & set(ml_by_geometry)),
        "review_candidates": sum(
            row["confidence"] == "low" for row in union
        ),
    })
    counts.update({
        f"teacher_{key}": value for key, value in teacher_counts.items()
    })
    return {
        "schema_version": 1,
        "phase": "assisted",
        "juan": juan,
        "active_learning_round": 3,
        "diagnostic_only": True,
        "human_review_scope": "low_confidence_and_model_disagreement",
        "initial_annotations": initial_annotations,
        "initial_decisions": initial_decisions,
        "candidates": union,
        "note_evidence": [],
    }, counts


def prepare_active_review(
    active_dir: Path,
    teacher_batches_dir: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"active review exists: {output_dir}")
    git_commit = _git_commit_clean()
    source_tasks = active_dir / "tasks"
    source_seeds = active_dir / "ml-seeds"
    source_manifest_path = source_tasks / "manifest.json"
    _, source_manifest, source_manifest_sha256 = _snapshot(
        source_manifest_path
    )
    selections = source_manifest.get("selected", [])
    if (
        source_manifest.get("status")
        != "round3_active_learning_teacher_tasks"
        or source_manifest.get("model_sha256") != EXPECTED_MODEL_SHA256
        or len(selections) != 60
        or len({int(row["juan"]) for row in selections}) != 60
    ):
        raise ValueError("active teacher manifest must contain 60 unique juans")
    teacher_paths = list(teacher_batches_dir.glob(
        "batch-*/assisted_juan_*.json"
    ))
    teacher_files = {path.name: path for path in teacher_paths}
    expected_names = {
        f"assisted_juan_{int(row['juan']):03d}.json"
        for row in selections
    }
    if (
        len(teacher_paths) != 60
        or len(teacher_files) != len(teacher_paths)
        or set(teacher_files) != expected_names
    ):
        raise ValueError("teacher batch files differ from active selection")

    totals = Counter()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        tasks_dir = staging / "tasks"
        assisted_dir = staging / "assisted"
        tasks_dir.mkdir()
        assisted_dir.mkdir()
        selected = []
        teacher_hashes = {}
        for selection in selections:
            juan = int(selection["juan"])
            task_source = source_tasks / str(selection["task"])
            seed_source = source_seeds / str(selection["ml_seed"])
            teacher_source = teacher_files[
                f"assisted_juan_{juan:03d}.json"
            ]
            task_raw, task, task_sha256 = _snapshot(task_source)
            _, teacher, teacher_sha256 = _snapshot(teacher_source)
            _, ml_seed, seed_sha256 = _snapshot(seed_source)
            if (
                task_sha256 != selection.get("task_sha256")
                or seed_sha256 != selection.get("ml_seed_sha256")
            ):
                raise ValueError(f"active task/seed hash differs: {juan}")
            pack, counts = build_review_pack(
                task, teacher, ml_seed, juan
            )
            totals.update(counts)
            task_target = tasks_dir / task_source.name
            pack_target = assisted_dir / f"assisted_juan_{juan:03d}.json"
            task_target.write_bytes(task_raw)
            pack_target.write_text(
                json.dumps(pack, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            teacher_hashes[str(juan)] = teacher_sha256
            selected.append({
                "juan": juan,
                "mode": "active_assisted",
                "role": "round3_active_learning",
                "task": task_target.name,
                "task_sha256": _sha256(task_target),
                "pack_sha256": _sha256(pack_target),
            })
        manifest = {
            "schema_version": 1,
            "status": "round3_active_learning_human_review",
            "formal_evaluation": False,
            "eligible_for_training_after_human_review": True,
            "human_review_scope": "low_confidence_and_model_disagreement",
            "source_manifest_sha256": source_manifest_sha256,
            "teacher_sha256_by_juan": teacher_hashes,
            "git_commit": git_commit,
            "counts": dict(totals),
            "selected": selected,
        }
        manifest_path = tasks_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for directory in (tasks_dir, assisted_dir):
            for path in directory.iterdir():
                path.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge Round 3 teachers into a focused review workflow."
    )
    parser.add_argument("--active", type=Path, required=True)
    parser.add_argument("--teacher-batches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_active_review(
        args.active, args.teacher_batches, args.output
    )
    print(json.dumps({
        "tasks": len(manifest["selected"]),
        "counts": manifest["counts"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
