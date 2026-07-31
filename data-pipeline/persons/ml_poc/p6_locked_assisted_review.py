from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import secrets
import shutil
import stat
import tempfile
from collections import Counter
from pathlib import Path

from p3_compact import _git_commit_clean
from p3_diagnostic import _validate_pack
from p6_locked_assisted_tasks import evaluation_claim_metadata


EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "b020541f12207244d110f38133e0c2dd1a43a554c18c952a1644a6cef45ec401"
)
AUDIT_RATE = 0.10


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _geometry(row: dict) -> tuple[int, int, int]:
    return int(row["para_id"]), int(row["start"]), int(row["end"])


def _validate_teacher(
    task: dict,
    pack: dict,
    juan: int,
    expected_pass: str,
    expected_channel: str,
) -> dict[tuple[int, int, int], dict]:
    _validate_pack(task, pack, juan)
    if (
        pack.get("teacher_pass") != expected_pass
        or any(
            row.get("channels") != [expected_channel]
            for row in pack.get("candidates", [])
        )
    ):
        raise ValueError(f"teacher provenance differs for juan {juan}")
    return {_geometry(row): row for row in pack["candidates"]}


def build_review_pack(
    task: dict,
    pass_a: dict,
    pass_b: dict,
    juan: int,
    audited: set[tuple[int, int, int]],
) -> tuple[dict, Counter]:
    a = _validate_teacher(
        task, pass_a, juan, "A-recall-first", "copilot_independent_a"
    )
    b = _validate_teacher(
        task, pass_b, juan, "B-boundary-first", "copilot_independent_b"
    )
    candidates = []
    initial_annotations = []
    initial_decisions = {}
    counts = Counter()
    for geometry in sorted(set(a) | set(b)):
        sources = []
        if geometry in a:
            sources.append("copilot_independent_a")
        if geometry in b:
            sources.append("copilot_independent_b")
        source = a.get(geometry) or b[geometry]
        agreed = geometry in a and geometry in b
        explicit_low = any(
            row.get("confidence") == "low"
            for row in (a.get(geometry), b.get(geometry))
            if row is not None
        )
        audit = geometry in audited
        if agreed and not explicit_low and not audit:
            confidence = "high"
            reason = ""
            counts["auto_accepted_consensus"] += 1
        elif audit:
            confidence = "low"
            reason = "Predeclared random audit of eligible A/B consensus."
            counts["consensus_audit_review"] += 1
        elif agreed:
            confidence = "low"
            reason = "Both Copilot passes agree, but at least one marked it low."
            counts["explicit_low_review"] += 1
        else:
            confidence = "low"
            reason = "Independent candidate-blind Copilot passes disagree."
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
            initial_annotations.append({
                key: candidate[key]
                for key in ("para_id", "start", "end", "surface")
            })
            if confidence == "high":
                initial_decisions[candidate["id"]] = "accept"
    counts["pass_a_candidates"] = len(a)
    counts["pass_b_candidates"] = len(b)
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
        "candidate_model_blind": True,
        "human_review_scope": (
            "teacher_disagreement_explicit_low_and_consensus_audit"
        ),
        "initial_annotations": initial_annotations,
        "initial_decisions": initial_decisions,
        "candidates": candidates,
        "note_evidence": [],
    }, counts


def prepare_locked_assisted_review(
    tasks_root: Path,
    teachers_root: Path,
    output_dir: Path,
    *,
    audit_seed: int | None = None,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"locked assisted review exists: {output_dir}")
    source_manifest_path = tasks_root / "manifest.json"
    if _sha256(source_manifest_path) != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("locked task manifest hash differs")
    source_manifest = _read(source_manifest_path)
    selections = source_manifest.get("selected", [])
    private_jies = {
        (int(row["juan"]), int(row["jie_index"]))
        for row in source_manifest.get("private_selected_jies", [])
    }
    expected_names = {
        f"assisted_juan_{int(row['juan']):03d}.json"
        for row in selections
    }
    pass_paths = {}
    for pass_name in ("pass-a", "pass-b"):
        paths = list((teachers_root / pass_name).glob("batch-*/*.json"))
        by_name = {path.name: path for path in paths}
        if len(by_name) != len(paths) or set(by_name) != expected_names:
            raise ValueError(f"{pass_name} teacher inventory differs")
        pass_paths[pass_name] = by_name
    if (
        source_manifest.get("status")
        != "candidate_blind_copilot_double_pass_tasks_before_labeling"
        or source_manifest.get("formal_evaluation") is not False
        or source_manifest.get("eligible_for_promotion") is not False
        or source_manifest.get("candidate_model_blind") is not True
        or source_manifest.get("model_predictions_generated") is not False
        or len(private_jies) != 20
    ):
        raise ValueError("source is not an untouched locked diagnostic task set")

    loaded = {}
    eligible_consensus = []
    processed_jies = set()
    for selection in selections:
        juan = int(selection["juan"])
        task_path = tasks_root / str(selection["task"])
        if _sha256(task_path) != selection["task_sha256"]:
            raise ValueError(f"task hash differs for juan {juan}")
        task = _read(task_path)
        task_jies = {
            (juan, int(jie["jie_index"])) for jie in task["jies"]
        }
        expected_task_jies = {key for key in private_jies if key[0] == juan}
        if (
            task_jies != expected_task_jies
            or len(task_jies) != int(selection["sampled_jies"])
        ):
            raise ValueError(f"task jie inventory differs for juan {juan}")
        processed_jies.update(task_jies)
        a_path = pass_paths["pass-a"][f"assisted_juan_{juan:03d}.json"]
        b_path = pass_paths["pass-b"][f"assisted_juan_{juan:03d}.json"]
        pass_a = _read(a_path)
        pass_b = _read(b_path)
        a = _validate_teacher(
            task, pass_a, juan, "A-recall-first", "copilot_independent_a"
        )
        b = _validate_teacher(
            task, pass_b, juan, "B-boundary-first", "copilot_independent_b"
        )
        for geometry in set(a) & set(b):
            if all(
                row.get("confidence") != "low"
                for row in (a[geometry], b[geometry])
            ):
                eligible_consensus.append((juan, *geometry))
        loaded[juan] = (task_path, task, a_path, pass_a, b_path, pass_b)
    if processed_jies != private_jies:
        raise ValueError("teacher tasks do not cover every sampled jie")

    selection_seed = (
        audit_seed if audit_seed is not None else secrets.randbits(128)
    )
    audit_count = (
        max(1, math.ceil(len(eligible_consensus) * AUDIT_RATE))
        if eligible_consensus else 0
    )
    audited_global = set(random.Random(selection_seed).sample(
        sorted(eligible_consensus), audit_count
    ))

    totals = Counter()
    teacher_hashes = {}
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
        for juan in sorted(loaded):
            task_path, task, a_path, pass_a, b_path, pass_b = loaded[juan]
            audited = {
                (para_id, start, end)
                for audit_juan, para_id, start, end in audited_global
                if audit_juan == juan
            }
            review_pack, counts = build_review_pack(
                task, pass_a, pass_b, juan, audited
            )
            totals.update(counts)
            task_target = tasks_output / task_path.name
            pack_target = assisted_output / f"assisted_juan_{juan:03d}.json"
            shutil.copyfile(task_path, task_target)
            pack_target.write_text(
                json.dumps(review_pack, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            teacher_hashes[str(juan)] = {
                "pass_a": _sha256(a_path),
                "pass_b": _sha256(b_path),
            }
            selected.append({
                "juan": juan,
                "mode": "diagnostic_assisted",
                "task": task_target.name,
                "task_sha256": _sha256(task_target),
                "pack_sha256": _sha256(pack_target),
            })
        manifest = {
            "schema_version": 1,
            "status": "candidate_blind_copilot_double_pass_focused_review",
            **evaluation_claim_metadata(),
            "candidate_model_blind": True,
            "reference_locked": False,
            "provenance": "copilot_double_pass_blind_diagnostic",
            "source_manifest_sha256": _sha256(source_manifest_path),
            "teacher_sha256_by_juan": teacher_hashes,
            "audit": {
                "policy": "uniform 10% ceil sample of eligible exact consensus",
                "seed": selection_seed,
                "eligible_consensus": len(eligible_consensus),
                "sampled": audit_count,
                "geometries": [
                    {
                        "juan": row[0],
                        "para_id": row[1],
                        "start": row[2],
                        "end": row[3],
                    }
                    for row in sorted(audited_global)
                ],
            },
            "counts": dict(totals),
            "selected": selected,
            "git_commit": _git_commit_clean(),
        }
        (tasks_output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for directory in (tasks_output, assisted_output):
            for path in directory.iterdir():
                path.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build focused review from candidate-blind Copilot A/B."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--teachers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_locked_assisted_review(
        args.tasks, args.teachers, args.output
    )
    print(json.dumps({
        "counts": manifest["counts"],
        "audit": {
            key: value for key, value in manifest["audit"].items()
            if key != "geometries"
        },
        "tasks": len(manifest["selected"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
