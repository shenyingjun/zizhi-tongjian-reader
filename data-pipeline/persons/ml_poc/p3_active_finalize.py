from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tempfile
from collections import Counter
from pathlib import Path

from p1_dataset import build_examples
from p2_round import _spans
from p3_compact import _git_commit_clean
from report import geometry_delta


def _snapshot(path: Path) -> tuple[bytes, dict, str]:
    raw = path.read_bytes()
    return raw, json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _geometry(row: dict) -> tuple[int, int, int]:
    return int(row["para_id"]), int(row["start"]), int(row["end"])


def _inventory(directory: Path) -> set[str]:
    entries = list(directory.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise ValueError(f"active review inventory is not regular files: {directory}")
    return {path.name for path in entries}


def _validate_decisions(
    juan: int,
    task: dict,
    pack: dict,
    assisted: dict,
) -> Counter:
    candidates = pack.get("candidates", [])
    candidate_ids = [str(row["id"]) for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"active pack duplicates candidate IDs: {juan}")
    decisions = assisted.get("decisions")
    if (
        not isinstance(decisions, dict)
        or set(decisions) != set(candidate_ids)
        or any(value not in {"accept", "reject"} for value in decisions.values())
    ):
        raise ValueError(f"active decisions are incomplete or invalid: {juan}")
    annotation_geometry = {
        (*_geometry(row), str(row["surface"]))
        for row in assisted.get("annotations", [])
    }
    text_by_para = {}
    for jie in task["jies"]:
        text = str(jie["text"])
        for segment in jie["segments"]:
            start = int(segment["assembled_start"])
            end = int(segment["assembled_end"])
            text_by_para[int(segment["para_id"])] = text[start:end]
    seen_geometry = set()
    for candidate in candidates:
        para_id, start, end = _geometry(candidate)
        geometry = (para_id, start, end)
        paragraph = text_by_para.get(para_id)
        if (
            candidate.get("id") != f"copilot:{para_id}:{start}:{end}"
            or geometry in seen_geometry
            or paragraph is None
            or not 0 <= start < end <= len(paragraph)
            or paragraph[start:end] != candidate.get("surface")
        ):
            raise ValueError(
                f"active candidate geometry is invalid: "
                f"{juan} {candidate.get('id')}"
            )
        seen_geometry.add(geometry)
        accepted = decisions[str(candidate["id"])] == "accept"
        normalized = (*geometry, str(candidate["surface"]))
        if accepted != (normalized in annotation_geometry):
            raise ValueError(
                f"active decision differs from annotation geometry: "
                f"{juan} {candidate['id']}"
            )
    return Counter(decisions.values())


def _aggregate_deltas(per_juan: dict[str, dict]) -> dict:
    fields = (
        "raw_additions",
        "removals",
        "net_growth",
        "geometry_replacements",
        "pure_additions",
        "pure_removals",
    )
    result = {
        field: sum(row[field] for row in per_juan.values())
        for field in fields
    }
    result["replacement_examples"] = [
        {"juan": int(juan), **example}
        for juan, row in per_juan.items()
        for example in row["replacement_examples"]
    ][:50]
    return result


def finalize_active_review(
    review_dir: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"active training freeze exists: {output_dir}")
    git_commit = _git_commit_clean()
    tasks_dir = review_dir / "tasks"
    assisted_dir = review_dir / "assisted"
    state_dir = review_dir / "state"
    manifest_path = tasks_dir / "manifest.json"
    _, manifest, manifest_sha256 = _snapshot(manifest_path)
    selections = manifest.get("selected", [])
    juans = [int(row["juan"]) for row in selections]
    if (
        manifest.get("status") != "round3_active_learning_human_review"
        or manifest.get("formal_evaluation") is not False
        or manifest.get("eligible_for_training_after_human_review") is not True
        or len(juans) != 60
        or len(set(juans)) != 60
        or any(row.get("mode") != "active_assisted" for row in selections)
    ):
        raise ValueError("invalid Round 3 active review manifest")
    expected_tasks_names = {
        str(row["task"]) for row in selections
    } | {"manifest.json"}
    expected_pack_names = {
        f"assisted_juan_{juan:03d}.json" for juan in juans
    }
    expected_state_names = {
        f"juan_{juan:03d}.json" for juan in juans
    }
    if (
        _inventory(tasks_dir) != expected_tasks_names
        or _inventory(assisted_dir) != expected_pack_names
        or _inventory(state_dir) != expected_state_names
    ):
        raise ValueError("active review file inventory differs from manifest")

    examples = []
    inputs = {}
    decisions_total = Counter()
    delta_by_juan = {}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        for selection in selections:
            juan = int(selection["juan"])
            task_path = tasks_dir / str(selection["task"])
            pack_path = assisted_dir / f"assisted_juan_{juan:03d}.json"
            state_path = state_dir / f"juan_{juan:03d}.json"
            _, task, task_sha256 = _snapshot(task_path)
            _, pack, pack_sha256 = _snapshot(pack_path)
            _, state, state_sha256 = _snapshot(state_path)
            if (
                selection.get("task") != f"blind_juan_{juan:03d}.json"
                or
                task_sha256 != selection.get("task_sha256")
                or pack_sha256 != selection.get("pack_sha256")
                or task.get("juan") != juan
                or task.get("phase") != "assisted"
                or pack.get("juan") != juan
                or pack.get("phase") != "assisted"
                or pack.get("active_learning_round") != 3
                or pack.get("diagnostic_only") is not True
            ):
                raise ValueError(f"active frozen input binding differs: {juan}")
            assisted = state.get("assisted", {})
            if (
                assisted.get("complete") is not True
                or assisted.get("pack_sha256") != pack_sha256
            ):
                raise ValueError(f"active review is not locked and bound: {juan}")
            decisions_total.update(
                _validate_decisions(juan, task, pack, assisted)
            )
            teacher_examples = build_examples(
                juan,
                task,
                {
                    "role_audit": {
                        "complete": True,
                        "annotations": pack.get("initial_annotations", []),
                    },
                },
                label_provenance="copilot_round3_teacher_initial",
            )
            final_examples = build_examples(
                juan,
                task,
                {
                    "role_audit": {
                        "complete": True,
                        "annotations": assisted.get("annotations", []),
                    },
                },
                label_provenance="human_audited_round3_active_training",
            )
            delta_by_juan[str(juan)] = geometry_delta(
                _spans(pack.get("initial_annotations", [])),
                _spans(assisted.get("annotations", [])),
            )
            if len(teacher_examples) != len(final_examples):
                raise ValueError(f"active example inventory differs: {juan}")
            examples.extend(final_examples)
            inputs[str(juan)] = {
                "task_sha256": task_sha256,
                "pack_sha256": pack_sha256,
                "state_sha256": state_sha256,
            }
        identities = [
            (int(row["juan"]), int(row["jie_index"])) for row in examples
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("active examples duplicate a juan/jie identity")
        examples.sort(key=lambda row: (row["juan"], row["jie_index"]))
        train_path = staging / "train_round3_active.jsonl"
        _write_jsonl(train_path, examples)
        train_sha256 = hashlib.sha256(train_path.read_bytes()).hexdigest()
        delta = _aggregate_deltas(delta_by_juan)
        report = {
            "schema_version": 1,
            "status": "frozen_round3_active_training",
            "formal_evaluation": False,
            "eligible_for_formal_metric": False,
            "label_provenance": "human_audited_round3_active_training",
            "git_commit": git_commit,
            "examples": len(examples),
            "characters": sum(len(row["text"]) for row in examples),
            "spans": sum(int(row["span_count"]) for row in examples),
            "juans": sorted(set(juans)),
            "candidate_decisions": dict(decisions_total),
            "teacher_to_final_geometry": delta,
            "teacher_to_final_geometry_by_juan": delta_by_juan,
            "frozen_inputs": inputs,
            "source_manifest_sha256": manifest_sha256,
            "outputs": {
                "train_round3_active_sha256": train_sha256,
            },
        }
        report_path = staging / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in staging.iterdir():
            path.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze locked Round 3 active labels as training BIO."
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = finalize_active_review(args.review, args.output)
    print(json.dumps({
        "examples": report["examples"],
        "characters": report["characters"],
        "spans": report["spans"],
        "candidate_decisions": report["candidate_decisions"],
        "teacher_to_final_geometry": report["teacher_to_final_geometry"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
