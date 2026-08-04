from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from p3_compact import _git_commit_clean
from p3_diagnostic import _validate_pack
from p4_assisted_review import build_review_pack


EXPECTED_TASK_MANIFEST_SHA256 = (
    "5b5c6b46375ef09ae7447b2af800d3f37eff2905ca28942c532bc14359e7e937"
)


def _snapshot(path: Path) -> tuple[dict, str, bytes]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest(), raw


def _teacher_name(task_name: str) -> str:
    if not task_name.startswith("blind_"):
        raise ValueError(f"unexpected challenge task name: {task_name}")
    return "assisted_" + task_name.removeprefix("blind_")


def _normalize_teacher(
    payload: dict, *, juan: int, jie_index: int, pass_name: str
) -> dict:
    expected_pass, channel = {
        "pass-a": ("A-recall-first", "copilot_independent_a"),
        "pass-b": ("B-exact-boundary-first", "copilot_independent_b"),
    }[pass_name]
    if (
        payload.get("schema_version") != 1
        or payload.get("phase") != "assisted"
        or payload.get("supplementary_challenge_evidence_only") is not True
        or payload.get("juan") != juan
        or payload.get("jie_index") != jie_index
        or payload.get("teacher_pass") != expected_pass
        or not isinstance(payload.get("candidates"), list)
        or any(row.get("channels") != [channel] for row in payload["candidates"])
    ):
        raise ValueError(
            f"Round 13 {pass_name} provenance differs: {juan}/{jie_index}"
        )
    normalized = copy.deepcopy(payload)
    normalized["diagnostic_only"] = True
    if pass_name == "pass-b":
        normalized["teacher_pass"] = "B-boundary-first"
    return normalized


def prepare_compact_challenge_review(
    task_root: Path,
    teacher_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"compact challenge review exists: {output_dir}")
    manifest, manifest_sha256, _ = _snapshot(task_root / "manifest.json")
    selections = manifest.get("selected", [])
    private_jies = {
        (int(row["juan"]), int(row["jie_index"]))
        for row in manifest.get("private_selected_jies", [])
    }
    protocol = manifest.get("labeling_protocol", {})
    if (
        manifest_sha256 != EXPECTED_TASK_MANIFEST_SHA256
        or manifest.get("status")
        != "round13_compact_challenge_tasks_before_labeling"
        or manifest.get("supplementary_challenge_evidence_only") is not True
        or manifest.get("formal_probability_metric") is not False
        or manifest.get("eligible_for_training") is not False
        or manifest.get("eligible_to_reverse_failed_precision_gate") is not False
        or manifest.get("candidate_model_blind") is not True
        or manifest.get("challenger_locked")
        != "round7_2_of_3_exact_geometry_ensemble"
        or manifest.get("model_predictions_generated") is not False
        or len(selections) != 8
        or len(private_jies) != 8
        or protocol.get("pass_visibility")
        != "mutually_hidden_candidate_free_raw_text_only"
        or protocol.get("focused_review")
        != "A_B_disagreement_and_explicit_low_only"
        or protocol.get("model_omission_candidates") is not False
    ):
        raise ValueError("compact challenge task manifest binding differs")
    expected_names = {
        _teacher_name(Path(str(row["task"])).name) for row in selections
    }
    teacher_snapshots: dict[str, dict[str, dict]] = {}
    teacher_hashes: dict[str, dict[str, str]] = {}
    for directory in ("pass-a", "pass-b"):
        paths = sorted((teacher_root / directory).glob("batch-*/*.json"))
        by_name = {path.name: path for path in paths}
        if len(paths) != len(by_name) or set(by_name) != expected_names:
            raise ValueError(f"Round 13 {directory} inventory differs")
        teacher_snapshots[directory] = {}
        for name, path in by_name.items():
            payload, digest, _ = _snapshot(path)
            teacher_snapshots[directory][name] = payload
            teacher_hashes.setdefault(name, {})[directory] = digest

    grouped: dict[int, list[tuple[dict, dict, dict]]] = defaultdict(list)
    for selection in selections:
        task_path = task_root / str(selection["task"])
        task, task_sha256, _ = _snapshot(task_path)
        if (
            task_sha256 != selection.get("task_sha256")
            or task.get("candidate_model_blind") is not True
            or len(task.get("jies", [])) != 1
        ):
            raise ValueError(f"Round 13 task differs: {task_path.name}")
        juan = int(task["juan"])
        jie_index = int(task["jies"][0]["jie_index"])
        if (juan, jie_index) not in private_jies:
            raise ValueError(f"Round 13 task jie differs: {juan}/{jie_index}")
        name = _teacher_name(task_path.name)
        pass_a = _normalize_teacher(
            teacher_snapshots["pass-a"][name],
            juan=juan,
            jie_index=jie_index,
            pass_name="pass-a",
        )
        pass_b = _normalize_teacher(
            teacher_snapshots["pass-b"][name],
            juan=juan,
            jie_index=jie_index,
            pass_name="pass-b",
        )
        _validate_pack(task, pass_a, juan)
        _validate_pack(task, pass_b, juan)
        grouped[juan].append((
            task,
            pass_a,
            pass_b,
        ))

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
        review_selections = []
        for juan, rows in sorted(grouped.items()):
            merged_task = copy.deepcopy(rows[0][0])
            merged_task["jies"] = [
                jie
                for task, _, _ in rows
                for jie in task["jies"]
            ]
            merged_a = copy.deepcopy(rows[0][1])
            merged_a["candidates"] = [
                candidate
                for _, pass_a, _ in rows
                for candidate in pass_a["candidates"]
            ]
            merged_b = copy.deepcopy(rows[0][2])
            merged_b["candidates"] = [
                candidate
                for _, _, pass_b in rows
                for candidate in pass_b["candidates"]
            ]
            pack, counts = build_review_pack(
                merged_task, merged_a, merged_b, [], juan
            )
            pack["supplementary_challenge_evidence_only"] = True
            pack["candidate_model_blind"] = True
            pack["human_review_scope"] = (
                "teacher_disagreement_and_explicit_low_only"
            )
            totals.update(counts)
            task_target = tasks_output / f"blind_juan_{juan:03d}.json"
            task_target.write_text(
                json.dumps(merged_task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            pack_target = assisted_output / f"assisted_juan_{juan:03d}.json"
            pack_target.write_text(
                json.dumps(pack, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            review_selections.append({
                "juan": juan,
                "role": "supplementary_challenge_evidence",
                "mode": "active_assisted",
                "task": task_target.name,
                "task_sha256": hashlib.sha256(
                    task_target.read_bytes()
                ).hexdigest(),
                "pack_sha256": hashlib.sha256(
                    pack_target.read_bytes()
                ).hexdigest(),
                "sampled_jies": len(rows),
            })
        if totals.get("model_only_review", 0):
            raise ValueError("model candidates entered Round 13 review")
        review_manifest = {
            "schema_version": 1,
            "status": "round13_compact_challenge_focused_review",
            "supplementary_challenge_evidence_only": True,
            "formal_probability_metric": False,
            "eligible_for_training": False,
            "eligible_to_reverse_failed_precision_gate": False,
            "candidate_model_blind": True,
            "model_predictions_used": False,
            "model_only_candidates": 0,
            "challenger_locked": "round7_2_of_3_exact_geometry_ensemble",
            "human_review_scope": (
                "teacher_disagreement_and_explicit_low_only"
            ),
            "source_manifest_sha256": manifest_sha256,
            "teacher_sha256_by_file": teacher_hashes,
            "git_commit": _git_commit_clean(),
            "counts": dict(totals),
            "selected": review_selections,
        }
        (tasks_output / "manifest.json").write_text(
            json.dumps(review_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        published = list(staging.rglob("*"))
        for path in published:
            if path.is_file():
                path.chmod(0o444)
        for path in sorted(
            (path for path in published if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            path.chmod(0o555)
        staging.chmod(0o555)
        staging.replace(output_dir)
    return review_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Round 13 compact challenge focused review."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--teachers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_compact_challenge_review(
        args.tasks, args.teachers, args.output
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
