from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

from p3_compact import _git_commit_clean
from p4_assisted_review import build_review_pack


EXPECTED_TASK_MANIFEST_SHA256 = (
    "a23614bc61b19093b0db5865825b57329363c06c1ccea75bed27e35eb17660de"
)


def _snapshot(path: Path) -> tuple[dict, str, bytes]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest(), raw


def prepare_independent_dev_review(
    task_root: Path,
    teacher_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"independent dev review exists: {output_dir}")
    manifest, manifest_sha256, _ = _snapshot(task_root / "manifest.json")
    selections = manifest.get("selected", [])
    protocol = manifest.get("labeling_protocol", {})
    selection_keys = {
        (int(row["juan"]), str(row["task"])) for row in selections
    }
    if (
        manifest_sha256 != EXPECTED_TASK_MANIFEST_SHA256
        or manifest.get("status")
        != "round10_independent_dev_tasks_before_labeling"
        or manifest.get("development_only") is not True
        or manifest.get("eligible_for_training") is not False
        or manifest.get("eligible_for_promotion") is not False
        or manifest.get("candidate_model_blind") is not True
        or manifest.get("model_predictions_generated") is not False
        or len(selections) != 20
        or len(selection_keys) != 20
        or protocol.get("pass_visibility")
        != "mutually_hidden_candidate_free_raw_text_only"
        or protocol.get("focused_review")
        != "A_B_disagreement_and_explicit_low_only"
        or protocol.get("model_omission_candidates") is not False
    ):
        raise ValueError("independent dev task manifest binding differs")

    expected_names = {
        f"assisted_juan_{int(row['juan']):03d}.json" for row in selections
    }
    teacher_snapshots = {}
    teacher_hashes = {}
    for directory in ("pass-a", "pass-b"):
        paths = sorted((teacher_root / directory).glob("batch-*/*.json"))
        by_name = {path.name: path for path in paths}
        if len(paths) != len(by_name) or set(by_name) != expected_names:
            raise ValueError(f"independent dev {directory} inventory differs")
        teacher_snapshots[directory] = {}
        for name, path in by_name.items():
            payload, digest, _ = _snapshot(path)
            teacher_snapshots[directory][name] = payload
            teacher_hashes.setdefault(name, {})[directory] = digest

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
            task_path = task_root / str(selection["task"])
            task, task_sha256, task_raw = _snapshot(task_path)
            if task_sha256 != selection.get("task_sha256"):
                raise ValueError(f"independent dev task hash differs: {juan}")
            name = f"assisted_juan_{juan:03d}.json"
            pack, counts = build_review_pack(
                task,
                teacher_snapshots["pass-a"][name],
                teacher_snapshots["pass-b"][name],
                [],
                juan,
            )
            pack["development_only"] = True
            pack["candidate_model_blind"] = True
            pack["human_review_scope"] = (
                "teacher_disagreement_and_explicit_low_only"
            )
            totals.update(counts)
            task_target = tasks_output / task_path.name
            task_target.write_bytes(task_raw)
            pack_target = assisted_output / name
            pack_target.write_text(
                json.dumps(pack, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            selected.append({
                "juan": juan,
                "role": "independent_development",
                "mode": "active_assisted",
                "task": task_target.name,
                "task_sha256": hashlib.sha256(task_raw).hexdigest(),
                "pack_sha256": hashlib.sha256(
                    pack_target.read_bytes()
                ).hexdigest(),
            })
        if totals.get("model_only_review", 0):
            raise ValueError("model candidates entered independent dev review")
        review_manifest = {
            "schema_version": 1,
            "status": "round10_independent_dev_focused_review",
            "development_only": True,
            "formal_evaluation": False,
            "eligible_for_training": False,
            "eligible_for_promotion": False,
            "candidate_model_blind": True,
            "model_predictions_used": False,
            "model_only_candidates": 0,
            "human_review_scope": (
                "teacher_disagreement_and_explicit_low_only"
            ),
            "source_manifest_sha256": manifest_sha256,
            "teacher_sha256_by_file": teacher_hashes,
            "git_commit": _git_commit_clean(),
            "counts": dict(totals),
            "selected": selected,
        }
        (tasks_output / "manifest.json").write_text(
            json.dumps(review_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        published_paths = list(staging.rglob("*"))
        for path in published_paths:
            if path.is_file():
                path.chmod(0o444)
        for path in sorted(
            (path for path in published_paths if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            path.chmod(0o555)
        staging.replace(output_dir)
    return review_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build candidate-blind independent dev focused review."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--teachers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_independent_dev_review(
        args.tasks, args.teachers, args.output
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
