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
from p3_active_finalize import (
    _aggregate_deltas,
    _inventory,
    _snapshot,
    _validate_decisions,
    _write_jsonl,
)
from p3_compact import _git_commit_clean
from report import geometry_delta


def finalize_assisted_review(review_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"assisted freeze exists: {output_dir}")
    git_commit = _git_commit_clean()
    tasks_dir = review_dir / "tasks"
    assisted_dir = review_dir / "assisted"
    state_dir = review_dir / "state"
    manifest_path = tasks_dir / "manifest.json"
    _, manifest, manifest_sha256 = _snapshot(manifest_path)
    selections = manifest.get("selected", [])
    juans = [int(row["juan"]) for row in selections]
    round5 = (
        manifest.get("status")
        == "round5_copilot_assisted_diagnostic_review"
    )
    expected_examples = 60 if round5 else 20
    if (
        (
            not round5
            and manifest.get("status")
            != "round3_copilot_assisted_diagnostic_review"
        )
        or manifest.get("formal_evaluation") is not False
        or manifest.get("candidate_blind") is not False
        or manifest.get("eligible_for_training_after_human_review") is not True
        or len(selections) != expected_examples
        or len(set(juans)) != expected_examples
        or any(row.get("mode") != "active_assisted" for row in selections)
    ):
        raise ValueError("invalid assisted review manifest")
    task_names = {str(row["task"]) for row in selections}
    pack_names = {f"assisted_juan_{juan:03d}.json" for juan in juans}
    state_names = {f"juan_{juan:03d}.json" for juan in juans}
    if (
        _inventory(tasks_dir) != task_names | {"manifest.json"}
        or _inventory(assisted_dir) != pack_names
        or _inventory(state_dir) != state_names
    ):
        raise ValueError("assisted review inventory differs")

    examples = []
    inputs = {}
    decisions_total = Counter()
    review_decisions = Counter()
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
                task_sha256 != selection.get("task_sha256")
                or pack_sha256 != selection.get("pack_sha256")
                or task.get("juan") != juan
                or pack.get("juan") != juan
                or pack.get("phase") != "assisted"
                or pack.get("diagnostic_only") is not True
            ):
                raise ValueError(f"assisted input binding differs: {juan}")
            assisted = state.get("assisted", {})
            if (
                assisted.get("complete") is not True
                or assisted.get("pack_sha256") != pack_sha256
            ):
                raise ValueError(f"assisted review is not locked: {juan}")
            decisions_total.update(
                _validate_decisions(juan, task, pack, assisted)
            )
            by_id = {row["id"]: row for row in pack["candidates"]}
            for candidate_id, decision in assisted["decisions"].items():
                if by_id[candidate_id]["confidence"] == "low":
                    review_decisions[decision] += 1
            final_examples = build_examples(
                juan,
                task,
                {"role_audit": {
                    "complete": True,
                    "annotations": assisted["annotations"],
                }},
                label_provenance=(
                    "human_reviewed_copilot_double_pass_diagnostic"
                ),
            )
            examples.extend(final_examples)
            delta_by_juan[str(juan)] = geometry_delta(
                _spans(pack["initial_annotations"]),
                _spans(assisted["annotations"]),
            )
            inputs[str(juan)] = {
                "task_sha256": task_sha256,
                "pack_sha256": pack_sha256,
                "state_sha256": state_sha256,
            }
        identities = {
            (int(row["juan"]), int(row["jie_index"])) for row in examples
        }
        if (
            len(examples) != expected_examples
            or len(identities) != expected_examples
        ):
            raise ValueError(
                "frozen assisted set has the wrong unique jie count"
            )
        examples.sort(key=lambda row: (row["juan"], row["jie_index"]))
        labels_name = (
            "train_assisted_round5.jsonl"
            if round5 else "train_assisted_round4.jsonl"
        )
        labels_path = staging / labels_name
        _write_jsonl(labels_path, examples)
        report = {
            "schema_version": 1,
            "status": "frozen_copilot_double_pass_diagnostic",
            "formal_evaluation": False,
            "eligible_for_formal_metric": False,
            "eligible_for_training": True,
            "label_provenance": (
                "human_reviewed_copilot_double_pass_diagnostic"
            ),
            "git_commit": git_commit,
            "examples": len(examples),
            "characters": sum(len(row["text"]) for row in examples),
            "spans": sum(int(row["span_count"]) for row in examples),
            "juans": sorted(juans),
            "candidate_decisions": dict(decisions_total),
            "focused_review_decisions": dict(review_decisions),
            "consensus_to_final_geometry": _aggregate_deltas(delta_by_juan),
            "consensus_to_final_geometry_by_juan": delta_by_juan,
            "frozen_inputs": inputs,
            "source_manifest_sha256": manifest_sha256,
            "outputs": {
                labels_name.replace(".jsonl", "_sha256"): hashlib.sha256(
                    labels_path.read_bytes()
                ).hexdigest()
            },
        }
        (staging / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in staging.iterdir():
            path.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze reviewed double-pass Copilot labels."
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = finalize_assisted_review(args.review, args.output)
    print(json.dumps({
        key: report[key]
        for key in (
            "examples", "characters", "spans", "candidate_decisions",
            "focused_review_decisions", "consensus_to_final_geometry",
        )
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
