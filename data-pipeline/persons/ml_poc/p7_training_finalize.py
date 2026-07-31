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


EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "38ee21a4f57cffdbab962a61d0528beb0080387d67fb239c58b8e81ece4b633c"
)
EXPECTED_REVIEW_MANIFEST_SHA256 = (
    "900c33a728a4ebe018fd3e2e848bacaa47b8d47311ae069ac855b8999fa9181c"
)
EXPECTED_MODEL_ARTIFACT_SHA256 = (
    "fb611bee98cdc672e19163ca84aef49612815cab285889c7458cff07ba098cbd"
)


def finalize_training_labels(
    review_dir: Path,
    state_dir: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Round 7 training freeze exists: {output_dir}")
    tasks_dir = review_dir / "tasks"
    assisted_dir = review_dir / "assisted"
    manifest_path = tasks_dir / "manifest.json"
    _, manifest, manifest_sha256 = _snapshot(manifest_path)
    selections = manifest.get("selected", [])
    juans = [int(row["juan"]) for row in selections]
    if (
        manifest_sha256 != EXPECTED_REVIEW_MANIFEST_SHA256
        or manifest.get("status") != "round7_copilot_assisted_training_review"
        or manifest.get("training_only") is not True
        or manifest.get("formal_evaluation") is not False
        or manifest.get("eligible_for_training_after_focused_review") is not True
        or manifest.get("eligible_for_evaluation") is not False
        or manifest.get("source_manifest_sha256")
        != EXPECTED_SOURCE_MANIFEST_SHA256
        or manifest.get("model_artifact_sha256")
        != EXPECTED_MODEL_ARTIFACT_SHA256
        or manifest.get("counts")
        != {
            "candidate_union": 475,
            "initial_annotations": 438,
            "review_candidates": 37,
            "auto_accepted_consensus": 438,
            "model_only_review": 14,
            "teacher_disagreement_review": 23,
        }
        or len(selections) != 40
        or len(set(juans)) != 40
        or any(row.get("mode") != "active_assisted" for row in selections)
    ):
        raise ValueError("invalid Round 7 training review manifest")

    task_names = {str(row["task"]) for row in selections}
    pack_names = {f"assisted_juan_{juan:03d}.json" for juan in juans}
    state_names = {f"juan_{juan:03d}.json" for juan in juans}
    if (
        _inventory(tasks_dir) != task_names | {"manifest.json"}
        or _inventory(assisted_dir) != pack_names
        or _inventory(state_dir) != state_names
    ):
        raise ValueError("Round 7 training review inventory differs")

    examples = []
    inputs = {}
    decisions_total = Counter()
    focused_decisions = Counter()
    initial_decision_count = 0
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
                or pack.get("human_review_scope")
                != "teacher_disagreement_low_and_model_only"
            ):
                raise ValueError(f"Round 7 input binding differs: {juan}")
            assisted = state.get("assisted", {})
            if (
                assisted.get("complete") is not True
                or assisted.get("pack_sha256") != pack_sha256
            ):
                raise ValueError(f"Round 7 review is not locked: {juan}")
            decisions_total.update(
                _validate_decisions(juan, task, pack, assisted)
            )
            initial_decisions = pack.get("initial_decisions", {})
            initial_decision_count += len(initial_decisions)
            for candidate_id, decision in assisted["decisions"].items():
                if candidate_id not in initial_decisions:
                    focused_decisions[decision] += 1
            final_examples = build_examples(
                juan,
                task,
                {
                    "role_audit": {
                        "complete": True,
                        "annotations": assisted["annotations"],
                    }
                },
                label_provenance=(
                    "focused_reviewed_copilot_double_pass_round7_training"
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
        if len(examples) != 40 or len(identities) != 40:
            raise ValueError("Round 7 freeze has the wrong unique jie count")
        if (
            initial_decision_count != 438
            or sum(focused_decisions.values()) != 37
            or sum(decisions_total.values()) != 475
        ):
            raise ValueError("Round 7 decision inventory differs")
        examples.sort(key=lambda row: (row["juan"], row["jie_index"]))
        labels_path = staging / "train_assisted_round7.jsonl"
        _write_jsonl(labels_path, examples)
        report = {
            "schema_version": 1,
            "status": "frozen_round7_copilot_assisted_training",
            "training_only": True,
            "formal_evaluation": False,
            "eligible_for_training": True,
            "eligible_for_evaluation": False,
            "label_provenance": (
                "focused_reviewed_copilot_double_pass_round7_training"
            ),
            "git_commit": _git_commit_clean(),
            "examples": len(examples),
            "characters": sum(len(row["text"]) for row in examples),
            "spans": sum(int(row["span_count"]) for row in examples),
            "juans": sorted(juans),
            "candidate_decisions": dict(decisions_total),
            "focused_review_decisions": dict(focused_decisions),
            "consensus_to_final_geometry": _aggregate_deltas(delta_by_juan),
            "consensus_to_final_geometry_by_juan": delta_by_juan,
            "frozen_inputs": inputs,
            "review_manifest_sha256": manifest_sha256,
            "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
            "outputs": {
                "train_assisted_round7_sha256": hashlib.sha256(
                    labels_path.read_bytes()
                ).hexdigest()
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
        description="Freeze focused-reviewed Round 7 training labels."
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = finalize_training_labels(args.review, args.state, args.output)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "examples",
                    "characters",
                    "spans",
                    "candidate_decisions",
                    "focused_review_decisions",
                    "consensus_to_final_geometry",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
