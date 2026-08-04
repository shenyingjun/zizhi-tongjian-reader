from __future__ import annotations

import argparse
import hashlib
import json
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


EXPECTED_REVIEW_MANIFEST_SHA256 = (
    "5fb2f5e1549d8f0dd639cf0533d109c4df73a1fb2982c1531d0e787dcb0268ba"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "a23614bc61b19093b0db5865825b57329363c06c1ccea75bed27e35eb17660de"
)
EXPECTED_REVIEW_COUNTS = {
    "auto_accepted_consensus": 184,
    "candidate_union": 187,
    "initial_annotations": 184,
    "review_candidates": 3,
    "teacher_disagreement_review": 3,
}
EXPECTED_LOCKED_STATE_ROOT_SHA256 = (
    "87166b56ff087e9cc374940ee82bcf2a4cb0c4155f552ab2655cf989910d6f9a"
)


def finalize_independent_dev(
    review_dir: Path,
    state_dir: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"independent dev reference exists: {output_dir}")
    tasks_dir = review_dir / "tasks"
    assisted_dir = review_dir / "assisted"
    _, manifest, manifest_sha256 = _snapshot(tasks_dir / "manifest.json")
    selections = manifest.get("selected", [])
    juans = [int(row["juan"]) for row in selections]
    if (
        manifest_sha256 != EXPECTED_REVIEW_MANIFEST_SHA256
        or manifest.get("status")
        != "round10_independent_dev_focused_review"
        or manifest.get("development_only") is not True
        or manifest.get("formal_evaluation") is not False
        or manifest.get("eligible_for_training") is not False
        or manifest.get("eligible_for_promotion") is not False
        or manifest.get("candidate_model_blind") is not True
        or manifest.get("model_predictions_used") is not False
        or manifest.get("model_only_candidates") != 0
        or manifest.get("source_manifest_sha256")
        != EXPECTED_SOURCE_MANIFEST_SHA256
        or manifest.get("counts") != EXPECTED_REVIEW_COUNTS
        or len(juans) != 20
        or len(set(juans)) != 20
        or any(row.get("mode") != "active_assisted" for row in selections)
    ):
        raise ValueError("invalid Round 10 independent dev review manifest")

    task_names = {str(row["task"]) for row in selections}
    pack_names = {f"assisted_juan_{juan:03d}.json" for juan in juans}
    state_names = {f"juan_{juan:03d}.json" for juan in juans}
    if (
        _inventory(tasks_dir) != task_names | {"manifest.json"}
        or _inventory(assisted_dir) != pack_names
        or _inventory(state_dir) != state_names
    ):
        raise ValueError("Round 10 independent dev inventory differs")
    state_snapshots = {
        name: _snapshot(state_dir / name) for name in sorted(state_names)
    }
    state_hashes = {
        name: snapshot[2] for name, snapshot in state_snapshots.items()
    }
    state_root_sha256 = hashlib.sha256(json.dumps(
        state_hashes, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if state_root_sha256 != EXPECTED_LOCKED_STATE_ROOT_SHA256:
        raise ValueError("Round 10 locked state root hash differs")

    examples = []
    frozen_inputs = {}
    decisions_total = Counter()
    focused_decisions = Counter()
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
            _, state, state_sha256 = state_snapshots[state_path.name]
            if (
                task_sha256 != selection.get("task_sha256")
                or pack_sha256 != selection.get("pack_sha256")
                or task.get("juan") != juan
                or pack.get("juan") != juan
                or pack.get("development_only") is not True
                or pack.get("candidate_model_blind") is not True
                or pack.get("human_review_scope")
                != "teacher_disagreement_and_explicit_low_only"
                or any(
                    "round3_ml_omission_check" in row.get("channels", [])
                    for row in pack.get("candidates", [])
                )
            ):
                raise ValueError(f"Round 10 input binding differs: {juan}")
            assisted = state.get("assisted", {})
            if (
                assisted.get("complete") is not True
                or assisted.get("pack_sha256") != pack_sha256
            ):
                raise ValueError(f"Round 10 review is not locked: {juan}")
            decisions_total.update(
                _validate_decisions(juan, task, pack, assisted)
            )
            initial_decisions = pack.get("initial_decisions", {})
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
                    "candidate_blind_copilot_double_pass_focused_review_round10"
                ),
            )
            examples.extend(final_examples)
            delta_by_juan[str(juan)] = geometry_delta(
                _spans(pack["initial_annotations"]),
                _spans(assisted["annotations"]),
            )
            frozen_inputs[str(juan)] = {
                "task_sha256": task_sha256,
                "pack_sha256": pack_sha256,
                "state_sha256": state_sha256,
            }

        identities = {
            (int(row["juan"]), int(row["jie_index"])) for row in examples
        }
        if (
            len(examples) != 20
            or len(identities) != 20
            or sum(decisions_total.values()) != 187
            or sum(focused_decisions.values()) != 3
        ):
            raise ValueError("Round 10 decision/reference inventory differs")
        examples.sort(key=lambda row: (row["juan"], row["jie_index"]))
        reference_path = staging / "development_reference.jsonl"
        _write_jsonl(reference_path, examples)
        report = {
            "schema_version": 1,
            "status": "frozen_round10_independent_development_reference",
            "development_only": True,
            "formal_evaluation": False,
            "eligible_for_training": False,
            "eligible_for_promotion": False,
            "candidate_model_blind": True,
            "model_predictions_used_in_labeling": False,
            "label_provenance": (
                "candidate_blind_copilot_double_pass_focused_review_round10"
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
            "frozen_inputs": frozen_inputs,
            "review_manifest_sha256": manifest_sha256,
            "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
            "locked_state_root_sha256": state_root_sha256,
            "outputs": {
                "development_reference_sha256": hashlib.sha256(
                    reference_path.read_bytes()
                ).hexdigest()
            },
        }
        (staging / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in staging.iterdir():
            path.chmod(0o444)
        staging.chmod(0o555)
        staging.replace(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the candidate-model-blind Round 10 dev reference."
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = finalize_independent_dev(args.review, args.state, args.output)
    print(json.dumps({
        key: report[key] for key in (
            "examples",
            "characters",
            "spans",
            "candidate_decisions",
            "focused_review_decisions",
            "consensus_to_final_geometry",
        )
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
