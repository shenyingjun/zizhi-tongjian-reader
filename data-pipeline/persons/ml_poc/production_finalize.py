from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tempfile
from collections import Counter
from pathlib import Path

from p1_dataset import build_examples
from p3_compact import _git_commit_clean
from production_review import _read, _sha256
from production_review_server import (
    EXPECTED_TASKS,
    REDUCED_STATUS,
    ProductionReviewStore,
    _artifact_file,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _summary(rows: list[dict]) -> dict:
    return {
        "examples": len(rows),
        "characters": sum(len(row["text"]) for row in rows),
        "spans": sum(int(row["span_count"]) for row in rows),
        "juans": len({int(row["juan"]) for row in rows}),
    }


def freeze_dataset(
    review_root: Path,
    state_root: Path,
    round_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"production dataset output exists: {output_dir}")
    review_manifest_path = review_root / "manifest.json"
    review_manifest = _read(review_manifest_path)
    round_manifest_path = round_root / "manifest.json"
    round_manifest = _read(round_manifest_path)
    roles_path = round_root / "private" / "selection.json"
    roles = _read(roles_path)
    if (
        review_manifest.get("schema_version") != 1
        or review_manifest.get("status") != REDUCED_STATUS
        or float(review_manifest.get("consensus_audit_rate", -1)) != 0
        or review_manifest.get("source_manifest_sha256")
        != _sha256(round_manifest_path)
        or len(review_manifest.get("selected", [])) != EXPECTED_TASKS
        or round_manifest.get("schema_version") != 1
        or round_manifest.get("status")
        != "ml_production_round_tasks_before_labeling"
        or len(round_manifest.get("tasks", [])) != EXPECTED_TASKS
        or round_manifest.get("private_selection_sha256")
        != _sha256(roles_path)
        or roles.get("schema_version") != 1
        or roles.get("status") != "ml_production_private_task_roles"
        or len(roles.get("selected_jies", [])) != EXPECTED_TASKS
    ):
        raise ValueError("production dataset source binding differs")
    round_number = 2 if round_manifest.get("replacement_round") is True else 1
    review_rows = {
        str(row["task_id"]): row
        for row in review_manifest["selected"]
    }
    round_rows = {
        str(row["task_id"]): row for row in round_manifest["tasks"]
    }
    role_rows = {
        str(row["task_id"]): row for row in roles["selected_jies"]
    }
    if (
        len(review_rows) != EXPECTED_TASKS
        or set(review_rows) != set(round_rows)
        or set(review_rows) != set(role_rows)
        or Counter(
            str(row["split"]) for row in role_rows.values()
        ) != {"train": 140, "development": 40}
    ):
        raise ValueError("production dataset task roles differ")
    for task_id, review_row in review_rows.items():
        round_row = round_rows[task_id]
        if (
            review_row["task_sha256"] != round_row["task_sha256"]
            or int(role_rows[task_id]["juan"])
            != int(_read(_artifact_file(
                review_root, review_row["task"], "review task"
            ))["juan"])
        ):
            raise ValueError(f"production task provenance differs: {task_id}")

    store = ProductionReviewStore(review_root, state_root)
    index = store.index()
    index_required_ids = {
        str(row["task_id"])
        for row in index["tasks"]
        if int(row["required"]) > 0
    }
    required_ids = {
        str(row["task_id"])
        for row in review_manifest["selected"]
        if int(row.get("review_candidates", 0)) > 0
    }
    state_names = {path.name for path in state_root.glob("*.json")}
    receipt_root = state_root / "completed"
    receipt_entries = (
        list(receipt_root.iterdir())
        if receipt_root.is_dir()
        else []
    )
    receipt_names = {path.name for path in receipt_entries}
    expected_names = {f"task_{task_id}.json" for task_id in required_ids}
    if (
        not required_ids
        or index_required_ids != required_ids
        or state_names != expected_names
        or receipt_names != expected_names
        or any(
            not path.is_file() or path.is_symlink()
            for path in receipt_entries
        )
    ):
        raise ValueError("production human-state inventory differs")

    split_rows = {"train": [], "development": []}
    frozen_inputs = {}
    final_decisions = Counter()
    human_decisions = Counter()
    for task_id, review_row in review_rows.items():
        payload = store.payload(task_id)
        state = payload["state"]
        if task_id in required_ids and not state["complete"]:
            raise ValueError(f"production review is not locked: {task_id}")
        candidates = {
            str(row["id"]): row for row in payload["review"]["candidates"]
        }
        decisions = state["effective_decisions"]
        if set(decisions) != set(candidates):
            raise ValueError(
                f"production candidate inventory unresolved: {task_id}"
            )
        final_decisions.update(decisions.values())
        human_decisions.update(state["human_decisions"].values())
        task = payload["task"]
        examples = build_examples(
            int(task["juan"]),
            task,
            {
                "role_audit": {
                    "complete": True,
                    "annotations": state["annotations"],
                }
            },
            label_provenance=(
                f"production_round{round_number}_teacher_high_confidence_"
                "and_focused_human"
            ),
        )
        if len(examples) != 1:
            raise ValueError(f"production task is not one jie: {task_id}")
        split = str(role_rows[task_id]["split"])
        split_rows[split].extend(examples)
        state_path = state_root / f"task_{task_id}.json"
        receipt_path = receipt_root / f"task_{task_id}.json"
        frozen_inputs[task_id] = {
            "task_sha256": review_row["task_sha256"],
            "review_sha256": review_row["review_sha256"],
            "state_sha256": (
                _sha256(state_path) if state_path.is_file() else None
            ),
            "receipt_sha256": (
                _sha256(receipt_path) if receipt_path.is_file() else None
            ),
            "split": split,
        }

    for rows in split_rows.values():
        rows.sort(key=lambda row: (
            int(row["juan"]), int(row["jie_index"])
        ))
    identities = {
        split: {
            (int(row["juan"]), int(row["jie_index"])) for row in rows
        }
        for split, rows in split_rows.items()
    }
    if (
        len(split_rows["train"]) != 140
        or len(split_rows["development"]) != 40
        or identities["train"] & identities["development"]
    ):
        raise ValueError("production dataset split geometry differs")

    git_commit = _git_commit_clean()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        train_path = staging / "train.jsonl"
        development_path = staging / "development.jsonl"
        _write_jsonl(train_path, split_rows["train"])
        _write_jsonl(development_path, split_rows["development"])
        manifest = {
            "schema_version": 1,
            "status": f"ml_production_round{round_number}_frozen_dataset",
            "round": round_number,
            "replacement_round": round_number == 2,
            "training_only": False,
            "formal_evaluation": False,
            "eligible_for_training": True,
            "eligible_for_checkpoint_selection": True,
            "eligible_for_promotion": False,
            "git_commit": git_commit,
            "label_provenance": (
                "teacher_high_confidence_plus_focused_human_review"
            ),
            "splits": {
                split: _summary(rows)
                for split, rows in split_rows.items()
            },
            "candidate_decisions": dict(final_decisions),
            "focused_human_decisions": dict(human_decisions),
            "inputs": {
                "review_manifest_sha256": _sha256(review_manifest_path),
                "round_manifest_sha256": _sha256(round_manifest_path),
                "private_roles_sha256": _sha256(roles_path),
                "tasks": frozen_inputs,
            },
            "outputs": {
                "train_sha256": _sha256(train_path),
                "development_sha256": _sha256(development_path),
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in staging.iterdir():
            path.chmod(0o444)
        staging.chmod(0o555)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze production training/development labels."
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--round", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_dataset(
        args.review, args.state, args.round, args.output
    )
    print(json.dumps({
        "splits": manifest["splits"],
        "candidate_decisions": manifest["candidate_decisions"],
        "focused_human_decisions": manifest["focused_human_decisions"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
