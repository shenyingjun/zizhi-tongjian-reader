from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import stat
import tempfile
from pathlib import Path

from production_review import _read, _sha256
from production_review_server import ProductionReviewStore
from production_third_teacher import (
    EXPECTED_TASKS,
    OUTPUT_STATUS,
    _artifact_path,
)


REDUCED_STATUS = "ml_production_focused_review_with_reduced_audit"
REDUCED_AUDIT_RATE = 0.05


def _is_consensus(candidate: dict) -> bool:
    channels = set(candidate.get("channels", []))
    confidence = candidate.get("pass_confidence", {})
    return (
        {"copilot_independent_a", "copilot_independent_b"}
        .issubset(channels)
        and confidence.get("a") in {"high", "medium"}
        and confidence.get("b") in {"high", "medium"}
    )


def _annotation(candidate: dict) -> dict:
    return {
        key: candidate[key]
        for key in ("para_id", "start", "end", "surface")
    }


def _set_annotation(
    annotations: list[dict], candidate: dict, accepted: bool
) -> None:
    geometry = (
        int(candidate["para_id"]),
        int(candidate["start"]),
        int(candidate["end"]),
    )
    annotations[:] = [
        row for row in annotations
        if (
            int(row["para_id"]),
            int(row["start"]),
            int(row["end"]),
        ) != geometry
    ]
    if accepted:
        annotation = _annotation(candidate)
        if any(
            int(annotation["para_id"]) == int(other["para_id"])
            and int(annotation["start"]) < int(other["end"])
            and int(other["start"]) < int(annotation["end"])
            for other in annotations
        ):
            raise ValueError("reduced audit acceptance overlaps annotation")
        annotations.append(annotation)


def reduce_audit(
    review_root: Path,
    state_root: Path,
    output_dir: Path,
    *,
    audit_rate: float = REDUCED_AUDIT_RATE,
    expected_carried_decisions: int,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"reduced-audit output exists: {output_dir}")
    manifest_path = review_root / "manifest.json"
    manifest = _read(manifest_path)
    source_audit_rate = float(manifest.get("consensus_audit_rate", -1))
    if not 0 <= audit_rate < source_audit_rate <= 1:
        raise ValueError("reduced audit rate must be below the source rate")
    private_path = review_root / "private" / "selection.json"
    private = _read(private_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != OUTPUT_STATUS
        or len(manifest.get("selected", [])) != EXPECTED_TASKS
        or manifest.get("private_selection_sha256") != _sha256(private_path)
        or private.get("schema_version") != 1
        or not isinstance(private.get("audit_seed"), int)
    ):
        raise ValueError("reduced-audit source review differs")
    store = ProductionReviewStore(review_root, state_root)
    state_files = list(state_root.glob("*.json"))
    state_by_task = {}
    for path in state_files:
        name = path.stem
        if (
            not name.startswith("task_")
            or len(name) != 25
            or any(
                char not in "0123456789abcdef"
                for char in name.removeprefix("task_")
            )
        ):
            raise ValueError(f"unexpected human state file: {path.name}")
        task_id = name.removeprefix("task_")
        if task_id in state_by_task:
            raise ValueError(f"duplicate human state task: {task_id}")
        state_by_task[task_id] = path
    selected_ids = {
        str(row["task_id"]) for row in manifest["selected"]
    }
    if not set(state_by_task).issubset(selected_ids):
        raise ValueError("human state inventory differs")

    loaded = {}
    consensus = []
    carried_state = {}
    for row in manifest["selected"]:
        task_id = str(row["task_id"])
        review_path = _artifact_path(
            review_root, row["review"], "source review"
        )
        review = _read(review_path)
        payload = store.payload(task_id)
        state = payload["state"]
        if state["expanded_full_union"] or state["complete"]:
            raise ValueError(
                "cannot reduce audit after expansion or task completion"
            )
        human = dict(state["human_decisions"])
        if any(decision != "accept" for decision in human.values()):
            raise ValueError(
                "cannot reduce audit after a rejected human decision"
            )
        state_path = state_by_task.get(task_id)
        if state_path is not None:
            carried_state[task_id] = state_path
        for candidate in review["candidates"]:
            if _is_consensus(candidate):
                consensus.append((
                    task_id,
                    int(candidate["para_id"]),
                    int(candidate["start"]),
                    int(candidate["end"]),
                ))
        loaded[task_id] = (row, review, human)
    actual_carried_decisions = sum(
        len(human) for _, _, human in loaded.values()
    )
    if actual_carried_decisions != expected_carried_decisions:
        raise ValueError(
            "carried human decision count differs: "
            f"expected {expected_carried_decisions}, "
            f"found {actual_carried_decisions}"
        )

    audit_count = math.ceil(len(consensus) * audit_rate)
    rng = random.Random(int(private["audit_seed"]))
    audited = set(rng.sample(sorted(consensus), audit_count))
    output = {
        **manifest,
        "status": REDUCED_STATUS,
        "source_review_manifest": (
            "provenance/source-review-manifest.json"
        ),
        "source_review_manifest_sha256": _sha256(manifest_path),
        "source_consensus_audit_rate": source_audit_rate,
        "consensus_audit_rate": audit_rate,
        "prior_human_state_inventory": {},
        "expected_carried_human_decisions": expected_carried_decisions,
        "counts": dict(manifest["counts"]),
    }
    output["counts"]["consensus_audit_review"] = audit_count
    carried_decisions = 0
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        for name in ("tasks", "review", "private"):
            shutil.copytree(review_root / name, staging / name)
        provenance = staging / "provenance"
        provenance.mkdir()
        shutil.copyfile(
            manifest_path, provenance / "source-review-manifest.json"
        )
        state_snapshot = staging / "prior-human-state"
        state_snapshot.mkdir()
        for task_id, path in carried_state.items():
            target = state_snapshot / path.name
            shutil.copyfile(path, target)
            output["prior_human_state_inventory"][task_id] = {
                "path": str(Path("prior-human-state") / target.name),
                "sha256": _sha256(target),
            }

        selected_by_id = {
            str(row["task_id"]): row for row in output["selected"]
        }
        for task_id, (source_row, review, human) in loaded.items():
            initial = dict(review["initial_decisions"])
            annotations = list(review["initial_annotations"])
            for candidate in review["candidates"]:
                candidate_id = str(candidate["id"])
                geometry = (
                    task_id,
                    int(candidate["para_id"]),
                    int(candidate["start"]),
                    int(candidate["end"]),
                )
                if candidate_id in human:
                    initial[candidate_id] = human[candidate_id]
                    _set_annotation(
                        annotations, candidate, human[candidate_id] == "accept"
                    )
                    candidate["carried_human_decision"] = human[candidate_id]
                    carried_decisions += 1
                if not _is_consensus(candidate):
                    continue
                if geometry in audited and candidate_id not in human:
                    initial.pop(candidate_id, None)
                    _set_annotation(annotations, candidate, False)
                    candidate["confidence"] = "low"
                    candidate["review_reason"] = (
                        f"Predeclared {audit_rate:.0%} audit of exact "
                        "non-low A/B consensus."
                    )
                elif geometry not in audited:
                    initial[candidate_id] = "accept"
                    _set_annotation(annotations, candidate, True)
                    candidate["confidence"] = "high"
                    candidate["review_reason"] = ""
            annotations.sort(key=lambda row: (
                int(row["para_id"]), int(row["start"]), int(row["end"])
            ))
            review["initial_decisions"] = initial
            review["initial_annotations"] = annotations
            review["human_review_scope"] = (
                "third_teacher_non_high_5_percent_consensus_audit_and_"
                "negative_recall_audit"
            )
            target_row = selected_by_id[task_id]
            review_path = staging / Path(str(target_row["review"]))
            review_path.chmod(stat.S_IWRITE)
            review_path.write_text(
                json.dumps(review, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            target_row["review_sha256"] = _sha256(review_path)
            target_row["review_candidates"] = (
                len(review["candidates"]) - len(initial)
            )

        output["counts"]["carried_human_decisions"] = carried_decisions
        output["counts"]["human_review_candidates"] = sum(
            int(row["review_candidates"]) for row in output["selected"]
        )
        new_private = {
            **private,
            "source_consensus_audit_rate": source_audit_rate,
            "consensus_audit_rate": audit_rate,
            "prior_audited_consensus": private.get("audited_consensus", []),
            "audited_consensus": [
                {
                    "task_id": task_id,
                    "para_id": para_id,
                    "start": start,
                    "end": end,
                }
                for task_id, para_id, start, end in sorted(audited)
            ],
        }
        private_target = staging / "private" / "selection.json"
        private_target.chmod(stat.S_IWRITE)
        private_target.write_text(
            json.dumps(new_private, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output["private_selection_sha256"] = _sha256(private_target)
        output_manifest = staging / "manifest.json"
        output_manifest.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in staging.rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reduce training/development consensus audit before freeze."
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-rate", type=float, default=REDUCED_AUDIT_RATE)
    parser.add_argument(
        "--expected-carried-decisions", type=int, required=True
    )
    args = parser.parse_args()
    manifest = reduce_audit(
        args.review,
        args.state,
        args.output,
        audit_rate=args.audit_rate,
        expected_carried_decisions=args.expected_carried_decisions,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
