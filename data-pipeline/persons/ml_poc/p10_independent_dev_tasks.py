from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tempfile
from collections import defaultdict
from pathlib import Path

from p3_compact import _git_commit_clean
from p6_locked_assisted_tasks import _manifest_juans
from p7_training_tasks import (
    POLICY_PATHS,
    _eligible_jies,
    select_unique_juan_rows,
)
from pilot import TEXT, _load


SAMPLED_JIES = 20
SELECTION_SEED = 20260803
MIN_PROMOTION_RESERVE = 35
EXPECTED_INPUTS = {
    "historical_exclusion_manifest": (
        "b020541f12207244d110f38133e0c2dd1a43a554c18c952a1644a6cef45ec401"
    ),
    "round7_dataset_manifest": (
        "09c2724e346b8df5b0ace423016155167790bc7324bf8876f1bfa5942e958eb5"
    ),
    "boundary_guide": (
        "f7c1dff1e97c41fc37846374867840ebde81b9b4fa601e3adbde2db9f1de8247"
    ),
    "spec": "ae0c837353adcdd88d0977acfea2836956ed377accc556ff02e21948923ea8f6",
    "spec_zh": "33f13ddefbb4fd46e5842abd1f476be291abf39140f286181287d28d2240fc66",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_independent_dev_tasks(
    output_dir: Path,
    historical_exclusion_manifest: Path,
    round7_dataset_manifest: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"independent dev task output exists: {output_dir}")
    input_paths = {
        "historical_exclusion_manifest": historical_exclusion_manifest,
        "round7_dataset_manifest": round7_dataset_manifest,
        **POLICY_PATHS,
    }
    input_hashes = {name: _sha256(path) for name, path in input_paths.items()}
    if input_hashes != EXPECTED_INPUTS:
        raise ValueError("independent dev task input hashes differ")
    historical = _load(historical_exclusion_manifest)
    dataset = _load(round7_dataset_manifest)
    if (
        historical.get("status")
        != "candidate_blind_copilot_double_pass_tasks_before_labeling"
        or dataset.get("status") != "round7_controlled_training_dataset"
        or dataset.get("formal_evaluation") is not False
    ):
        raise ValueError("independent dev exclusion manifest status differs")

    excluded_juans = set(int(row) for row in historical["excluded_juans"])
    excluded_juans.update(_manifest_juans(historical))
    for split in dataset.get("splits", {}).values():
        excluded_juans.update(int(juan) for juan in split.get("juans", []))
    sources = {}
    source_paths = {}
    for juan in range(1, 295):
        path = TEXT / f"juan_{juan:03d}.json"
        if path.is_file() and juan not in excluded_juans:
            sources[juan] = _load(path)
            source_paths[juan] = path
    frame = _eligible_jies(sources, excluded_juans)
    selected = select_unique_juan_rows(
        frame, seed=SELECTION_SEED, count=SAMPLED_JIES
    )
    eligible_juans = len({int(row["juan"]) for row in frame})
    remaining = eligible_juans - SAMPLED_JIES
    if remaining < MIN_PROMOTION_RESERVE:
        raise ValueError(
            f"independent dev sample leaves only {remaining} promotion juans"
        )
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in selected:
        grouped[int(row["juan"])].append(row)

    manifest = {
        "schema_version": 1,
        "status": "round10_independent_dev_tasks_before_labeling",
        "development_only": True,
        "formal_evaluation": False,
        "eligible_for_training": False,
        "eligible_for_promotion": False,
        "selection_seed": SELECTION_SEED,
        "selection_policy": (
            "uniformly sample 20 wholly unused eligible juans without replacement, "
            "then uniformly sample one eligible numbered jie within each juan"
        ),
        "promotion_reserve_policy": (
            f"leave at least {MIN_PROMOTION_RESERVE} wholly unused eligible juans "
            "for a future independently frozen promotion evaluation"
        ),
        "candidate_model_blind": True,
        "labeling_protocol": {
            "pass_a": "independent_recall_first",
            "pass_b": "independent_exact_boundary_first",
            "pass_visibility": "mutually_hidden_candidate_free_raw_text_only",
            "focused_review": "A_B_disagreement_and_explicit_low_only",
            "model_omission_candidates": False,
        },
        "sampling_frame": {
            "eligible_jies": len(frame),
            "eligible_juans": eligible_juans,
            "sampled_juans": SAMPLED_JIES,
            "remaining_eligible_juans": remaining,
        },
        "excluded_juans": sorted(excluded_juans),
        "inputs": input_hashes,
        "v1_used_for_selection": False,
        "rules_used_for_selection": False,
        "model_predictions_used_for_selection": False,
        "locked_p3_outputs_used_for_selection": False,
        "model_predictions_generated": False,
        "git_commit": _git_commit_clean(),
        "selected": [],
        "selected_jies": [
            {
                key: value
                for key, value in row.items()
                if key not in {"text", "segments"}
            }
            for row in selected
        ],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        tasks_dir = staging / "tasks"
        tasks_dir.mkdir()
        for juan, rows in sorted(grouped.items()):
            task = {
                "schema_version": 1,
                "phase": "copilot_double_pass",
                "development_only": True,
                "candidate_model_blind": True,
                "juan": juan,
                "instructions": (
                    "Independently mark every main-text person span in the sampled "
                    "jie. Apply BOUNDARY_GUIDE.md using target-jie evidence only. "
                    "Do not read another pass, any model/rule/v1 prediction, "
                    "translations, notes, identity data, historical evaluation, "
                    "or text outside this task."
                ),
                "jies": [
                    {
                        "jie_index": row["jie_index"],
                        "jie_number": row["jie_number"],
                        "text": row["text"],
                        "segments": row["segments"],
                        "annotations": [],
                    }
                    for row in rows
                ],
            }
            task_path = tasks_dir / f"blind_juan_{juan:03d}.json"
            task_path.write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest["selected"].append({
                "juan": juan,
                "task": str(Path("tasks") / task_path.name),
                "task_sha256": _sha256(task_path),
                "source_sha256": _sha256(source_paths[juan]),
                "sampled_jies": len(rows),
            })
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in tasks_dir.iterdir():
            path.chmod(stat.S_IREAD)
        manifest_path.chmod(stat.S_IREAD)
        tasks_dir.chmod(stat.S_IREAD)
        staging.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze candidate-blind independent development tasks."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--historical-exclusion-manifest", type=Path, required=True
    )
    parser.add_argument("--round7-dataset-manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_independent_dev_tasks(
        args.output,
        args.historical_exclusion_manifest,
        args.round7_dataset_manifest,
    )
    print(json.dumps({
        "sampled_jies": len(manifest["selected_jies"]),
        "characters": sum(
            row["characters"] for row in manifest["selected_jies"]
        ),
        "eligible_juans": manifest["sampling_frame"]["eligible_juans"],
        "remaining_eligible_juans": manifest["sampling_frame"][
            "remaining_eligible_juans"
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
