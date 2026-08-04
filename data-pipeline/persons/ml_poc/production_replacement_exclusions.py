from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def freeze_replacement_exclusions(
    base_path: Path,
    round_dir: Path,
    dataset_dir: Path,
    selection_dir: Path,
    output_path: Path,
) -> dict:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"replacement exclusions exist: {output_path}")
    base = _load(base_path)
    round_manifest_path = round_dir / "manifest.json"
    private_path = round_dir / "private" / "selection.json"
    dataset_manifest_path = dataset_dir / "manifest.json"
    selection_report_path = selection_dir / "report.json"
    round_manifest = _load(round_manifest_path)
    private = _load(private_path)
    dataset = _load(dataset_manifest_path)
    selection = _load(selection_report_path)
    if (
        base.get("status") != "ml_production_exact_jie_exclusions"
        or base.get("complete") is not True
        or round_manifest.get("status")
        != "ml_production_round_tasks_before_labeling"
        or round_manifest.get("exclusion_manifest_sha256") != _sha256(base_path)
        or round_manifest.get("private_selection_sha256") != _sha256(private_path)
        or private.get("status") != "ml_production_private_task_roles"
        or len(private.get("selected_jies", [])) != 180
        or dataset.get("status") != "ml_production_round1_frozen_dataset"
        or dataset.get("inputs", {}).get("round_manifest_sha256")
        != _sha256(round_manifest_path)
        or dataset.get("inputs", {}).get("private_roles_sha256")
        != _sha256(private_path)
        or selection.get("status")
        != "ml_production_round1_development_selection"
        or selection.get("decision") != "start_new_training_data_round"
        or selection.get("development_comparison_consumed") is not True
        or selection.get("fresh_formal_evaluation_created") is not False
        or selection.get("inputs", {}).get("dataset_manifest_sha256")
        != _sha256(dataset_manifest_path)
        or selection.get("inputs", {}).get("private_roles_sha256")
        != _sha256(private_path)
    ):
        raise ValueError("replacement-round provenance differs")
    existing = {
        (int(row["juan"]), int(row["jie_index"]))
        for row in base.get("consumed", [])
    }
    selected = {
        (int(row["juan"]), int(row["jie_index"]))
        for row in private["selected_jies"]
    }
    if len(selected) != 180 or existing & selected:
        raise ValueError("round-1 consumed geometry differs")
    result = copy.deepcopy(base)
    result["program_round"] = 2
    result["replacement_round_authorized"] = True
    result["git_commit"] = _git_commit_clean()
    result["inputs"].extend([
        {
            "path": str(round_manifest_path),
            "sha256": _sha256(round_manifest_path),
            "statuses": [round_manifest["status"]],
        },
        {
            "path": str(private_path),
            "sha256": _sha256(private_path),
            "statuses": [private["status"]],
        },
        {
            "path": str(dataset_manifest_path),
            "sha256": _sha256(dataset_manifest_path),
            "statuses": [dataset["status"]],
        },
        {
            "path": str(selection_report_path),
            "sha256": _sha256(selection_report_path),
            "statuses": [selection["status"], selection["decision"]],
        },
    ])
    result["consumed"] = sorted(
        [
            *result["consumed"],
            *[
                {
                    "juan": juan,
                    "jie_index": jie_index,
                    "reason": "ml_production_round1_train_or_development",
                }
                for juan, jie_index in selected
            ],
        ],
        key=lambda row: (int(row["juan"]), int(row["jie_index"])),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.name}-", dir=output_path.parent
    ) as temporary:
        staging = Path(temporary) / output_path.name
        staging.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.chmod(0o444)
        staging.replace(output_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exclude Round 1 before the reserved replacement round."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--round", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze_replacement_exclusions(
        args.base, args.round, args.dataset, args.selection, args.output
    )
    print(json.dumps({
        "consumed_jies": len(result["consumed"]),
        "replacement_round_authorized": result["replacement_round_authorized"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
