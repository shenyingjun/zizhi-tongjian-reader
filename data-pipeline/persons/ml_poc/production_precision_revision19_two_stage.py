from __future__ import annotations

import argparse
import json
from pathlib import Path

from production_precision_revision14_two_stage import run_revision14_two_stage
from production_precision_revision19_overlay import READY_STATUS


REVISION = 19
FINETUNE_STATUS_BLOCKED = "ml_production_precision_revision19_two_stage_blocked"
FINETUNE_STATUS_SELECTED = "ml_production_precision_revision19_two_stage_selected"


def run_revision19_two_stage(
    inventory_root: Path,
    grouped_root: Path,
    revision9_root: Path,
    augmentation_root: Path,
    output_dir: Path,
) -> dict:
    return run_revision14_two_stage(
        inventory_root,
        grouped_root,
        revision9_root,
        output_dir,
        experiment_revision=REVISION,
        status_blocked=FINETUNE_STATUS_BLOCKED,
        status_selected=FINETUNE_STATUS_SELECTED,
        stage1_real_only=True,
        stage1_structural_negatives=True,
        stage1_three_stratum_balanced=True,
        greedy_resolution=True,
        augmentation_root=augmentation_root,
        augmentation_status=READY_STATUS,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Revision-19 reviewed-augmentation two-stage OOF."
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--grouped-data", type=Path, required=True)
    parser.add_argument("--revision-9", type=Path, required=True)
    parser.add_argument("--augmentation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_revision19_two_stage(
        args.inventory,
        args.grouped_data,
        args.revision_9,
        args.augmentation,
        args.output,
    )
    print(json.dumps({
        "status": manifest["status"],
        "augmentation": manifest["augmentation"],
        "selected": manifest["selected"],
        "table": manifest["table"],
    }, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == FINETUNE_STATUS_SELECTED else 1


if __name__ == "__main__":
    raise SystemExit(main())
