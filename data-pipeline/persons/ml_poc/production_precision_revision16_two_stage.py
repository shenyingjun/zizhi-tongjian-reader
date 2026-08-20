from __future__ import annotations

import argparse
import json
from pathlib import Path

from production_precision_revision14_two_stage import run_revision14_two_stage


REVISION = 16
FINETUNE_STATUS_BLOCKED = "ml_production_precision_revision16_two_stage_blocked"
FINETUNE_STATUS_SELECTED = "ml_production_precision_revision16_two_stage_selected"


def run_revision16_two_stage(
    inventory_root: Path,
    grouped_root: Path,
    revision9_root: Path,
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
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Revision-16 hard-negative two-stage OOF experiment."
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--grouped-data", type=Path, required=True)
    parser.add_argument("--revision-9", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_revision16_two_stage(
        args.inventory,
        args.grouped_data,
        args.revision_9,
        args.output,
    )
    print(json.dumps({
        "status": manifest["status"],
        "selected": manifest["selected"],
        "table": manifest["table"],
    }, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == FINETUNE_STATUS_SELECTED else 1


if __name__ == "__main__":
    raise SystemExit(main())
