from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from p3_compact import _git_commit_clean
from production_precision_encoder_finetune import (
    FOLDS,
    SEED,
    _tokenize_record,
    _validate_sentinels,
)
from production_precision_grouped_verifier import _read, _read_jsonl, _sha256
from production_precision_revision14_two_stage import (
    _assemble_inventory,
    _fit_stage1,
    _fold_local_augmentation_indices,
    _load_inputs,
    _prepare_training_augmentation,
    _score_stage1,
    _seed_everything,
    _write_jsonl,
    end_to_end_metrics,
)
from production_precision_revision15_two_stage import (
    FINETUNE_STATUS_BLOCKED as REVISION15_BLOCKED_STATUS,
)
from production_precision_revision19_overlay import READY_STATUS
from production_precision_same_jie_attention import _folds
from production_train import _make_read_only


REVISION = 21
STATUS_BLOCKED = "ml_production_precision_revision21_specialists_blocked"
STATUS_SELECTED = "ml_production_precision_revision21_specialists_selected"


def _specialist_weights(
    labels: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, dict]:
    positive = indices[labels[indices] == 1]
    negative = indices[labels[indices] == 0]
    if not len(positive) or not len(negative):
        raise ValueError("Revision-21 specialist fold class is empty")
    weights = np.ones(len(labels), dtype=np.float32)
    weights[positive] = np.float32(0.5 / len(positive))
    weights[negative] = np.float32(0.5 / len(negative))
    return weights, {
        "positive_rows": len(positive),
        "negative_rows": len(negative),
        "positive_target_mass": 0.5,
        "negative_target_mass": 0.5,
        "positive_weight": 0.5 / len(positive),
        "negative_weight": 0.5 / len(negative),
    }


def _specialist_training_indices(
    base_train: np.ndarray,
    base_positive: np.ndarray,
    base_negative: np.ndarray,
    augmentation_positive: np.ndarray,
    augmentation_negative: np.ndarray,
) -> np.ndarray:
    base_train_set = set(base_train.tolist())
    return np.asarray(
        sorted(
            base_train_set.intersection(base_positive.tolist())
            | base_train_set.intersection(base_negative.tolist())
            | set(augmentation_positive.tolist())
            | set(augmentation_negative.tolist())
        ),
        dtype=np.int64,
    )


def _load_anchor(anchor_root: Path, evaluation_rows: list[dict]) -> dict:
    manifest_path = anchor_root / "manifest.json"
    stage1_path = anchor_root / "oof-stage1-scores.jsonl"
    stage2_path = anchor_root / "oof-stage2-scores.jsonl"
    manifest = _read(manifest_path)
    if (
        manifest.get("status") != REVISION15_BLOCKED_STATUS
        or manifest.get("revision") != 15
        or manifest.get("confirmation_read") is not False
        or manifest.get("outputs", {}).get("oof_stage1_scores_sha256")
        != _sha256(stage1_path)
        or manifest.get("outputs", {}).get("oof_stage2_scores_sha256")
        != _sha256(stage2_path)
    ):
        raise ValueError("Revision-21 anchor manifest differs")
    stage1_rows = _read_jsonl(stage1_path)
    stage2_rows = _read_jsonl(stage2_path)
    if (
        len(stage1_rows) != len(evaluation_rows)
        or len(stage2_rows) != len(evaluation_rows)
    ):
        raise ValueError("Revision-21 anchor row count differs")
    fields = ("id", "juan", "jie_index", "para_id", "start", "end", "class")
    for expected, stage1, stage2 in zip(
        evaluation_rows, stage1_rows, stage2_rows
    ):
        if (
            any(expected[key] != stage1[key] for key in fields)
            or any(expected[key] != stage2[key] for key in fields)
        ):
            raise ValueError("Revision-21 anchor geometry differs")
    return {
        "manifest_path": manifest_path,
        "stage1_path": stage1_path,
        "stage2_path": stage2_path,
        "stage1": np.asarray(
            [row["oof_stage1_probability"] for row in stage1_rows],
            dtype=np.float32,
        ),
        "stage2": np.asarray(
            [row["oof_stage2_score"] for row in stage2_rows],
            dtype=np.float32,
        ),
    }


def run_revision21_specialists(
    inventory_root: Path,
    grouped_root: Path,
    revision9_root: Path,
    augmentation_root: Path,
    anchor_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Revision-21 output exists: {output_dir}")
    (
        inventory_manifest,
        grouped_manifest,
        example_rows,
        inventory_rows,
        encoder_dir,
    ) = _load_inputs(inventory_root, grouped_root, revision9_root)
    base_examples = {str(row["id"]): row for row in example_rows}
    fold_by_juan = _folds([int(row["juan"]) for row in example_rows])
    inventory = _assemble_inventory(
        base_examples,
        inventory_rows["references"],
        inventory_rows["existence"],
        inventory_rows["rank_pairs"],
        inventory_rows["easy_negatives"],
        inventory_rows["mined_boundaries"],
        inventory_rows["candidate_lattice"],
        fold_by_juan,
    )
    evaluation_rows = inventory["rows"]
    evaluation_classes = inventory["classes"]
    fold_ids = inventory["fold_ids"]
    evaluation_real_labels = inventory["real_labels"]
    evaluation_count = len(evaluation_rows)

    augmentation_manifest_path = augmentation_root / "manifest.json"
    augmentation_manifest = _read(augmentation_manifest_path)
    augmentation_paths = {
        "examples": augmentation_root / "examples.jsonl",
        "exact_additions": augmentation_root / "exact-additions.jsonl",
        "semantic_negatives": augmentation_root / "semantic-negatives.jsonl",
        "rank_pairs": augmentation_root / "rank-pairs.jsonl",
    }
    if (
        augmentation_manifest.get("status") != READY_STATUS
        or augmentation_manifest.get("confirmation_read") is not False
        or augmentation_manifest.get("formal_reserve_text_read") is not False
        or augmentation_manifest.get("counts", {}).get("conflicts") != 0
        or any(
            augmentation_manifest.get("outputs", {}).get(f"{key}_sha256")
            != _sha256(path)
            for key, path in augmentation_paths.items()
        )
    ):
        raise ValueError("Revision-21 augmentation binding differs")
    augmentation = _prepare_training_augmentation(
        inventory,
        base_examples,
        _read_jsonl(augmentation_paths["examples"]),
        _read_jsonl(augmentation_paths["exact_additions"]),
        _read_jsonl(augmentation_paths["semantic_negatives"]),
        _read_jsonl(augmentation_paths["rank_pairs"]),
        boundary_stage1_positive=True,
    )
    rows = augmentation["rows"]
    examples = augmentation["examples"]
    anchor = _load_anchor(anchor_root, evaluation_rows)

    specialist_labels = np.full(len(rows), -1, dtype=np.float32)
    base_positive = np.asarray(
        [
            index for index, value in enumerate(evaluation_classes)
            if value in {"exact_reference", "boundary_alternative"}
        ],
        dtype=np.int64,
    )
    base_semantic = np.asarray(
        [
            index for index, value in enumerate(evaluation_classes)
            if value == "semantic_negative"
        ],
        dtype=np.int64,
    )
    base_structural = np.asarray(
        [
            index for index, value in enumerate(evaluation_classes)
            if value in {"easy_negative", "reconciled_nonoverlap"}
        ],
        dtype=np.int64,
    )
    if set(augmentation["semantic_indices"].tolist()).intersection(
        base_structural.tolist()
    ):
        raise ValueError(
            "Revision-21 semantic augmentation collides with structural class"
        )
    all_positive = np.asarray(
        sorted(set(base_positive.tolist()).union(
            augmentation["positive_indices"].tolist()
        )),
        dtype=np.int64,
    )
    specialist_labels[all_positive] = 1
    specialist_labels[base_semantic] = 0
    specialist_labels[base_structural] = 0
    specialist_labels[augmentation["semantic_indices"]] = 0

    git_commit = _git_commit_clean()
    import torch
    from transformers import AutoTokenizer

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Revision-21 specialists require CUDA")
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir, use_fast=True)
    _validate_sentinels(tokenizer, examples)
    records = [
        {
            **row,
            **_tokenize_record(tokenizer, examples[str(row["id"])], row),
        }
        for row in rows
    ]

    semantic_oof = np.zeros(evaluation_count, dtype=np.float32)
    structural_oof = np.zeros(evaluation_count, dtype=np.float32)
    fold_inventory = []
    for fold in range(FOLDS):
        heldout = np.flatnonzero(fold_ids == fold)
        base_train = np.flatnonzero(fold_ids != fold)
        fold_positive = _fold_local_augmentation_indices(
            augmentation["positive_indices"], rows, fold_by_juan, fold
        )
        fold_semantic = _fold_local_augmentation_indices(
            augmentation["semantic_indices"], rows, fold_by_juan, fold
        )
        semantic_train = _specialist_training_indices(
            base_train,
            base_positive,
            base_semantic,
            fold_positive,
            fold_semantic,
        )
        structural_train = _specialist_training_indices(
            base_train,
            base_positive,
            base_structural,
            fold_positive,
            np.asarray([], dtype=np.int64),
        )

        semantic_weights, semantic_counts = _specialist_weights(
            specialist_labels, semantic_train
        )
        semantic_model = _fit_stage1(
            records,
            specialist_labels,
            semantic_train,
            encoder_dir,
            device,
            row_weights=semantic_weights,
        )
        semantic_oof[heldout] = _score_stage1(
            semantic_model, records, heldout, device
        )
        del semantic_model
        torch.cuda.empty_cache()

        structural_weights, structural_counts = _specialist_weights(
            specialist_labels, structural_train
        )
        structural_model = _fit_stage1(
            records,
            specialist_labels,
            structural_train,
            encoder_dir,
            device,
            row_weights=structural_weights,
        )
        structural_oof[heldout] = _score_stage1(
            structural_model, records, heldout, device
        )
        del structural_model
        torch.cuda.empty_cache()
        fold_inventory.append({
            "fold": fold,
            "heldout_rows": len(heldout),
            "semantic": semantic_counts,
            "structural": structural_counts,
            "augmentation_positive_rows": len(fold_positive),
            "augmentation_semantic_rows": len(fold_semantic),
        })

    combined = np.minimum(
        anchor["stage1"],
        np.minimum(semantic_oof, structural_oof),
    )
    table = end_to_end_metrics(
        evaluation_rows,
        combined,
        anchor["stage2"],
        evaluation_classes,
        fold_ids,
        evaluation_real_labels,
        set(inventory["exact_by_geometry"]),
        set(inventory["boundary_by_geometry"]),
        greedy_resolution=True,
    )
    eligible = [row for row in table if row["eligible"]]
    selected = max(
        eligible,
        key=lambda row: (
            row["e2e_exact_recall"],
            row["e2e_real_precision"],
            row["threshold"],
        ),
        default=None,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        semantic_path = staging / "oof-semantic-specialist.jsonl"
        structural_path = staging / "oof-structural-specialist.jsonl"
        combined_path = staging / "oof-combined-stage1.jsonl"
        def score_rows(values: np.ndarray, key: str) -> list[dict]:
            return [
                {
                    "id": row["id"],
                    "juan": row["juan"],
                    "jie_index": row["jie_index"],
                    "para_id": row["para_id"],
                    "start": row["start"],
                    "end": row["end"],
                    "class": row["class"],
                    "fold": int(fold_ids[index]),
                    key: float(values[index]),
                }
                for index, row in enumerate(evaluation_rows)
            ]
        _write_jsonl(
            semantic_path,
            score_rows(semantic_oof, "semantic_overlap_probability"),
        )
        _write_jsonl(
            structural_path,
            score_rows(structural_oof, "structural_overlap_probability"),
        )
        _write_jsonl(
            combined_path,
            score_rows(combined, "combined_admission_probability"),
        )
        manifest = {
            "schema_version": 1,
            "status": STATUS_SELECTED if selected else STATUS_BLOCKED,
            "revision": REVISION,
            "formal_grade": False,
            "eligible_for_production": False,
            "fit_only": True,
            "confirmation_read": False,
            "formal_reserve_text_read": False,
            "git_commit": git_commit,
            "bindings": {
                "inventory_manifest_sha256": _sha256(
                    inventory_root / "manifest.json"
                ),
                "grouped_manifest_sha256": _sha256(
                    grouped_root / "manifest.json"
                ),
                "augmentation_manifest_sha256": _sha256(
                    augmentation_manifest_path
                ),
                "augmentation_outputs": {
                    key: _sha256(path)
                    for key, path in augmentation_paths.items()
                },
                "anchor_manifest_sha256": _sha256(
                    anchor["manifest_path"]
                ),
                "anchor_stage1_sha256": _sha256(anchor["stage1_path"]),
                "anchor_stage2_sha256": _sha256(anchor["stage2_path"]),
            },
            "control": {
                "seed": SEED,
                "folds": FOLDS,
                "specialists_share_fitted_state": False,
                "specialist_class_masses": {
                    "overlap_positive": 0.5,
                    "veto_negative": 0.5,
                },
                "combination": (
                    "min_revision15_semantic_overlap_structural_overlap"
                ),
                "stage2": "frozen_revision15_oof",
                "final_full_fit": False,
            },
            "inventory": {
                "evaluation_rows": evaluation_count,
                "evaluation_rows_added": 0,
                "augmentation_positive_rows": len(
                    augmentation["positive_indices"]
                ),
                "augmentation_semantic_rows": len(
                    augmentation["semantic_indices"]
                ),
            },
            "folds": fold_inventory,
            "table": table,
            "selected": selected,
            "outputs": {
                "semantic_specialist_sha256": _sha256(semantic_path),
                "structural_specialist_sha256": _sha256(structural_path),
                "combined_stage1_sha256": _sha256(combined_path),
            },
            "claim_limit": (
                "Adaptive fit-only OOF specialist architecture diagnostic; "
                "no final fit, confirmation, or formal reserve read."
            ),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Revision-21 independent specialist OOF heads."
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--grouped-data", type=Path, required=True)
    parser.add_argument("--revision-9", type=Path, required=True)
    parser.add_argument("--augmentation", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_revision21_specialists(
        args.inventory,
        args.grouped_data,
        args.revision_9,
        args.augmentation,
        args.anchor,
        args.output,
    )
    print(json.dumps({
        "status": manifest["status"],
        "selected": manifest["selected"],
        "table": manifest["table"],
    }, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == STATUS_SELECTED else 1


if __name__ == "__main__":
    raise SystemExit(main())
