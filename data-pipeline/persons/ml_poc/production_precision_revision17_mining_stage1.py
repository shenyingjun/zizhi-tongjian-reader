from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from production_precision_encoder_finetune import (
    ACCUMULATION_STEPS,
    CLASSIFIER_LEARNING_RATE,
    DROPOUT,
    ENCODER_LEARNING_RATE,
    EPOCHS,
    MAX_GRAD_NORM,
    MAX_LENGTH,
    PHYSICAL_BATCH_SIZE,
    SEED,
    SENTINEL_IDS,
    WEIGHT_DECAY,
    _tokenize_record,
    _validate_sentinels,
)
from production_precision_revision14_two_stage import (
    _assemble_inventory,
    _fit_stage1,
    _load_inputs,
    _score_stage1,
    _stage1_training_indices,
    _three_stratum_weights,
)
from production_precision_revision17_candidates import (
    CANDIDATE_STATUS,
    _read_jsonl,
)
from production_precision_revision17_plan import PLAN_STATUS, _read, _sha256
from production_precision_same_jie_attention import _folds
from production_train import _make_read_only


REVISION = 17
MINING_STAGE1_STATUS = "ml_production_precision_revision17_mining_stage1"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def attach_scores(candidates: list[dict], scores: np.ndarray) -> list[dict]:
    if len(candidates) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Revision-17 Stage-1 score coverage differs")
    return [
        {**row, "stage1_probability": float(np.float32(score))}
        for row, score in zip(candidates, scores)
    ]


def fit_and_score(
    inventory_root: Path,
    grouped_root: Path,
    revision9_root: Path,
    plan_root: Path,
    candidate_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Revision-17 mining Stage-1 exists: {output_dir}")

    (
        inventory_manifest,
        grouped_manifest,
        fit_example_rows,
        inv_rows,
        encoder_dir,
    ) = _load_inputs(inventory_root, grouped_root, revision9_root)
    fit_examples = {str(row["id"]): row for row in fit_example_rows}
    fold_by_juan = _folds([int(row["juan"]) for row in fit_example_rows])
    inventory = _assemble_inventory(
        fit_examples,
        inv_rows["references"],
        inv_rows["existence"],
        inv_rows["rank_pairs"],
        inv_rows["easy_negatives"],
        inv_rows["mined_boundaries"],
        inv_rows["candidate_lattice"],
        fold_by_juan,
    )

    plan_manifest_path = plan_root / "manifest.json"
    mining_path = plan_root / "mining.jsonl"
    plan = _read(plan_manifest_path)
    candidate_manifest_path = candidate_root / "manifest.json"
    candidates_path = candidate_root / "candidates.jsonl"
    candidate_manifest = _read(candidate_manifest_path)
    if (
        plan.get("status") != PLAN_STATUS
        or plan.get("confirmation_read") is not False
        or plan.get("formal_reserve_text_read") is not False
        or plan.get("outputs", {}).get("mining_sha256") != _sha256(mining_path)
        or candidate_manifest.get("status") != CANDIDATE_STATUS
        or candidate_manifest.get("confirmation_read") is not False
        or candidate_manifest.get("formal_reserve_text_read") is not False
        or candidate_manifest.get("bindings", {}).get("plan_manifest_sha256")
        != _sha256(plan_manifest_path)
        or candidate_manifest.get("bindings", {}).get("mining_sha256")
        != _sha256(mining_path)
        or candidate_manifest.get("outputs", {}).get("candidates_sha256")
        != _sha256(candidates_path)
    ):
        raise ValueError("Revision-17 mining Stage-1 source binding differs")

    mining_example_rows = _read_jsonl(mining_path)
    mining_examples = {str(row["id"]): row for row in mining_example_rows}
    candidates = _read_jsonl(candidates_path)
    if (
        len(mining_examples) != len(mining_example_rows)
        or len(mining_examples) != 1098
        or len(candidates)
        != int(candidate_manifest.get("counts", {}).get("candidates", -1))
    ):
        raise ValueError("Revision-17 mining Stage-1 inventory differs")
    if set(fit_examples) & set(mining_examples):
        raise ValueError("Revision-17 fit and mining examples overlap")
    if any(str(row["id"]) not in mining_examples for row in candidates):
        raise ValueError("Revision-17 candidate example is absent")

    git_commit = _git_commit_clean()
    import safetensors
    import torch
    import transformers
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Revision-17 mining Stage-1 requires CUDA")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir, use_fast=True)
    all_examples = {**fit_examples, **mining_examples}
    _validate_sentinels(tokenizer, all_examples)

    training_rows = inventory["rows"]
    training_records = [
        {
            **row,
            **_tokenize_record(tokenizer, fit_examples[str(row["id"])], row),
        }
        for row in training_rows
    ]
    candidate_records = [
        {
            **row,
            **_tokenize_record(tokenizer, mining_examples[str(row["id"])], row),
        }
        for row in candidates
    ]
    records = [*training_records, *candidate_records]

    labels = np.concatenate((
        inventory["binary_labels"],
        np.zeros(len(candidate_records), dtype=np.float32),
    ))
    classes = [
        *inventory["classes"],
        *(["unlabeled_mining_candidate"] * len(candidate_records)),
    ]
    all_training_indices = np.arange(len(training_rows), dtype=np.int64)
    train_indices = _stage1_training_indices(
        all_training_indices,
        classes,
        inventory["real_labels"],
        real_only=True,
        structural_negatives=True,
    )
    weights, weight_inventory = _three_stratum_weights(
        labels, classes, train_indices
    )
    model = _fit_stage1(
        records,
        labels,
        train_indices,
        encoder_dir,
        device,
        row_weights=weights,
    )
    score_indices = np.arange(
        len(training_rows), len(records), dtype=np.int64
    )
    scores = _score_stage1(model, records, score_indices, device)
    scored = attach_scores(candidates, scores)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        scores_path = staging / "scored-candidates.jsonl"
        _write_jsonl(scores_path, scored)
        tokens_path = staging / "candidate-input-tokens.jsonl"
        _write_jsonl(
            tokens_path,
            [
                {
                    "id": str(row["id"]),
                    "juan": int(row["juan"]),
                    "jie_index": int(row["jie_index"]),
                    "para_id": int(row["para_id"]),
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "input_ids": row["input_ids"],
                    "attention_mask": row["attention_mask"],
                    "segment_a_indices": row["segment_a_indices"],
                    "occurrence_indices": row["occurrence_indices"],
                    "slice_start": int(row["slice_start"]),
                    "slice_end": int(row["slice_end"]),
                }
                for row in candidate_records
            ],
        )
        encoder_output = staging / "encoder"
        head_path = staging / "head.safetensors"
        model.encoder.save_pretrained(encoder_output, safe_serialization=True)
        save_file(
            {
                key: value.detach().cpu()
                for key, value in model.head.state_dict().items()
            },
            head_path,
        )
        del model
        torch.cuda.empty_cache()

        manifest = {
            "schema_version": 1,
            "status": MINING_STAGE1_STATUS,
            "revision": REVISION,
            "fit_only": True,
            "mining_only": True,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "formal_reserve_text_read": False,
            "new_judgments_read": False,
            "git_commit": git_commit,
            "bindings": {
                "inventory_manifest_sha256": _sha256(
                    inventory_root / "manifest.json"
                ),
                "grouped_manifest_sha256": _sha256(
                    grouped_root / "manifest.json"
                ),
                "plan_manifest_sha256": _sha256(plan_manifest_path),
                "mining_sha256": _sha256(mining_path),
                "candidate_manifest_sha256": _sha256(
                    candidate_manifest_path
                ),
                "candidates_sha256": _sha256(candidates_path),
                "revision9_base_encoder": _model_artifact(encoder_dir),
            },
            "control": {
                "seed": SEED,
                "max_length": MAX_LENGTH,
                "epochs": EPOCHS,
                "physical_batch_size": PHYSICAL_BATCH_SIZE,
                "gradient_accumulation_steps": ACCUMULATION_STEPS,
                "encoder_learning_rate": ENCODER_LEARNING_RATE,
                "classifier_learning_rate": CLASSIFIER_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "dropout": DROPOUT,
                "max_grad_norm": MAX_GRAD_NORM,
                "stage1_loss": (
                    "BCEWithLogitsLoss_full_fit_three_stratum_balanced"
                ),
                "sentinels": SENTINEL_IDS,
            },
            "counts": {
                "training_rows": len(train_indices),
                "scored_candidates": len(scored),
                "training_weights": weight_inventory,
            },
            "model": {
                "encoder": _model_artifact(encoder_output),
                "head_sha256": _sha256(head_path),
            },
            "environment": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "transformers": transformers.__version__,
                "safetensors": safetensors.__version__,
                "device": torch.cuda.get_device_name(device),
            },
            "outputs": {
                "scores_sha256": _sha256(scores_path),
                "candidate_input_tokens_sha256": _sha256(tokens_path),
            },
            "claim_limit": (
                "Adaptive fit-only mining score. It is not an OOF metric, a "
                "deployment model, or fresh generalization evidence."
            ),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit Revision-16 Stage 1 and score Revision-17 candidates."
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--grouped-data", type=Path, required=True)
    parser.add_argument("--revision-9", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = fit_and_score(
        args.inventory,
        args.grouped_data,
        args.revision_9,
        args.plan,
        args.candidates,
        args.output,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
