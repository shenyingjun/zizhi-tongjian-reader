from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from production_precision_grouped_data import GROUPED_DATA_STATUS
from production_precision_grouped_verifier import (
    _read,
    _read_jsonl,
    _sha256,
)
from production_precision_negative_audit_freeze import (
    EXPECTED_SAFE,
    FROZEN_STATUS,
)
from production_precision_same_jie_attention import _folds
from production_precision_select import _metric
from production_span_verifier import THRESHOLDS, _assembled_bounds
from production_train import _make_read_only


FINETUNE_STATUS_BLOCKED = "ml_production_encoder_finetune_blocked"
FINETUNE_STATUS_SELECTED = "ml_production_encoder_finetune_selected"
REVISION = 11
SEED = 20260818
FOLDS = 7
MAX_LENGTH = 384
EPOCHS = 3
PHYSICAL_BATCH_SIZE = 1
ACCUMULATION_STEPS = 32
ENCODER_LEARNING_RATE = 1e-5
CLASSIFIER_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
DROPOUT = 0.10
MAX_GRAD_NORM = 1.0
LEFT_SENTINEL = "㈠"
RIGHT_SENTINEL = "㈡"
PARAGRAPH_SENTINEL = "㈢"
SENTINEL_IDS = {
    LEFT_SENTINEL: 8680,
    RIGHT_SENTINEL: 11019,
    PARAGRAPH_SENTINEL: 12821,
}
LABEL_NAMES = ("exact_person", "boundary_alternative", "not_person")
LABEL_BY_NAME = {name: index for index, name in enumerate(LABEL_NAMES)}
STRATUM_MASSES = {
    "exact_person": 0.45,
    "boundary_alternative": 0.15,
    "real_not_person": 0.20,
    "mined_not_person": 0.20,
}
EXPECTED_EXACT = 2393
EXPECTED_BOUNDARY = 132
EXPECTED_REAL_NEGATIVE = 171


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _classify_real_row(row: dict) -> tuple[str, int]:
    if not int(row["label"]):
        return "real_not_person", LABEL_BY_NAME["not_person"]
    exact = any(
        int(reference["para_id"]) == int(row["para_id"])
        and int(reference["start"]) == int(row["start"])
        and int(reference["end"]) == int(row["end"])
        for reference in row["overlapping_references"]
    )
    if exact:
        return "exact_person", LABEL_BY_NAME["exact_person"]
    return "boundary_alternative", LABEL_BY_NAME["boundary_alternative"]


def _stratum_weights(strata: list[str]) -> tuple[np.ndarray, dict]:
    total = len(strata)
    counts = {name: strata.count(name) for name in STRATUM_MASSES}
    if not total or any(not counts[name] for name in STRATUM_MASSES):
        raise ValueError("fine-tune loss strata must all be non-empty")
    weights = np.asarray(
        [
            STRATUM_MASSES[stratum] * total / counts[stratum]
            for stratum in strata
        ],
        dtype=np.float32,
    )
    return weights, {
        name: {
            "rows": counts[name],
            "aggregate_mass": STRATUM_MASSES[name],
            "row_weight": STRATUM_MASSES[name] * total / counts[name],
        }
        for name in STRATUM_MASSES
    }


def _expand_bounds(
    text: str,
    candidate_start: int,
    candidate_end: int,
    fits: Callable[[int, int], bool],
) -> tuple[int, int]:
    if not (0 <= candidate_start < candidate_end <= len(text)):
        raise ValueError("invalid candidate bounds")
    if fits(0, len(text)):
        return 0, len(text)
    if not fits(candidate_start, candidate_end):
        raise ValueError("candidate-only marked input exceeds token limit")

    def bounds_after_steps(steps: int) -> tuple[int, int]:
        start = candidate_start
        end = candidate_end
        prefer_left = True
        for _ in range(steps):
            if start > 0 and (prefer_left or end >= len(text)):
                start -= 1
                prefer_left = False
            elif end < len(text):
                end += 1
                prefer_left = True
            else:
                raise ValueError("fine-tune expansion exceeds jie")
        return start, end

    maximum = len(text) - (candidate_end - candidate_start)
    lower = 0
    upper = maximum
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if fits(*bounds_after_steps(middle)):
            lower = middle
        else:
            upper = middle - 1
    selected = bounds_after_steps(lower)
    if not fits(*selected):
        raise ValueError("fine-tune selected slice exceeds token limit")
    if lower < maximum and fits(*bounds_after_steps(lower + 1)):
        raise ValueError("fine-tune token length is not monotone")
    return selected


def _marked_text(
    text: str, candidate_start: int, candidate_end: int, start: int, end: int
) -> str:
    paragraph_marker = f" {PARAGRAPH_SENTINEL} "
    left = text[start:candidate_start].replace("\n", paragraph_marker)
    candidate = text[candidate_start:candidate_end].replace(
        "\n", paragraph_marker
    )
    right = text[candidate_end:end].replace("\n", paragraph_marker)
    return (
        left
        + f" {LEFT_SENTINEL} "
        + candidate
        + f" {RIGHT_SENTINEL} "
        + right
    )


def _tokenize_record(tokenizer, example: dict, row: dict) -> dict:
    text = str(example["text"])
    candidate_start, candidate_end, _ = _assembled_bounds(example, row)
    surface = str(row["surface"])
    if text[candidate_start:candidate_end] != surface:
        raise ValueError(f"fine-tune source mismatch: {row['id']}")

    def encode(start: int, end: int):
        marked = _marked_text(
            text, candidate_start, candidate_end, start, end
        )
        return tokenizer(
            surface,
            marked,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=True,
        )

    def fits(start: int, end: int) -> bool:
        return len(encode(start, end)["input_ids"]) <= MAX_LENGTH

    slice_start, slice_end = _expand_bounds(
        text, candidate_start, candidate_end, fits
    )
    encoded = encode(slice_start, slice_end)
    input_ids = [int(value) for value in encoded["input_ids"]]
    sequence_ids = encoded.sequence_ids()
    marker_positions = {}
    for sentinel in (LEFT_SENTINEL, RIGHT_SENTINEL):
        token_id = SENTINEL_IDS[sentinel]
        positions = [
            index
            for index, (value, sequence_id) in enumerate(
                zip(input_ids, sequence_ids)
            )
            if value == token_id and sequence_id == 1
        ]
        if len(positions) != 1:
            raise ValueError(
                f"fine-tune marker count differs for {row['id']}: {sentinel}"
            )
        marker_positions[sentinel] = positions[0]
    paragraph_count = sum(
        1
        for value, sequence_id in zip(input_ids, sequence_ids)
        if value == SENTINEL_IDS[PARAGRAPH_SENTINEL] and sequence_id == 1
    )
    if paragraph_count != text[slice_start:slice_end].count("\n"):
        raise ValueError(
            f"fine-tune paragraph marker count differs: {row['id']}"
        )
    left_marker = marker_positions[LEFT_SENTINEL]
    right_marker = marker_positions[RIGHT_SENTINEL]
    segment_a = [
        index for index, sequence_id in enumerate(sequence_ids)
        if sequence_id == 0
    ]
    occurrence = [
        index
        for index, sequence_id in enumerate(sequence_ids)
        if sequence_id == 1 and left_marker < index < right_marker
    ]
    if (
        not segment_a
        or not occurrence
        or left_marker >= right_marker
        or len(input_ids) > MAX_LENGTH
    ):
        raise ValueError(f"fine-tune token geometry differs: {row['id']}")
    return {
        "input_ids": input_ids,
        "attention_mask": [
            int(value) for value in encoded["attention_mask"]
        ],
        "segment_a_indices": segment_a,
        "occurrence_indices": occurrence,
        "slice_start": slice_start,
        "slice_end": slice_end,
        "candidate_assembled_start": candidate_start,
        "candidate_assembled_end": candidate_end,
    }


def _validate_sentinels(tokenizer, examples: dict[str, dict]) -> None:
    for sentinel, expected_id in SENTINEL_IDS.items():
        actual_id = tokenizer.get_vocab().get(sentinel)
        tokenized = tokenizer(
            sentinel, add_special_tokens=False, truncation=False
        )["input_ids"]
        if actual_id != expected_id or tokenized != [expected_id]:
            raise ValueError(f"fine-tune sentinel binding differs: {sentinel}")
        if any(sentinel in str(example["text"]) for example in examples.values()):
            raise ValueError(f"fine-tune sentinel occurs in fit text: {sentinel}")


def _collate(records: list[dict], indices: np.ndarray, device):
    import torch

    chosen = [records[int(index)] for index in indices]
    max_length = max(len(record["input_ids"]) for record in chosen)
    input_ids = np.ones((len(chosen), max_length), dtype=np.int64)
    attention_mask = np.zeros((len(chosen), max_length), dtype=np.int64)
    segment_a_mask = np.zeros((len(chosen), max_length), dtype=bool)
    occurrence_mask = np.zeros((len(chosen), max_length), dtype=bool)
    for batch_index, record in enumerate(chosen):
        length = len(record["input_ids"])
        input_ids[batch_index, :length] = record["input_ids"]
        attention_mask[batch_index, :length] = record["attention_mask"]
        segment_a_mask[batch_index, record["segment_a_indices"]] = True
        occurrence_mask[batch_index, record["occurrence_indices"]] = True
    return (
        torch.from_numpy(input_ids).long().to(device),
        torch.from_numpy(attention_mask).long().to(device),
        torch.from_numpy(segment_a_mask).bool().to(device),
        torch.from_numpy(occurrence_mask).bool().to(device),
    )


def _build_classifier(encoder_dir: Path, device):
    import torch
    from torch import nn
    from transformers import AutoModel

    class CandidateMarkedClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(encoder_dir)
            self.encoder.gradient_checkpointing_enable()
            hidden_size = int(self.encoder.config.hidden_size)
            self.dropout = nn.Dropout(DROPOUT)
            self.classifier = nn.Linear(hidden_size * 3, len(LABEL_NAMES))

        def forward(
            self, input_ids, attention_mask, segment_a_mask, occurrence_mask
        ):
            hidden = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            segment_a = (
                hidden * segment_a_mask.unsqueeze(-1)
            ).sum(dim=1) / segment_a_mask.sum(dim=1, keepdim=True)
            occurrence = (
                hidden * occurrence_mask.unsqueeze(-1)
            ).sum(dim=1) / occurrence_mask.sum(dim=1, keepdim=True)
            pooled = torch.cat((hidden[:, 0], segment_a, occurrence), dim=1)
            return self.classifier(self.dropout(pooled))

    return CandidateMarkedClassifier().to(device)


def _fit(
    records: list[dict],
    labels: np.ndarray,
    strata: list[str],
    train_indices: np.ndarray,
    encoder_dir: Path,
    device,
) -> tuple[object, dict]:
    import torch
    from torch.nn import functional as F

    _seed_everything(SEED)
    model = _build_classifier(encoder_dir, device)
    train_strata = [strata[int(index)] for index in train_indices]
    local_weights, inventory = _stratum_weights(train_strata)
    weights = np.zeros(len(records), dtype=np.float32)
    weights[train_indices] = local_weights
    from transformers import Adafactor

    optimizer = Adafactor(
        [
            {
                "params": model.encoder.parameters(),
                "lr": ENCODER_LEARNING_RATE,
            },
            {
                "params": model.classifier.parameters(),
                "lr": CLASSIFIER_LEARNING_RATE,
            },
        ],
        weight_decay=WEIGHT_DECAY,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        beta1=None,
    )
    generator = np.random.default_rng(SEED)
    group_size = PHYSICAL_BATCH_SIZE * ACCUMULATION_STEPS
    for _ in range(EPOCHS):
        ordered = generator.permutation(train_indices)
        model.train()
        for group_start in range(0, len(ordered), group_size):
            group = ordered[group_start:group_start + group_size]
            model.zero_grad(set_to_none=True)
            divisor = len(group)
            for start in range(0, divisor, PHYSICAL_BATCH_SIZE):
                indices = group[start:start + PHYSICAL_BATCH_SIZE]
                logits = model(*_collate(records, indices, device))
                losses = F.cross_entropy(
                    logits,
                    torch.from_numpy(labels[indices]).long().to(device),
                    reduction="none",
                )
                row_weights = torch.from_numpy(weights[indices]).float().to(
                    device
                )
                ((losses * row_weights).sum() / divisor).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), MAX_GRAD_NORM, foreach=False
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    model.eval()
    return model, inventory


def _score(model, records: list[dict], indices: np.ndarray, device):
    import torch

    scores = np.zeros(len(indices), dtype=np.float32)
    predictions = np.zeros(len(indices), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), PHYSICAL_BATCH_SIZE):
            batch_indices = indices[start:start + PHYSICAL_BATCH_SIZE]
            logits = model(*_collate(records, batch_indices, device))
            probabilities = torch.softmax(logits, dim=1)
            scores[start:start + len(batch_indices)] = (
                probabilities[:, LABEL_BY_NAME["exact_person"]]
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            predictions[start:start + len(batch_indices)] = (
                probabilities.argmax(dim=1).cpu().numpy().astype(np.int64)
            )
    if not np.isfinite(scores).all():
        raise ValueError("fine-tune model emitted non-finite scores")
    return scores, predictions


def _table(
    scores: np.ndarray,
    labels: np.ndarray,
    strata: list[str],
    fold_ids: np.ndarray,
) -> list[dict]:
    exact_mask = labels == LABEL_BY_NAME["exact_person"]
    reference = {int(index) for index in np.flatnonzero(exact_mask)}
    mined_mask = np.asarray(
        [value == "mined_not_person" for value in strata]
    )
    real_negative_mask = np.asarray(
        [value == "real_not_person" for value in strata]
    )
    boundary_mask = labels == LABEL_BY_NAME["boundary_alternative"]
    rows = []
    for threshold in THRESHOLDS:
        prediction = {
            int(index) for index in np.flatnonzero(scores >= threshold)
        }
        true_positive = len(reference & prediction)
        precision = true_positive / len(prediction) if prediction else 0.0
        recall = true_positive / len(reference)
        wilson = _metric(reference, prediction)[
            "wilson_precision_lower_one_sided_95"
        ]
        fold_recalls = [
            float(np.mean(scores[exact_mask & (fold_ids == fold)] >= threshold))
            for fold in range(FOLDS)
        ]
        mined_rejection = float(np.mean(scores[mined_mask] < threshold))
        real_negative_rejection = float(
            np.mean(scores[real_negative_mask] < threshold)
        )
        boundary_rejection = float(
            np.mean(scores[boundary_mask] < threshold)
        )
        rows.append({
            "threshold": threshold,
            "rows": len(labels),
            "prediction_rows": len(prediction),
            "true_positive": true_positive,
            "precision": precision,
            "recall": recall,
            "wilson_precision_lower_one_sided_95": wilson,
            "mined_negative_rejection": mined_rejection,
            "real_negative_rejection": real_negative_rejection,
            "boundary_alternative_rejection": boundary_rejection,
            "fold_positive_recalls": fold_recalls,
            "eligible": (
                recall >= 0.98
                and precision >= 0.99
                and wilson >= 0.985
                and mined_rejection >= 0.99
                and real_negative_rejection >= 0.98
                and boundary_rejection >= 0.90
                and min(fold_recalls) >= 0.95
            ),
        })
    return rows


def _confusion(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        actual_name: {
            predicted_name: int(np.sum(
                (labels == actual_index) & (predictions == predicted_index)
            ))
            for predicted_index, predicted_name in enumerate(LABEL_NAMES)
        }
        for actual_index, actual_name in enumerate(LABEL_NAMES)
    }


def _score_distributions(scores: np.ndarray, strata: list[str]) -> dict:
    result = {}
    for stratum in STRATUM_MASSES:
        values = scores[np.asarray([value == stratum for value in strata])]
        result[stratum] = {
            "rows": len(values),
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "maximum": float(values.max()),
            "mean": float(values.mean()),
        }
    return result


def run_encoder_finetune(
    grouped_root: Path,
    safe_root: Path,
    revision9_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"fine-tune output exists: {output_dir}")
    grouped_manifest_path = grouped_root / "manifest.json"
    grouped_manifest = _read(grouped_manifest_path)
    examples_path = grouped_root / "examples.jsonl"
    existence_path = grouped_root / "existence.jsonl"
    examples = {
        str(row["id"]): row for row in _read_jsonl(examples_path)
    }
    real_rows = _read_jsonl(existence_path)
    if (
        grouped_manifest.get("status") != GROUPED_DATA_STATUS
        or grouped_manifest.get("confirmation_read") is not False
        or grouped_manifest.get("outputs", {}).get("examples_sha256")
        != _sha256(examples_path)
        or grouped_manifest.get("outputs", {}).get("existence_sha256")
        != _sha256(existence_path)
    ):
        raise ValueError("fine-tune grouped-data binding differs")

    safe_manifest_path = safe_root / "manifest.json"
    safe_manifest = _read(safe_manifest_path)
    safe_path = safe_root / "safe-negatives.jsonl"
    safe_rows = _read_jsonl(safe_path)
    if (
        safe_manifest.get("status") != FROZEN_STATUS
        or safe_manifest.get("confirmation_read") is not False
        or safe_manifest.get("outputs", {}).get("safe_negatives_sha256")
        != _sha256(safe_path)
        or safe_manifest.get("bindings", {}).get("grouped_manifest_sha256")
        != _sha256(grouped_manifest_path)
        or len(safe_rows) != EXPECTED_SAFE
    ):
        raise ValueError("fine-tune safe-negative binding differs")

    revision9_manifest_path = revision9_root / "manifest.json"
    revision9_manifest = _read(revision9_manifest_path)
    encoder_dir = revision9_root / "encoder"
    if (
        revision9_manifest.get("status")
        != "ml_production_precision_lexical_verifier_blocked_ai_assisted"
        or revision9_manifest.get("revision") != 9
        or revision9_manifest.get("confirmation_read") is not False
        or revision9_manifest.get("encoder_artifact")
        != _model_artifact(encoder_dir)
        or revision9_manifest.get("bindings", {}).get(
            "grouped_manifest_sha256"
        ) != _sha256(grouped_manifest_path)
        or revision9_manifest.get("bindings", {}).get(
            "safe_manifest_sha256"
        ) != _sha256(safe_manifest_path)
        or revision9_manifest.get("fit_inventory", {}).get(
            "rank_head_sha256"
        ) != _sha256(revision9_root / "ranker" / "head.safetensors")
        or revision9_manifest.get("fit_inventory", {}).get(
            "rank_scaler_sha256"
        ) != _sha256(revision9_root / "ranker" / "scaler.npz")
    ):
        raise ValueError("fine-tune encoder binding differs")

    rows = [*real_rows, *safe_rows]
    classified = [_classify_real_row(row) for row in real_rows]
    strata = [value[0] for value in classified] + [
        "mined_not_person"
    ] * len(safe_rows)
    labels = np.asarray(
        [value[1] for value in classified]
        + [LABEL_BY_NAME["not_person"]] * len(safe_rows),
        dtype=np.int64,
    )
    if (
        int(np.sum(labels == LABEL_BY_NAME["exact_person"]))
        != EXPECTED_EXACT
        or int(np.sum(labels == LABEL_BY_NAME["boundary_alternative"]))
        != EXPECTED_BOUNDARY
        or strata.count("real_not_person") != EXPECTED_REAL_NEGATIVE
        or any(str(row["id"]) not in examples for row in rows)
    ):
        raise ValueError("fine-tune row inventory differs")
    geometry = {
        (
            int(row["juan"]),
            int(row["jie_index"]),
            int(row["para_id"]),
            int(row["start"]),
            int(row["end"]),
        )
        for row in rows
    }
    if len(geometry) != len(rows):
        raise ValueError("fine-tune geometry collision")

    git_commit = _git_commit_clean()
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir, use_fast=True)
    _validate_sentinels(tokenizer, examples)

    records = []
    for row in rows:
        tokenized = _tokenize_record(
            tokenizer, examples[str(row["id"])], row
        )
        records.append({**row, **tokenized})

    fold_by_juan = _folds([int(row["juan"]) for row in examples.values()])
    fold_ids = np.asarray(
        [fold_by_juan[int(row["juan"])] for row in rows],
        dtype=np.int64,
    )
    oof_scores = np.zeros(len(rows), dtype=np.float32)
    oof_predictions = np.zeros(len(rows), dtype=np.int64)
    fold_inventory = []
    for fold in range(FOLDS):
        train_indices = np.flatnonzero(fold_ids != fold)
        heldout_indices = np.flatnonzero(fold_ids == fold)
        model, weight_inventory = _fit(
            records,
            labels,
            strata,
            train_indices,
            encoder_dir,
            device,
        )
        scores, predictions = _score(
            model, records, heldout_indices, device
        )
        oof_scores[heldout_indices] = scores
        oof_predictions[heldout_indices] = predictions
        fold_inventory.append({
            "fold": fold,
            "train_juans": sorted(
                juan for juan, value in fold_by_juan.items()
                if value != fold
            ),
            "heldout_juans": sorted(
                juan for juan, value in fold_by_juan.items()
                if value == fold
            ),
            "train_rows": len(train_indices),
            "heldout_rows": len(heldout_indices),
            "stratum_weights": weight_inventory,
        })
        del model
        torch.cuda.empty_cache()

    table = _table(oof_scores, labels, strata, fold_ids)
    eligible = [row for row in table if row["eligible"]]
    selected = max(
        eligible,
        key=lambda row: (row["recall"], row["precision"], row["threshold"]),
        default=None,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        input_path = staging / "inputs.jsonl"
        input_path.write_text(
            "".join(
                json.dumps(
                    {
                        "id": str(row["id"]),
                        "juan": int(row["juan"]),
                        "jie_index": int(row["jie_index"]),
                        "para_id": int(row["para_id"]),
                        "start": int(row["start"]),
                        "end": int(row["end"]),
                        "slice_start": int(record["slice_start"]),
                        "slice_end": int(record["slice_end"]),
                        "input_ids": record["input_ids"],
                        "segment_a_indices": record["segment_a_indices"],
                        "occurrence_indices": record["occurrence_indices"],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
                for row, record in zip(rows, records)
            ),
            encoding="utf-8",
        )
        scores_path = staging / "oof-scores.jsonl"
        scores_path.write_text(
            "".join(
                json.dumps(
                    {
                        **row,
                        "stratum": stratum,
                        "class_label": LABEL_NAMES[int(label)],
                        "fold": int(fold),
                        "oof_exact_probability": float(score),
                        "oof_argmax": LABEL_NAMES[int(prediction)],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
                for row, stratum, label, fold, score, prediction in zip(
                    rows,
                    strata,
                    labels,
                    fold_ids,
                    oof_scores,
                    oof_predictions,
                )
            ),
            encoding="utf-8",
        )
        final_inventory = None
        if selected is not None:
            all_indices = np.arange(len(records), dtype=np.int64)
            model, weight_inventory = _fit(
                records,
                labels,
                strata,
                all_indices,
                encoder_dir,
                device,
            )
            model.encoder.save_pretrained(
                staging / "admission-encoder", safe_serialization=True
            )
            save_file(
                {
                    key: value.detach().cpu()
                    for key, value in model.classifier.state_dict().items()
                },
                staging / "admission-classifier.safetensors",
            )
            pinned_encoder = staging / "rank-encoder"
            ranker = staging / "ranker"
            shutil.copytree(
                encoder_dir, pinned_encoder, copy_function=shutil.copy2
            )
            shutil.copytree(
                revision9_root / "ranker",
                ranker,
                copy_function=shutil.copy2,
            )
            if (
                _sha256(ranker / "head.safetensors")
                != revision9_manifest["fit_inventory"]["rank_head_sha256"]
                or _sha256(ranker / "scaler.npz")
                != revision9_manifest["fit_inventory"]["rank_scaler_sha256"]
            ):
                raise RuntimeError("fine-tune copied ranker differs")
            final_inventory = {
                "rows": len(records),
                "stratum_weights": weight_inventory,
                "admission_encoder": _model_artifact(
                    staging / "admission-encoder"
                ),
                "classifier_sha256": _sha256(
                    staging / "admission-classifier.safetensors"
                ),
                "rank_encoder": _model_artifact(pinned_encoder),
                "rank_head_sha256": _sha256(
                    ranker / "head.safetensors"
                ),
                "rank_scaler_sha256": _sha256(ranker / "scaler.npz"),
            }
        manifest = {
            "schema_version": 1,
            "status": (
                FINETUNE_STATUS_SELECTED
                if selected is not None
                else FINETUNE_STATUS_BLOCKED
            ),
            "revision": REVISION,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "git_commit": git_commit,
            "bindings": {
                "grouped_manifest_sha256": _sha256(
                    grouped_manifest_path
                ),
                "safe_manifest_sha256": _sha256(safe_manifest_path),
                "revision9_manifest_sha256": _sha256(
                    revision9_manifest_path
                ),
            },
            "control": {
                "seed": SEED,
                "folds": FOLDS,
                "max_length": MAX_LENGTH,
                "epochs": EPOCHS,
                "physical_batch_size": PHYSICAL_BATCH_SIZE,
                "gradient_accumulation_steps": ACCUMULATION_STEPS,
                "encoder_learning_rate": ENCODER_LEARNING_RATE,
                "classifier_learning_rate": CLASSIFIER_LEARNING_RATE,
                "optimizer": "adafactor",
                "weight_decay": WEIGHT_DECAY,
                "dropout": DROPOUT,
                "max_grad_norm": MAX_GRAD_NORM,
                "sentinel_ids": SENTINEL_IDS,
                "encoder_frozen": False,
                "gradient_checkpointing": True,
                "optimizer_state_device": "cuda",
                "labels": list(LABEL_NAMES),
                "stratum_masses": STRATUM_MASSES,
            },
            "folds": fold_inventory,
            "confusion": _confusion(labels, oof_predictions),
            "score_distributions": _score_distributions(
                oof_scores, strata
            ),
            "table": table,
            "selected": selected,
            "final_fit": final_inventory,
            "outputs": {
                "inputs_sha256": _sha256(input_path),
                "oof_scores_sha256": _sha256(scores_path),
            },
            "claim_limit": (
                "Fit-only candidate-marked encoder fine-tuning; consumed "
                "calibration and sealed confirmation were not loaded."
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
        description="Run Revision-11 fit-only encoder fine-tuning."
    )
    parser.add_argument("--grouped-data", type=Path, required=True)
    parser.add_argument("--safe-negatives", type=Path, required=True)
    parser.add_argument("--revision-9", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_encoder_finetune(
        args.grouped_data,
        args.safe_negatives,
        args.revision_9,
        args.output,
    )
    print(json.dumps({
        "status": manifest["status"],
        "selected": manifest["selected"],
        "confusion": manifest["confusion"],
        "table": manifest["table"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
