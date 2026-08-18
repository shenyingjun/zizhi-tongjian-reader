from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
import unicodedata
from pathlib import Path

import numpy as np

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from p1_windows import build_windows
from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from production_precision_grouped_data import GROUPED_DATA_STATUS
from production_precision_grouped_verifier import (
    BATCH_SIZE,
    DROPOUT,
    EPOCHS,
    HIDDEN_WIDTH,
    LEARNING_RATE,
    WEIGHT_DECAY,
    _read,
    _read_jsonl,
    _sha256,
)
from production_precision_negative_audit_freeze import (
    EXPECTED_SAFE,
    FROZEN_STATUS,
)
from production_precision_select import _metric
from production_span_verifier import (
    BOUNDARY_CATEGORIES,
    THRESHOLDS,
    _assembled_bounds,
    _category,
)
from production_train import _make_read_only


ATTENTION_STATUS_BLOCKED = "ml_production_same_jie_attention_blocked"
ATTENTION_STATUS_SELECTED = "ml_production_same_jie_attention_selected"
SEED = 20260817
ATTENTION_WIDTH = 128
FOLDS = 7
EXPECTED_JUANS = 28
EXPECTED_POSITIVES = 2525
EXPECTED_REAL_NEGATIVES = 171
STRATUM_MASSES = {
    "real_positive": 0.50,
    "real_negative": 0.25,
    "mined_negative": 0.25,
}
DISTANCE_BUCKETS = (
    "inside", "1", "2-4", "5-16", "17-64", "65+", "other_paragraph"
)


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _folds(juans: list[int]) -> dict[int, int]:
    unique = sorted(set(juans))
    if len(unique) != EXPECTED_JUANS:
        raise ValueError(f"expected {EXPECTED_JUANS} fit juans")
    return {juan: index % FOLDS for index, juan in enumerate(unique)}


def _distance_bucket(
    position: int,
    candidate_start: int,
    candidate_end: int,
    position_paragraph: int,
    candidate_paragraph: int,
) -> int:
    if position_paragraph != candidate_paragraph:
        return DISTANCE_BUCKETS.index("other_paragraph")
    if candidate_start <= position < candidate_end:
        return DISTANCE_BUCKETS.index("inside")
    distance = (
        candidate_start - position
        if position < candidate_start
        else position - candidate_end + 1
    )
    if distance == 1:
        return DISTANCE_BUCKETS.index("1")
    if distance <= 4:
        return DISTANCE_BUCKETS.index("2-4")
    if distance <= 16:
        return DISTANCE_BUCKETS.index("5-16")
    if distance <= 64:
        return DISTANCE_BUCKETS.index("17-64")
    return DISTANCE_BUCKETS.index("65+")


def _scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def _stratum_weights(strata: list[str]) -> tuple[np.ndarray, dict]:
    total = len(strata)
    counts = {name: strata.count(name) for name in STRATUM_MASSES}
    if not total or any(not counts[name] for name in STRATUM_MASSES):
        raise ValueError("attention loss strata must all be non-empty")
    weights = np.asarray([
        STRATUM_MASSES[name] * total / counts[name] for name in strata
    ], dtype=np.float32)
    return weights, {
        name: {
            "rows": counts[name],
            "aggregate_mass": STRATUM_MASSES[name],
            "row_weight": STRATUM_MASSES[name] * total / counts[name],
        }
        for name in STRATUM_MASSES
    }


def _encode_examples(examples: dict[str, dict], encoder, tokenizer, device):
    import torch

    hidden_size = int(encoder.config.hidden_size)
    caches = {}
    encoder.eval()
    with torch.inference_mode():
        for identity in sorted(examples):
            example = examples[identity]
            text = str(example["text"])
            windows = build_windows(
                tokenizer, example, max_length=512, stride=128
            )
            hidden = np.zeros((len(text), hidden_size), dtype=np.float32)
            owned = np.zeros(len(text), dtype=bool)
            for window in windows:
                inputs = {
                    "input_ids": torch.tensor(
                        [window.input_ids], dtype=torch.long, device=device
                    ),
                    "attention_mask": torch.tensor(
                        [window.attention_mask], dtype=torch.long, device=device
                    ),
                }
                if window.token_type_ids is not None:
                    inputs["token_type_ids"] = torch.tensor(
                        [window.token_type_ids],
                        dtype=torch.long,
                        device=device,
                    )
                output = encoder(**inputs).last_hidden_state[0].cpu().numpy()
                for token_index, (start, end) in enumerate(window.offsets):
                    if not window.owned_tokens[token_index]:
                        continue
                    for character_index in range(start, end):
                        if owned[character_index]:
                            raise ValueError(
                                "character has multiple attention-cache owners"
                            )
                        hidden[character_index] = output[token_index]
                        owned[character_index] = True
            paragraph_by_position = np.full(len(text), -1, dtype=np.int32)
            paragraph_ordinals = {}
            paragraph_means = {}
            for ordinal, segment in enumerate(example["segments"]):
                start = int(segment["assembled_start"])
                end = int(segment["assembled_end"])
                para_id = int(segment["para_id"])
                positions = np.asarray([
                    index for index in range(start, end)
                    if owned[index] and text[index] != "\n"
                ], dtype=np.int32)
                if not len(positions):
                    raise ValueError(f"paragraph has no owned text: {identity}")
                paragraph_by_position[positions] = para_id
                paragraph_ordinals[para_id] = ordinal
                paragraph_means[para_id] = hidden[positions].mean(axis=0)
            positions = np.asarray([
                index for index, is_owned in enumerate(owned)
                if is_owned and text[index] != "\n"
            ], dtype=np.int32)
            if (
                not len(positions)
                or (paragraph_by_position[positions] < 0).any()
            ):
                raise ValueError(f"invalid same-jie attention cache: {identity}")
            caches[identity] = {
                "hidden": hidden,
                "owned": owned,
                "positions": positions,
                "context_hidden": hidden[positions],
                "context_paragraphs": paragraph_by_position[positions],
                "paragraph_means": paragraph_means,
                "paragraph_ordinals": paragraph_ordinals,
                "paragraph_count": len(example["segments"]),
                "jie_mean": hidden[positions].mean(axis=0),
            }
    return hidden_size, caches


def _numeric_features(
    example: dict, row: dict, assembled_start: int, assembled_end: int,
    segment: dict, paragraph_ordinal: int, paragraph_count: int,
) -> np.ndarray:
    text = str(example["text"])
    segment_start = int(segment["assembled_start"])
    segment_end = int(segment["assembled_end"])
    left_index = assembled_start - 1 if assembled_start > segment_start else None
    right_index = assembled_end if assembled_end < segment_end else None
    numeric = [math.log1p(assembled_end - assembled_start)]
    for character in (
        text[left_index] if left_index is not None else None,
        text[right_index] if right_index is not None else None,
    ):
        category = _category(character)
        numeric.extend(
            float(category == expected) for expected in BOUNDARY_CATEGORIES
        )
    numeric.extend([
        float(left_index is None),
        float(right_index is None),
    ])
    paragraph_length = segment_end - segment_start
    numeric.extend([
        int(row["start"]) / paragraph_length,
        int(row["end"]) / paragraph_length,
        paragraph_ordinal / max(1, paragraph_count - 1),
        math.log1p(paragraph_count),
    ])
    return np.asarray(numeric, dtype=np.float32)


def _records(
    examples: dict[str, dict], rows: list[dict], caches: dict
) -> list[dict]:
    records = []
    for row in rows:
        identity = str(row["id"])
        example = examples[identity]
        cache = caches[identity]
        start, end, segment = _assembled_bounds(example, row)
        if not cache["owned"][start:end].all():
            raise ValueError(f"candidate not attention-cache owned: {identity}")
        para_id = int(row["para_id"])
        segment_start = int(segment["assembled_start"])
        segment_end = int(segment["assembled_end"])
        left_index = start - 1 if start > segment_start else None
        right_index = end if end < segment_end else None
        left = (
            cache["hidden"][left_index]
            if left_index is not None and cache["owned"][left_index]
            else None
        )
        right = (
            cache["hidden"][right_index]
            if right_index is not None and cache["owned"][right_index]
            else None
        )
        buckets = np.asarray([
            _distance_bucket(
                int(position), start, end, int(position_paragraph), para_id
            )
            for position, position_paragraph in zip(
                cache["positions"], cache["context_paragraphs"]
            )
        ], dtype=np.int64)
        records.append({
            "row": row,
            "identity": identity,
            "candidate": cache["hidden"][start:end].mean(axis=0),
            "left": left,
            "right": right,
            "paragraph": cache["paragraph_means"][para_id],
            "jie": cache["jie_mean"],
            "numeric": _numeric_features(
                example,
                row,
                start,
                end,
                segment,
                cache["paragraph_ordinals"][para_id],
                cache["paragraph_count"],
            ),
            "buckets": buckets,
        })
    return records


def _fit_scalers(records: list[dict], caches: dict) -> dict:
    identities = sorted({record["identity"] for record in records})
    hidden_values = np.concatenate([
        caches[identity]["context_hidden"] for identity in identities
    ], axis=0)
    numeric_values = np.stack([record["numeric"] for record in records])
    hidden_mean, hidden_scale = _scale(hidden_values)
    numeric_mean, numeric_scale = _scale(numeric_values)
    return {
        "hidden_mean": hidden_mean,
        "hidden_scale": hidden_scale,
        "numeric_mean": numeric_mean,
        "numeric_scale": numeric_scale,
    }


def _build_model(hidden_size: int, numeric_size: int):
    import torch
    from torch import nn

    class SameJieAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(hidden_size, ATTENTION_WIDTH)
            self.key = nn.Linear(hidden_size, ATTENTION_WIDTH)
            self.bucket_bias = nn.Embedding(len(DISTANCE_BUCKETS), 1)
            self.head = nn.Sequential(
                nn.Linear(hidden_size * 6 + numeric_size, HIDDEN_WIDTH),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_WIDTH, 1),
            )

        def forward(
            self, candidate, left, right, paragraph, jie, numeric,
            context, buckets, mask,
        ):
            query = self.query(candidate)
            keys = self.key(context)
            scores = torch.einsum("bd,bld->bl", query, keys)
            scores = scores / math.sqrt(ATTENTION_WIDTH)
            scores = scores + self.bucket_bias(buckets).squeeze(-1)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            attention = torch.softmax(scores, dim=1)
            attended = torch.bmm(attention.unsqueeze(1), context).squeeze(1)
            return self.head(torch.cat((
                candidate, left, right, attended, paragraph, jie, numeric
            ), dim=1)).squeeze(-1)

    return SameJieAttention()


def _batch(
    indices: np.ndarray,
    records: list[dict],
    caches: dict,
    scalers: dict,
    labels: np.ndarray | None,
    weights: np.ndarray | None,
    device,
) -> tuple:
    import torch

    chosen = [records[int(index)] for index in indices]
    hidden_mean = scalers["hidden_mean"]
    hidden_scale = scalers["hidden_scale"]
    max_length = max(len(record["buckets"]) for record in chosen)
    hidden_size = len(hidden_mean)
    context = np.zeros(
        (len(chosen), max_length, hidden_size), dtype=np.float32
    )
    buckets = np.zeros((len(chosen), max_length), dtype=np.int64)
    mask = np.zeros((len(chosen), max_length), dtype=bool)
    vectors = {name: [] for name in (
        "candidate", "left", "right", "paragraph", "jie"
    )}
    numeric = []
    for batch_index, record in enumerate(chosen):
        raw_context = caches[record["identity"]]["context_hidden"]
        length = len(raw_context)
        context[batch_index, :length] = (
            raw_context - hidden_mean
        ) / hidden_scale
        buckets[batch_index, :length] = record["buckets"]
        mask[batch_index, :length] = True
        for name in ("candidate", "paragraph", "jie"):
            vectors[name].append(
                (record[name] - hidden_mean) / hidden_scale
            )
        for name in ("left", "right"):
            vectors[name].append(
                np.zeros(hidden_size, dtype=np.float32)
                if record[name] is None
                else (record[name] - hidden_mean) / hidden_scale
            )
        numeric.append(
            (record["numeric"] - scalers["numeric_mean"])
            / scalers["numeric_scale"]
        )
    tensors = [
        torch.from_numpy(np.stack(vectors[name])).float().to(device)
        for name in ("candidate", "left", "right", "paragraph", "jie")
    ]
    tensors.extend([
        torch.from_numpy(np.stack(numeric)).float().to(device),
        torch.from_numpy(context).float().to(device),
        torch.from_numpy(buckets).long().to(device),
        torch.from_numpy(mask).bool().to(device),
    ])
    if labels is not None:
        tensors.append(torch.from_numpy(labels[indices]).float().to(device))
    if weights is not None:
        tensors.append(torch.from_numpy(weights[indices]).float().to(device))
    return tuple(tensors)


def _fit(
    records: list[dict],
    caches: dict,
    labels: np.ndarray,
    strata: list[str],
    train_indices: np.ndarray,
    hidden_size: int,
    device,
) -> tuple[object, dict, dict]:
    import torch
    from torch.nn import functional as F

    _seed_everything(SEED)
    train_records = [records[int(index)] for index in train_indices]
    scalers = _fit_scalers(train_records, caches)
    train_strata = [strata[int(index)] for index in train_indices]
    local_weights, weight_inventory = _stratum_weights(train_strata)
    weights = np.zeros(len(records), dtype=np.float32)
    weights[train_indices] = local_weights
    model = _build_model(hidden_size, len(records[0]["numeric"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = np.random.default_rng(SEED)
    for _ in range(EPOCHS):
        ordered = generator.permutation(train_indices)
        model.train()
        for start in range(0, len(ordered), BATCH_SIZE):
            batch_indices = ordered[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            values = _batch(
                batch_indices, records, caches, scalers, labels, weights, device
            )
            logits = model(*values[:9])
            loss = F.binary_cross_entropy_with_logits(
                logits, values[9], reduction="none"
            )
            loss = (loss * values[10]).mean()
            loss.backward()
            optimizer.step()
    model.eval()
    return model, scalers, weight_inventory


def _score(
    model, records: list[dict], caches: dict, scalers: dict,
    indices: np.ndarray, device,
) -> np.ndarray:
    import torch

    result = np.zeros(len(indices), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for output_start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[output_start:output_start + BATCH_SIZE]
            values = _batch(
                batch_indices, records, caches, scalers, None, None, device
            )
            result[output_start:output_start + len(batch_indices)] = (
                model(*values).sigmoid().cpu().numpy().astype(np.float32)
            )
    if not np.isfinite(result).all():
        raise ValueError("same-jie attention emitted non-finite scores")
    return result


def _table(
    scores: np.ndarray,
    labels: np.ndarray,
    strata: list[str],
    fold_ids: np.ndarray,
) -> list[dict]:
    reference = {int(index) for index in np.flatnonzero(labels == 1)}
    rows = []
    for threshold in THRESHOLDS:
        prediction = {
            int(index) for index in np.flatnonzero(scores >= threshold)
        }
        true_positive = len(reference & prediction)
        precision = (
            true_positive / len(prediction) if prediction else 0.0
        )
        recall = true_positive / len(reference)
        wilson = _metric(reference, prediction)[
            "wilson_precision_lower_one_sided_95"
        ]
        mined = np.asarray([
            name == "mined_negative" for name in strata
        ])
        real_negative = np.asarray([
            name == "real_negative" for name in strata
        ])
        fold_recalls = []
        for fold in range(FOLDS):
            fold_positive = (labels == 1) & (fold_ids == fold)
            fold_recalls.append(float(np.mean(
                scores[fold_positive] >= threshold
            )))
        mined_rejection = float(np.mean(scores[mined] < threshold))
        real_negative_rejection = float(
            np.mean(scores[real_negative] < threshold)
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
            "fold_positive_recalls": fold_recalls,
            "eligible": (
                recall >= 0.98
                and precision >= 0.99
                and wilson >= 0.985
                and mined_rejection >= 0.99
                and real_negative_rejection >= 0.98
                and min(fold_recalls) >= 0.95
            ),
        })
    return rows


def run_same_jie_attention(
    grouped_root: Path,
    safe_root: Path,
    revision9_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"same-jie attention output exists: {output_dir}")
    grouped_manifest_path = grouped_root / "manifest.json"
    grouped_manifest = _read(grouped_manifest_path)
    examples_path = grouped_root / "examples.jsonl"
    existence_path = grouped_root / "existence.jsonl"
    examples = {
        str(row["id"]): row for row in _read_jsonl(examples_path)
    }
    real_rows = _read_jsonl(existence_path)
    grouped_outputs = grouped_manifest.get("outputs", {})
    if (
        grouped_manifest.get("status") != GROUPED_DATA_STATUS
        or grouped_manifest.get("confirmation_read") is not False
        or grouped_outputs.get("examples_sha256") != _sha256(examples_path)
        or grouped_outputs.get("existence_sha256") != _sha256(existence_path)
    ):
        raise ValueError("same-jie attention grouped-data binding differs")

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
        raise ValueError("same-jie attention safe-negative binding differs")

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
    ):
        raise ValueError("same-jie attention encoder binding differs")

    rows = [*real_rows, *safe_rows]
    labels = np.asarray(
        [int(row["label"]) for row in rows], dtype=np.float32
    )
    strata = [
        *[
            "real_positive" if int(row["label"]) else "real_negative"
            for row in real_rows
        ],
        *(["mined_negative"] * len(safe_rows)),
    ]
    if (
        int(labels.sum()) != EXPECTED_POSITIVES
        or strata.count("real_negative") != EXPECTED_REAL_NEGATIVES
        or any(str(row["id"]) not in examples for row in rows)
    ):
        raise ValueError("same-jie attention row inventory differs")
    geometry = {
        (
            int(row["juan"]), int(row["jie_index"]), int(row["para_id"]),
            int(row["start"]), int(row["end"]),
        )
        for row in rows
    }
    if len(geometry) != len(rows):
        raise ValueError("same-jie attention geometry collision")

    fold_by_juan = _folds([int(row["juan"]) for row in examples.values()])
    fold_ids = np.asarray([
        fold_by_juan[int(row["juan"])] for row in rows
    ], dtype=np.int64)
    git_commit = _git_commit_clean()
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModel, AutoTokenizer

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir, use_fast=True)
    encoder = AutoModel.from_pretrained(encoder_dir).to(device)
    hidden_size, caches = _encode_examples(
        examples, encoder, tokenizer, device
    )
    records = _records(examples, rows, caches)
    oof_scores = np.zeros(len(rows), dtype=np.float32)
    fold_inventory = []
    for fold in range(FOLDS):
        train_indices = np.flatnonzero(fold_ids != fold)
        heldout_indices = np.flatnonzero(fold_ids == fold)
        model, scalers, weight_inventory = _fit(
            records,
            caches,
            labels,
            strata,
            train_indices,
            hidden_size,
            device,
        )
        oof_scores[heldout_indices] = _score(
            model,
            records,
            caches,
            scalers,
            heldout_indices,
            device,
        )
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
        score_rows = [{
            **row,
            "stratum": stratum,
            "fold": int(fold),
            "oof_score": float(score),
        } for row, stratum, fold, score in zip(
            rows, strata, fold_ids, oof_scores
        )]
        scores_path = staging / "oof-scores.jsonl"
        scores_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n" for row in score_rows
            ),
            encoding="utf-8",
        )
        final_inventory = None
        if selected is not None:
            all_indices = np.arange(len(records), dtype=np.int64)
            model, scalers, weight_inventory = _fit(
                records,
                caches,
                labels,
                strata,
                all_indices,
                hidden_size,
                device,
            )
            model_dir = staging / "model"
            model_dir.mkdir()
            save_file(
                {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                model_dir / "attention.safetensors",
            )
            np.savez(
                model_dir / "scalers.npz",
                hidden_mean=scalers["hidden_mean"],
                hidden_scale=scalers["hidden_scale"],
                numeric_mean=scalers["numeric_mean"],
                numeric_scale=scalers["numeric_scale"],
            )
            rank_source = revision9_root / "ranker"
            rank_target = staging / "ranker"
            shutil.copytree(
                rank_source, rank_target, copy_function=shutil.copy2
            )
            encoder_target = staging / "encoder"
            shutil.copytree(
                encoder_dir, encoder_target, copy_function=shutil.copy2
            )
            if (
                _model_artifact(encoder_target)
                != revision9_manifest["encoder_artifact"]
                or _sha256(rank_target / "head.safetensors")
                != revision9_manifest["fit_inventory"]["rank_head_sha256"]
                or _sha256(rank_target / "scaler.npz")
                != revision9_manifest["fit_inventory"]["rank_scaler_sha256"]
            ):
                raise RuntimeError("same-jie copied artifact differs")
            final_inventory = {
                "rows": len(records),
                "stratum_weights": weight_inventory,
                "attention_sha256": _sha256(
                    model_dir / "attention.safetensors"
                ),
                "scalers_sha256": _sha256(model_dir / "scalers.npz"),
                "rank_head_sha256": _sha256(
                    rank_target / "head.safetensors"
                ),
                "rank_scaler_sha256": _sha256(
                    rank_target / "scaler.npz"
                ),
            }
        manifest = {
            "schema_version": 1,
            "status": (
                ATTENTION_STATUS_SELECTED
                if selected is not None
                else ATTENTION_STATUS_BLOCKED
            ),
            "revision": 10,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "git_commit": git_commit,
            "bindings": {
                "grouped_manifest_sha256": _sha256(grouped_manifest_path),
                "safe_manifest_sha256": _sha256(safe_manifest_path),
                "revision9_manifest_sha256": _sha256(
                    revision9_manifest_path
                ),
            },
            "control": {
                "seed": SEED,
                "folds": FOLDS,
                "attention_width": ATTENTION_WIDTH,
                "distance_buckets": list(DISTANCE_BUCKETS),
                "epochs": EPOCHS,
                "hidden_width": HIDDEN_WIDTH,
                "dropout": DROPOUT,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "encoder_frozen": True,
                "row_level_shuffling": True,
                "feature_dim": hidden_size * 6 + len(records[0]["numeric"]),
            },
            "folds": fold_inventory,
            "table": table,
            "selected": selected,
            "final_fit": final_inventory,
            "outputs": {"oof_scores_sha256": _sha256(scores_path)},
            "claim_limit": (
                "Fit-only same-jie attention selection; consumed calibration "
                "and sealed confirmation were not loaded."
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
        description="Run Revision-10 fit-only same-jie attention selection."
    )
    parser.add_argument("--grouped-data", type=Path, required=True)
    parser.add_argument("--safe-negatives", type=Path, required=True)
    parser.add_argument("--revision-9", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_same_jie_attention(
        args.grouped_data,
        args.safe_negatives,
        args.revision_9,
        args.output,
    )
    print(json.dumps({
        "status": manifest["status"],
        "selected": manifest["selected"],
        "table": manifest["table"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
