from __future__ import annotations

import argparse
import json
import os
import platform
import re
import tempfile
from pathlib import Path

import numpy as np

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from production_precision_encoder_finetune import (
    ACCUMULATION_STEPS,
    CLASSIFIER_LEARNING_RATE,
    DROPOUT,
    ENCODER_LEARNING_RATE,
    EPOCHS,
    FOLDS,
    LEFT_SENTINEL,
    MAX_GRAD_NORM,
    MAX_LENGTH,
    PARAGRAPH_SENTINEL,
    PHYSICAL_BATCH_SIZE,
    RIGHT_SENTINEL,
    SEED,
    SENTINEL_IDS,
    WEIGHT_DECAY,
    _collate,
    _tokenize_record,
    _validate_sentinels,
)
from production_precision_grouped_data import GROUPED_DATA_STATUS
from production_precision_grouped_verifier import _read, _read_jsonl, _sha256
from production_precision_revision14_inventory import (
    STATUS as INVENTORY_STATUS,
    EXPECTED_REFERENCES,
    EXPECTED_EXISTENCE,
    EXPECTED_SEMANTIC,
    EXPECTED_EASY,
    EXPECTED_BOUNDARY,
    EXPECTED_RANK_PAIRS,
    EXPECTED_LATTICE,
    EXPECTED_MINED_BOUNDARY,
    _key,
    _overlaps,
    _ordered,
)
from production_precision_same_jie_attention import _folds
from production_precision_select import _metric
from production_span_verifier import THRESHOLDS
from production_train import _make_read_only


REVISION = 14
FINETUNE_STATUS_BLOCKED = (
    "ml_production_precision_revision14_two_stage_blocked"
)
FINETUNE_STATUS_SELECTED = (
    "ml_production_precision_revision14_two_stage_selected"
)
REVISION9_BASE_ENCODER = {
    "files": {
        "config.json": "3b8a850017fe155135e85313829edae4548b98d3e994d99982525e9d5d65babd",
        "model.safetensors": "d1f871d3deba6fc8491e3980034482c343275510ab92e27c1978932b5e28064f",
        "special_tokens_map.json": "3c3507f36dff57bce437223db3b3081d1e2b52ec3e56ee55438193ecb2c94dd6",
        "tokenizer.json": "53a310380181ff91e8040963dfb600721407492fc981de8072c5d55a3be7cb9b",
        "tokenizer_config.json": "fcf430ac661d36e8dfccffd050a74e31ea1cfbf54c16c762826224f551dd9044",
        "vocab.txt": "d6d19121379d1d258ff28c2ae25e264d70a66ff3926fe3490e2f8977a4d3111e",
    },
    "combined_sha256": "c91ecf9d7fd5544984cc8c86f68e51914a96fc32d7847f0b3c3e7d257950d877",
}
MARGIN = 1.0
EXPECTED_RECONCILED_NONOVERLAP = 15

# Stage-1 gate thresholds
STAGE1_OVERLAP_RECALL_MIN = 0.99
STAGE1_SEMANTIC_REJECTION_MIN = 0.95
STAGE1_EASY_REJECTION_MIN = 0.99
STAGE1_MIN_FOLD_OVERLAP_RECALL = 0.97

# End-to-end gate thresholds
E2E_EXACT_RECALL_MIN = 0.98
E2E_REAL_PRECISION_MIN = 0.99
E2E_BOUNDARY_COMPONENT_ACCURACY_MIN = 0.95
E2E_MIN_FOLD_EXACT_RECALL = 0.95


def _seed_everything(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _geometry(row: dict) -> tuple[str, int, int, int]:
    return (
        str(row["id"]),
        int(row["para_id"]),
        int(row["start"]),
        int(row["end"]),
    )


def _ordered_key(row: dict) -> tuple[int, int, int, int, int, str]:
    return (
        int(row["juan"]),
        int(row["jie_index"]),
        int(row["para_id"]),
        int(row["start"]),
        int(row["end"]),
        str(row["id"]),
    )


def _validate_candidate(
    examples: dict[str, dict],
    row: dict,
    fold_by_juan: dict[int, int],
) -> dict:
    candidate = {
        "id": str(row["id"]),
        "juan": int(row["juan"]),
        "jie_index": int(row["jie_index"]),
        "para_id": int(row["para_id"]),
        "start": int(row["start"]),
        "end": int(row["end"]),
        "surface": str(row["surface"]),
    }
    example = examples.get(candidate["id"])
    if example is None:
        raise ValueError("revision14 two-stage candidate example missing")
    if (
        int(example["juan"]) != candidate["juan"]
        or int(example["jie_index"]) != candidate["jie_index"]
    ):
        raise ValueError("revision14 two-stage candidate jie binding differs")
    if fold_by_juan.get(candidate["juan"]) is None:
        raise ValueError("revision14 two-stage candidate juan outside folds")
    return candidate


def _assemble_inventory(
    examples: dict[str, dict],
    references_rows: list[dict],
    existence_rows: list[dict],
    rank_pairs_rows: list[dict],
    easy_negatives_rows: list[dict],
    mined_boundaries_rows: list[dict],
    candidate_lattice_rows: list[dict],
    fold_by_juan: dict[int, int],
) -> dict:
    """Assemble and validate disjoint class inventory from frozen files."""
    # Exact references
    exact_by_geometry = {}
    for row in references_rows:
        candidate = _validate_candidate(examples, row, fold_by_juan)
        key = _geometry(candidate)
        if key in exact_by_geometry:
            raise ValueError("revision14 two-stage duplicate reference")
        exact_by_geometry[key] = candidate
    if len(exact_by_geometry) != EXPECTED_REFERENCES:
        raise ValueError(
            f"revision14 two-stage reference count {len(exact_by_geometry)} "
            f"!= {EXPECTED_REFERENCES}"
        )

    # Existence file -> semantic negatives (label=0)
    existence_by_geometry = {}
    semantic_by_geometry = {}
    for row in existence_rows:
        candidate = _validate_candidate(examples, row, fold_by_juan)
        key = _geometry(candidate)
        if key in existence_by_geometry:
            raise ValueError("revision14 two-stage duplicate existence row")
        existence_by_geometry[key] = {**candidate, "label": int(row["label"])}
        if int(row["label"]) == 0:
            semantic_by_geometry[key] = candidate
    if len(existence_by_geometry) != EXPECTED_EXISTENCE:
        raise ValueError("revision14 two-stage existence count differs")
    if len(semantic_by_geometry) != EXPECTED_SEMANTIC:
        raise ValueError("revision14 two-stage semantic-negative count differs")

    # Rank pairs -> boundary alternatives (unique negatives)
    boundary_by_geometry = {}
    pair_records = []
    for pair in rank_pairs_rows:
        positive = _validate_candidate(examples, pair["positive"], fold_by_juan)
        negative = _validate_candidate(examples, pair["negative"], fold_by_juan)
        pos_key = _geometry(positive)
        neg_key = _geometry(negative)
        if pos_key not in exact_by_geometry:
            raise ValueError("revision14 two-stage rank positive not a reference")
        if neg_key in exact_by_geometry:
            raise ValueError("revision14 two-stage rank negative is a reference")
        boundary_by_geometry[neg_key] = negative
        pair_records.append((pos_key, neg_key))
    if len(boundary_by_geometry) != EXPECTED_BOUNDARY:
        raise ValueError(
            f"revision14 two-stage boundary count {len(boundary_by_geometry)} "
            f"!= {EXPECTED_BOUNDARY}"
        )
    pair_set = set(pair_records)
    if (
        len(pair_records) != EXPECTED_RANK_PAIRS
        or len(pair_set) != EXPECTED_RANK_PAIRS
    ):
        raise ValueError("revision14 two-stage rank-pair count differs")
    exact_by_paragraph: dict[tuple[str, int], list[tuple]] = {}
    for key in exact_by_geometry:
        exact_by_paragraph.setdefault((key[0], key[1]), []).append(key)
    exhaustive_pairs = {
        (exact_key, boundary_key)
        for boundary_key in boundary_by_geometry
        for exact_key in exact_by_paragraph.get(
            (boundary_key[0], boundary_key[1]), []
        )
        if _overlaps(exact_key, boundary_key)
    }
    if pair_set != exhaustive_pairs:
        raise ValueError("revision14 two-stage rank pairs are not exhaustive")

    # Easy negatives
    easy_by_geometry = {}
    for row in easy_negatives_rows:
        candidate = _validate_candidate(examples, row, fold_by_juan)
        key = _geometry(candidate)
        if key in easy_by_geometry:
            raise ValueError("revision14 two-stage duplicate easy negative")
        easy_by_geometry[key] = candidate
    if len(easy_by_geometry) != EXPECTED_EASY:
        raise ValueError("revision14 two-stage easy-negative count differs")

    # Mined boundaries (subset of boundary)
    mined_by_geometry = {}
    for row in mined_boundaries_rows:
        candidate = _validate_candidate(examples, row, fold_by_juan)
        key = _geometry(candidate)
        mined_by_geometry[key] = candidate
    if len(mined_by_geometry) != EXPECTED_MINED_BOUNDARY:
        raise ValueError("revision14 two-stage mined-boundary count differs")

    # Validate disjoint classes
    exact_set = set(exact_by_geometry)
    boundary_set = set(boundary_by_geometry)
    semantic_set = set(semantic_by_geometry)
    easy_set = set(easy_by_geometry)
    if exact_set & boundary_set:
        raise ValueError("revision14 two-stage exact/boundary overlap")
    if exact_set & semantic_set:
        raise ValueError("revision14 two-stage exact/semantic overlap")
    if exact_set & easy_set:
        raise ValueError("revision14 two-stage exact/easy overlap")
    if boundary_set & semantic_set:
        raise ValueError("revision14 two-stage boundary/semantic overlap")
    if boundary_set & easy_set:
        raise ValueError("revision14 two-stage boundary/easy overlap")
    if semantic_set & easy_set:
        raise ValueError("revision14 two-stage semantic/easy overlap")

    candidate_lattice = {}
    for row in candidate_lattice_rows:
        candidate = _validate_candidate(examples, row, fold_by_juan)
        key = _geometry(candidate)
        if key in candidate_lattice:
            raise ValueError("revision14 two-stage duplicate lattice geometry")
        candidate_lattice[key] = candidate
    if len(candidate_lattice) != EXPECTED_LATTICE:
        raise ValueError("revision14 two-stage complete lattice count differs")
    classified = exact_set | boundary_set | semantic_set | easy_set
    if not classified.issubset(candidate_lattice):
        raise ValueError("revision14 two-stage class is outside complete lattice")
    reconciled_nonoverlap = set(candidate_lattice) - classified
    if len(reconciled_nonoverlap) != EXPECTED_RECONCILED_NONOVERLAP:
        raise ValueError(
            "revision14 two-stage reconciled non-overlap count differs"
        )
    if any(
        any(_overlaps(key, exact_key) for exact_key in exact_set)
        for key in reconciled_nonoverlap
    ):
        raise ValueError("revision14 reconciled non-overlap still overlaps reference")

    # Build ordered candidate list
    rows = []
    existence_labels = {}  # 1=positive, 0=negative

    # exact -> existence positive
    for key, candidate in sorted(
        exact_by_geometry.items(), key=lambda item: _ordered_key(item[1])
    ):
        rows.append({
            **candidate,
            "class": "exact_reference",
            "existence_label": 1,
        })
        existence_labels[key] = 1

    # boundary -> existence positive
    for key, candidate in sorted(
        boundary_by_geometry.items(), key=lambda item: _ordered_key(item[1])
    ):
        rows.append({
            **candidate,
            "class": "boundary_alternative",
            "existence_label": 1,
        })
        existence_labels[key] = 1

    # semantic -> existence negative
    for key, candidate in sorted(
        semantic_by_geometry.items(), key=lambda item: _ordered_key(item[1])
    ):
        rows.append({
            **candidate,
            "class": "semantic_negative",
            "existence_label": 0,
        })
        existence_labels[key] = 0

    # easy -> existence negative
    for key, candidate in sorted(
        easy_by_geometry.items(), key=lambda item: _ordered_key(item[1])
    ):
        rows.append({
            **candidate,
            "class": "easy_negative",
            "existence_label": 0,
        })
        existence_labels[key] = 0

    for key in sorted(
        reconciled_nonoverlap,
        key=lambda value: _ordered_key(candidate_lattice[value]),
    ):
        candidate = candidate_lattice[key]
        rows.append({
            **candidate,
            "class": "reconciled_nonoverlap",
            "existence_label": 0,
        })
        existence_labels[key] = 0

    total = len(rows)
    n_exact = len(exact_by_geometry)
    n_boundary = len(boundary_by_geometry)
    n_semantic = len(semantic_by_geometry)
    n_easy = len(easy_by_geometry)
    if total != (
        n_exact
        + n_boundary
        + n_semantic
        + n_easy
        + EXPECTED_RECONCILED_NONOVERLAP
    ):
        raise ValueError("revision14 two-stage class cardinality differs")

    fold_ids = np.asarray(
        [fold_by_juan[int(row["juan"])] for row in rows], dtype=np.int64
    )
    binary_labels = np.asarray(
        [int(row["existence_label"]) for row in rows], dtype=np.float32
    )
    classes = [str(row["class"]) for row in rows]
    if len(rows) != EXPECTED_LATTICE:
        raise ValueError("revision14 two-stage ordered lattice count differs")

    # Real candidate labels for precision (from existence.jsonl)
    real_labels = np.full(len(rows), -1, dtype=np.int8)
    row_geometry_to_index = {_geometry(row): i for i, row in enumerate(rows)}
    for key, entry in existence_by_geometry.items():
        idx = row_geometry_to_index.get(key)
        if idx is not None:
            if key in exact_by_geometry:
                real_labels[idx] = 1
            else:
                real_labels[idx] = 0

    return {
        "rows": rows,
        "binary_labels": binary_labels,
        "classes": classes,
        "fold_ids": fold_ids,
        "real_labels": real_labels,
        "fold_by_juan": fold_by_juan,
        "exact_by_geometry": exact_by_geometry,
        "boundary_by_geometry": boundary_by_geometry,
        "semantic_by_geometry": semantic_by_geometry,
        "easy_by_geometry": easy_by_geometry,
        "pair_records": pair_records,
        "row_geometry_to_index": row_geometry_to_index,
    }


def _build_binary_head(encoder_dir: Path, device):
    """Stage-1: encoder + single-logit scalar binary head."""
    import torch
    from torch import nn
    from transformers import AutoModel

    class BinaryExistenceClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(encoder_dir)
            self.encoder.gradient_checkpointing_enable()
            hidden_size = int(self.encoder.config.hidden_size)
            self.dropout = nn.Dropout(DROPOUT)
            self.head = nn.Linear(hidden_size * 3, 1)

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
            return self.head(self.dropout(pooled)).squeeze(-1)

    return BinaryExistenceClassifier().to(device)


def _fit_stage1(
    records: list[dict],
    binary_labels: np.ndarray,
    train_indices: np.ndarray,
    encoder_dir: Path,
    device,
    row_weights: np.ndarray | None = None,
) -> object:
    """Train Stage-1 binary existence classifier with uniform per-row loss."""
    import torch

    _seed_everything(SEED)
    model = _build_binary_head(encoder_dir, device)
    from transformers import Adafactor

    optimizer = Adafactor(
        [
            {
                "params": model.encoder.parameters(),
                "lr": ENCODER_LEARNING_RATE,
            },
            {
                "params": model.head.parameters(),
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
    weighted_loss = row_weights is not None
    if row_weights is None:
        row_weights = np.ones(len(binary_labels), dtype=np.float32)
    if (
        row_weights.shape != binary_labels.shape
        or not np.isfinite(row_weights).all()
        or np.any(row_weights <= 0)
    ):
        raise ValueError("revision14 Stage-1 row weights are invalid")
    group_size = PHYSICAL_BATCH_SIZE * ACCUMULATION_STEPS
    for _ in range(EPOCHS):
        ordered = generator.permutation(train_indices)
        model.train()
        for group_start in range(0, len(ordered), group_size):
            group = ordered[group_start:group_start + group_size]
            model.zero_grad(set_to_none=True)
            divisor = len(group)
            weight_divisor = float(row_weights[group].sum())
            for start in range(0, divisor, PHYSICAL_BATCH_SIZE):
                indices = group[start:start + PHYSICAL_BATCH_SIZE]
                logits = model(*_collate(records, indices, device))
                targets = (
                    torch.from_numpy(binary_labels[indices])
                    .float()
                    .to(device)
                )
                if weighted_loss:
                    losses = (
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            logits, targets, reduction="none"
                        )
                    )
                    weights = (
                        torch.from_numpy(row_weights[indices]).float().to(device)
                    )
                    ((losses * weights).sum() / weight_divisor).backward()
                else:
                    loss = (
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            logits, targets, reduction="mean"
                        )
                    )
                    (loss / (divisor / len(indices))).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), MAX_GRAD_NORM, foreach=False
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    model.eval()
    return model


def _score_stage1(
    model, records: list[dict], indices: np.ndarray, device
) -> np.ndarray:
    """Score Stage-1: return sigmoid probabilities."""
    import torch

    scores = np.zeros(len(indices), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), PHYSICAL_BATCH_SIZE):
            batch_indices = indices[start:start + PHYSICAL_BATCH_SIZE]
            logits = model(*_collate(records, batch_indices, device))
            probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            scores[start:start + len(batch_indices)] = probabilities
    if not np.isfinite(scores).all():
        raise ValueError("revision14 Stage-1 non-finite scores")
    return scores


def _build_scalar_head(encoder_dir: Path, device):
    """Stage-2: encoder + scalar ranking head."""
    import torch
    from torch import nn
    from transformers import AutoModel

    class ScalarRankingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(encoder_dir)
            self.encoder.gradient_checkpointing_enable()
            hidden_size = int(self.encoder.config.hidden_size)
            self.dropout = nn.Dropout(DROPOUT)
            self.head = nn.Linear(hidden_size * 3, 1)

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
            return self.head(self.dropout(pooled)).squeeze(-1)

    return ScalarRankingModel().to(device)


def _fit_stage2(
    records: list[dict],
    train_pair_indices: list[tuple[int, int]],
    encoder_dir: Path,
    device,
) -> object:
    """Train Stage-2 margin-ranking model with uniform per-pair loss."""
    import torch

    _seed_everything(SEED)
    model = _build_scalar_head(encoder_dir, device)
    from transformers import Adafactor

    optimizer = Adafactor(
        [
            {
                "params": model.encoder.parameters(),
                "lr": ENCODER_LEARNING_RATE,
            },
            {
                "params": model.head.parameters(),
                "lr": CLASSIFIER_LEARNING_RATE,
            },
        ],
        weight_decay=WEIGHT_DECAY,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        beta1=None,
    )
    pair_array = np.asarray(train_pair_indices, dtype=np.int64)
    generator = np.random.default_rng(SEED)
    group_size = PHYSICAL_BATCH_SIZE * ACCUMULATION_STEPS
    for _ in range(EPOCHS):
        ordered = generator.permutation(len(pair_array))
        model.train()
        for group_start in range(0, len(ordered), group_size):
            group = ordered[group_start:group_start + group_size]
            model.zero_grad(set_to_none=True)
            divisor = len(group)
            for start in range(0, divisor, PHYSICAL_BATCH_SIZE):
                batch = group[start:start + PHYSICAL_BATCH_SIZE]
                pos_indices = pair_array[batch, 0]
                neg_indices = pair_array[batch, 1]
                pos_scores = model(
                    *_collate(records, pos_indices, device)
                )
                neg_scores = model(
                    *_collate(records, neg_indices, device)
                )
                # hinge loss: max(0, margin - pos + neg)
                loss = torch.clamp(
                    MARGIN - pos_scores + neg_scores, min=0.0
                ).mean()
                (loss / (divisor / len(batch))).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), MAX_GRAD_NORM, foreach=False
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    model.eval()
    return model


def _score_stage2(
    model, records: list[dict], indices: np.ndarray, device
) -> np.ndarray:
    """Score Stage-2: return raw scalar scores."""
    import torch

    scores = np.zeros(len(indices), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), PHYSICAL_BATCH_SIZE):
            batch_indices = indices[start:start + PHYSICAL_BATCH_SIZE]
            raw = model(*_collate(records, batch_indices, device))
            scores[start:start + len(batch_indices)] = (
                raw.cpu().numpy().astype(np.float32)
            )
    if not np.isfinite(scores).all():
        raise ValueError("revision14 Stage-2 non-finite scores")
    return scores


def overlap_components(
    candidates: list[dict],
) -> list[list[int]]:
    """Find connected components by same-paragraph interval overlap.

    Each candidate dict must have 'id' (jie id), 'para_id', 'start', 'end'.
    Returns list of components, each a sorted list of original indices.
    """
    paragraphs: dict[tuple[str, int], list[int]] = {}
    for index, candidate in enumerate(candidates):
        paragraphs.setdefault(
            (str(candidate["id"]), int(candidate["para_id"])), []
        ).append(index)

    components = []
    for paragraph in sorted(paragraphs):
        indices = sorted(
            paragraphs[paragraph],
            key=lambda index: (
                int(candidates[index]["start"]),
                int(candidates[index]["end"]),
                index,
            ),
        )
        current = []
        maximum_end = None
        for index in indices:
            start = int(candidates[index]["start"])
            end = int(candidates[index]["end"])
            if current and start >= maximum_end:
                components.append(sorted(current))
                current = []
                maximum_end = None
            current.append(index)
            maximum_end = end if maximum_end is None else max(maximum_end, end)
        if current:
            components.append(sorted(current))
    return components


def select_from_components(
    candidates: list[dict],
    stage2_scores: np.ndarray,
    exact_geometry_set: set,
) -> list[int]:
    """For each component, select highest Stage-2 score; tie-break by geometry order."""
    components = overlap_components(candidates)
    selected = []
    for component in components:
        best_idx = None
        best_score = None
        best_key = None
        for idx in component:
            score = float(stage2_scores[idx])
            geo_key = _ordered_key(candidates[idx])
            if best_idx is None or score > best_score or (
                score == best_score and geo_key < best_key
            ):
                best_idx = idx
                best_score = score
                best_key = geo_key
        selected.append(best_idx)
    return selected


def select_greedy_nonoverlap(
    candidates: list[dict],
    scores: np.ndarray,
    eligible_indices: list[int],
) -> list[int]:
    """Greedily retain ranked, pairwise non-overlapping paragraph candidates."""
    selected = []
    selected_by_paragraph: dict[tuple[str, int], list[int]] = {}
    ordered = sorted(
        eligible_indices,
        key=lambda index: (
            -float(scores[index]),
            _ordered_key(candidates[index]),
        ),
    )
    for index in ordered:
        candidate = candidates[index]
        paragraph = (str(candidate["id"]), int(candidate["para_id"]))
        if any(
            int(candidate["start"]) < int(candidates[prior]["end"])
            and int(candidates[prior]["start"]) < int(candidate["end"])
            for prior in selected_by_paragraph.get(paragraph, [])
        ):
            continue
        selected.append(index)
        selected_by_paragraph.setdefault(paragraph, []).append(index)
    return selected


def end_to_end_metrics(
    rows: list[dict],
    stage1_scores: np.ndarray,
    stage2_scores: np.ndarray,
    classes: list[str],
    fold_ids: np.ndarray,
    real_labels: np.ndarray,
    exact_geometry_set: set,
    boundary_geometry_set: set,
    greedy_resolution: bool = False,
) -> list[dict]:
    """Compute end-to-end metrics for all 15 thresholds."""
    n = len(rows)
    exact_mask = np.asarray(
        [c == "exact_reference" for c in classes], dtype=bool
    )
    boundary_mask = np.asarray(
        [c == "boundary_alternative" for c in classes], dtype=bool
    )
    semantic_mask = np.asarray(
        [c == "semantic_negative" for c in classes], dtype=bool
    )
    easy_mask = np.asarray(
        [c == "easy_negative" for c in classes], dtype=bool
    )
    # existence-positive = exact | boundary
    existence_positive_mask = exact_mask | boundary_mask
    real_mask = real_labels >= 0
    real_positive_mask = real_labels == 1
    components = overlap_components(rows)
    boundary_components = [
        component
        for component in components
        if any(exact_mask[i] for i in component)
        and any(boundary_mask[i] for i in component)
    ]

    table = []
    for threshold in THRESHOLDS:
        admitted = stage1_scores >= np.float32(threshold)

        # Stage-1 metrics
        s1_overlap_recall = (
            float(np.mean(admitted[existence_positive_mask]))
            if existence_positive_mask.any() else 0.0
        )
        s1_semantic_rejection = (
            float(np.mean(~admitted[semantic_mask]))
            if semantic_mask.any() else None
        )
        s1_easy_rejection = (
            float(np.mean(~admitted[easy_mask]))
            if easy_mask.any() else None
        )
        fold_overlap_recalls = []
        for fold in range(FOLDS):
            fold_mask = fold_ids == fold
            fold_pos = existence_positive_mask & fold_mask
            if fold_pos.any():
                fold_overlap_recalls.append(
                    float(np.mean(admitted[fold_pos]))
                )
        s1_min_fold_overlap_recall = (
            min(fold_overlap_recalls) if fold_overlap_recalls else None
        )

        if greedy_resolution:
            selected_global_indices = select_greedy_nonoverlap(
                rows,
                stage2_scores,
                np.flatnonzero(admitted).tolist(),
            )
        else:
            selected_global_indices = []
            for component in components:
                eligible = [i for i in component if admitted[i]]
                if not eligible:
                    continue
                selected_global_indices.append(
                    min(
                        eligible,
                        key=lambda i: (
                            -float(stage2_scores[i]),
                            _ordered_key(rows[i]),
                        ),
                    )
                )

        selected_set = set(selected_global_indices)
        exact_indices = {i for i in range(n) if exact_mask[i]}

        # End-to-end exact recall
        exact_selected = len(exact_indices & selected_set)
        e2e_exact_recall = (
            exact_selected / len(exact_indices)
            if exact_indices else 0.0
        )

        # Real precision
        real_selected = [i for i in selected_global_indices if real_labels[i] >= 0]
        real_tp = sum(1 for i in real_selected if real_labels[i] == 1)
        e2e_real_precision = (
            real_tp / len(real_selected) if real_selected else 0.0
        )

        # Evaluate against prospectively fixed full-lattice boundary components.
        boundary_component_total = len(boundary_components)
        boundary_component_correct = 0
        for component in boundary_components:
            winners = [i for i in component if i in selected_set]
            component_exact = {
                i for i in component
                if i in exact_indices and admitted[i]
            }
            if greedy_resolution:
                overlapping_nonexact = any(
                    winner not in exact_indices
                    and any(
                        int(rows[winner]["start"]) < int(rows[exact]["end"])
                        and int(rows[exact]["start"]) < int(rows[winner]["end"])
                        for exact in component_exact
                    )
                    for winner in winners
                )
                correct = (
                    component_exact.issubset(selected_set)
                    and not overlapping_nonexact
                )
            else:
                correct = len(winners) == 1 and winners[0] in exact_indices
            if correct:
                boundary_component_correct += 1
        e2e_boundary_accuracy = (
            boundary_component_correct / boundary_component_total
            if boundary_component_total else None
        )

        # Min-fold exact recall
        fold_exact_recalls = []
        for fold in range(FOLDS):
            fold_exact = [
                i for i in exact_indices if fold_ids[i] == fold
            ]
            if fold_exact:
                fold_exact_recalls.append(
                    sum(1 for i in fold_exact if i in selected_set)
                    / len(fold_exact)
                )
        e2e_min_fold_exact_recall = (
            min(fold_exact_recalls) if fold_exact_recalls else None
        )

        # Gate checks
        stage1_pass = (
            s1_overlap_recall >= STAGE1_OVERLAP_RECALL_MIN
            and s1_semantic_rejection is not None
            and s1_semantic_rejection >= STAGE1_SEMANTIC_REJECTION_MIN
            and s1_easy_rejection is not None
            and s1_easy_rejection >= STAGE1_EASY_REJECTION_MIN
            and s1_min_fold_overlap_recall is not None
            and s1_min_fold_overlap_recall >= STAGE1_MIN_FOLD_OVERLAP_RECALL
        )
        e2e_pass = (
            e2e_exact_recall >= E2E_EXACT_RECALL_MIN
            and e2e_real_precision >= E2E_REAL_PRECISION_MIN
            and (
                e2e_boundary_accuracy is not None
                and e2e_boundary_accuracy >= E2E_BOUNDARY_COMPONENT_ACCURACY_MIN
            )
            and e2e_min_fold_exact_recall is not None
            and e2e_min_fold_exact_recall >= E2E_MIN_FOLD_EXACT_RECALL
        )
        eligible = stage1_pass and e2e_pass

        table.append({
            "threshold": float(threshold),
            "stage1_overlap_recall": s1_overlap_recall,
            "stage1_semantic_rejection": s1_semantic_rejection,
            "stage1_easy_rejection": s1_easy_rejection,
            "stage1_min_fold_overlap_recall": s1_min_fold_overlap_recall,
            "e2e_exact_recall": e2e_exact_recall,
            "e2e_real_precision": e2e_real_precision,
            "e2e_boundary_component_accuracy": e2e_boundary_accuracy,
            "e2e_boundary_component_total": boundary_component_total,
            "e2e_boundary_component_correct": boundary_component_correct,
            "e2e_min_fold_exact_recall": e2e_min_fold_exact_recall,
            "e2e_selected_count": len(selected_global_indices),
            "e2e_admitted_count": int(np.sum(admitted)),
            "fold_overlap_recalls": fold_overlap_recalls,
            "fold_exact_recalls": fold_exact_recalls,
            "stage1_pass": stage1_pass,
            "e2e_pass": e2e_pass,
            "eligible": eligible,
        })
    return table


def _validate_sha_bindings(manifest: dict, keys: tuple[str, ...]) -> None:
    bindings = manifest.get("bindings", {})
    for key in keys:
        value = bindings.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(
                f"revision14 two-stage manifest binding differs: {key}"
            )


def _load_inputs(
    inventory_root: Path,
    grouped_root: Path,
    revision9_root: Path,
) -> tuple[dict, dict, list[dict], dict[str, list[dict]], Path]:
    inventory_manifest_path = inventory_root / "manifest.json"
    grouped_manifest_path = grouped_root / "manifest.json"
    inventory_manifest = _read(inventory_manifest_path)
    grouped_manifest = _read(grouped_manifest_path)
    examples_path = grouped_root / "examples.jsonl"

    if (
        inventory_manifest.get("status") != INVENTORY_STATUS
        or inventory_manifest.get("revision") != REVISION
        or inventory_manifest.get("confirmation_read") is not False
    ):
        raise ValueError("revision14 two-stage inventory binding differs")

    _validate_sha_bindings(
        inventory_manifest,
        (
            "grouped_manifest_sha256",
            "examples_sha256",
        ),
    )
    if (
        inventory_manifest["bindings"]["grouped_manifest_sha256"]
        != _sha256(grouped_manifest_path)
        or inventory_manifest["bindings"]["examples_sha256"]
        != _sha256(examples_path)
    ):
        raise ValueError("revision14 two-stage inventory grouped binding differs")

    from production_precision_revision14_inventory import OUTPUT_FILES

    inventory_rows = {}
    for key, name in OUTPUT_FILES.items():
        path = inventory_root / name
        expected = inventory_manifest.get("outputs", {}).get(f"{key}_sha256")
        if (
            not path.is_file()
            or not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
            or expected != _sha256(path)
        ):
            raise ValueError(
                f"revision14 two-stage output binding differs: {name}"
            )
        inventory_rows[key] = _read_jsonl(path)

    if (
        grouped_manifest.get("status") != GROUPED_DATA_STATUS
        or grouped_manifest.get("confirmation_read") is not False
        or grouped_manifest.get("outputs", {}).get("examples_sha256")
        != _sha256(examples_path)
    ):
        raise ValueError("revision14 two-stage grouped-data binding differs")
    examples = _read_jsonl(examples_path)

    encoder_dir = revision9_root / "encoder"
    encoder_artifact = _model_artifact(encoder_dir)
    if encoder_artifact != REVISION9_BASE_ENCODER:
        raise ValueError("revision14 two-stage Revision-9 base binding differs")
    return (
        inventory_manifest,
        grouped_manifest,
        examples,
        inventory_rows,
        encoder_dir,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _class_balanced_weights(
    labels: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    counts = {
        value: int(np.sum(labels[train_indices] == value))
        for value in (0.0, 1.0)
    }
    if not all(counts.values()):
        raise ValueError("Stage-1 training fold must contain both classes")
    total = len(train_indices)
    class_weights = {
        value: total / (2.0 * count) for value, count in counts.items()
    }
    weights = np.ones(len(labels), dtype=np.float32)
    for value, weight in class_weights.items():
        weights[
            train_indices[labels[train_indices] == value]
        ] = np.float32(weight)
    return weights, {
        "positive_rows": counts[1.0],
        "negative_rows": counts[0.0],
        "positive_weight": class_weights[1.0],
        "negative_weight": class_weights[0.0],
    }


def _three_stratum_weights(
    labels: np.ndarray,
    classes: list[str],
    train_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    strata = {
        "positive": np.asarray(
            [labels[index] == 1 for index in train_indices], dtype=bool
        ),
        "semantic_negative": np.asarray(
            [classes[index] == "semantic_negative" for index in train_indices],
            dtype=bool,
        ),
        "structural_negative": np.asarray(
            [
                classes[index] in {"easy_negative", "reconciled_nonoverlap"}
                for index in train_indices
            ],
            dtype=bool,
        ),
    }
    counts = {name: int(mask.sum()) for name, mask in strata.items()}
    if not all(counts.values()):
        raise ValueError("Stage-1 training fold must contain all three strata")
    if any(
        int(sum(mask[position] for mask in strata.values())) != 1
        for position in range(len(train_indices))
    ):
        raise ValueError("Stage-1 training row has invalid stratum ownership")
    target_masses = {
        "positive": 0.5,
        "semantic_negative": 0.25,
        "structural_negative": 0.25,
    }
    weights = np.ones(len(labels), dtype=np.float32)
    inventory: dict[str, float | int] = {}
    for name, mask in strata.items():
        weight = target_masses[name] / counts[name]
        weights[train_indices[mask]] = np.float32(weight)
        inventory[f"{name}_rows"] = counts[name]
        inventory[f"{name}_target_mass"] = target_masses[name]
        inventory[f"{name}_weight"] = weight
    return weights, inventory


def _stage1_training_indices(
    all_indices: np.ndarray,
    classes: list[str],
    real_labels: np.ndarray,
    *,
    real_only: bool,
    structural_negatives: bool,
) -> np.ndarray:
    if structural_negatives:
        return np.asarray(
            [
                index for index in all_indices
                if real_labels[index] >= 0
                or classes[index] in {
                    "easy_negative",
                    "reconciled_nonoverlap",
                }
            ],
            dtype=np.int64,
        )
    if real_only:
        return all_indices[real_labels[all_indices] >= 0]
    return all_indices


def _validate_augmentation_candidate(
    examples: dict[str, dict],
    row: dict,
) -> dict:
    candidate = {
        "id": str(row["id"]),
        "juan": int(row["juan"]),
        "jie_index": int(row["jie_index"]),
        "para_id": int(row["para_id"]),
        "start": int(row["start"]),
        "end": int(row["end"]),
        "surface": str(row["surface"]),
    }
    example = examples.get(candidate["id"])
    if (
        example is None
        or int(example["juan"]) != candidate["juan"]
        or int(example["jie_index"]) != candidate["jie_index"]
    ):
        raise ValueError("two-stage augmentation candidate jie differs")
    segment = next(
        (
            value for value in example["segments"]
            if int(value["para_id"]) == candidate["para_id"]
        ),
        None,
    )
    if segment is None:
        raise ValueError("two-stage augmentation paragraph differs")
    paragraph = example["text"][
        int(segment["assembled_start"]):int(segment["assembled_end"])
    ]
    if (
        not 0 <= candidate["start"] < candidate["end"] <= len(paragraph)
        or paragraph[candidate["start"]:candidate["end"]]
        != candidate["surface"]
    ):
        raise ValueError("two-stage augmentation source geometry differs")
    return candidate


def _prepare_training_augmentation(
    inventory: dict,
    base_examples: dict[str, dict],
    augmentation_examples: list[dict],
    exact_rows: list[dict],
    semantic_rows: list[dict],
    rank_pairs: list[dict],
    *,
    boundary_stage1_positive: bool = False,
) -> dict:
    examples = dict(base_examples)
    for example in augmentation_examples:
        example_id = str(example["id"])
        prior = examples.get(example_id)
        if prior is not None:
            for key in ("id", "juan", "jie_index", "text", "segments"):
                if prior[key] != example[key]:
                    raise ValueError(
                        "two-stage augmentation example binding differs"
                    )
        else:
            examples[example_id] = example

    rows = list(inventory["rows"])
    binary_labels = inventory["binary_labels"].tolist()
    classes = list(inventory["classes"])
    weight_classes = list(classes)
    real_labels = inventory["real_labels"].tolist()
    row_by_geometry = {
        _geometry(row): index for index, row in enumerate(rows)
    }
    if len(row_by_geometry) != len(rows):
        raise ValueError("two-stage augmentation base geometry differs")
    stage1_indices = set()
    augmentation_positive_indices = set()
    augmentation_semantic_indices = set()
    augmentation_boundary_indices = set()
    added_indices = set()
    exact_geometries = set()

    for source in exact_rows:
        candidate = _validate_augmentation_candidate(examples, source)
        geometry = _geometry(candidate)
        if geometry in exact_geometries:
            raise ValueError("two-stage augmentation duplicate exact geometry")
        exact_geometries.add(geometry)
        index = row_by_geometry.get(geometry)
        if index is not None:
            if binary_labels[index] != 1:
                raise ValueError(
                    "two-stage augmentation exact conflicts with base label"
                )
        else:
            index = len(rows)
            row_by_geometry[geometry] = index
            rows.append({
                **candidate,
                "class": "reviewed_exact_reference",
                "existence_label": 1,
            })
            binary_labels.append(1.0)
            classes.append("reviewed_exact_reference")
            weight_classes.append("reviewed_exact_reference")
            real_labels.append(-1)
            added_indices.add(index)
        stage1_indices.add(index)
        augmentation_positive_indices.add(index)

    semantic_geometries = set()
    for source in semantic_rows:
        candidate = _validate_augmentation_candidate(examples, source)
        geometry = _geometry(candidate)
        if (
            geometry in semantic_geometries
            or geometry in exact_geometries
        ):
            raise ValueError(
                "two-stage augmentation semantic geometry conflicts"
            )
        semantic_geometries.add(geometry)
        index = row_by_geometry.get(geometry)
        if index is not None:
            if binary_labels[index] != 0:
                raise ValueError(
                    "two-stage augmentation semantic conflicts with base label"
                )
            weight_classes[index] = "semantic_negative"
        else:
            index = len(rows)
            row_by_geometry[geometry] = index
            rows.append({
                **candidate,
                "class": "semantic_negative",
                "existence_label": 0,
            })
            binary_labels.append(0.0)
            classes.append("semantic_negative")
            weight_classes.append("semantic_negative")
            real_labels.append(-1)
            added_indices.add(index)
        stage1_indices.add(index)
        augmentation_semantic_indices.add(index)

    augmentation_pairs = []
    pair_keys = set()
    for pair in rank_pairs:
        positive = _validate_augmentation_candidate(
            examples, pair["positive"]
        )
        negative = _validate_augmentation_candidate(
            examples, pair["negative"]
        )
        positive_geometry = _geometry(positive)
        negative_geometry = _geometry(negative)
        if (
            positive_geometry not in exact_geometries
            or positive["id"] != negative["id"]
            or positive["para_id"] != negative["para_id"]
            or positive["juan"] != negative["juan"]
            or not _overlaps(positive_geometry, negative_geometry)
            or positive_geometry == negative_geometry
        ):
            raise ValueError("two-stage augmentation rank pair differs")
        positive_index = row_by_geometry[positive_geometry]
        negative_index = row_by_geometry.get(negative_geometry)
        if negative_index is not None:
            if (
                negative_geometry in exact_geometries
                or negative_geometry in inventory["exact_by_geometry"]
            ):
                raise ValueError(
                    "two-stage augmentation rank negative is exact"
                )
            if (
                boundary_stage1_positive
                and binary_labels[negative_index] != 1
            ):
                raise ValueError(
                    "two-stage augmentation overlap-positive conflicts "
                    "with base label"
                )
        else:
            negative_index = len(rows)
            row_by_geometry[negative_geometry] = negative_index
            rows.append({
                **negative,
                "class": "reviewed_boundary_alternative",
                "existence_label": 1,
            })
            binary_labels.append(1.0)
            classes.append("reviewed_boundary_alternative")
            weight_classes.append("reviewed_boundary_alternative")
            real_labels.append(-1)
            added_indices.add(negative_index)
        if boundary_stage1_positive:
            stage1_indices.add(negative_index)
            augmentation_positive_indices.add(negative_index)
            augmentation_boundary_indices.add(negative_index)
        pair_key = (positive_index, negative_index)
        if pair_key in pair_keys:
            raise ValueError("two-stage augmentation duplicate rank pair")
        pair_keys.add(pair_key)
        augmentation_pairs.append((
            positive_index,
            negative_index,
            int(positive["juan"]),
        ))

    return {
        "examples": examples,
        "rows": rows,
        "binary_labels": np.asarray(binary_labels, dtype=np.float32),
        "classes": classes,
        "weight_classes": weight_classes,
        "real_labels": np.asarray(real_labels, dtype=np.int8),
        "stage1_indices": np.asarray(
            sorted(stage1_indices), dtype=np.int64
        ),
        "positive_indices": np.asarray(
            sorted(augmentation_positive_indices), dtype=np.int64
        ),
        "semantic_indices": np.asarray(
            sorted(augmentation_semantic_indices), dtype=np.int64
        ),
        "boundary_indices": np.asarray(
            sorted(augmentation_boundary_indices), dtype=np.int64
        ),
        "pair_indices": augmentation_pairs,
        "added_indices": np.asarray(
            sorted(added_indices), dtype=np.int64
        ),
    }


def _fold_local_augmentation_indices(
    indices: np.ndarray,
    rows: list[dict],
    fold_by_juan: dict[int, int],
    heldout_fold: int,
) -> np.ndarray:
    return np.asarray(
        [
            int(index) for index in indices
            if fold_by_juan.get(int(rows[int(index)]["juan"]))
            != heldout_fold
        ],
        dtype=np.int64,
    )


def run_revision14_two_stage(
    inventory_root: Path,
    grouped_root: Path,
    revision9_root: Path,
    output_dir: Path,
    *,
    experiment_revision: int = REVISION,
    status_blocked: str = FINETUNE_STATUS_BLOCKED,
    status_selected: str = FINETUNE_STATUS_SELECTED,
    stage1_real_only: bool = False,
    stage1_class_balanced: bool = False,
    stage1_structural_negatives: bool = False,
    stage1_three_stratum_balanced: bool = False,
    greedy_resolution: bool = False,
    augmentation_root: Path | None = None,
    augmentation_status: str | None = None,
    augmentation_boundary_stage1_positive: bool = False,
) -> dict:
    if stage1_class_balanced and stage1_three_stratum_balanced:
        raise ValueError("Stage-1 weighting strategies are mutually exclusive")
    if stage1_three_stratum_balanced and not stage1_structural_negatives:
        raise ValueError(
            "Three-stratum weighting requires structural negatives"
        )
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"revision14 two-stage output exists: {output_dir}"
        )
    (
        inventory_manifest,
        grouped_manifest,
        example_rows,
        inv_rows,
        encoder_dir,
    ) = _load_inputs(inventory_root, grouped_root, revision9_root)

    examples = {str(row["id"]): row for row in example_rows}
    fold_by_juan = _folds([int(row["juan"]) for row in example_rows])

    inventory = _assemble_inventory(
        examples,
        inv_rows["references"],
        inv_rows["existence"],
        inv_rows["rank_pairs"],
        inv_rows["easy_negatives"],
        inv_rows["mined_boundaries"],
        inv_rows["candidate_lattice"],
        fold_by_juan,
    )
    evaluation_rows = inventory["rows"]
    evaluation_binary_labels = inventory["binary_labels"]
    evaluation_classes = inventory["classes"]
    fold_ids = inventory["fold_ids"]
    evaluation_real_labels = inventory["real_labels"]
    exact_by_geometry = inventory["exact_by_geometry"]
    boundary_by_geometry = inventory["boundary_by_geometry"]
    pair_records = inventory["pair_records"]
    row_geometry_to_index = inventory["row_geometry_to_index"]

    # Build pair index list: (positive_row_index, negative_row_index)
    pair_index_list = []
    for pos_key, neg_key in pair_records:
        pos_idx = row_geometry_to_index.get(pos_key)
        neg_idx = row_geometry_to_index.get(neg_key)
        if pos_idx is None or neg_idx is None:
            raise ValueError("revision14 two-stage pair index missing")
        pair_index_list.append((pos_idx, neg_idx))

    augmentation_manifest = None
    augmentation_paths = {}
    augmentation = {
        "examples": examples,
        "rows": evaluation_rows,
        "binary_labels": evaluation_binary_labels,
        "classes": evaluation_classes,
        "weight_classes": evaluation_classes,
        "real_labels": evaluation_real_labels,
        "stage1_indices": np.asarray([], dtype=np.int64),
        "pair_indices": [],
        "added_indices": np.asarray([], dtype=np.int64),
    }
    if augmentation_root is not None:
        if augmentation_status is None:
            raise ValueError("two-stage augmentation status is required")
        augmentation_manifest_path = augmentation_root / "manifest.json"
        augmentation_manifest = _read(augmentation_manifest_path)
        augmentation_names = {
            "examples": "examples.jsonl",
            "exact_additions": "exact-additions.jsonl",
            "semantic_negatives": "semantic-negatives.jsonl",
            "rank_pairs": "rank-pairs.jsonl",
        }
        augmentation_rows = {}
        for key, name in augmentation_names.items():
            path = augmentation_root / name
            if (
                augmentation_manifest.get("outputs", {}).get(
                    f"{key}_sha256"
                )
                != _sha256(path)
            ):
                raise ValueError(
                    "two-stage augmentation output binding differs"
                )
            augmentation_paths[key] = path
            augmentation_rows[key] = _read_jsonl(path)
        if (
            augmentation_manifest.get("status") != augmentation_status
            or augmentation_manifest.get("fit_only") is not True
            or augmentation_manifest.get("confirmation_read") is not False
            or augmentation_manifest.get("formal_reserve_text_read") is not False
            or augmentation_manifest.get("counts", {}).get("conflicts") != 0
        ):
            raise ValueError("two-stage augmentation manifest differs")
        augmentation = _prepare_training_augmentation(
            inventory,
            examples,
            augmentation_rows["examples"],
            augmentation_rows["exact_additions"],
            augmentation_rows["semantic_negatives"],
            augmentation_rows["rank_pairs"],
            boundary_stage1_positive=augmentation_boundary_stage1_positive,
        )

    examples = augmentation["examples"]
    rows = augmentation["rows"]
    binary_labels = augmentation["binary_labels"]
    classes = augmentation["classes"]
    weight_classes = augmentation["weight_classes"]
    real_labels = augmentation["real_labels"]
    evaluation_count = len(evaluation_rows)

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
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("revision14 two-stage requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir, use_fast=True)
    _validate_sentinels(tokenizer, examples)
    records = [
        {
            **row,
            **_tokenize_record(tokenizer, examples[str(row["id"])], row),
        }
        for row in rows
    ]

    all_evaluation_indices = np.arange(evaluation_count, dtype=np.int64)
    oof_stage1 = np.zeros(evaluation_count, dtype=np.float32)
    oof_stage2 = np.zeros(evaluation_count, dtype=np.float32)
    fold_inventory = []

    for fold in range(FOLDS):
        train_indices = np.flatnonzero(fold_ids != fold)
        heldout_indices = np.flatnonzero(fold_ids == fold)
        if not len(train_indices) or not len(heldout_indices):
            raise ValueError("revision14 two-stage fold is empty")

        stage1_train_indices = _stage1_training_indices(
            train_indices,
            evaluation_classes,
            evaluation_real_labels,
            real_only=stage1_real_only,
            structural_negatives=stage1_structural_negatives,
        )
        fold_augmentation_indices = _fold_local_augmentation_indices(
            augmentation["stage1_indices"],
            rows,
            fold_by_juan,
            fold,
        )
        stage1_train_indices = np.asarray(
            sorted(set(stage1_train_indices.tolist()).union(
                fold_augmentation_indices.tolist()
            )),
            dtype=np.int64,
        )
        stage1_weights = np.ones(len(rows), dtype=np.float32)
        stage1_weight_inventory = {
            "positive_rows": int(np.sum(
                binary_labels[stage1_train_indices] == 1
            )),
            "negative_rows": int(np.sum(
                binary_labels[stage1_train_indices] == 0
            )),
            "positive_weight": 1.0,
            "negative_weight": 1.0,
        }
        if stage1_class_balanced:
            stage1_weights, stage1_weight_inventory = (
                _class_balanced_weights(
                    binary_labels,
                    stage1_train_indices,
                )
            )
        elif stage1_three_stratum_balanced:
            stage1_weights, stage1_weight_inventory = (
                _three_stratum_weights(
                    binary_labels,
                    weight_classes,
                    stage1_train_indices,
                )
            )

        # Stage 1: binary existence
        model1 = _fit_stage1(
            records,
            binary_labels,
            stage1_train_indices,
            encoder_dir,
            device,
            row_weights=(
                stage1_weights
                if stage1_class_balanced or stage1_three_stratum_balanced
                else None
            ),
        )
        scores1 = _score_stage1(model1, records, heldout_indices, device)
        oof_stage1[heldout_indices] = scores1
        del model1
        torch.cuda.empty_cache()

        # Stage 2: margin ranking on training-fold pairs only
        train_set = set(train_indices.tolist())
        train_pairs = [
            (pos, neg)
            for pos, neg in pair_index_list
            if pos in train_set and neg in train_set
        ]
        train_pairs.extend(
            (pos, neg)
            for pos, neg, juan in augmentation["pair_indices"]
            if fold_by_juan.get(juan) != fold
        )
        train_pairs = sorted(set(train_pairs))
        if not train_pairs:
            raise ValueError("revision14 two-stage fold has no training pairs")
        model2 = _fit_stage2(records, train_pairs, encoder_dir, device)
        scores2 = _score_stage2(model2, records, heldout_indices, device)
        oof_stage2[heldout_indices] = scores2
        del model2
        torch.cuda.empty_cache()

        fold_inventory.append({
            "fold": fold,
            "train_rows": len(train_indices),
            "augmentation_stage1_rows": len(fold_augmentation_indices),
            "stage1_train_rows": len(stage1_train_indices),
            "stage1_class_weights": stage1_weight_inventory,
            "heldout_rows": len(heldout_indices),
            "train_pairs": len(train_pairs),
            "augmentation_train_pairs": sum(
                fold_by_juan.get(juan) != fold
                for _, _, juan in augmentation["pair_indices"]
            ),
        })

    # Compute metrics
    exact_geometry_set = set(exact_by_geometry)
    boundary_geometry_set = set(boundary_by_geometry)
    table = end_to_end_metrics(
        evaluation_rows,
        oof_stage1,
        oof_stage2,
        evaluation_classes,
        fold_ids,
        evaluation_real_labels,
        exact_geometry_set,
        boundary_geometry_set,
        greedy_resolution=greedy_resolution,
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

    # Write outputs
    inventory_manifest_path = inventory_root / "manifest.json"
    grouped_manifest_path = grouped_root / "manifest.json"

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        inventory_path = staging / "inventory.jsonl"
        _write_jsonl(inventory_path, evaluation_rows)
        pair_inventory_path = staging / "pair-inventory.jsonl"
        _write_jsonl(
            pair_inventory_path,
            [
                {"positive_index": int(pos), "negative_index": int(neg)}
                for pos, neg in pair_index_list
            ],
        )
        tokens_path = staging / "input-tokens.jsonl"
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
                    "slice_start": int(record["slice_start"]),
                    "slice_end": int(record["slice_end"]),
                    "input_ids": record["input_ids"],
                    "attention_mask": record["attention_mask"],
                    "segment_a_indices": record["segment_a_indices"],
                    "occurrence_indices": record["occurrence_indices"],
                }
                for row, record in zip(
                    evaluation_rows, records[:evaluation_count]
                )
            ],
        )
        augmentation_inventory_path = staging / "augmentation-inventory.jsonl"
        _write_jsonl(
            augmentation_inventory_path,
            [
                {
                    "row_index": int(index),
                    **rows[int(index)],
                    "stage1_training": (
                        int(index)
                        in set(augmentation["stage1_indices"].tolist())
                    ),
                }
                for index in augmentation["added_indices"]
            ],
        )
        augmentation_pairs_path = staging / "augmentation-pairs.jsonl"
        _write_jsonl(
            augmentation_pairs_path,
            [
                {
                    "positive_index": int(pos),
                    "negative_index": int(neg),
                    "juan": int(juan),
                }
                for pos, neg, juan in augmentation["pair_indices"]
            ],
        )
        augmentation_tokens_path = staging / "augmentation-input-tokens.jsonl"
        _write_jsonl(
            augmentation_tokens_path,
            [
                {
                    "row_index": int(index),
                    "id": str(rows[int(index)]["id"]),
                    "juan": int(rows[int(index)]["juan"]),
                    "jie_index": int(rows[int(index)]["jie_index"]),
                    "para_id": int(rows[int(index)]["para_id"]),
                    "start": int(rows[int(index)]["start"]),
                    "end": int(rows[int(index)]["end"]),
                    "slice_start": int(records[int(index)]["slice_start"]),
                    "slice_end": int(records[int(index)]["slice_end"]),
                    "input_ids": records[int(index)]["input_ids"],
                    "attention_mask": records[int(index)]["attention_mask"],
                    "segment_a_indices": records[int(index)][
                        "segment_a_indices"
                    ],
                    "occurrence_indices": records[int(index)][
                        "occurrence_indices"
                    ],
                }
                for index in augmentation["added_indices"]
            ],
        )
        stage1_scores_path = staging / "oof-stage1-scores.jsonl"
        _write_jsonl(
            stage1_scores_path,
            [
                {
                    "id": str(row["id"]),
                    "juan": int(row["juan"]),
                    "jie_index": int(row["jie_index"]),
                    "para_id": int(row["para_id"]),
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "class": str(row["class"]),
                    "existence_label": int(row["existence_label"]),
                    "fold": int(fold_ids[i]),
                    "oof_stage1_probability": float(oof_stage1[i]),
                }
                for i, row in enumerate(evaluation_rows)
            ],
        )
        stage2_scores_path = staging / "oof-stage2-scores.jsonl"
        _write_jsonl(
            stage2_scores_path,
            [
                {
                    "id": str(row["id"]),
                    "juan": int(row["juan"]),
                    "jie_index": int(row["jie_index"]),
                    "para_id": int(row["para_id"]),
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "class": str(row["class"]),
                    "fold": int(fold_ids[i]),
                    "oof_stage2_score": float(oof_stage2[i]),
                }
                for i, row in enumerate(evaluation_rows)
            ],
        )

        final_fit = None
        if selected is not None:
            final_stage1_indices = _stage1_training_indices(
                all_evaluation_indices,
                evaluation_classes,
                evaluation_real_labels,
                real_only=stage1_real_only,
                structural_negatives=stage1_structural_negatives,
            )
            final_stage1_indices = np.asarray(
                sorted(set(final_stage1_indices.tolist()).union(
                    augmentation["stage1_indices"].tolist()
                )),
                dtype=np.int64,
            )
            final_stage1_weights = np.ones(len(rows), dtype=np.float32)
            final_stage1_weight_inventory = {
                "positive_rows": int(np.sum(
                    binary_labels[final_stage1_indices] == 1
                )),
                "negative_rows": int(np.sum(
                    binary_labels[final_stage1_indices] == 0
                )),
                "positive_weight": 1.0,
                "negative_weight": 1.0,
            }
            if stage1_class_balanced:
                (
                    final_stage1_weights,
                    final_stage1_weight_inventory,
                ) = _class_balanced_weights(
                    binary_labels,
                    final_stage1_indices,
                )
            elif stage1_three_stratum_balanced:
                (
                    final_stage1_weights,
                    final_stage1_weight_inventory,
                ) = _three_stratum_weights(
                    binary_labels,
                    weight_classes,
                    final_stage1_indices,
                )
            # Final fit on all data
            model1_final = _fit_stage1(
                records,
                binary_labels,
                final_stage1_indices,
                encoder_dir,
                device,
                row_weights=(
                    final_stage1_weights
                    if stage1_class_balanced
                    or stage1_three_stratum_balanced
                    else None
                ),
            )
            stage1_encoder_dir = staging / "stage1-encoder"
            stage1_head_path = staging / "stage1-head.safetensors"
            model1_final.encoder.save_pretrained(
                stage1_encoder_dir, safe_serialization=True
            )
            save_file(
                {
                    key: value.detach().cpu()
                    for key, value in model1_final.head.state_dict().items()
                },
                stage1_head_path,
            )
            del model1_final
            torch.cuda.empty_cache()

            final_pair_indices = sorted(set(
                pair_index_list
                + [
                    (pos, neg)
                    for pos, neg, _ in augmentation["pair_indices"]
                ]
            ))
            model2_final = _fit_stage2(
                records, final_pair_indices, encoder_dir, device
            )
            stage2_encoder_dir = staging / "stage2-encoder"
            stage2_head_path = staging / "stage2-head.safetensors"
            model2_final.encoder.save_pretrained(
                stage2_encoder_dir, safe_serialization=True
            )
            save_file(
                {
                    key: value.detach().cpu()
                    for key, value in model2_final.head.state_dict().items()
                },
                stage2_head_path,
            )
            del model2_final
            torch.cuda.empty_cache()

            final_fit = {
                "stage1_encoder": _model_artifact(stage1_encoder_dir),
                "stage1_head_sha256": _sha256(stage1_head_path),
                "stage2_encoder": _model_artifact(stage2_encoder_dir),
                "stage2_head_sha256": _sha256(stage2_head_path),
                "threshold": selected["threshold"],
                "threshold_role": "eligibility_diagnostic_only",
                "stage1_train_rows": len(final_stage1_indices),
                "stage1_class_weights": final_stage1_weight_inventory,
                "stage2_train_pairs": len(final_pair_indices),
            }

        environment = {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "safetensors_version": safetensors.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_type": torch.cuda.get_device_name(device),
        }
        deterministic_flags = {
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        }
        manifest = {
            "schema_version": 1,
            "status": (
                status_selected
                if selected is not None
                else status_blocked
            ),
            "revision": experiment_revision,
            "formal_grade": False,
            "eligible_for_production": False,
            "fit_only": True,
            "confirmation_read": False,
            "git_commit": git_commit,
            "bindings": {
                "inventory_manifest_sha256": _sha256(inventory_manifest_path),
                "inventory_outputs": {
                    key: inventory_manifest["outputs"][key]
                    for key in inventory_manifest.get("outputs", {})
                    if key.endswith("_sha256")
                },
                "grouped_manifest_sha256": _sha256(grouped_manifest_path),
                "grouped_examples_sha256": grouped_manifest["outputs"][
                    "examples_sha256"
                ],
                "revision9_base_encoder": REVISION9_BASE_ENCODER,
                "ordered_inventory_sha256": _sha256(inventory_path),
                "pair_inventory_sha256": _sha256(pair_inventory_path),
                "input_tokens_sha256": _sha256(tokens_path),
                "oof_stage1_scores_sha256": _sha256(stage1_scores_path),
                "oof_stage2_scores_sha256": _sha256(stage2_scores_path),
                "augmentation_inventory_sha256": _sha256(
                    augmentation_inventory_path
                ),
                "augmentation_pairs_sha256": _sha256(
                    augmentation_pairs_path
                ),
                "augmentation_input_tokens_sha256": _sha256(
                    augmentation_tokens_path
                ),
                **(
                    {
                        "augmentation_manifest_sha256": _sha256(
                            augmentation_root / "manifest.json"
                        ),
                        "augmentation_outputs": {
                            key: _sha256(path)
                            for key, path in augmentation_paths.items()
                        },
                    }
                    if augmentation_manifest is not None
                    else {}
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
                "stage1_loss": (
                    "BCEWithLogitsLoss_fold_three_stratum_balanced"
                    if stage1_three_stratum_balanced
                    else "BCEWithLogitsLoss_fold_class_balanced_real_candidates"
                    if stage1_class_balanced
                    else "BCEWithLogitsLoss_uniform_per_row"
                ),
                "stage1_real_candidates_only": (
                    stage1_real_only and not stage1_structural_negatives
                ),
                "stage1_real_candidates_included": (
                    stage1_real_only or stage1_structural_negatives
                ),
                "stage1_structural_negatives": stage1_structural_negatives,
                "training_augmentation": augmentation_manifest is not None,
                "augmentation_fold_policy": (
                    "exclude_same_heldout_juan_else_train_every_fold"
                    if augmentation_manifest is not None
                    else "absent"
                ),
                "augmentation_boundary_stage1_positive": (
                    augmentation_boundary_stage1_positive
                ),
                "stage2_loss": "margin_ranking_hinge_margin_1.0_uniform_per_pair",
                "overlap_resolution": (
                    "greedy_nonoverlap_rank_score_then_geometry"
                    if greedy_resolution
                    else "one_winner_per_transitive_component"
                ),
                "margin": MARGIN,
                "sentinels": {
                    "left": LEFT_SENTINEL,
                    "right": RIGHT_SENTINEL,
                    "paragraph": PARAGRAPH_SENTINEL,
                    "ids": SENTINEL_IDS,
                },
                "thresholds": list(THRESHOLDS),
                "threshold_comparison": "float32_score_greater_than_or_equal",
                "encoder_frozen_during_fit": False,
                "gradient_checkpointing": True,
                "optimizer_state_device": "cuda",
                "deterministic_flags": deterministic_flags,
                "veto_field": "absent",
            },
            "environment": environment,
            "inventory": {
                "rows": len(evaluation_rows),
                "exact_reference": sum(
                    1 for c in evaluation_classes if c == "exact_reference"
                ),
                "boundary_alternative": sum(
                    1
                    for c in evaluation_classes
                    if c == "boundary_alternative"
                ),
                "semantic_negative": sum(
                    1
                    for c in evaluation_classes
                    if c == "semantic_negative"
                ),
                "stage1_real_candidates": int(np.sum(
                    evaluation_real_labels >= 0
                )),
                "stage1_real_positive": int(np.sum(
                    (evaluation_real_labels >= 0)
                    & (evaluation_binary_labels == 1)
                )),
                "stage1_real_negative": int(np.sum(
                    (evaluation_real_labels >= 0)
                    & (evaluation_binary_labels == 0)
                )),
                "easy_negative": sum(
                    1 for c in evaluation_classes if c == "easy_negative"
                ),
                "total_rank_pairs": len(pair_index_list),
                "candidate_lattice": inventory_manifest.get(
                    "counts", {}
                ).get("candidate_lattice"),
            },
            "augmentation": {
                "enabled": augmentation_manifest is not None,
                "source_exact_rows": (
                    augmentation_manifest.get("counts", {}).get(
                        "unique_exact_additions"
                    )
                    if augmentation_manifest is not None
                    else 0
                ),
                "source_semantic_rows": (
                    augmentation_manifest.get("counts", {}).get(
                        "not_person_decisions"
                    )
                    if augmentation_manifest is not None
                    else 0
                ),
                "source_rank_pairs": (
                    augmentation_manifest.get("counts", {}).get("rank_pairs")
                    if augmentation_manifest is not None
                    else 0
                ),
                "distinct_stage1_indices": len(
                    augmentation["stage1_indices"]
                ),
                "added_training_rows": len(augmentation["added_indices"]),
                "rank_pairs": len(augmentation["pair_indices"]),
                "evaluation_rows_added": 0,
            },
            "folds": fold_inventory,
            "table": table,
            "selected": selected,
            "final_fit": final_fit,
            "outputs": {
                "inventory_sha256": _sha256(inventory_path),
                "pair_inventory_sha256": _sha256(pair_inventory_path),
                "input_tokens_sha256": _sha256(tokens_path),
                "oof_stage1_scores_sha256": _sha256(stage1_scores_path),
                "oof_stage2_scores_sha256": _sha256(stage2_scores_path),
                "augmentation_inventory_sha256": _sha256(
                    augmentation_inventory_path
                ),
                "augmentation_pairs_sha256": _sha256(
                    augmentation_pairs_path
                ),
                "augmentation_input_tokens_sha256": _sha256(
                    augmentation_tokens_path
                ),
            },
            "claim_limit": (
                "Fit-only, non-formal, non-production two-stage "
                "candidate-admission diagnostic. Calibration and confirmation "
                "were not loaded; any selected threshold is eligibility-only. "
                "No candidate-level veto field exists in the frozen inventory."
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
        description="Run Revision-14 two-stage encoder OOF experiment."
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--grouped-data", type=Path, required=True)
    parser.add_argument("--revision-9", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_revision14_two_stage(
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
