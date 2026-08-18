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
from production_precision_corrected_inventory import CORRECTED_STATUS
from production_precision_encoder_finetune import (
    ACCUMULATION_STEPS,
    CLASSIFIER_LEARNING_RATE,
    DROPOUT,
    ENCODER_LEARNING_RATE,
    EPOCHS,
    FOLDS,
    LABEL_BY_NAME,
    LABEL_NAMES,
    LEFT_SENTINEL,
    MAX_GRAD_NORM,
    MAX_LENGTH,
    PARAGRAPH_SENTINEL,
    PHYSICAL_BATCH_SIZE,
    RIGHT_SENTINEL,
    SEED,
    SENTINEL_IDS,
    STRATUM_MASSES,
    WEIGHT_DECAY,
    _confusion,
    _fit,
    _score,
    _score_distributions,
    _tokenize_record,
    _validate_sentinels,
)
from production_precision_grouped_data import GROUPED_DATA_STATUS
from production_precision_grouped_verifier import _read, _read_jsonl, _sha256
from production_precision_same_jie_attention import _folds
from production_precision_select import _metric
from production_span_verifier import THRESHOLDS, _assembled_bounds
from production_train import _make_read_only


REVISION = 13
FINETUNE_STATUS_BLOCKED = (
    "ml_production_precision_corrected_encoder_finetune_blocked"
)
FINETUNE_STATUS_SELECTED = (
    "ml_production_precision_corrected_encoder_finetune_selected"
)
REVISION9_STATUS = (
    "ml_production_precision_lexical_verifier_blocked_ai_assisted"
)

EXPECTED_REFERENCES = 2566
EXPECTED_EXISTENCE = 2696
EXPECTED_RANK_PAIRS = 7205
EXPECTED_RANK_BOUNDARY = 7197
EXPECTED_BOUNDARY = 7233
EXPECTED_SEMANTIC = 59
EXPECTED_EASY = 3115
EXPECTED_MINED_BOUNDARIES = 11
EXPECTED_INVENTORY = 12973

OUTPUT_FILES = {
    "references": "references.jsonl",
    "existence": "existence.jsonl",
    "rank_pairs": "rank-pairs.jsonl",
    "mandatory_pair_counts": "mandatory-pair-counts.jsonl",
    "easy_negatives": "easy-negatives.jsonl",
    "mined_boundaries": "mined-boundaries.jsonl",
    "corrections": "corrections.jsonl",
}


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


def _candidate(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "juan": int(row["juan"]),
        "jie_index": int(row["jie_index"]),
        "para_id": int(row["para_id"]),
        "start": int(row["start"]),
        "end": int(row["end"]),
        "surface": str(row["surface"]),
    }


def _validate_candidate(
    examples: dict[str, dict],
    row: dict,
    fold_by_juan: dict[int, int],
) -> dict:
    candidate = _candidate(row)
    example = examples.get(candidate["id"])
    if example is None:
        raise ValueError("corrected encoder candidate example missing")
    if (
        int(example["juan"]) != candidate["juan"]
        or int(example["jie_index"]) != candidate["jie_index"]
    ):
        raise ValueError("corrected encoder candidate jie binding differs")
    _assembled_bounds(example, candidate)
    expected_fold = fold_by_juan.get(candidate["juan"])
    if expected_fold is None:
        raise ValueError("corrected encoder candidate juan is outside folds")
    return candidate


def _unique_candidates(
    examples: dict[str, dict],
    rows: list[dict],
    fold_by_juan: dict[int, int],
    description: str,
) -> dict[tuple[str, int, int, int], dict]:
    result = {}
    for row in rows:
        candidate = _validate_candidate(examples, row, fold_by_juan)
        key = _geometry(candidate)
        prior = result.get(key)
        if prior is not None and prior != candidate:
            raise ValueError(f"{description} duplicate geometry differs")
        result[key] = candidate
    return result


def _assemble_inventory(
    examples_rows: list[dict],
    references: list[dict],
    existence: list[dict],
    rank_pairs: list[dict],
    easy_negatives: list[dict],
    mined_boundaries: list[dict],
) -> dict:
    examples = {str(row["id"]): row for row in examples_rows}
    if len(examples) != len(examples_rows):
        raise ValueError("corrected encoder duplicate grouped example")
    jie_examples = {
        (int(row["juan"]), int(row["jie_index"])) for row in examples_rows
    }
    if len(jie_examples) != len(examples_rows):
        raise ValueError("corrected encoder grouped jie is duplicated")
    fold_by_juan = _folds([int(row["juan"]) for row in examples_rows])

    if len(references) != EXPECTED_REFERENCES:
        raise ValueError("corrected encoder reference count differs")
    if len(existence) != EXPECTED_EXISTENCE:
        raise ValueError("corrected encoder existence count differs")
    if len(rank_pairs) != EXPECTED_RANK_PAIRS:
        raise ValueError("corrected encoder rank-pair count differs")
    if len(easy_negatives) != EXPECTED_EASY:
        raise ValueError("corrected encoder easy-negative count differs")
    if len(mined_boundaries) != EXPECTED_MINED_BOUNDARIES:
        raise ValueError("corrected encoder mined-boundary count differs")

    exact = _unique_candidates(
        examples, references, fold_by_juan, "corrected reference"
    )
    if len(exact) != EXPECTED_REFERENCES:
        raise ValueError("corrected encoder duplicate reference geometry")

    pair_positive_rows = []
    pair_negative_rows = []
    for pair in rank_pairs:
        if not isinstance(pair, dict) or set(("positive", "negative")) - set(pair):
            raise ValueError("corrected encoder malformed rank pair")
        positive = _validate_candidate(
            examples, pair["positive"], fold_by_juan
        )
        negative = _validate_candidate(
            examples, pair["negative"], fold_by_juan
        )
        if (
            positive["id"] != negative["id"]
            or positive["juan"] != negative["juan"]
            or positive["jie_index"] != negative["jie_index"]
            or int(positive["para_id"]) != int(negative["para_id"])
        ):
            raise ValueError("corrected encoder rank pair crosses a jie")
        pair_positive_rows.append(positive)
        pair_negative_rows.append(negative)
    pair_positives = _unique_candidates(
        examples, pair_positive_rows, fold_by_juan, "rank-pair positive"
    )
    boundary = _unique_candidates(
        examples, pair_negative_rows, fold_by_juan, "rank-pair negative"
    )
    if not set(pair_positives).issubset(exact):
        raise ValueError("corrected encoder rank positive is not a reference")
    if len(boundary) != EXPECTED_RANK_BOUNDARY:
        raise ValueError("corrected encoder unique rank-negative count differs")

    mined = _unique_candidates(
        examples,
        mined_boundaries,
        fold_by_juan,
        "corrected mined boundary",
    )
    if len(mined) != EXPECTED_MINED_BOUNDARIES:
        raise ValueError("corrected encoder unique mined-boundary count differs")
    if set(mined) - set(boundary):
        raise ValueError(
            "corrected mined boundary is not a rank-negative geometry"
        )

    existence_by_geometry = _unique_candidates(
        examples, existence, fold_by_juan, "corrected existence"
    )
    if len(existence_by_geometry) != EXPECTED_EXISTENCE:
        raise ValueError("corrected encoder duplicate existence geometry")
    real_boundary_rows = [
        row
        for row in existence
        if int(row["label"]) == 1 and _geometry(row) not in exact
    ]
    real_boundary = _unique_candidates(
        examples,
        real_boundary_rows,
        fold_by_juan,
        "corrected real boundary",
    )
    for key, candidate in real_boundary.items():
        prior = boundary.get(key)
        if prior is not None and prior != candidate:
            raise ValueError("corrected boundary candidate geometry differs")
        boundary[key] = candidate
    if len(boundary) != EXPECTED_BOUNDARY:
        raise ValueError("corrected encoder complete boundary count differs")
    semantic_rows = [row for row in existence if int(row["label"]) == 0]
    semantic = _unique_candidates(
        examples, semantic_rows, fold_by_juan, "corrected semantic negative"
    )
    if len(semantic) != EXPECTED_SEMANTIC:
        raise ValueError("corrected encoder semantic-negative count differs")

    easy = _unique_candidates(
        examples,
        easy_negatives,
        fold_by_juan,
        "corrected easy negative",
    )
    if len(easy) != EXPECTED_EASY:
        raise ValueError("corrected encoder duplicate easy-negative geometry")

    geometry_sets = {
        "exact_person": set(exact),
        "boundary_alternative": set(boundary),
        "real_not_person": set(semantic),
        "mined_not_person": set(easy),
    }
    names = list(geometry_sets)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            if geometry_sets[left_name] & geometry_sets[right_name]:
                raise ValueError(
                    "corrected encoder class/stratum geometry collision: "
                    f"{left_name}/{right_name}"
                )

    reachable_candidate_geometry = (
        set(existence_by_geometry)
        | set(pair_positives)
        | set(boundary)
        | set(easy)
        | set(mined)
    )
    if not set(exact).issubset(reachable_candidate_geometry):
        raise ValueError("corrected encoder reference is unreachable")
    inventory_by_geometry = set().union(*geometry_sets.values())
    if (
        len(inventory_by_geometry) != EXPECTED_INVENTORY
    ):
        raise ValueError("corrected encoder inventory differs")

    real_labels = {}
    for row in existence:
        key = _geometry(row)
        if key in exact:
            if int(row["label"]) != 1:
                raise ValueError(
                    "corrected exact real candidate has a negative label"
                )
            value = 1
        elif int(row["label"]) == 0:
            if key not in semantic:
                raise ValueError("corrected semantic mapping differs")
            value = 0
        else:
            if key not in boundary:
                raise ValueError(
                    "corrected real boundary is absent from rank negatives"
                )
            value = 0
        if key in real_labels:
            raise ValueError("corrected encoder real mapping is duplicated")
        real_labels[key] = value
    if set(real_labels) != set(existence_by_geometry):
        raise ValueError("corrected encoder real-candidate mapping differs")

    rows = []
    specifications = (
        (exact, "exact_person", "exact_person", "corrected_reference"),
        (
            boundary,
            "boundary_alternative",
            "boundary_alternative",
            "corrected_rank_pair_negative",
        ),
        (semantic, "not_person", "real_not_person", "corrected_semantic"),
        (easy, "not_person", "mined_not_person", "corrected_easy"),
    )
    for candidates, class_name, stratum, source in specifications:
        for key, candidate in sorted(
            candidates.items(), key=lambda value: _ordered_key(value[1])
        ):
            rows.append({
                **candidate,
                "class_label": class_name,
                "stratum": stratum,
                "inventory_source": source,
                "real_candidate_label": real_labels.get(key),
            })
    if len(rows) != EXPECTED_INVENTORY:
        raise ValueError("corrected encoder ordered inventory count differs")

    fold_ids = np.asarray(
        [fold_by_juan[int(row["juan"])] for row in rows], dtype=np.int64
    )
    labels = np.asarray(
        [LABEL_BY_NAME[str(row["class_label"])] for row in rows],
        dtype=np.int64,
    )
    strata = [str(row["stratum"]) for row in rows]
    real_label_array = np.asarray(
        [
            -1
            if row["real_candidate_label"] is None
            else int(row["real_candidate_label"])
            for row in rows
        ],
        dtype=np.int8,
    )
    if int(np.sum(real_label_array >= 0)) != EXPECTED_EXISTENCE:
        raise ValueError("corrected encoder real diagnostic subset differs")
    for fold in range(FOLDS):
        train_strata = {
            strata[index]
            for index in np.flatnonzero(fold_ids != fold)
        }
        if train_strata != set(STRATUM_MASSES):
            raise ValueError(
                "corrected encoder training fold lacks a loss stratum"
            )
    fold_by_jie = {}
    for row, fold in zip(rows, fold_ids):
        key = (int(row["juan"]), int(row["jie_index"]))
        prior = fold_by_jie.setdefault(key, int(fold))
        if prior != int(fold):
            raise ValueError("corrected encoder jie crosses folds")
    return {
        "rows": rows,
        "labels": labels,
        "strata": strata,
        "real_labels": real_label_array,
        "fold_ids": fold_ids,
        "fold_by_juan": fold_by_juan,
    }


def _rate(mask: np.ndarray, accepted: np.ndarray, rejection: bool) -> float | None:
    count = int(mask.sum())
    if count == 0:
        return None
    values = ~accepted[mask] if rejection else accepted[mask]
    return float(np.mean(values))


def _threshold_table(
    scores: np.ndarray,
    labels: np.ndarray,
    strata: list[str],
    fold_ids: np.ndarray,
    real_labels: np.ndarray,
) -> list[dict]:
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    fold_ids = np.asarray(fold_ids, dtype=np.int64)
    real_labels = np.asarray(real_labels, dtype=np.int8)
    if not (
        len(scores)
        == len(labels)
        == len(strata)
        == len(fold_ids)
        == len(real_labels)
    ):
        raise ValueError("corrected encoder metric arrays differ")
    exact_mask = labels == LABEL_BY_NAME["exact_person"]
    boundary_mask = np.asarray(
        [value == "boundary_alternative" for value in strata], dtype=bool
    )
    semantic_mask = np.asarray(
        [value == "real_not_person" for value in strata], dtype=bool
    )
    easy_mask = np.asarray(
        [value == "mined_not_person" for value in strata], dtype=bool
    )
    real_mask = real_labels >= 0
    real_positive_mask = real_labels == 1
    if not int(exact_mask.sum()) or not int(real_mask.sum()):
        raise ValueError("corrected encoder metric denominator is empty")

    exact_indices = set(int(value) for value in np.flatnonzero(exact_mask))
    real_positive_indices = set(
        int(value) for value in np.flatnonzero(real_positive_mask)
    )
    rows = []
    for threshold in THRESHOLDS:
        accepted = scores >= np.float32(threshold)
        accepted_indices = set(
            int(value) for value in np.flatnonzero(accepted)
        )
        real_accepted_indices = set(
            int(value) for value in np.flatnonzero(accepted & real_mask)
        )
        exact_true_positive = int((accepted & exact_mask).sum())
        real_true_positive = int((accepted & real_positive_mask).sum())
        exact_recall = exact_true_positive / int(exact_mask.sum())
        all_row_precision = (
            exact_true_positive / int(accepted.sum())
            if int(accepted.sum())
            else 0.0
        )
        real_precision = (
            real_true_positive / len(real_accepted_indices)
            if real_accepted_indices
            else 0.0
        )
        all_metric = _metric(exact_indices, accepted_indices)
        real_metric = _metric(
            real_positive_indices, real_accepted_indices
        )

        fold_metrics = []
        for fold in range(FOLDS):
            fold_mask = fold_ids == fold
            fold_exact = exact_mask & fold_mask
            fold_boundary = boundary_mask & fold_mask
            fold_semantic = semantic_mask & fold_mask
            fold_metrics.append({
                "fold": fold,
                "exact_rows": int(fold_exact.sum()),
                "exact_recall": _rate(
                    fold_exact, accepted, rejection=False
                ),
                "boundary_rows": int(fold_boundary.sum()),
                "boundary_rejection": _rate(
                    fold_boundary, accepted, rejection=True
                ),
                "semantic_rows": int(fold_semantic.sum()),
                "semantic_rejection": _rate(
                    fold_semantic, accepted, rejection=True
                ),
            })
        fold_exact_recalls = [
            row["exact_recall"] for row in fold_metrics
        ]
        fold_boundary_rejections = [
            row["boundary_rejection"]
            for row in fold_metrics
            if row["boundary_rejection"] is not None
        ]
        fold_semantic_rejections = [
            row["semantic_rejection"]
            for row in fold_metrics
            if row["semantic_rejection"] is not None
        ]
        minimum_fold_exact = (
            min(fold_exact_recalls)
            if all(value is not None for value in fold_exact_recalls)
            else None
        )
        boundary_rejection = _rate(
            boundary_mask, accepted, rejection=True
        )
        semantic_rejection = _rate(
            semantic_mask, accepted, rejection=True
        )
        easy_rejection = _rate(easy_mask, accepted, rejection=True)
        eligible = (
            boundary_rejection is not None
            and semantic_rejection is not None
            and easy_rejection is not None
            and minimum_fold_exact is not None
            and exact_recall >= 0.98
            and real_precision >= 0.99
            and easy_rejection >= 0.99
            and semantic_rejection >= 0.98
            and boundary_rejection >= 0.95
            and minimum_fold_exact >= 0.95
        )
        rows.append({
            "threshold": float(threshold),
            "rows": len(labels),
            "accepted_rows": int(accepted.sum()),
            "exact_true_positive": exact_true_positive,
            "exact_recall": exact_recall,
            "real_candidate_rows": int(real_mask.sum()),
            "real_candidate_accepted": len(real_accepted_indices),
            "real_candidate_true_positive": real_true_positive,
            "real_candidate_precision": real_precision,
            "real_candidate_wilson_precision_lower_one_sided_95": (
                real_metric["wilson_precision_lower_one_sided_95"]
            ),
            "all_row_precision": all_row_precision,
            "all_row_wilson_precision_lower_one_sided_95": (
                all_metric["wilson_precision_lower_one_sided_95"]
            ),
            "easy_negative_rejection": easy_rejection,
            "semantic_negative_rejection": semantic_rejection,
            "boundary_alternative_rejection": boundary_rejection,
            "minimum_fold_exact_recall": minimum_fold_exact,
            "worst_fold_boundary_rejection": (
                min(fold_boundary_rejections)
                if fold_boundary_rejections
                else None
            ),
            "worst_fold_semantic_rejection": (
                min(fold_semantic_rejections)
                if fold_semantic_rejections
                else None
            ),
            "fold_metrics": fold_metrics,
            "eligible": eligible,
        })
    return rows


def _validate_sha_bindings(manifest: dict, keys: tuple[str, ...]) -> None:
    bindings = manifest.get("bindings", {})
    for key in keys:
        value = bindings.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"corrected encoder manifest binding differs: {key}")


def _load_inputs(
    corrected_root: Path,
    grouped_root: Path,
    revision9_root: Path,
) -> tuple[dict, dict, dict, list[dict], dict[str, list[dict]], Path]:
    corrected_manifest_path = corrected_root / "manifest.json"
    grouped_manifest_path = grouped_root / "manifest.json"
    revision9_manifest_path = revision9_root / "manifest.json"
    corrected_manifest = _read(corrected_manifest_path)
    grouped_manifest = _read(grouped_manifest_path)
    revision9_manifest = _read(revision9_manifest_path)
    examples_path = grouped_root / "examples.jsonl"

    _validate_sha_bindings(
        corrected_manifest,
        (
            "grouped_manifest_sha256",
            "audit_manifest_sha256",
            "targets_manifest_sha256",
            "conflicts_manifest_sha256",
            "safe_manifest_sha256",
            "examples_sha256",
        ),
    )
    if (
        corrected_manifest.get("status") != CORRECTED_STATUS
        or corrected_manifest.get("revision") != REVISION
        or corrected_manifest.get("confirmation_read") is not False
        or corrected_manifest["bindings"]["grouped_manifest_sha256"]
        != _sha256(grouped_manifest_path)
        or corrected_manifest["bindings"]["examples_sha256"]
        != _sha256(examples_path)
    ):
        raise ValueError("corrected encoder corrected-inventory binding differs")
    corrected_counts = corrected_manifest.get("counts", {})
    if (
        corrected_counts.get("corrected_references") != EXPECTED_REFERENCES
        or corrected_counts.get("real_rows") != EXPECTED_EXISTENCE
        or corrected_counts.get("corrected_rank_pairs")
        != EXPECTED_RANK_PAIRS
        or corrected_counts.get("semantic_negatives") != EXPECTED_SEMANTIC
        or corrected_counts.get("easy_negatives") != EXPECTED_EASY
        or corrected_counts.get("mined_boundary_alternatives")
        != EXPECTED_MINED_BOUNDARIES
    ):
        raise ValueError("corrected encoder corrected manifest counts differ")
    corrected_rows = {}
    for key, name in OUTPUT_FILES.items():
        path = corrected_root / name
        expected = corrected_manifest.get("outputs", {}).get(
            f"{key}_sha256"
        )
        if expected != _sha256(path):
            raise ValueError(
                f"corrected encoder corrected output differs: {name}"
            )
        corrected_rows[key] = _read_jsonl(path)

    if (
        grouped_manifest.get("status") != GROUPED_DATA_STATUS
        or grouped_manifest.get("confirmation_read") is not False
        or grouped_manifest.get("outputs", {}).get("examples_sha256")
        != _sha256(examples_path)
    ):
        raise ValueError("corrected encoder grouped-data binding differs")
    examples = _read_jsonl(examples_path)

    encoder_dir = revision9_root / "encoder"
    encoder_artifact = _model_artifact(encoder_dir)
    if (
        revision9_manifest.get("status") != REVISION9_STATUS
        or revision9_manifest.get("revision") != 9
        or revision9_manifest.get("confirmation_read") is not False
        or revision9_manifest.get("encoder_artifact") != encoder_artifact
        or revision9_manifest.get("bindings", {}).get(
            "grouped_manifest_sha256"
        )
        != _sha256(grouped_manifest_path)
    ):
        raise ValueError("corrected encoder Revision-9 base binding differs")
    return (
        corrected_manifest,
        grouped_manifest,
        revision9_manifest,
        examples,
        corrected_rows,
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


def run_corrected_encoder_finetune(
    corrected_root: Path,
    grouped_root: Path,
    revision9_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"corrected encoder fine-tune output exists: {output_dir}"
        )
    (
        corrected_manifest,
        grouped_manifest,
        revision9_manifest,
        example_rows,
        corrected,
        encoder_dir,
    ) = _load_inputs(corrected_root, grouped_root, revision9_root)
    inventory = _assemble_inventory(
        example_rows,
        corrected["references"],
        corrected["existence"],
        corrected["rank_pairs"],
        corrected["easy_negatives"],
        corrected["mined_boundaries"],
    )
    rows = inventory["rows"]
    labels = inventory["labels"]
    strata = inventory["strata"]
    real_labels = inventory["real_labels"]
    fold_ids = inventory["fold_ids"]
    fold_by_juan = inventory["fold_by_juan"]
    examples = {str(row["id"]): row for row in example_rows}

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
        raise RuntimeError("corrected encoder fine-tune requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir, use_fast=True)
    _validate_sentinels(tokenizer, examples)
    records = [
        {
            **row,
            **_tokenize_record(tokenizer, examples[str(row["id"])], row),
        }
        for row in rows
    ]

    oof_scores = np.zeros(len(rows), dtype=np.float32)
    oof_predictions = np.zeros(len(rows), dtype=np.int64)
    fold_inventory = []
    for fold in range(FOLDS):
        train_indices = np.flatnonzero(fold_ids != fold)
        heldout_indices = np.flatnonzero(fold_ids == fold)
        if not len(train_indices) or not len(heldout_indices):
            raise ValueError("corrected encoder fold is empty")
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
        heldout_jies = sorted({
            str(rows[int(index)]["id"]) for index in heldout_indices
        })
        train_jies = sorted({
            str(rows[int(index)]["id"]) for index in train_indices
        })
        if set(heldout_jies) & set(train_jies):
            raise ValueError("corrected encoder OOF jie leakage")
        fold_inventory.append({
            "fold": fold,
            "train_juans": sorted(
                juan
                for juan, value in fold_by_juan.items()
                if value != fold
            ),
            "heldout_juans": sorted(
                juan
                for juan, value in fold_by_juan.items()
                if value == fold
            ),
            "train_jies": train_jies,
            "heldout_jies": heldout_jies,
            "train_rows": len(train_indices),
            "heldout_rows": len(heldout_indices),
            "train_stratum_rows": {
                name: sum(
                    strata[int(index)] == name for index in train_indices
                )
                for name in STRATUM_MASSES
            },
            "heldout_stratum_rows": {
                name: sum(
                    strata[int(index)] == name for index in heldout_indices
                )
                for name in STRATUM_MASSES
            },
            "fold_local_stratum_weights": weight_inventory,
        })
        del model
        torch.cuda.empty_cache()

    table = _threshold_table(
        oof_scores, labels, strata, fold_ids, real_labels
    )
    eligible = [row for row in table if row["eligible"]]
    selected = max(
        eligible,
        key=lambda row: (
            row["exact_recall"],
            row["real_candidate_precision"],
            row["threshold"],
        ),
        default=None,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        inventory_path = staging / "inventory.jsonl"
        _write_jsonl(inventory_path, rows)
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
                for row, record in zip(rows, records)
            ],
        )
        scores_path = staging / "oof-scores.jsonl"
        _write_jsonl(
            scores_path,
            [
                {
                    **row,
                    "fold": int(fold),
                    "oof_exact_probability": float(score),
                    "oof_argmax": LABEL_NAMES[int(prediction)],
                }
                for row, fold, score, prediction in zip(
                    rows, fold_ids, oof_scores, oof_predictions
                )
            ],
        )

        final_fit = None
        if selected is not None:
            model, full_weights = _fit(
                records,
                labels,
                strata,
                np.arange(len(records), dtype=np.int64),
                encoder_dir,
                device,
            )
            admission_encoder = staging / "admission-encoder"
            classifier_path = (
                staging / "admission-classifier.safetensors"
            )
            model.encoder.save_pretrained(
                admission_encoder, safe_serialization=True
            )
            save_file(
                {
                    key: value.detach().cpu()
                    for key, value in model.classifier.state_dict().items()
                },
                classifier_path,
            )
            final_fit = {
                "rows": len(records),
                "stratum_weights": full_weights,
                "admission_encoder": _model_artifact(admission_encoder),
                "classifier_sha256": _sha256(classifier_path),
                "threshold": selected["threshold"],
                "threshold_role": "eligibility_diagnostic_only",
            }
            del model
            torch.cuda.empty_cache()

        corrected_manifest_path = corrected_root / "manifest.json"
        grouped_manifest_path = grouped_root / "manifest.json"
        revision9_manifest_path = revision9_root / "manifest.json"
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
            "transformers_offline": os.environ.get(
                "TRANSFORMERS_OFFLINE"
            ),
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
            "fit_only": True,
            "confirmation_read": False,
            "git_commit": git_commit,
            "bindings": {
                "corrected_manifest_sha256": _sha256(
                    corrected_manifest_path
                ),
                "corrected_outputs": {
                    key: corrected_manifest["outputs"][f"{key}_sha256"]
                    for key in OUTPUT_FILES
                },
                "grouped_manifest_sha256": _sha256(
                    grouped_manifest_path
                ),
                "grouped_examples_sha256": grouped_manifest["outputs"][
                    "examples_sha256"
                ],
                "revision9_manifest_sha256": _sha256(
                    revision9_manifest_path
                ),
                "revision9_base_encoder": revision9_manifest[
                    "encoder_artifact"
                ],
                "ordered_inventory_sha256": _sha256(inventory_path),
                "input_tokens_sha256": _sha256(tokens_path),
                "oof_scores_sha256": _sha256(scores_path),
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
                "gradient_clipping_frequency": {
                    "available": False,
                    "reason": (
                        "Revision-11 fixed training helper does not expose "
                        "per-step clipping norms."
                    ),
                },
                "sentinels": {
                    "left": LEFT_SENTINEL,
                    "right": RIGHT_SENTINEL,
                    "paragraph": PARAGRAPH_SENTINEL,
                    "ids": SENTINEL_IDS,
                },
                "labels": list(LABEL_NAMES),
                "stratum_masses": STRATUM_MASSES,
                "thresholds": list(THRESHOLDS),
                "threshold_comparison": "float32_score_greater_than_or_equal",
                "encoder_frozen_during_fit": False,
                "gradient_checkpointing": True,
                "optimizer_state_device": "cuda",
                "deterministic_flags": deterministic_flags,
            },
            "environment": environment,
            "inventory": {
                "rows": len(rows),
                "exact_person": strata.count("exact_person"),
                "boundary_alternative": strata.count(
                    "boundary_alternative"
                ),
                "real_not_person": strata.count("real_not_person"),
                "mined_not_person": strata.count("mined_not_person"),
                "real_candidate_rows": int((real_labels >= 0).sum()),
                "rank_pairs": len(corrected["rank_pairs"]),
                "mined_boundaries": len(corrected["mined_boundaries"]),
            },
            "folds": fold_inventory,
            "confusion": _confusion(labels, oof_predictions),
            "score_distributions": _score_distributions(
                oof_scores, strata
            ),
            "table": table,
            "selected": selected,
            "final_fit": final_fit,
            "outputs": {
                "inventory_sha256": _sha256(inventory_path),
                "input_tokens_sha256": _sha256(tokens_path),
                "oof_scores_sha256": _sha256(scores_path),
            },
            "claim_limit": (
                "Fit-only, non-formal, non-production corrected-label "
                "candidate-admission diagnostic. Calibration and confirmation "
                "were not loaded; any selected threshold is eligibility-only."
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
        description=(
            "Run Revision-13 corrected-label encoder seven-fold OOF."
        )
    )
    parser.add_argument("--corrected-inventory", type=Path, required=True)
    parser.add_argument("--grouped-data", type=Path, required=True)
    parser.add_argument("--revision-9", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_corrected_encoder_finetune(
        args.corrected_inventory,
        args.grouped_data,
        args.revision_9,
        args.output,
    )
    print(json.dumps({
        "status": manifest["status"],
        "selected": manifest["selected"],
        "table": manifest["table"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
