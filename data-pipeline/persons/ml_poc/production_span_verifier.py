from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import stat
import tempfile
import unicodedata
from bisect import bisect_right
from pathlib import Path

import numpy as np

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from p1_train import MODEL_NAME, MODEL_REVISION
from p1_windows import build_windows
from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from production_precision_select import _metric
from production_train import _make_read_only


SEED = 20260810
THRESHOLDS = (
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
    0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99,
)
BOUNDARY_CATEGORIES = ("edge", "letter", "number", "punctuation", "symbol", "other")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _category(character: str | None) -> str:
    if character is None:
        return "edge"
    prefix = unicodedata.category(character)[0]
    return {
        "L": "letter",
        "N": "number",
        "P": "punctuation",
        "S": "symbol",
    }.get(prefix, "other")


def _candidate_key(row: dict) -> tuple:
    return (
        str(row["id"]),
        int(row["para_id"]),
        int(row["start"]),
        int(row["end"]),
        str(row["surface"]),
    )


def _assembled_bounds(example: dict, row: dict) -> tuple[int, int, dict]:
    matches = [
        segment for segment in example["segments"]
        if int(segment["para_id"]) == int(row["para_id"])
    ]
    if len(matches) != 1:
        raise ValueError(f"candidate segment differs: {_candidate_key(row)}")
    segment = matches[0]
    start = int(segment["assembled_start"]) + int(row["start"])
    end = int(segment["assembled_start"]) + int(row["end"])
    if (
        not int(segment["assembled_start"])
        <= start
        < end
        <= int(segment["assembled_end"])
        or str(example["text"])[start:end] != row["surface"]
    ):
        raise ValueError(f"candidate source differs: {_candidate_key(row)}")
    return start, end, segment


def _pool_candidate_encodings(
    examples: dict[str, dict],
    candidates: list[dict],
    model,
    tokenizer,
    device,
) -> tuple[int, list[dict]]:
    """Pool the frozen encoder for every candidate.

    Returns the encoder hidden size and, for each candidate index, the shared
    candidate/left/right/context poolings plus the immediate boundary characters,
    span length, and paragraph-edge bits. This is deterministic and free of any
    generator metadata, so revision-3 and revision-4 verifier feature builders can
    reuse it without duplicating unsafe encoder logic.
    """
    import torch

    by_id = {}
    for index, row in enumerate(candidates):
        by_id.setdefault(str(row["id"]), []).append((index, row))
    hidden_size = int(model.config.hidden_size)
    pooled: list[dict] = [None] * len(candidates)
    model.eval()
    with torch.inference_mode():
        for identity in sorted(by_id):
            example = examples[identity]
            windows = build_windows(tokenizer, example, max_length=512, stride=128)
            character_hidden = np.zeros(
                (len(example["text"]), hidden_size), dtype=np.float32
            )
            owned = np.zeros(len(example["text"]), dtype=bool)
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
                        [window.token_type_ids], dtype=torch.long, device=device
                    )
                hidden = model(**inputs).last_hidden_state[0].cpu().numpy()
                for token_index, (start, end) in enumerate(window.offsets):
                    if not window.owned_tokens[token_index]:
                        continue
                    for character_index in range(start, end):
                        if owned[character_index]:
                            raise ValueError("character has multiple encoder owners")
                        character_hidden[character_index] = hidden[token_index]
                        owned[character_index] = True
            context_positions = [
                index
                for index, (character, is_owned) in enumerate(
                    zip(example["text"], owned)
                )
                if is_owned and character != "\n"
            ]
            if not context_positions:
                raise ValueError(f"encoder owns no target text: {identity}")
            context_mean = character_hidden[context_positions].mean(axis=0)
            for output_index, row in by_id[identity]:
                start, end, segment = _assembled_bounds(example, row)
                if not owned[start:end].all():
                    raise ValueError(f"candidate not encoder-owned: {_candidate_key(row)}")
                candidate_mean = character_hidden[start:end].mean(axis=0)
                segment_start = int(segment["assembled_start"])
                segment_end = int(segment["assembled_end"])
                left_index = start - 1 if start > segment_start else None
                right_index = end if end < segment_end else None
                left_hidden = (
                    character_hidden[left_index]
                    if left_index is not None and owned[left_index]
                    else np.zeros(hidden_size, dtype=np.float32)
                )
                right_hidden = (
                    character_hidden[right_index]
                    if right_index is not None and owned[right_index]
                    else np.zeros(hidden_size, dtype=np.float32)
                )
                left_character = (
                    str(example["text"])[left_index] if left_index is not None else None
                )
                right_character = (
                    str(example["text"])[right_index]
                    if right_index is not None
                    else None
                )
                pooled[output_index] = {
                    "candidate_mean": candidate_mean,
                    "left_hidden": left_hidden,
                    "right_hidden": right_hidden,
                    "context_mean": context_mean,
                    "left_character": left_character,
                    "right_character": right_character,
                    "length": end - start,
                    "starts_paragraph": left_index is None,
                    "ends_paragraph": right_index is None,
                }
    return hidden_size, pooled


def _extract_features(
    examples: dict[str, dict],
    candidates: list[dict],
    model,
    tokenizer,
    device,
) -> np.ndarray:
    hidden_size, pooled = _pool_candidate_encodings(
        examples, candidates, model, tokenizer, device
    )
    numeric_size = 1 + 1 + 3 + len(BOUNDARY_CATEGORIES) * 2
    features = np.zeros(
        (len(candidates), hidden_size * 4 + numeric_size),
        dtype=np.float32,
    )
    for output_index, row in enumerate(candidates):
        parts = pooled[output_index]
        numeric = [
            math.log1p(parts["length"]),
            int(row["support_count"]) / 3,
            *[
                float(row["seed_confidences"][str(seed)])
                for seed in (20260727, 20260728, 20260729)
            ],
        ]
        for value in (parts["left_character"], parts["right_character"]):
            category = _category(value)
            numeric.extend(
                float(category == expected)
                for expected in BOUNDARY_CATEGORIES
            )
        features[output_index] = np.concatenate((
            parts["candidate_mean"],
            parts["left_hidden"],
            parts["right_hidden"],
            parts["context_mean"],
            np.asarray(numeric, dtype=np.float32),
        ))
    return features


def _selection_better(left: dict, right: dict) -> bool:
    left_key = (
        left["weight"],
        left["support"],
        left["confidence"],
        left["covered"],
        -len(left["indices"]),
    )
    right_key = (
        right["weight"],
        right["support"],
        right["confidence"],
        right["covered"],
        -len(right["indices"]),
    )
    if left_key != right_key:
        return left_key > right_key
    return left["geometry"] < right["geometry"]


def _resolve_group(rows: list[dict], threshold: float) -> list[dict]:
    if not rows:
        return []
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["end"]),
            int(row["start"]),
            -int(row["support_count"]),
            -min(
                value for value in row["seed_confidences"].values() if value > 0
            ),
            str(row["surface"]),
        ),
    )
    ends = [int(row["end"]) for row in ordered]
    predecessors = [
        bisect_right(ends, int(row["start"]), hi=index) - 1
        for index, row in enumerate(ordered)
    ]
    empty = {
        "weight": 0,
        "support": 0,
        "confidence": 0,
        "covered": 0,
        "indices": (),
        "geometry": (),
    }
    best = []
    logit_threshold = math.log(threshold / (1 - threshold))
    for index, row in enumerate(ordered):
        previous = best[predecessors[index]] if predecessors[index] >= 0 else empty
        score = min(max(float(row["score"]), 1e-6), 1 - 1e-6)
        confidence = min(
            value for value in row["seed_confidences"].values() if value > 0
        )
        included_indices = previous["indices"] + (index,)
        included = {
            "weight": previous["weight"] + round(
                1_000_000 * (math.log(score / (1 - score)) - logit_threshold)
            ),
            "support": previous["support"] + int(row["support_count"]),
            "confidence": previous["confidence"] + round(1_000_000 * confidence),
            "covered": previous["covered"] + int(row["end"]) - int(row["start"]),
            "indices": included_indices,
            "geometry": tuple(
                sorted(
                    previous["geometry"]
                    + ((int(row["start"]), int(row["end"])),)
                )
            ),
        }
        excluded = best[index - 1] if index else empty
        best.append(included if _selection_better(included, excluded) else excluded)
    return [ordered[index] for index in best[-1]["indices"]]


def _resolve(candidates: list[dict], threshold: float) -> set[tuple]:
    groups = {}
    for row in candidates:
        if (
            float(row["score"]) < threshold
            or row.get("intrinsic_hard_vetoes")
        ):
            continue
        groups.setdefault((str(row["id"]), int(row["para_id"])), []).append(row)
    selected = set()
    for rows in groups.values():
        for row in _resolve_group(rows, threshold):
            selected.add(_candidate_key(row))
    return selected


def _train_head(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    score_indices: np.ndarray,
    output_dir: Path,
) -> np.ndarray:
    import torch
    from safetensors.torch import save_file
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    mean = features[train_indices].mean(axis=0)
    scale = features[train_indices].std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (features - mean) / scale

    class Head(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(width, 256),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(256, 1),
            )

        def forward(self, values):
            return self.network(values).squeeze(-1)

    model = Head(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.01
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(1.0, device=device)
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized[train_indices]).float(),
            torch.from_numpy(labels[train_indices]).float(),
        ),
        batch_size=32,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    for _ in range(20):
        model.train()
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features.to(device))
            loss = loss_function(logits, batch_labels.to(device))
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        scores = (
            model(
                torch.from_numpy(normalized[score_indices]).float().to(device)
            )
            .sigmoid()
            .cpu()
            .numpy()
        )
    output_dir.mkdir(parents=True)
    save_file(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        output_dir / "head.safetensors",
    )
    np.savez(
        output_dir / "scaler.npz",
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
    )
    return scores


def train_verifier(
    lattice_root: Path,
    reference_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"span verifier output exists: {output_dir}")
    lattice_manifest_path = lattice_root / "manifest.json"
    lattice_manifest = _read(lattice_manifest_path)
    lattice_path = lattice_root / "lattice.jsonl"
    candidates = [
        json.loads(line)
        for line in lattice_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    reference_manifest_path = reference_root / "manifest.json"
    reference_manifest = _read(reference_manifest_path)
    calibration_path = reference_root / "calibration.jsonl"
    examples_list = [
        json.loads(line)
        for line in calibration_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    examples = {str(row["id"]): row for row in examples_list}
    if (
        lattice_manifest.get("status")
        != "ml_production_verifier_lattice_ai_assisted"
        or lattice_manifest.get("candidate_recall", 0) < 0.98
        or lattice_manifest.get("lattice_sha256") != _sha256(lattice_path)
        or len(candidates) != 517
        or reference_manifest.get("status")
        != "ml_production_precision_reference_ai_assisted"
        or reference_manifest.get("outputs", {}).get("calibration_sha256")
        != _sha256(calibration_path)
        or len(examples) != 45
        or not {str(row["id"]) for row in candidates}.issubset(examples)
    ):
        raise ValueError("span verifier input binding differs")
    git_commit = _git_commit_clean()

    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, use_fast=True
    )
    encoder = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    encoder.to(device)
    features = _extract_features(
        examples, candidates, encoder, tokenizer, device
    )
    labels = np.asarray([int(row["label"]) for row in candidates], dtype=np.float32)
    folds = np.asarray([int(row["fold"]) for row in candidates], dtype=np.int64)
    oof_scores = np.zeros(len(candidates), dtype=np.float32)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        encoder_dir = staging / "encoder"
        encoder.save_pretrained(encoder_dir)
        tokenizer.save_pretrained(encoder_dir)
        fold_inventory = {}
        for fold in range(1, 5):
            train_indices = np.flatnonzero(folds != fold)
            score_indices = np.flatnonzero(folds == fold)
            if not len(train_indices) or not len(score_indices):
                raise ValueError(f"empty verifier fold: {fold}")
            fold_dir = staging / "folds" / f"fold_{fold}"
            scores = _train_head(
                features,
                labels,
                train_indices,
                score_indices,
                fold_dir,
            )
            oof_scores[score_indices] = scores
            fold_inventory[str(fold)] = {
                "train_candidates": len(train_indices),
                "score_candidates": len(score_indices),
                "head_sha256": _sha256(fold_dir / "head.safetensors"),
                "scaler_sha256": _sha256(fold_dir / "scaler.npz"),
            }

        scored_candidates = []
        for row, score in zip(candidates, oof_scores):
            scored_candidates.append({**row, "score": float(score)})
        reference = {
            _candidate_key(row)
            for row in candidates
            if int(row["label"]) == 1
        }
        missed_reference_spans = int(
            lattice_manifest["counts"]["missed_reference_spans"]
        )
        table = []
        for threshold in THRESHOLDS:
            prediction = _resolve(scored_candidates, threshold)
            true_positive = len(reference & prediction)
            reference_count = len(reference) + missed_reference_spans
            prediction_count = len(prediction)
            precision = true_positive / prediction_count if prediction_count else 0.0
            recall = true_positive / reference_count if reference_count else 0.0
            wilson = _metric(reference, prediction)[
                "wilson_precision_lower_one_sided_95"
            ]
            table.append({
                "threshold": threshold,
                "reference_spans": reference_count,
                "prediction_spans": prediction_count,
                "true_positive": true_positive,
                "precision": precision,
                "recall": recall,
                "wilson_precision_lower_one_sided_95": wilson,
                "eligible": (
                    prediction_count >= 300
                    and precision >= 0.99
                    and recall >= 0.95
                    and wilson >= 0.98
                ),
            })
        eligible = [row for row in table if row["eligible"]]
        selected = max(
            eligible,
            key=lambda row: (
                row["recall"],
                row["precision"],
                row["threshold"],
            ),
            default=None,
        )
        final_inventory = None
        if selected is not None:
            final_dir = staging / "final"
            all_indices = np.arange(len(candidates))
            _train_head(
                features,
                labels,
                all_indices,
                all_indices,
                final_dir,
            )
            final_inventory = {
                "train_candidates": len(candidates),
                "head_sha256": _sha256(final_dir / "head.safetensors"),
                "scaler_sha256": _sha256(final_dir / "scaler.npz"),
            }
        scores_path = staging / "oof-scores.json"
        scores_path.write_text(
            json.dumps(scored_candidates, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        features_path = staging / "features.npz"
        np.savez_compressed(features_path, features=features)
        manifest = {
            "schema_version": 1,
            "status": (
                "ml_production_span_verifier_selected_ai_assisted"
                if selected is not None
                else "ml_production_span_verifier_blocked_ai_assisted"
            ),
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "git_commit": git_commit,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "encoder_artifact": _model_artifact(encoder_dir),
            "lattice_manifest_sha256": _sha256(lattice_manifest_path),
            "reference_manifest_sha256": _sha256(reference_manifest_path),
            "calibration_sha256": _sha256(calibration_path),
            "control": {
                "seed": SEED,
                "folds": 4,
                "epochs": 20,
                "hidden_width": 256,
                "dropout": 0.10,
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "batch_size": 32,
                "positive_class_weight": 1.0,
                "encoder_frozen": True,
            },
            "fold_inventory": fold_inventory,
            "final_inventory": final_inventory,
            "thresholds": list(THRESHOLDS),
            "table": table,
            "selected": selected,
            "oof_scores_sha256": _sha256(scores_path),
            "features_sha256": _sha256(features_path),
            "claim_limit": (
                "AI-assisted out-of-fold diagnostic only; confirmation remains unread."
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--lattice", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = train_verifier(args.lattice, args.reference, args.output)
    print(json.dumps({
        "status": manifest["status"],
        "selected": manifest["selected"],
        "table": manifest["table"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
