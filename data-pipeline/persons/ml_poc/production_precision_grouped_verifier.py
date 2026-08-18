from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from p1_train import MODEL_NAME, MODEL_REVISION
from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from production_precision_grouped_data import GROUPED_DATA_STATUS
from production_precision_select import _metric
from production_precision_verifier import (
    BATCH_SIZE,
    DROPOUT,
    EPOCHS,
    HIDDEN_WIDTH,
    LEARNING_RATE,
    NUMERIC_SIZE,
    WEIGHT_DECAY,
    _build_head,
    _extract_features,
)
from production_span_verifier import THRESHOLDS, _candidate_key
from production_train import _make_read_only


EXISTENCE_SEED = 20260814
RANK_SEED = 20260815
LATTICE_STATUS = "ml_production_verifier_lattice_ai_assisted"
REFERENCE_STATUS = "ml_production_precision_reference_ai_assisted"
CALIBRATION_REFERENCE_SPANS = 469


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _feature_key(row: dict) -> tuple[str, int, int, int]:
    return (
        str(row["id"]),
        int(row["para_id"]),
        int(row["start"]),
        int(row["end"]),
    )


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _fit_scaler(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _fit_existence(
    features: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    device,
) -> tuple[object, np.ndarray, np.ndarray, dict]:
    import torch
    from safetensors.torch import save_file
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    _seed_everything(EXISTENCE_SEED)
    mean, scale = _fit_scaler(features)
    normalized = (features - mean) / scale
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        raise ValueError("existence inventory must contain both classes")
    class_weights = np.asarray([
        len(labels) / (2 * (positive_count if label else negative_count))
        for label in labels
    ], dtype=np.float32)

    model = _build_head(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator().manual_seed(EXISTENCE_SEED)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized).float(),
            torch.from_numpy(labels).float(),
            torch.from_numpy(class_weights).float(),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    for _ in range(EPOCHS):
        model.train()
        for batch_features, batch_labels, batch_weights in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features.to(device)).squeeze(-1)
            weights = batch_weights.to(device)
            losses = F.binary_cross_entropy_with_logits(
                logits, batch_labels.to(device), reduction="none"
            )
            loss = (losses * weights).sum() / weights.sum()
            loss.backward()
            optimizer.step()
    output_dir.mkdir(parents=True)
    save_file(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        output_dir / "head.safetensors",
    )
    np.savez(output_dir / "scaler.npz", mean=mean, scale=scale)
    model.eval()
    return model, mean, scale, {
        "positive_row_weight": len(labels) / (2 * positive_count),
        "negative_row_weight": len(labels) / (2 * negative_count),
    }


def _fit_ranker(
    candidate_features: np.ndarray,
    pair_indices: np.ndarray,
    output_dir: Path,
    device,
) -> tuple[object, np.ndarray, np.ndarray]:
    import torch
    from safetensors.torch import save_file
    from torch.utils.data import DataLoader, TensorDataset

    _seed_everything(RANK_SEED)
    mean, scale = _fit_scaler(candidate_features)
    normalized = (candidate_features - mean) / scale
    model = _build_head(candidate_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator().manual_seed(RANK_SEED)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(pair_indices).long()),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    feature_tensor = torch.from_numpy(normalized).float().to(device)
    for _ in range(EPOCHS):
        model.train()
        for (indices,) in loader:
            optimizer.zero_grad(set_to_none=True)
            indices = indices.to(device)
            positive = model(feature_tensor[indices[:, 0]]).squeeze(-1)
            negative = model(feature_tensor[indices[:, 1]]).squeeze(-1)
            loss = torch.clamp(1.0 - positive + negative, min=0.0).mean()
            loss.backward()
            optimizer.step()
    output_dir.mkdir(parents=True)
    save_file(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        output_dir / "head.safetensors",
    )
    np.savez(output_dir / "scaler.npz", mean=mean, scale=scale)
    model.eval()
    return model, mean, scale


def _score(
    model, mean: np.ndarray, scale: np.ndarray, features: np.ndarray, device,
    *, sigmoid: bool,
) -> np.ndarray:
    import torch

    normalized = (features - mean) / scale
    with torch.inference_mode():
        scores = model(
            torch.from_numpy(normalized).float().to(device)
        ).squeeze(-1)
        if sigmoid:
            scores = scores.sigmoid()
        result = scores.cpu().numpy().astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("verifier emitted non-finite scores")
    return result


def _confidence_key(row: dict) -> int:
    return round(
        1_000_000
        * sum(float(value) for value in row["seed_confidences"].values())
    )


def _resolve_group(rows: list[dict], threshold: float) -> list[dict]:
    admitted = [
        row
        for row in rows
        if float(row["existence_score"]) >= threshold
        and not row.get("intrinsic_hard_vetoes")
    ]
    for row in admitted:
        if not math.isfinite(float(row["rank_logit"])):
            raise ValueError("rank logit must be finite")
    ordered = sorted(admitted, key=lambda row: (
        -float(row["rank_logit"]),
        -int(row["support_count"]),
        -_confidence_key(row),
        -(int(row["end"]) - int(row["start"])),
        int(row["start"]),
        int(row["end"]),
    ))
    selected = []
    for row in ordered:
        if all(
            int(row["end"]) <= int(other["start"])
            or int(other["end"]) <= int(row["start"])
            for other in selected
        ):
            selected.append(row)
    return sorted(selected, key=lambda row: (int(row["start"]), int(row["end"])))


def _resolve(candidates: list[dict], threshold: float) -> set[tuple]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in candidates:
        groups.setdefault(
            (str(row["id"]), int(row["para_id"])), []
        ).append(row)
    return {
        _candidate_key(row)
        for rows in groups.values()
        for row in _resolve_group(rows, threshold)
    }


def train_grouped_verifier(
    grouped_root: Path,
    lattice_root: Path,
    reference_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"grouped verifier output exists: {output_dir}")

    grouped_manifest_path = grouped_root / "manifest.json"
    grouped_manifest = _read(grouped_manifest_path)
    examples_path = grouped_root / "examples.jsonl"
    existence_path = grouped_root / "existence.jsonl"
    rank_pairs_path = grouped_root / "rank-pairs.jsonl"
    fit_examples = {
        str(row["id"]): row for row in _read_jsonl(examples_path)
    }
    existence_rows = _read_jsonl(existence_path)
    rank_pairs = _read_jsonl(rank_pairs_path)
    grouped_outputs = grouped_manifest.get("outputs", {})
    if (
        grouped_manifest.get("status") != GROUPED_DATA_STATUS
        or grouped_manifest.get("confirmation_read") is not False
        or not all(
            gate.get("passed")
            for gate in grouped_manifest.get("floors", {}).values()
        )
        or grouped_outputs.get("examples_sha256") != _sha256(examples_path)
        or grouped_outputs.get("existence_sha256") != _sha256(existence_path)
        or grouped_outputs.get("rank_pairs_sha256") != _sha256(rank_pairs_path)
    ):
        raise ValueError("grouped verifier training-data binding differs")

    lattice_manifest_path = lattice_root / "manifest.json"
    lattice_manifest = _read(lattice_manifest_path)
    lattice_path = lattice_root / "lattice.jsonl"
    calibration_candidates = _read_jsonl(lattice_path)
    reference_manifest_path = reference_root / "manifest.json"
    reference_manifest = _read(reference_manifest_path)
    calibration_path = reference_root / "calibration.jsonl"
    calibration_examples = {
        str(row["id"]): row for row in _read_jsonl(calibration_path)
    }
    if (
        lattice_manifest.get("status") != LATTICE_STATUS
        or lattice_manifest.get("candidate_recall", 0) < 0.98
        or lattice_manifest.get("lattice_sha256") != _sha256(lattice_path)
        or len(calibration_candidates) != 517
        or reference_manifest.get("status") != REFERENCE_STATUS
        or reference_manifest.get("outputs", {}).get("calibration_sha256")
        != _sha256(calibration_path)
        or len(calibration_examples) != 45
    ):
        raise ValueError("grouped verifier calibration binding differs")
    if {int(row["juan"]) for row in fit_examples.values()} & {
        int(row["juan"]) for row in calibration_examples.values()
    }:
        raise ValueError("grouped verifier fit/calibration juans overlap")

    rank_candidates_by_key: dict[tuple, dict] = {}
    pair_keys = []
    for pair in rank_pairs:
        positive_key = _feature_key(pair["positive"])
        negative_key = _feature_key(pair["negative"])
        rank_candidates_by_key.setdefault(positive_key, pair["positive"])
        rank_candidates_by_key.setdefault(negative_key, pair["negative"])
        pair_keys.append((positive_key, negative_key))
    rank_keys = sorted(rank_candidates_by_key)
    rank_candidates = [rank_candidates_by_key[key] for key in rank_keys]
    rank_index = {key: index for index, key in enumerate(rank_keys)}
    pair_indices = np.asarray([
        (rank_index[positive], rank_index[negative])
        for positive, negative in pair_keys
    ], dtype=np.int64)

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

    existence_features = _extract_features(
        fit_examples, existence_rows, encoder, tokenizer, device
    )
    existence_labels = np.asarray(
        [int(row["label"]) for row in existence_rows], dtype=np.float32
    )
    rank_features = _extract_features(
        fit_examples, rank_candidates, encoder, tokenizer, device
    )
    calibration_features = _extract_features(
        calibration_examples, calibration_candidates, encoder, tokenizer, device
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        encoder_dir = staging / "encoder"
        encoder.save_pretrained(encoder_dir)
        tokenizer.save_pretrained(encoder_dir)

        existence_dir = staging / "existence"
        existence_model, existence_mean, existence_scale, class_weights = (
            _fit_existence(
                existence_features,
                existence_labels,
                existence_dir,
                device,
            )
        )
        rank_dir = staging / "ranker"
        rank_model, rank_mean, rank_scale = _fit_ranker(
            rank_features, pair_indices, rank_dir, device
        )
        existence_scores = _score(
            existence_model,
            existence_mean,
            existence_scale,
            calibration_features,
            device,
            sigmoid=True,
        )
        rank_scores = _score(
            rank_model,
            rank_mean,
            rank_scale,
            calibration_features,
            device,
            sigmoid=False,
        )
        scored_candidates = [
            {
                **row,
                "existence_score": float(existence_score),
                "rank_logit": float(rank_score),
            }
            for row, existence_score, rank_score in zip(
                calibration_candidates, existence_scores, rank_scores
            )
        ]
        reference = {
            _candidate_key(row)
            for row in calibration_candidates
            if int(row["label"]) == 1
        }
        missed = int(lattice_manifest["counts"]["missed_reference_spans"])
        reference_count = len(reference) + missed
        if reference_count != CALIBRATION_REFERENCE_SPANS:
            raise RuntimeError(
                f"calibration reference denominator differs: {reference_count}"
            )

        table = []
        for threshold in THRESHOLDS:
            prediction = _resolve(scored_candidates, threshold)
            true_positive = len(reference & prediction)
            prediction_count = len(prediction)
            precision = (
                true_positive / prediction_count if prediction_count else 0.0
            )
            recall = true_positive / reference_count
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
            key=lambda row: (row["recall"], row["precision"], row["threshold"]),
            default=None,
        )

        scores_path = staging / "calibration-scores.json"
        scores_path.write_text(
            json.dumps(scored_candidates, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "status": (
                "ml_production_precision_grouped_verifier_selected_ai_assisted"
                if selected is not None
                else "ml_production_precision_grouped_verifier_blocked_ai_assisted"
            ),
            "revision": 6,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "git_commit": git_commit,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "encoder_artifact": _model_artifact(encoder_dir),
            "grouped_manifest_sha256": _sha256(grouped_manifest_path),
            "lattice_manifest_sha256": _sha256(lattice_manifest_path),
            "reference_manifest_sha256": _sha256(reference_manifest_path),
            "calibration_sha256": _sha256(calibration_path),
            "control": {
                "existence_seed": EXISTENCE_SEED,
                "rank_seed": RANK_SEED,
                "epochs": EPOCHS,
                "hidden_width": HIDDEN_WIDTH,
                "dropout": DROPOUT,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "encoder_frozen": True,
                "feature_dim": int(existence_features.shape[1]),
                "numeric_feature_size": NUMERIC_SIZE,
                "generator_metadata_in_learned_features": False,
                "existence_class_weights": class_weights,
                "rank_margin": 1.0,
                "rank_output": "unbounded_logit",
                "resolver": "descending_rank_greedy_nonoverlap",
            },
            "fit_inventory": {
                "existence_candidates": len(existence_rows),
                "existence_positives": int(existence_labels.sum()),
                "existence_negatives": int((existence_labels == 0).sum()),
                "rank_candidates": len(rank_candidates),
                "rank_pairs": len(rank_pairs),
                "existence_head_sha256": _sha256(
                    existence_dir / "head.safetensors"
                ),
                "existence_scaler_sha256": _sha256(
                    existence_dir / "scaler.npz"
                ),
                "rank_head_sha256": _sha256(rank_dir / "head.safetensors"),
                "rank_scaler_sha256": _sha256(rank_dir / "scaler.npz"),
            },
            "thresholds": list(THRESHOLDS),
            "table": table,
            "selected": selected,
            "calibration_scores_sha256": _sha256(scores_path),
            "claim_limit": (
                "AI-assisted diagnostic revision-6 grouped verifier; "
                "calibration metrics only, confirmation unread."
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
        description="Train and select the revision-6 grouped span verifier."
    )
    parser.add_argument("--grouped-data", type=Path, required=True)
    parser.add_argument("--lattice", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = train_grouped_verifier(
        args.grouped_data, args.lattice, args.reference, args.output
    )
    print(json.dumps({
        "status": manifest["status"],
        "selected": manifest["selected"],
        "table": manifest["table"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
