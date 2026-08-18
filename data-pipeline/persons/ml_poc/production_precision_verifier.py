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
from production_precision_select import _metric
from production_span_verifier import (
    BOUNDARY_CATEGORIES,
    THRESHOLDS,
    _candidate_key,
    _category,
    _pool_candidate_encodings,
    _resolve,
)
from production_train import _make_read_only


# Revision-4 fixed verifier controls (section 5.3.3). The encoder is frozen; only
# the scaler and the two-layer head are learned, once, on the fit inventory.
VERIFIER_SEED = 20260812
EPOCHS = 20
HIDDEN_WIDTH = 256
DROPOUT = 0.10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 32
POSITIVE_CLASS_WEIGHT = 1.0

# log1p(length) + left/right boundary one-hot + two paragraph-edge bits.
NUMERIC_SIZE = 1 + len(BOUNDARY_CATEGORIES) * 2 + 2

HARD_NEGATIVE_STATUS = "ml_production_precision_mining_hard_negatives_ai_assisted"
LATTICE_STATUS = "ml_production_verifier_lattice_ai_assisted"
REFERENCE_STATUS = "ml_production_precision_reference_ai_assisted"
CALIBRATION_REFERENCE_SPANS = 469


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _extract_features(
    examples: dict[str, dict],
    candidates: list[dict],
    encoder,
    tokenizer,
    device,
) -> np.ndarray:
    """Revision-4 verifier features.

    Encoder candidate/left/right/context poolings (shared, frozen) plus the only
    non-encoder inputs allowed by section 5.3.3: log1p(span length), one-hot
    boundary categories for the immediate neighbours, and two paragraph-edge bits.
    Generator support count and seed confidences are deliberately excluded so that
    synthetic negatives cannot be told apart by missing generator metadata.
    """
    hidden_size, pooled = _pool_candidate_encodings(
        examples, candidates, encoder, tokenizer, device
    )
    features = np.zeros(
        (len(candidates), hidden_size * 4 + NUMERIC_SIZE),
        dtype=np.float32,
    )
    for output_index in range(len(candidates)):
        parts = pooled[output_index]
        numeric = [math.log1p(parts["length"])]
        for value in (parts["left_character"], parts["right_character"]):
            category = _category(value)
            numeric.extend(
                float(category == expected) for expected in BOUNDARY_CATEGORIES
            )
        numeric.append(float(parts["starts_paragraph"]))
        numeric.append(float(parts["ends_paragraph"]))
        features[output_index] = np.concatenate((
            parts["candidate_mean"],
            parts["left_hidden"],
            parts["right_hidden"],
            parts["context_mean"],
            np.asarray(numeric, dtype=np.float32),
        ))
    return features


def _build_head(width: int):
    from torch import nn

    return nn.Sequential(
        nn.Linear(width, HIDDEN_WIDTH),
        nn.GELU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN_WIDTH, 1),
    )


def _fit_head(
    features: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    device,
) -> tuple[object, np.ndarray, np.ndarray]:
    """Fit the scaler and head once on the fit inventory only."""
    import torch
    from safetensors.torch import save_file
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(VERIFIER_SEED)
    np.random.seed(VERIFIER_SEED)
    torch.manual_seed(VERIFIER_SEED)
    torch.cuda.manual_seed_all(VERIFIER_SEED)

    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (features - mean) / scale

    model = _build_head(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(POSITIVE_CLASS_WEIGHT, device=device)
    )
    generator = torch.Generator().manual_seed(VERIFIER_SEED)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized).float(),
            torch.from_numpy(labels).float(),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    for _ in range(EPOCHS):
        model.train()
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features.to(device)).squeeze(-1)
            loss = loss_function(logits, batch_labels.to(device))
            loss.backward()
            optimizer.step()
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
    model.eval()
    return model, mean, scale


def _score(model, mean, scale, features: np.ndarray, device) -> np.ndarray:
    import torch

    normalized = (features - mean) / scale
    with torch.inference_mode():
        return (
            model(torch.from_numpy(normalized).float().to(device))
            .squeeze(-1)
            .sigmoid()
            .cpu()
            .numpy()
        )


def train_verifier(
    negatives_root: Path,
    lattice_root: Path,
    reference_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"revision-4 verifier output exists: {output_dir}")

    negatives_manifest_path = negatives_root / "manifest.json"
    negatives_manifest = _read(negatives_manifest_path)
    fit_candidates_path = negatives_root / "candidates.jsonl"
    fit_examples_path = negatives_root / "examples.jsonl"
    fit_candidates = _read_jsonl(fit_candidates_path)
    fit_examples = {str(row["id"]): row for row in _read_jsonl(fit_examples_path)}

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

    floors = negatives_manifest.get("floors", {})
    outputs = negatives_manifest.get("outputs", {})
    if (
        negatives_manifest.get("status") != HARD_NEGATIVE_STATUS
        or negatives_manifest.get("mining_only") is not True
        or negatives_manifest.get("confirmation_read") is not False
        or not negatives_manifest.get("oof_recall_gate", {}).get("passed")
        or not all(gate.get("passed") for gate in floors.values())
        or outputs.get("candidates_sha256") != _sha256(fit_candidates_path)
        or outputs.get("examples_sha256") != _sha256(fit_examples_path)
        or lattice_manifest.get("status") != LATTICE_STATUS
        or lattice_manifest.get("candidate_recall", 0) < 0.98
        or lattice_manifest.get("lattice_sha256") != _sha256(lattice_path)
        or len(calibration_candidates) != 517
        or reference_manifest.get("status") != REFERENCE_STATUS
        or reference_manifest.get("outputs", {}).get("calibration_sha256")
        != _sha256(calibration_path)
        or len(calibration_examples) != 45
    ):
        raise ValueError("revision-4 verifier input binding differs")

    # Fit / calibration separation: fit candidates only reference fit examples and
    # calibration candidates only reference calibration examples.
    if not {str(row["id"]) for row in fit_candidates}.issubset(fit_examples):
        raise ValueError("fit candidate references a non-fit example")
    if not {str(row["id"]) for row in calibration_candidates}.issubset(
        calibration_examples
    ):
        raise ValueError("calibration candidate references a non-calibration example")
    fit_juans = {int(row["juan"]) for row in fit_examples.values()}
    calibration_juans = {int(row["juan"]) for row in calibration_examples.values()}
    if fit_juans & calibration_juans:
        raise ValueError("fit and calibration juans overlap")

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

    fit_features = _extract_features(
        fit_examples, fit_candidates, encoder, tokenizer, device
    )
    fit_labels = np.asarray(
        [int(row["label"]) for row in fit_candidates], dtype=np.float32
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

        verifier_dir = staging / "verifier"
        model, mean, scale = _fit_head(
            fit_features, fit_labels, verifier_dir, device
        )
        calibration_scores = _score(
            model, mean, scale, calibration_features, device
        )

        scored_candidates = [
            {**row, "score": float(score)}
            for row, score in zip(calibration_candidates, calibration_scores)
        ]
        reference = {
            _candidate_key(row)
            for row in calibration_candidates
            if int(row["label"]) == 1
        }
        missed_reference_spans = int(
            lattice_manifest["counts"]["missed_reference_spans"]
        )
        reference_count = len(reference) + missed_reference_spans
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
        fit_features_path = staging / "fit-features.npz"
        np.savez_compressed(fit_features_path, features=fit_features)
        manifest = {
            "schema_version": 1,
            "status": (
                "ml_production_precision_verifier_selected_ai_assisted"
                if selected is not None
                else "ml_production_precision_verifier_blocked_ai_assisted"
            ),
            "revision": 4,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "git_commit": git_commit,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "encoder_artifact": _model_artifact(encoder_dir),
            "negatives_manifest_sha256": _sha256(negatives_manifest_path),
            "lattice_manifest_sha256": _sha256(lattice_manifest_path),
            "reference_manifest_sha256": _sha256(reference_manifest_path),
            "calibration_sha256": _sha256(calibration_path),
            "control": {
                "seed": VERIFIER_SEED,
                "epochs": EPOCHS,
                "hidden_width": HIDDEN_WIDTH,
                "dropout": DROPOUT,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "positive_class_weight": POSITIVE_CLASS_WEIGHT,
                "encoder_frozen": True,
                "single_training_on_fit_only": True,
                "scaler_fit_on_fit_only": True,
                "feature_dim": int(fit_features.shape[1]),
                "numeric_feature_size": NUMERIC_SIZE,
                "generator_metadata_in_features": False,
            },
            "fit_inventory": {
                "candidates": len(fit_candidates),
                "positives": int(fit_labels.sum()),
                "negatives": int((fit_labels == 0).sum()),
                "head_sha256": _sha256(verifier_dir / "head.safetensors"),
                "scaler_sha256": _sha256(verifier_dir / "scaler.npz"),
            },
            "thresholds": list(THRESHOLDS),
            "table": table,
            "selected": selected,
            "calibration_scores_sha256": _sha256(scores_path),
            "fit_features_sha256": _sha256(fit_features_path),
            "claim_limit": (
                "AI-assisted diagnostic verifier trained on fit-only mining hard "
                "negatives; calibration metrics only, confirmation unread, no "
                "production authority."
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
            "Train and select the revision-4 fit-only hard-negative span verifier."
        )
    )
    parser.add_argument("--negatives", type=Path, required=True)
    parser.add_argument("--lattice", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = train_verifier(
        args.negatives, args.lattice, args.reference, args.output
    )
    print(json.dumps({
        "status": manifest["status"],
        "selected": manifest["selected"],
        "table": manifest["table"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
