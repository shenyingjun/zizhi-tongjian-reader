from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from p1_train import MODEL_REVISION
from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from production_precision_grouped_data import GROUPED_DATA_STATUS
from production_precision_grouped_verifier import (
    BATCH_SIZE,
    CALIBRATION_REFERENCE_SPANS,
    DROPOUT,
    EPOCHS,
    HIDDEN_WIDTH,
    LATTICE_STATUS,
    LEARNING_RATE,
    NUMERIC_SIZE,
    REFERENCE_STATUS,
    WEIGHT_DECAY,
    _feature_key,
    _fit_scaler,
    _read,
    _read_jsonl,
    _resolve,
    _score,
    _sha256,
)
from production_precision_negative_audit_freeze import (
    EXPECTED_SAFE,
    FROZEN_STATUS,
)
from production_precision_select import _metric
from production_precision_verifier import _build_head, _extract_features
from production_span_verifier import THRESHOLDS, _candidate_key
from production_train import _make_read_only


REVISION6_STATUS = "ml_production_precision_grouped_verifier_blocked_ai_assisted"
EXISTENCE_SEED = 20260816
EXPECTED_POSITIVES = 2525
EXPECTED_REAL_NEGATIVES = 171
STRATUM_MASSES = {
    "real_positive": 0.50,
    "real_negative": 0.25,
    "mined_negative": 0.25,
}


def _canonical_key(row: dict) -> tuple[int, int, int, int, int]:
    return (
        int(row["juan"]),
        int(row["jie_index"]),
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


def _stratum_weights(strata: list[str]) -> tuple[np.ndarray, dict]:
    total = len(strata)
    counts = {name: strata.count(name) for name in STRATUM_MASSES}
    if not total or any(not counts[name] for name in STRATUM_MASSES):
        raise ValueError("existence loss strata must all be non-empty")
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


def _fit_existence(
    features: np.ndarray,
    labels: np.ndarray,
    strata: list[str],
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
    row_weights, weight_inventory = _stratum_weights(strata)
    model = _build_head(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator().manual_seed(EXISTENCE_SEED)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized).float(),
            torch.from_numpy(labels).float(),
            torch.from_numpy(row_weights).float(),
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
            losses = F.binary_cross_entropy_with_logits(
                logits, batch_labels.to(device), reduction="none"
            )
            loss = (losses * batch_weights.to(device)).mean()
            loss.backward()
            optimizer.step()
    output_dir.mkdir(parents=True)
    save_file(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        output_dir / "head.safetensors",
    )
    np.savez(output_dir / "scaler.npz", mean=mean, scale=scale)
    model.eval()
    return model, mean, scale, weight_inventory


def _load_ranker(revision6_root: Path, feature_dim: int, device):
    from safetensors.torch import load_file

    rank_dir = revision6_root / "ranker"
    model = _build_head(feature_dim).to(device)
    model.load_state_dict(
        load_file(rank_dir / "head.safetensors", device=str(device))
    )
    model.eval()
    scaler = np.load(rank_dir / "scaler.npz")
    return (
        model,
        scaler["mean"].astype(np.float32),
        scaler["scale"].astype(np.float32),
    )


def train_lexical_verifier(
    grouped_root: Path,
    safe_root: Path,
    revision6_root: Path,
    lattice_root: Path,
    reference_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"lexical verifier output exists: {output_dir}")
    grouped_manifest_path = grouped_root / "manifest.json"
    grouped_manifest = _read(grouped_manifest_path)
    examples_path = grouped_root / "examples.jsonl"
    existence_path = grouped_root / "existence.jsonl"
    fit_examples = {
        str(row["id"]): row for row in _read_jsonl(examples_path)
    }
    real_rows = _read_jsonl(existence_path)
    grouped_outputs = grouped_manifest.get("outputs", {})
    positive_count = sum(int(row["label"]) == 1 for row in real_rows)
    real_negative_count = len(real_rows) - positive_count
    if (
        grouped_manifest.get("status") != GROUPED_DATA_STATUS
        or grouped_manifest.get("confirmation_read") is not False
        or grouped_outputs.get("examples_sha256") != _sha256(examples_path)
        or grouped_outputs.get("existence_sha256") != _sha256(existence_path)
        or positive_count != EXPECTED_POSITIVES
        or real_negative_count != EXPECTED_REAL_NEGATIVES
    ):
        raise ValueError("Revision-9 grouped-data binding differs")

    safe_manifest_path = safe_root / "manifest.json"
    safe_manifest = _read(safe_manifest_path)
    safe_path = safe_root / "safe-negatives.jsonl"
    safe_rows = _read_jsonl(safe_path)
    if (
        safe_manifest.get("status") != FROZEN_STATUS
        or safe_manifest.get("confirmation_read") is not False
        or safe_manifest.get("counts", {}).get("audit_exclusions") != 0
        or safe_manifest.get("outputs", {}).get("safe_negatives_sha256")
        != _sha256(safe_path)
        or safe_manifest.get("bindings", {}).get("grouped_manifest_sha256")
        != _sha256(grouped_manifest_path)
        or len(safe_rows) != EXPECTED_SAFE
    ):
        raise ValueError("Revision-9 safe-negative binding differs")

    revision6_manifest_path = revision6_root / "manifest.json"
    revision6_manifest = _read(revision6_manifest_path)
    encoder_source = revision6_root / "encoder"
    rank_source = revision6_root / "ranker"
    feature_dim = int(revision6_manifest.get("control", {}).get(
        "feature_dim", -1
    ))
    if (
        revision6_manifest.get("status") != REVISION6_STATUS
        or revision6_manifest.get("revision") != 6
        or revision6_manifest.get("confirmation_read") is not False
        or revision6_manifest.get("model_revision") != MODEL_REVISION
        or revision6_manifest.get("grouped_manifest_sha256")
        != _sha256(grouped_manifest_path)
        or revision6_manifest.get("encoder_artifact")
        != _model_artifact(encoder_source)
        or revision6_manifest.get("fit_inventory", {}).get(
            "rank_head_sha256"
        ) != _sha256(rank_source / "head.safetensors")
        or revision6_manifest.get("fit_inventory", {}).get(
            "rank_scaler_sha256"
        ) != _sha256(rank_source / "scaler.npz")
        or safe_manifest.get("bindings", {}).get("revision6_manifest_sha256")
        != _sha256(revision6_manifest_path)
    ):
        raise ValueError("Revision-9 ranker binding differs")

    all_keys = set()
    for row in [*real_rows, *safe_rows]:
        key = _canonical_key(row)
        if key in all_keys:
            raise ValueError(f"existence geometry collision: {key}")
        all_keys.add(key)
        identity = str(row["id"])
        if (
            identity not in fit_examples
            or int(fit_examples[identity]["juan"]) != int(row["juan"])
            or int(fit_examples[identity]["jie_index"]) != int(row["jie_index"])
        ):
            raise ValueError(f"existence example binding differs: {key}")
    existence_rows = [
        *real_rows,
        *[{**row, "label": 0} for row in safe_rows],
    ]
    existence_labels = np.asarray(
        [int(row["label"]) for row in existence_rows], dtype=np.float32
    )
    strata = [
        *[
            "real_positive" if int(row["label"]) else "real_negative"
            for row in real_rows
        ],
        *(["mined_negative"] * len(safe_rows)),
    ]

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
        or {int(row["juan"]) for row in fit_examples.values()} & {
            int(row["juan"]) for row in calibration_examples.values()
        }
    ):
        raise ValueError("Revision-9 calibration binding differs")

    git_commit = _git_commit_clean()
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(encoder_source, use_fast=True)
    encoder = AutoModel.from_pretrained(encoder_source).to(device)
    existence_features = _extract_features(
        fit_examples, existence_rows, encoder, tokenizer, device
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
        rank_dir = staging / "ranker"
        shutil.copytree(encoder_source, encoder_dir, copy_function=shutil.copy2)
        shutil.copytree(rank_source, rank_dir, copy_function=shutil.copy2)
        if (
            _model_artifact(encoder_dir)
            != revision6_manifest["encoder_artifact"]
            or _sha256(rank_dir / "head.safetensors")
            != revision6_manifest["fit_inventory"]["rank_head_sha256"]
            or _sha256(rank_dir / "scaler.npz")
            != revision6_manifest["fit_inventory"]["rank_scaler_sha256"]
        ):
            raise RuntimeError("copied Revision-6 rank artifact differs")

        existence_dir = staging / "existence"
        existence_model, mean, scale, weight_inventory = _fit_existence(
            existence_features,
            existence_labels,
            strata,
            existence_dir,
            device,
        )
        positive_indices = np.flatnonzero(existence_labels == 1)
        fit_positive_scores = _score(
            existence_model,
            mean,
            scale,
            existence_features[positive_indices],
            device,
            sigmoid=True,
        )
        fit_positive_recall = float(
            np.mean(fit_positive_scores >= np.float32(0.50))
        )
        if fit_positive_recall < 0.95:
            raise RuntimeError(
                "Revision-9 fit-positive sanity tripwire failed: "
                f"{fit_positive_recall}"
            )

        rank_model, rank_mean, rank_scale = _load_ranker(
            revision6_root, feature_dim, device
        )
        existence_scores = _score(
            existence_model,
            mean,
            scale,
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
        reference_count = (
            len(reference)
            + int(lattice_manifest["counts"]["missed_reference_spans"])
        )
        if reference_count != CALIBRATION_REFERENCE_SPANS:
            raise RuntimeError("Revision-9 calibration denominator differs")
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
                "ml_production_precision_lexical_verifier_selected_ai_assisted"
                if selected is not None
                else "ml_production_precision_lexical_verifier_blocked_ai_assisted"
            ),
            "revision": 9,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "git_commit": git_commit,
            "model_revision": MODEL_REVISION,
            "encoder_artifact": _model_artifact(encoder_dir),
            "bindings": {
                "grouped_manifest_sha256": _sha256(grouped_manifest_path),
                "safe_manifest_sha256": _sha256(safe_manifest_path),
                "revision6_manifest_sha256": _sha256(
                    revision6_manifest_path
                ),
                "lattice_manifest_sha256": _sha256(lattice_manifest_path),
                "reference_manifest_sha256": _sha256(
                    reference_manifest_path
                ),
            },
            "control": {
                "existence_seed": EXISTENCE_SEED,
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
                "stratum_weights": weight_inventory,
                "ranker_copied_from_revision": 6,
            },
            "fit_inventory": {
                "existence_candidates": len(existence_rows),
                "real_positives": positive_count,
                "real_negatives": real_negative_count,
                "mined_negatives": len(safe_rows),
                "fit_positive_recall_at_0_50": fit_positive_recall,
                "existence_head_sha256": _sha256(
                    existence_dir / "head.safetensors"
                ),
                "existence_scaler_sha256": _sha256(
                    existence_dir / "scaler.npz"
                ),
                "rank_head_sha256": _sha256(
                    rank_dir / "head.safetensors"
                ),
                "rank_scaler_sha256": _sha256(rank_dir / "scaler.npz"),
            },
            "thresholds": list(THRESHOLDS),
            "table": table,
            "selected": selected,
            "calibration_scores_sha256": _sha256(scores_path),
            "claim_limit": (
                "AI-assisted diagnostic Revision-9 existence retraining; "
                "unchanged calibration only, confirmation unread."
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
        description="Train the Revision-9 lexical existence verifier."
    )
    parser.add_argument("--grouped-data", type=Path, required=True)
    parser.add_argument("--safe-negatives", type=Path, required=True)
    parser.add_argument("--revision-6", type=Path, required=True)
    parser.add_argument("--lattice", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = train_lexical_verifier(
        args.grouped_data,
        args.safe_negatives,
        args.revision_6,
        args.lattice,
        args.reference,
        args.output,
    )
    print(json.dumps({
        "status": manifest["status"],
        "fit_inventory": manifest["fit_inventory"],
        "selected": manifest["selected"],
        "table": manifest["table"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
