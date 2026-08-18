from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from p1_train import MODEL_NAME, MODEL_REVISION, train
from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from p3_round3_train import CONTROL
from production_precision_mining_plan import FOLD_NUMBERS, MINING_ORDER_SEED
from production_train import SEEDS, _make_read_only


PLAN_STATUS = "ml_production_precision_mining_plan"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _line_count(path: Path) -> int:
    return sum(bool(line) for line in path.read_text(encoding="utf-8").splitlines())


def run_mining_training(
    plan_root: Path,
    output_dir: Path,
    *,
    fold: int,
    seed: int,
) -> dict:
    """Train one mining model for ``(fold, seed)`` on that fold's train rows.

    The model is trained for exactly epoch 5 with no checkpoint selection and is a
    verifier-training artifact only: never a deployment or production candidate.
    The held-out fold is never read during training; it is evaluated only later by
    the separate out-of-fold inference path.
    """
    if fold not in FOLD_NUMBERS:
        raise ValueError(f"mining fold is not one of 1..5: {fold}")
    if seed not in SEEDS:
        raise ValueError(f"mining seed is not predeclared: {seed}")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"mining model output exists: {output_dir}")

    plan_manifest_path = plan_root / "manifest.json"
    plan_manifest = _read(plan_manifest_path)
    fold_dir = plan_root / "folds" / f"fold-{fold}"
    train_path = fold_dir / "train.jsonl"
    holdout_path = fold_dir / "holdout.jsonl"
    fold_outputs = plan_manifest.get("outputs", {}).get(str(fold), {})
    if (
        plan_manifest.get("status") != PLAN_STATUS
        or plan_manifest.get("mining_only") is not True
        or plan_manifest.get("eligible_for_deployment") is not False
        or plan_manifest.get("order_seed") != MINING_ORDER_SEED
        or fold_outputs.get("train_sha256") != _sha256(train_path)
        or fold_outputs.get("holdout_sha256") != _sha256(holdout_path)
        or _line_count(train_path) < 1
        or _line_count(holdout_path) < 1
    ):
        raise ValueError("mining training input binding differs")
    git_commit = _git_commit_clean()

    import torch
    import transformers

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    control = {**CONTROL, "seed": seed}

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        inputs = staging / "inputs"
        inputs.mkdir()
        for source, name in (
            (train_path, "train.jsonl"),
            (holdout_path, "holdout.jsonl"),
            (plan_manifest_path, "plan-manifest.json"),
        ):
            (inputs / name).write_bytes(source.read_bytes())
        # The held-out fold is passed only as an in-sample-free diagnostic evaluation
        # target that cannot select a checkpoint; training data is the fold's train
        # rows and the fold's own jies/juans are never in the training set.
        report = train(argparse.Namespace(
            dataset=None,
            train_file=inputs / "train.jsonl",
            dev_file=inputs / "train.jsonl",
            evaluation_file=inputs / "train.jsonl",
            evaluation_name="mining_train_insample_diagnostic",
            output=staging,
            model=MODEL_NAME,
            model_revision=MODEL_REVISION,
            checkpoint_selection="final_epoch",
            epoch_evaluation=False,
            **control,
        ))
        if (
            report.get("config", {}).get("selected_epoch") != 5
            or report.get("config", {}).get("checkpoint_selection") != "final_epoch"
            or report.get("config", {}).get("epoch_evaluation") is not False
            or report.get("inputs", {}).get("train_sha256")
            != _sha256(inputs / "train.jsonl")
        ):
            raise RuntimeError("mining training report differs")
        report["mining_control"] = {
            "git_commit": git_commit,
            "fold": fold,
            "seed": seed,
            "plan_manifest_sha256": _sha256(inputs / "plan-manifest.json"),
            "train_sha256": _sha256(inputs / "train.jsonl"),
            "holdout_sha256": _sha256(inputs / "holdout.jsonl"),
            "base_model": MODEL_NAME,
            "base_model_revision": MODEL_REVISION,
            "full_control": control,
            "checkpoint_selection": "fixed_epoch_5",
            "deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "tf32": False,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "transformers_version": transformers.__version__,
            "model_artifact": _model_artifact(staging / "model"),
            "mining_only": True,
            "eligible_for_deployment": False,
            "eligible_for_production": False,
            "eligible_for_production_precision_claim": False,
            "formal_grade": False,
            "formal_evaluation": False,
        }
        report["inputs"].update({
            "train": str(output_dir / "inputs" / "train.jsonl"),
            "dev": str(output_dir / "inputs" / "train.jsonl"),
            "evaluation": str(output_dir / "inputs" / "train.jsonl"),
            "holdout": str(output_dir / "inputs" / "holdout.jsonl"),
        })
        report["claim_limit"] = (
            "Mining out-of-fold generator model only. It is a verifier-training "
            "artifact, trained on four of five fit folds, and cannot authorize "
            "deployment or production promotion."
        )
        (staging / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if _model_artifact(staging / "model") != report["mining_control"][
            "model_artifact"
        ]:
            raise RuntimeError("mining model artifact differs")
        _make_read_only(staging)
        staging.replace(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train one revision-4 mining out-of-fold generator model (fold, seed)."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=FOLD_NUMBERS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    report = run_mining_training(
        args.plan,
        args.output,
        fold=args.fold,
        seed=args.seed,
    )
    print(json.dumps({
        "fold": args.fold,
        "seed": args.seed,
        "selected_epoch": report["config"]["selected_epoch"],
        "timing": report["timing"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
