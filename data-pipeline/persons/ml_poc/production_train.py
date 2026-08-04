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


SEEDS = (20260727, 20260728, 20260729)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def run_training(
    dataset_dir: Path,
    output_dir: Path,
    *,
    seed: int,
) -> dict:
    if seed not in SEEDS:
        raise ValueError(f"production seed is not predeclared: {seed}")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"production model output exists: {output_dir}")
    expected_names = {"train.jsonl", "development.jsonl", "manifest.json"}
    paths = list(dataset_dir.iterdir())
    if (
        {path.name for path in paths} != expected_names
        or any(not path.is_file() or path.is_symlink() for path in paths)
    ):
        raise ValueError("production dataset inventory differs")
    snapshots = {path.name: path.read_bytes() for path in paths}
    manifest = json.loads(snapshots["manifest.json"])
    actual_examples = {
        split: len([
            line for line in snapshots[f"{split}.jsonl"].splitlines() if line.strip()
        ])
        for split in ("train", "development")
    }
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "ml_production_round1_frozen_dataset"
        or manifest.get("eligible_for_training") is not True
        or manifest.get("eligible_for_checkpoint_selection") is not True
        or manifest.get("formal_evaluation") is not False
        or manifest.get("splits", {}).get("train", {}).get("examples") != 140
        or manifest.get("splits", {}).get("development", {}).get("examples") != 40
        or actual_examples != {"train": 140, "development": 40}
        or manifest.get("outputs", {}).get("train_sha256")
        != hashlib.sha256(snapshots["train.jsonl"]).hexdigest()
        or manifest.get("outputs", {}).get("development_sha256")
        != hashlib.sha256(snapshots["development.jsonl"]).hexdigest()
    ):
        raise ValueError("production dataset binding differs")
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
        for name, raw in snapshots.items():
            (inputs / name).write_bytes(raw)
        report = train(argparse.Namespace(
            dataset=None,
            train_file=inputs / "train.jsonl",
            dev_file=inputs / "development.jsonl",
            evaluation_file=inputs / "development.jsonl",
            evaluation_name="development_duplicate_not_independent",
            output=staging,
            model=MODEL_NAME,
            model_revision=MODEL_REVISION,
            **control,
        ))
        if (
            report.get("model") != MODEL_NAME
            or report.get("model_revision") != MODEL_REVISION
            or report.get("config", {}).get("seed") != seed
            or report.get("config", {}).get("learning_rate") != 3e-5
            or report.get("config", {}).get("context_mode") != "target_only"
            or report.get("inputs", {}).get("train_sha256")
            != _sha256(inputs / "train.jsonl")
            or report.get("inputs", {}).get("dev_sha256")
            != _sha256(inputs / "development.jsonl")
        ):
            raise RuntimeError("production training report differs")
        report["production_control"] = {
            "git_commit": git_commit,
            "dataset_manifest_sha256": _sha256(inputs / "manifest.json"),
            "seed": seed,
            "base_model": MODEL_NAME,
            "base_model_revision": MODEL_REVISION,
            "full_control": control,
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
            "formal_evaluation": False,
            "eligible_for_development_selection": True,
            "eligible_for_promotion": False,
        }
        report["inputs"].update({
            "train": str(output_dir / "inputs" / "train.jsonl"),
            "dev": str(output_dir / "inputs" / "development.jsonl"),
            "evaluation": str(output_dir / "inputs" / "development.jsonl"),
        })
        report["claim_limit"] = (
            "Fresh development checkpoint selection only; formal evaluation "
            "has not been created or consumed."
        )
        report_path = staging / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        artifact_names = {
            "history.json",
            "dev_predictions.json",
            "evaluation_predictions.json",
            "report.json",
        }
        if (
            {path.name for path in staging.iterdir()}
            != {"model", "inputs", *artifact_names}
            or _model_artifact(staging / "model")
            != report["production_control"]["model_artifact"]
        ):
            raise RuntimeError("production training artifact inventory differs")
        _make_read_only(staging)
        staging.replace(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train one deterministic production candidate seed."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    report = run_training(args.dataset, args.output, seed=args.seed)
    print(json.dumps({
        "seed": args.seed,
        "selected_epoch": report["config"]["selected_epoch"],
        "development": report["dev_challenge"]["exact"],
        "timing": report["timing"],
        "model_artifact": report["production_control"]["model_artifact"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
