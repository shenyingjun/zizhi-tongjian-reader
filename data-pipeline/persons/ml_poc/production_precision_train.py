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
from production_train import SEEDS, _make_read_only


PARTITION_STATUS = "ml_production_precision_partition"
REFERENCE_STATUS = "ml_production_precision_reference_ai_assisted"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _line_count(path: Path) -> int:
    return sum(bool(line) for line in path.read_text(encoding="utf-8").splitlines())


def run_training(
    partition_root: Path,
    reference_root: Path,
    output_dir: Path,
    *,
    seed: int,
) -> dict:
    if seed not in SEEDS:
        raise ValueError(f"precision seed is not predeclared: {seed}")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"precision model output exists: {output_dir}")
    partition_manifest_path = partition_root / "manifest.json"
    partition_manifest = _read(partition_manifest_path)
    fit_path = partition_root / "fit.jsonl"
    reference_manifest_path = reference_root / "manifest.json"
    reference_manifest = _read(reference_manifest_path)
    calibration_path = reference_root / "calibration.jsonl"
    if (
        partition_manifest.get("status") != PARTITION_STATUS
        or partition_manifest.get("outputs", {}).get("fit_sha256")
        != _sha256(fit_path)
        or partition_manifest.get("partitions", {}).get("fit", {}).get("examples")
        != 189
        or _line_count(fit_path) != 189
        or reference_manifest.get("status") != REFERENCE_STATUS
        or reference_manifest.get("formal_grade") is not False
        or reference_manifest.get("eligible_for_production_precision_claim") is not False
        or reference_manifest.get("outputs", {}).get("calibration_sha256")
        != _sha256(calibration_path)
        or reference_manifest.get("splits", {}).get("calibration", {}).get("examples")
        != 45
        or _line_count(calibration_path) != 45
    ):
        raise ValueError("precision training input binding differs")
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
            (fit_path, "fit.jsonl"),
            (partition_manifest_path, "partition-manifest.json"),
            (calibration_path, "calibration.jsonl"),
            (reference_manifest_path, "reference-manifest.json"),
        ):
            (inputs / name).write_bytes(source.read_bytes())
        report = train(argparse.Namespace(
            dataset=None,
            train_file=inputs / "fit.jsonl",
            dev_file=inputs / "calibration.jsonl",
            evaluation_file=inputs / "calibration.jsonl",
            evaluation_name="calibration_duplicate_diagnostic",
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
            != _sha256(inputs / "fit.jsonl")
            or report.get("inputs", {}).get("dev_sha256")
            != _sha256(inputs / "calibration.jsonl")
        ):
            raise RuntimeError("precision training report differs")
        report["precision_control"] = {
            "git_commit": git_commit,
            "seed": seed,
            "partition_manifest_sha256": _sha256(
                inputs / "partition-manifest.json"
            ),
            "reference_manifest_sha256": _sha256(
                inputs / "reference-manifest.json"
            ),
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
            "formal_grade": False,
            "formal_evaluation": False,
            "eligible_for_production_precision_claim": False,
        }
        report["inputs"].update({
            "train": str(output_dir / "inputs" / "fit.jsonl"),
            "dev": str(output_dir / "inputs" / "calibration.jsonl"),
            "evaluation": str(output_dir / "inputs" / "calibration.jsonl"),
        })
        report["claim_limit"] = (
            "AI-assisted diagnostic calibration only; confirmation is untouched and "
            "this model cannot authorize production promotion."
        )
        (staging / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if _model_artifact(staging / "model") != report["precision_control"][
            "model_artifact"
        ]:
            raise RuntimeError("precision model artifact differs")
        _make_read_only(staging)
        staging.replace(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    report = run_training(
        args.partition,
        args.reference,
        args.output,
        seed=args.seed,
    )
    print(json.dumps({
        "seed": args.seed,
        "selected_epoch": report["config"]["selected_epoch"],
        "calibration": report["dev_challenge"]["exact"],
        "timing": report["timing"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
