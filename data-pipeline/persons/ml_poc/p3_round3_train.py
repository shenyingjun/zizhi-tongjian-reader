from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tempfile
from pathlib import Path

from p1_train import MODEL_NAME, MODEL_REVISION, train
from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact


CONTROL = {
    "epochs": 5,
    "max_length": 512,
    "stride": 128,
    "micro_batch_size": 1,
    "eval_batch_size": 2,
    "gradient_accumulation": 8,
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "seed": 20260727,
    "context_mode": "target_only",
}
EXPECTED_DATASET_MANIFEST_SHA256 = "PIN_AFTER_DATASET_BUILD"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(stat.S_IREAD)
    root.chmod(stat.S_IREAD)


def run_controlled_training(dataset_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Round 3 model output exists: {output_dir}")
    git_commit = _git_commit_clean()
    expected_names = {
        "train.jsonl", "dev.jsonl", "evaluation.jsonl", "manifest.json",
    }
    entries = list(dataset_dir.iterdir())
    if (
        {path.name for path in entries} != expected_names
        or any(not path.is_file() or path.is_symlink() for path in entries)
    ):
        raise ValueError("Round 3 dataset inventory differs")
    snapshots = {path.name: path.read_bytes() for path in entries}
    manifest_sha256 = hashlib.sha256(snapshots["manifest.json"]).hexdigest()
    manifest = json.loads(snapshots["manifest.json"].decode("utf-8"))
    if (
        manifest_sha256 != EXPECTED_DATASET_MANIFEST_SHA256
        or
        manifest.get("status") != "round3_controlled_training_dataset"
        or manifest.get("formal_evaluation") is not False
    ):
        raise ValueError("invalid Round 3 dataset manifest")
    names = {
        "train": "train.jsonl",
        "dev": "dev.jsonl",
        "evaluation": "evaluation.jsonl",
    }
    if any(
        hashlib.sha256(snapshots[name]).hexdigest()
        != manifest.get("outputs", {}).get(name)
        for name in names.values()
    ):
        raise ValueError("Round 3 dataset output hash differs")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        frozen_inputs = staging / "inputs"
        frozen_inputs.mkdir()
        for name in expected_names:
            (frozen_inputs / name).write_bytes(snapshots[name])
        paths = {
            key: frozen_inputs / name for key, name in names.items()
        }
        arguments = argparse.Namespace(
            dataset=None,
            train_file=paths["train"],
            dev_file=paths["dev"],
            evaluation_file=paths["evaluation"],
            evaluation_name="locked_blind_anchor_diagnostic",
            output=staging,
            model=MODEL_NAME,
            model_revision=MODEL_REVISION,
            **CONTROL,
        )
        report = train(arguments)
        config = report.get("config", {})
        required_report_config = {
            key: value for key, value in CONTROL.items()
            if key not in {"eval_batch_size", "max_grad_norm"}
        }
        if (
            report.get("model") != MODEL_NAME
            or report.get("model_revision") != MODEL_REVISION
            or report.get("evaluation", {}).get("name")
            != "locked_blind_anchor_diagnostic"
            or any(
                config.get(key) != value
                for key, value in required_report_config.items()
            )
            or any(
                report.get("inputs", {}).get(f"{name}_sha256")
                != _sha256(path)
                for name, path in paths.items()
            )
        ):
            raise RuntimeError("controlled training report differs")
        report["round3_control"] = {
            "git_commit": git_commit,
            "dataset_manifest_sha256": manifest_sha256,
            "base_model": MODEL_NAME,
            "base_model_revision": MODEL_REVISION,
            "full_control": dict(CONTROL),
            "model_artifact": _model_artifact(staging / "model"),
            "formal_evaluation": False,
            "eligible_for_promotion_without_fresh_sealed_set": False,
        }
        report["inputs"].update({
            name: str(output_dir / "inputs" / path.name)
            for name, path in paths.items()
        })
        run_artifact_names = (
            "history.json",
            "dev_predictions.json",
            "evaluation_predictions.json",
        )
        report["round3_control"]["run_artifacts"] = {
            name: _sha256(staging / name) for name in run_artifact_names
        }
        report["round3_control"]["input_snapshots"] = {
            name: hashlib.sha256(raw).hexdigest()
            for name, raw in snapshots.items()
        }
        report["claim_limit"] = (
            "Round 3 diagnostic comparison only; the locked blind anchor has "
            "been reused and cannot authorize formal promotion"
        )
        report_path = staging / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        expected_top_level = {
            "model", "inputs", "report.json", *run_artifact_names,
        }
        if (
            {path.name for path in staging.iterdir()} != expected_top_level
            or _model_artifact(staging / "model")
            != report["round3_control"]["model_artifact"]
            or any(
                _sha256(staging / name) != digest
                for name, digest
                in report["round3_control"]["run_artifacts"].items()
            )
        ):
            raise RuntimeError("Round 3 run artifact changed before publication")
        _make_read_only(staging)
        staging.replace(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the controlled Round 3 model training."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_controlled_training(args.dataset, args.output)
    print(json.dumps({
        "selected_epoch": report["config"]["selected_epoch"],
        "dev": report["dev_challenge"]["exact"],
        "evaluation": report["evaluation"]["exact"],
        "model_artifact": report["round3_control"]["model_artifact"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
