from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

import torch

from p1_train import MODEL_NAME, MODEL_REVISION, train
from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from p3_round3_train import CONTROL


DATASETS = {
    "round4": {
        "manifest_sha256": (
            "84c244e7719e891f9a53a618ffb5bca7de1756a596104ab29d974dc837496a7b"
        ),
        "status": "round4_controlled_training_dataset",
    },
    "round6": {
        "manifest_sha256": (
            "7788c8d3341e56a17eaa89fd582df6fcc930d9f5c723d1bd5a0c78b61f3852a1"
        ),
        "status": "round6_controlled_training_dataset",
    },
    "round7": {
        "manifest_sha256": (
            "09c2724e346b8df5b0ace423016155167790bc7324bf8876f1bfa5942e958eb5"
        ),
        "status": "round7_controlled_training_dataset",
    },
}
REPLICATION_SEEDS = (20260727, 20260728, 20260729)
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
EXPERIMENTS = {
    "baseline": {},
    "round8-lr2e-5": {"learning_rate": 2e-5},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(stat.S_IREAD)
    root.chmod(stat.S_IREAD)


def _configure_determinism() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_seed_replication(
    dataset_dir: Path,
    dataset_kind: str,
    seed: int,
    output_dir: Path,
    experiment: str = "baseline",
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"seed replication output exists: {output_dir}")
    if dataset_kind not in DATASETS:
        raise ValueError(f"unsupported dataset kind: {dataset_kind}")
    if seed not in REPLICATION_SEEDS:
        raise ValueError(f"unsupported replication seed: {seed}")
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unsupported replication experiment: {experiment}")
    if experiment != "baseline" and dataset_kind != "round7":
        raise ValueError(f"{experiment} requires the Round 7 dataset")
    git_commit = _git_commit_clean()
    expected_names = {
        "train.jsonl", "dev.jsonl", "evaluation.jsonl", "manifest.json",
    }
    entries = list(dataset_dir.iterdir())
    if (
        {path.name for path in entries} != expected_names
        or any(not path.is_file() or path.is_symlink() for path in entries)
    ):
        raise ValueError(f"{dataset_kind} dataset inventory differs")
    snapshots = {path.name: path.read_bytes() for path in entries}
    manifest_sha256 = hashlib.sha256(snapshots["manifest.json"]).hexdigest()
    manifest = json.loads(snapshots["manifest.json"])
    expected_dataset = DATASETS[dataset_kind]
    if (
        manifest_sha256 != expected_dataset["manifest_sha256"]
        or manifest.get("status") != expected_dataset["status"]
        or manifest.get("formal_evaluation") is not False
        or any(
            hashlib.sha256(snapshots[name]).hexdigest()
            != manifest.get("outputs", {}).get(name)
            for name in ("train.jsonl", "dev.jsonl", "evaluation.jsonl")
        )
    ):
        raise ValueError(f"{dataset_kind} dataset provenance differs")

    control = {**CONTROL, **EXPERIMENTS[experiment], "seed": seed}
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
            key: frozen_inputs / name
            for key, name in {
                "train": "train.jsonl",
                "dev": "dev.jsonl",
                "evaluation": "evaluation.jsonl",
            }.items()
        }
        _configure_determinism()
        report = train(argparse.Namespace(
            dataset=None,
            train_file=paths["train"],
            dev_file=paths["dev"],
            evaluation_file=paths["evaluation"],
            evaluation_name="locked_blind_anchor_diagnostic",
            output=staging,
            model=MODEL_NAME,
            model_revision=MODEL_REVISION,
            **control,
        ))
        required_config = {
            key: value for key, value in control.items()
            if key not in {"eval_batch_size", "max_grad_norm"}
        }
        if (
            report.get("model") != MODEL_NAME
            or report.get("model_revision") != MODEL_REVISION
            or report.get("evaluation", {}).get("name")
            != "locked_blind_anchor_diagnostic"
            or any(
                report.get("config", {}).get(key) != value
                for key, value in required_config.items()
            )
            or any(
                report.get("inputs", {}).get(f"{name}_sha256") != _sha256(path)
                for name, path in paths.items()
            )
        ):
            raise RuntimeError("seed replication report differs")
        artifact_names = (
            "history.json", "dev_predictions.json", "evaluation_predictions.json",
        )
        replication_control = {
            "git_commit": git_commit,
            "dataset_kind": dataset_kind,
            "dataset_manifest_sha256": manifest_sha256,
            "base_model": MODEL_NAME,
            "base_model_revision": MODEL_REVISION,
            "full_control": control,
            "determinism": {
                "torch_deterministic_algorithms": (
                    torch.are_deterministic_algorithms_enabled()
                ),
                "warn_only": (
                    torch.is_deterministic_algorithms_warn_only_enabled()
                ),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cublas_workspace_config": os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG"
                ),
            },
            "model_artifact": _model_artifact(staging / "model"),
            "run_artifacts": {
                name: _sha256(staging / name) for name in artifact_names
            },
            "input_snapshots": {
                name: hashlib.sha256(raw).hexdigest()
                for name, raw in snapshots.items()
            },
            "formal_evaluation": False,
            "eligible_for_promotion": False,
        }
        if experiment != "baseline":
            replication_control["experiment"] = experiment
        report["seed_replication_control"] = replication_control
        report["inputs"].update({
            name: str(output_dir / "inputs" / path.name)
            for name, path in paths.items()
        })
        report["claim_limit"] = (
            "Matched-seed replication diagnostic only; Juan 27 and Juan 76 "
            "are reused and cannot authorize promotion"
            if experiment == "baseline"
            else f"{experiment} matched-seed diagnostic only; Juan 27 and "
            "Juan 76 are reused and cannot authorize promotion"
        )
        (staging / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if (
            {path.name for path in staging.iterdir()}
            != {"model", "inputs", "report.json", *artifact_names}
            or _model_artifact(staging / "model")
            != report["seed_replication_control"]["model_artifact"]
            or report["seed_replication_control"]["determinism"] != {
                "torch_deterministic_algorithms": True,
                "warn_only": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
            }
            or any(
                _sha256(staging / name) != digest
                for name, digest
                in report["seed_replication_control"]["run_artifacts"].items()
            )
            or {path.name for path in frozen_inputs.iterdir()} != expected_names
            or any(
                not path.is_file()
                or path.is_symlink()
                or _sha256(path)
                != report["seed_replication_control"]["input_snapshots"].get(
                    path.name
                )
                for path in frozen_inputs.iterdir()
            )
        ):
            raise RuntimeError("seed replication changed before publication")
        _make_read_only(staging)
        staging.replace(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train one controlled matched-seed replication."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-kind", choices=sorted(DATASETS), required=True)
    parser.add_argument(
        "--seed", type=int, choices=REPLICATION_SEEDS, required=True
    )
    parser.add_argument(
        "--experiment", choices=sorted(EXPERIMENTS), default="baseline"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_seed_replication(
        args.dataset, args.dataset_kind, args.seed, args.output, args.experiment
    )
    result = {
        "dataset_kind": args.dataset_kind,
        "seed": args.seed,
        "selected_epoch": report["config"]["selected_epoch"],
        "dev": report["dev_challenge"]["exact"],
        "evaluation": report["evaluation"]["exact"],
        "model_artifact": report["seed_replication_control"]["model_artifact"],
    }
    if args.experiment != "baseline":
        result["experiment"] = args.experiment
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
