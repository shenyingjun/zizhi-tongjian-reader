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
from p3_round3_train import CONTROL


EXPECTED_DATASET_MANIFEST_SHA256 = (
    "84c244e7719e891f9a53a618ffb5bca7de1756a596104ab29d974dc837496a7b"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(stat.S_IREAD)
    root.chmod(stat.S_IREAD)


def run_round4_training(dataset_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Round 4 model output exists: {output_dir}")
    git_commit = _git_commit_clean()
    expected_names = {
        "train.jsonl", "dev.jsonl", "evaluation.jsonl", "manifest.json",
    }
    entries = list(dataset_dir.iterdir())
    if (
        {path.name for path in entries} != expected_names
        or any(not path.is_file() or path.is_symlink() for path in entries)
    ):
        raise ValueError("Round 4 dataset inventory differs")
    snapshots = {path.name: path.read_bytes() for path in entries}
    manifest_sha256 = hashlib.sha256(snapshots["manifest.json"]).hexdigest()
    manifest = json.loads(snapshots["manifest.json"].decode("utf-8"))
    if (
        manifest_sha256 != EXPECTED_DATASET_MANIFEST_SHA256
        or manifest.get("status") != "round4_controlled_training_dataset"
        or manifest.get("formal_evaluation") is not False
    ):
        raise ValueError("invalid Round 4 dataset manifest")
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
        raise ValueError("Round 4 dataset output hash differs")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        frozen_inputs = staging / "inputs"
        frozen_inputs.mkdir()
        for name in expected_names:
            (frozen_inputs / name).write_bytes(snapshots[name])
        paths = {key: frozen_inputs / name for key, name in names.items()}
        report = train(argparse.Namespace(
            dataset=None,
            train_file=paths["train"],
            dev_file=paths["dev"],
            evaluation_file=paths["evaluation"],
            evaluation_name="locked_blind_anchor_diagnostic",
            output=staging,
            model=MODEL_NAME,
            model_revision=MODEL_REVISION,
            **CONTROL,
        ))
        config = report.get("config", {})
        required_config = {
            key: value for key, value in CONTROL.items()
            if key not in {"eval_batch_size", "max_grad_norm"}
        }
        if (
            report.get("model") != MODEL_NAME
            or report.get("model_revision") != MODEL_REVISION
            or report.get("evaluation", {}).get("name")
            != "locked_blind_anchor_diagnostic"
            or any(config.get(key) != value for key, value in required_config.items())
            or any(
                report.get("inputs", {}).get(f"{name}_sha256") != _sha256(path)
                for name, path in paths.items()
            )
        ):
            raise RuntimeError("controlled Round 4 report differs")
        report["round4_control"] = {
            "git_commit": git_commit,
            "dataset_manifest_sha256": manifest_sha256,
            "base_model": MODEL_NAME,
            "base_model_revision": MODEL_REVISION,
            "full_control": dict(CONTROL),
            "model_artifact": _model_artifact(staging / "model"),
            "formal_evaluation": False,
            "eligible_for_promotion": False,
        }
        report["inputs"].update({
            name: str(output_dir / "inputs" / path.name)
            for name, path in paths.items()
        })
        artifact_names = (
            "history.json", "dev_predictions.json", "evaluation_predictions.json",
        )
        report["round4_control"]["run_artifacts"] = {
            name: _sha256(staging / name) for name in artifact_names
        }
        report["round4_control"]["input_snapshots"] = {
            name: hashlib.sha256(raw).hexdigest()
            for name, raw in snapshots.items()
        }
        report["claim_limit"] = (
            "Round 4 diagnostic comparison only; Juan 27 and Juan 76 are reused"
        )
        (staging / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if (
            {path.name for path in staging.iterdir()}
            != {"model", "inputs", "report.json", *artifact_names}
            or _model_artifact(staging / "model")
            != report["round4_control"]["model_artifact"]
            or any(
                _sha256(staging / name) != digest
                for name, digest
                in report["round4_control"]["run_artifacts"].items()
            )
        ):
            raise RuntimeError("Round 4 run artifact changed before publication")
        _make_read_only(staging)
        staging.replace(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run controlled Round 4 model training."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_round4_training(args.dataset, args.output)
    print(json.dumps({
        "selected_epoch": report["config"]["selected_epoch"],
        "dev": report["dev_challenge"]["exact"],
        "evaluation": report["evaluation"]["exact"],
        "model_artifact": report["round4_control"]["model_artifact"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
