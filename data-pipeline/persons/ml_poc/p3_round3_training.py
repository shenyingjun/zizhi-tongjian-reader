from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean


EXPECTED_BASE_TRAIN_SHA256 = (
    "94ce32c2643e8a447934bc58772e6f78dc17b0a3c1e61eced280232efcd88aca"
)
EXPECTED_DEV_SHA256 = (
    "5fb15836f52baa989b1e65c17c19f2090efb9ca96b5ce075a484484de411a09f"
)
EXPECTED_EVALUATION_SHA256 = (
    "5a6ce12b0f07579ab12b3d93eef6da34f336f4e20a0d173a1d78758fab9e804c"
)
EXPECTED_BASE_REPORT_SHA256 = (
    "e5bf3104adf0016331eb5c9a061844f7afa4ead8895f95313579c1261f182e50"
)
EXPECTED_ACTIVE_REPORT_SHA256 = (
    "6514edb54edc0ac54b108fd340fbf699d223f277fa01f9def18c7f8c61bf005a"
)
EXPECTED_CONFIG = {
    "epochs": 5,
    "max_length": 512,
    "stride": 128,
    "micro_batch_size": 1,
    "gradient_accumulation": 8,
    "effective_batch_size": 8,
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "seed": 20260727,
    "precision": "fp32",
    "context_mode": "target_only",
}


def _snapshot(path: Path) -> tuple[bytes, str]:
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def _read_json(raw: bytes) -> dict:
    return json.loads(raw.decode("utf-8"))


def _read_jsonl(raw: bytes, source: str) -> list[dict]:
    rows = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line
    ]
    identities = []
    for row in rows:
        juan = int(row["juan"])
        jie_index = int(row["jie_index"])
        labels = list(row["labels"])
        legal_bio = all(
            label != "I-PER"
            or index > 0
            and labels[index - 1] in {"B-PER", "I-PER"}
            for index, label in enumerate(labels)
        )
        if (
            row.get("id") != f"juan-{juan:03d}-jie-{jie_index:04d}"
            or len(str(row["text"])) != len(labels)
            or any(label not in {"O", "B-PER", "I-PER"} for label in labels)
            or not legal_bio
            or int(row["span_count"]) != labels.count("B-PER")
        ):
            raise ValueError(f"invalid BIO example in {source}: {row.get('id')}")
        identities.append((juan, jie_index))
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate juan/jie identity in {source}")
    return rows


def _summary(rows: list[dict]) -> dict:
    return {
        "examples": len(rows),
        "characters": sum(len(str(row["text"])) for row in rows),
        "spans": sum(int(row["span_count"]) for row in rows),
        "juans": sorted({int(row["juan"]) for row in rows}),
    }


def prepare_round3_training(
    base_train: Path,
    active_train: Path,
    dev: Path,
    evaluation: Path,
    base_report: Path,
    active_report: Path,
    boundary_guide: Path,
    spec: Path,
    spec_zh: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Round 3 training data exists: {output_dir}")
    git_commit = _git_commit_clean()
    inputs = {}
    snapshots = {}
    for name, path in (
        ("base_train", base_train),
        ("active_train", active_train),
        ("dev", dev),
        ("evaluation", evaluation),
        ("base_report", base_report),
        ("active_report", active_report),
        ("boundary_guide", boundary_guide),
        ("spec", spec),
        ("spec_zh", spec_zh),
    ):
        raw, digest = _snapshot(path)
        snapshots[name] = raw
        inputs[name] = {"path": str(path), "sha256": digest}
    active_metadata = _read_json(snapshots["active_report"])
    base_metadata = _read_json(snapshots["base_report"])
    if (
        inputs["active_report"]["sha256"] != EXPECTED_ACTIVE_REPORT_SHA256
        or
        active_metadata.get("status") != "frozen_round3_active_training"
        or active_metadata.get("formal_evaluation") is not False
        or active_metadata.get("examples") != 60
        or active_metadata.get("spans") != 1460
        or active_metadata.get("candidate_decisions")
        != {"accept": 1453, "reject": 183}
        or active_metadata.get("outputs", {}).get(
            "train_round3_active_sha256"
        ) != inputs["active_train"]["sha256"]
    ):
        raise ValueError("active training artifact binding differs")
    if (
        inputs["base_train"]["sha256"] != EXPECTED_BASE_TRAIN_SHA256
        or inputs["dev"]["sha256"] != EXPECTED_DEV_SHA256
        or inputs["evaluation"]["sha256"] != EXPECTED_EVALUATION_SHA256
        or inputs["base_report"]["sha256"] != EXPECTED_BASE_REPORT_SHA256
        or base_metadata.get("model")
        != "KoichiYasuoka/roberta-classical-chinese-base-char"
        or base_metadata.get("evaluation", {}).get("name")
        != "locked_blind_anchor_diagnostic"
        or any(
            base_metadata.get("config", {}).get(key) != value
            for key, value in EXPECTED_CONFIG.items()
        )
        or Path(base_metadata.get("inputs", {}).get("train", "")).resolve()
        != base_train.resolve()
        or Path(base_metadata.get("inputs", {}).get("dev", "")).resolve()
        != dev.resolve()
        or Path(base_metadata.get("inputs", {}).get("evaluation", "")).resolve()
        != evaluation.resolve()
    ):
        raise ValueError("frozen Round 2 baseline binding differs")
    base = _read_jsonl(snapshots["base_train"], "base_train")
    active = _read_jsonl(snapshots["active_train"], "active_train")
    dev_rows = _read_jsonl(snapshots["dev"], "dev")
    evaluation_rows = _read_jsonl(snapshots["evaluation"], "evaluation")
    split_juans = {
        "base_train": {int(row["juan"]) for row in base},
        "active_train": {int(row["juan"]) for row in active},
        "dev": {int(row["juan"]) for row in dev_rows},
        "evaluation": {int(row["juan"]) for row in evaluation_rows},
    }
    names = list(split_juans)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            overlap = split_juans[left] & split_juans[right]
            if overlap:
                raise ValueError(
                    f"whole-juan split overlap {left}/{right}: "
                    f"{sorted(overlap)}"
                )
    combined = sorted(
        base + active,
        key=lambda row: (int(row["juan"]), int(row["jie_index"])),
    )
    identities = [
        (int(row["juan"]), int(row["jie_index"])) for row in combined
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("combined Round 3 train duplicates a juan/jie")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        train_path = staging / "train.jsonl"
        train_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
                for row in combined
            ),
            encoding="utf-8",
        )
        (staging / "dev.jsonl").write_bytes(snapshots["dev"])
        (staging / "evaluation.jsonl").write_bytes(snapshots["evaluation"])
        outputs = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in staging.iterdir()
        }
        manifest = {
            "schema_version": 1,
            "status": "round3_controlled_training_dataset",
            "formal_evaluation": False,
            "git_commit": git_commit,
            "split_policy": (
                "retrain from the same base model on prior Round 2 train plus "
                "60 audited active jies; retain the unchanged challenge dev "
                "and locked blind-anchor diagnostic"
            ),
            "splits": {
                "train": _summary(combined),
                "base_train_component": _summary(base),
                "active_train_component": _summary(active),
                "dev": _summary(dev_rows),
                "evaluation": _summary(evaluation_rows),
            },
            "inputs": inputs,
            "outputs": outputs,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in staging.iterdir():
            path.chmod(stat.S_IREAD)
        staging.chmod(stat.S_IREAD)
        if (
            {path.name for path in staging.iterdir()}
            != {"train.jsonl", "dev.jsonl", "evaluation.jsonl", "manifest.json"}
            or any(
                hashlib.sha256((staging / name).read_bytes()).hexdigest()
                != digest
                for name, digest in outputs.items()
            )
        ):
            raise RuntimeError("Round 3 dataset changed before publication")
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the controlled Round 3 training dataset."
    )
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--active-train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--active-report", type=Path, required=True)
    parser.add_argument("--boundary-guide", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-zh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_round3_training(
        args.base_train,
        args.active_train,
        args.dev,
        args.evaluation,
        args.base_report,
        args.active_report,
        args.boundary_guide,
        args.spec,
        args.spec_zh,
        args.output,
    )
    print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
