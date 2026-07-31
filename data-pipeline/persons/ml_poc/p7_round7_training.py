from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from p3_round3_training import _read_jsonl, _summary


EXPECTED_BASE = {
    "train.jsonl": "6d46c47b37f72c06549b89655000c793fd0eb5b168cb8e392feba170dc7ec769",
    "dev.jsonl": "5fb15836f52baa989b1e65c17c19f2090efb9ca96b5ce075a484484de411a09f",
    "evaluation.jsonl": "5a6ce12b0f07579ab12b3d93eef6da34f336f4e20a0d173a1d78758fab9e804c",
    "manifest.json": "7788c8d3341e56a17eaa89fd582df6fcc930d9f5c723d1bd5a0c78b61f3852a1",
}
EXPECTED_ROUND7 = {
    "train": "ff61a25a49c254d23f6de161c698d828741ced57bb039c0a23a8895704441345",
    "report": "1b361683ed8e823e3a949de25d78a7be78ca5b4a938c823b639efe367e7a834a",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_round7_training(
    base_dataset: Path,
    round7_freeze: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Round 7 dataset exists: {output_dir}")
    git_commit = _git_commit_clean()
    base_paths = {
        name: base_dataset / name
        for name in ("train.jsonl", "dev.jsonl", "evaluation.jsonl", "manifest.json")
    }
    round7_paths = {
        "train": round7_freeze / "train_assisted_round7.jsonl",
        "report": round7_freeze / "report.json",
    }
    snapshots = {
        "base": {name: path.read_bytes() for name, path in base_paths.items()},
        "round7": {name: path.read_bytes() for name, path in round7_paths.items()},
    }
    hashes = {
        group: {
            name: hashlib.sha256(raw).hexdigest()
            for name, raw in values.items()
        }
        for group, values in snapshots.items()
    }
    if hashes["base"] != EXPECTED_BASE or hashes["round7"] != EXPECTED_ROUND7:
        raise ValueError("Round 7 dataset input binding differs")
    base_manifest = json.loads(snapshots["base"]["manifest.json"])
    round7_report = json.loads(snapshots["round7"]["report"])
    if (
        base_manifest.get("status") != "round6_controlled_training_dataset"
        or base_manifest.get("formal_evaluation") is not False
        or round7_report.get("status")
        != "frozen_round7_copilot_assisted_training"
        or round7_report.get("training_only") is not True
        or round7_report.get("eligible_for_training") is not True
        or round7_report.get("eligible_for_evaluation") is not False
        or round7_report.get("examples") != 40
        or round7_report.get("characters") != 6183
        or round7_report.get("spans") != 460
        or round7_report.get("outputs", {}).get(
            "train_assisted_round7_sha256"
        )
        != EXPECTED_ROUND7["train"]
    ):
        raise ValueError("Round 7 frozen-label provenance differs")

    base_rows = _read_jsonl(snapshots["base"]["train.jsonl"], "base")
    round7_rows = _read_jsonl(snapshots["round7"]["train"], "round7")
    dev_rows = _read_jsonl(snapshots["base"]["dev.jsonl"], "dev")
    evaluation_rows = _read_jsonl(
        snapshots["base"]["evaluation.jsonl"], "evaluation"
    )
    splits = {
        "base": {int(row["juan"]) for row in base_rows},
        "round7": {int(row["juan"]) for row in round7_rows},
        "dev": {int(row["juan"]) for row in dev_rows},
        "evaluation": {int(row["juan"]) for row in evaluation_rows},
    }
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = splits[left] & splits[right]
            if overlap:
                raise ValueError(
                    f"whole-juan split overlap {left}/{right}: {sorted(overlap)}"
                )
    combined = sorted(
        base_rows + round7_rows,
        key=lambda row: (int(row["juan"]), int(row["jie_index"])),
    )
    identities = [
        (int(row["juan"]), int(row["jie_index"])) for row in combined
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Round 7 training examples duplicate a juan/jie")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        train_path = staging / "train.jsonl"
        train_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in combined
            ),
            encoding="utf-8",
        )
        (staging / "dev.jsonl").write_bytes(snapshots["base"]["dev.jsonl"])
        (staging / "evaluation.jsonl").write_bytes(
            snapshots["base"]["evaluation.jsonl"]
        )
        outputs = {path.name: _sha256(path) for path in staging.iterdir()}
        manifest = {
            "schema_version": 1,
            "status": "round7_controlled_training_dataset",
            "formal_evaluation": False,
            "eligible_for_promotion": False,
            "git_commit": git_commit,
            "split_policy": (
                "Round 6 train plus 40 fresh reviewed Round 7 training-only jies; "
                "unchanged reused Juan 27 dev and Juan 76 diagnostic evaluation"
            ),
            "splits": {
                "train": _summary(combined),
                "round6_train_component": _summary(base_rows),
                "round7_fresh_component": _summary(round7_rows),
                "dev": _summary(dev_rows),
                "evaluation": _summary(evaluation_rows),
            },
            "inputs": hashes,
            "outputs": outputs,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in staging.iterdir():
            path.chmod(stat.S_IREAD)
            if path.stat().st_mode & (
                stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
            ):
                raise PermissionError(f"failed to freeze dataset file: {path}")
        staging.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the controlled Round 7 training dataset."
    )
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--round7-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_round7_training(
        args.base_dataset, args.round7_freeze, args.output
    )
    print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
