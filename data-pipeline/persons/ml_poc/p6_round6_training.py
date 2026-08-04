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
    "train.jsonl": "78995e7a26f2712a6a189283d9d64a1b82a6bc6945d5d9e43f03688571fa2c87",
    "dev.jsonl": "5fb15836f52baa989b1e65c17c19f2090efb9ca96b5ce075a484484de411a09f",
    "evaluation.jsonl": "5a6ce12b0f07579ab12b3d93eef6da34f336f4e20a0d173a1d78758fab9e804c",
    "manifest.json": "84c244e7719e891f9a53a618ffb5bca7de1756a596104ab29d974dc837496a7b",
}
EXPECTED_ASSISTED = {
    "train": "c1642bcfa0e67a4972b651248890d52ccc9e653b9628109eae821ef783923e69",
    "report": "00ae224fd3a84f9564eba66337d51fcb676b9b36221b7d46658b47f759ae01a4",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_round6_training(
    base_dataset: Path,
    assisted_freeze: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Round 6 dataset exists: {output_dir}")
    git_commit = _git_commit_clean()
    base_paths = {
        name: base_dataset / name
        for name in ("train.jsonl", "dev.jsonl", "evaluation.jsonl", "manifest.json")
    }
    assisted_paths = {
        "train": assisted_freeze / "train_assisted_round5.jsonl",
        "report": assisted_freeze / "report.json",
    }
    snapshots = {
        "base": {name: path.read_bytes() for name, path in base_paths.items()},
        "assisted": {
            name: path.read_bytes() for name, path in assisted_paths.items()
        },
    }
    hashes = {
        group: {
            name: hashlib.sha256(raw).hexdigest()
            for name, raw in values.items()
        }
        for group, values in snapshots.items()
    }
    if hashes["base"] != EXPECTED_BASE:
        raise ValueError("Round 4 base dataset binding differs")
    base_manifest = json.loads(snapshots["base"]["manifest.json"].decode("utf-8"))
    assisted_report = json.loads(snapshots["assisted"]["report"].decode("utf-8"))
    if (
        base_manifest.get("status") != "round4_controlled_training_dataset"
        or base_manifest.get("formal_evaluation") is not False
        or hashes["assisted"] != EXPECTED_ASSISTED
        or assisted_report.get("status")
        != "frozen_copilot_double_pass_diagnostic"
        or assisted_report.get("formal_evaluation") is not False
        or assisted_report.get("eligible_for_training") is not True
        or assisted_report.get("examples") != 60
        or assisted_report.get("spans") != 483
        or assisted_report.get("outputs", {}).get(
            "train_assisted_round5_sha256"
        ) != EXPECTED_ASSISTED["train"]
    ):
        raise ValueError("Round 5 assisted freeze binding differs")

    base_rows = _read_jsonl(snapshots["base"]["train.jsonl"], "base")
    assisted_rows = _read_jsonl(
        snapshots["assisted"]["train"], "assisted"
    )
    dev_rows = _read_jsonl(snapshots["base"]["dev.jsonl"], "dev")
    evaluation_rows = _read_jsonl(
        snapshots["base"]["evaluation.jsonl"], "evaluation"
    )
    splits = {
        "base": {int(row["juan"]) for row in base_rows},
        "assisted": {int(row["juan"]) for row in assisted_rows},
        "dev": {int(row["juan"]) for row in dev_rows},
        "evaluation": {int(row["juan"]) for row in evaluation_rows},
    }
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = splits[left] & splits[right]
            if overlap:
                raise ValueError(
                    f"whole-juan split overlap {left}/{right}: {sorted(overlap)}"
                )
    combined = sorted(
        base_rows + assisted_rows,
        key=lambda row: (int(row["juan"]), int(row["jie_index"])),
    )
    identities = [
        (int(row["juan"]), int(row["jie_index"])) for row in combined
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Round 6 training examples duplicate a juan/jie")

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
        outputs = {
            path.name: _sha256(path) for path in staging.iterdir()
        }
        manifest = {
            "schema_version": 1,
            "status": "round6_controlled_training_dataset",
            "formal_evaluation": False,
            "git_commit": git_commit,
            "split_policy": (
                "prior Round 4 train plus 60 reviewed double-pass assisted jies; "
                "unchanged Juan 27 dev and reused Juan 76 diagnostic evaluation"
            ),
            "splits": {
                "train": _summary(combined),
                "round4_train_component": _summary(base_rows),
                "round5_assisted_component": _summary(assisted_rows),
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
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the controlled Round 6 training dataset."
    )
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--assisted-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_round6_training(
        args.base_dataset, args.assisted_freeze, args.output
    )
    print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
