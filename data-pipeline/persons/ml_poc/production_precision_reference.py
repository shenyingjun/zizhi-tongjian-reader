from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tempfile
from pathlib import Path


REVIEW_STATUS = "ml_production_precision_reference_review_ai_assisted"
OUTPUT_STATUS = "ml_production_precision_reference_ai_assisted"


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> dict[tuple[int, int], dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (int(row["juan"]), int(row["jie_index"]))
        if key in result:
            raise ValueError(f"duplicate dataset pair: {key}")
        result[key] = row
    return result


def _dataset(root: Path) -> tuple[dict, str]:
    manifest_path = root / "manifest.json"
    manifest = _read(manifest_path)
    train_path = root / "train.jsonl"
    if (
        manifest.get("status")
        not in {
            "ml_production_round1_frozen_dataset",
            "ml_production_round2_frozen_dataset",
        }
        or manifest.get("outputs", {}).get("train_sha256") != _sha256(train_path)
    ):
        raise ValueError(f"source dataset binding differs: {root}")
    return _rows(train_path), _sha256(manifest_path)


def _labels_from_annotations(row: dict, task: dict, annotations: list[dict]) -> list[str]:
    text = str(row["text"])
    jie = task["jies"][0]
    if str(jie["text"]) != text or row["segments"] != jie["segments"]:
        raise ValueError(f"task and dataset text differ: {row['id']}")
    segments = {int(segment["para_id"]): segment for segment in row["segments"]}
    labels = ["O"] * len(text)
    previous = None
    for annotation in annotations:
        para_id = int(annotation["para_id"])
        start = int(annotation["start"])
        end = int(annotation["end"])
        geometry = (para_id, start, end)
        segment = segments.get(para_id)
        if segment is None or previous is not None and geometry < previous:
            raise ValueError(f"invalid annotation order: {row['id']}")
        assembled_start = int(segment["assembled_start"]) + start
        assembled_end = int(segment["assembled_start"]) + end
        if (
            not int(segment["assembled_start"])
            <= assembled_start
            < assembled_end
            <= int(segment["assembled_end"])
            or text[assembled_start:assembled_end] != annotation.get("surface")
            or any(label != "O" for label in labels[assembled_start:assembled_end])
        ):
            raise ValueError(f"invalid annotation geometry: {row['id']} {geometry}")
        labels[assembled_start] = "B-PER"
        labels[assembled_start + 1 : assembled_end] = ["I-PER"] * (
            assembled_end - assembled_start - 1
        )
        previous = geometry
    return labels


def freeze_references(
    review_root: Path,
    state_root: Path,
    round1_dataset: Path,
    round2_dataset: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"precision reference output exists: {output_dir}")
    review_manifest_path = review_root / "manifest.json"
    review_manifest = _read(review_manifest_path)
    review_manifest_sha = _sha256(review_manifest_path)
    private_path = review_root / "private" / "selection.json"
    private = _read(private_path)
    if (
        review_manifest.get("status") != REVIEW_STATUS
        or review_manifest.get("formal_grade") is not False
        or review_manifest.get("eligible_for_production_precision_claim") is not False
        or review_manifest.get("private_selection_sha256") != _sha256(private_path)
        or private.get("status") != "ml_production_precision_reference_private"
    ):
        raise ValueError("AI-assisted review binding differs")

    datasets = {}
    dataset_hashes = {}
    for number, root in ((1, round1_dataset), (2, round2_dataset)):
        datasets[number], dataset_hashes[number] = _dataset(root)
    selected = {
        str(row["task_id"]): row for row in review_manifest["selected"]
    }
    roles = {str(row["task_id"]): row for row in private["rows"]}
    expected_names = {f"task_{task_id}.json" for task_id in selected}
    state_paths = list(state_root.glob("task_*.json"))
    receipt_root = state_root / "completed"
    receipt_paths = list(receipt_root.glob("task_*.json"))
    if (
        len(selected) != 91
        or set(roles) != set(selected)
        or {path.name for path in state_paths} != expected_names
        or {path.name for path in receipt_paths} != expected_names
    ):
        raise ValueError("completed review state inventory differs")

    output_rows = {"calibration": [], "confirmation": []}
    state_inventory = {}
    for task_id in sorted(selected):
        selection = selected[task_id]
        role = roles[task_id]
        state_path = state_root / f"task_{task_id}.json"
        receipt_path = receipt_root / f"task_{task_id}.json"
        state = _read(state_path)
        receipt = _read(receipt_path)
        task_path = review_root / str(selection["task"])
        review_path = review_root / str(selection["review"])
        if (
            state.get("complete") is not True
            or state.get("source_manifest_sha256") != review_manifest_sha
            or state.get("task_sha256") != selection["task_sha256"]
            or state.get("review_sha256") != selection["review_sha256"]
            or receipt.get("source_manifest_sha256") != review_manifest_sha
            or receipt.get("state_sha256") != _sha256(state_path)
            or _sha256(task_path) != selection["task_sha256"]
            or _sha256(review_path) != selection["review_sha256"]
        ):
            raise ValueError(f"completed review binding differs: {task_id}")
        key = (int(role["juan"]), int(role["jie_index"]))
        row = dict(datasets[int(role["round"])][key])
        labels = _labels_from_annotations(
            row,
            _read(task_path),
            state["annotations"],
        )
        row["labels"] = labels
        row["span_count"] = labels.count("B-PER")
        row["label_provenance"] = "ai_assisted_precision_reference"
        output_rows[str(role["partition"])].append(row)
        state_inventory[task_id] = {
            "state_sha256": _sha256(state_path),
            "receipt_sha256": _sha256(receipt_path),
        }

    for rows in output_rows.values():
        rows.sort(key=lambda row: (int(row["juan"]), int(row["jie_index"])))
    if {key: len(value) for key, value in output_rows.items()} != {
        "calibration": 45,
        "confirmation": 46,
    }:
        raise ValueError("precision reference partition counts differ")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        outputs = {}
        splits = {}
        for split, rows in output_rows.items():
            path = staging / f"{split}.jsonl"
            path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            outputs[f"{split}_sha256"] = _sha256(path)
            splits[split] = {
                "examples": len(rows),
                "spans": sum(int(row["span_count"]) for row in rows),
            }
        manifest = {
            "schema_version": 1,
            "status": OUTPUT_STATUS,
            "formal_grade": False,
            "formal_evaluation": False,
            "eligible_for_training": False,
            "eligible_for_controller_calibration": True,
            "eligible_for_production_precision_claim": False,
            "review_manifest_sha256": review_manifest_sha,
            "private_selection_sha256": _sha256(private_path),
            "source_dataset_manifests": {
                str(number): digest for number, digest in dataset_hashes.items()
            },
            "splits": splits,
            "state_inventory": state_inventory,
            "outputs": outputs,
            "claim_limit": (
                "AI-assisted diagnostic evidence only; cannot authorize production "
                "precision, promotion, or formal evaluation."
            ),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in staging.rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--round1-dataset", type=Path, required=True)
    parser.add_argument("--round2-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_references(
        args.review,
        args.state,
        args.round1_dataset,
        args.round2_dataset,
        args.output,
    )
    print(json.dumps(manifest["splits"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
