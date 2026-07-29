from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from p1_dataset import build_examples
from p2_round import (
    _spans,
    aggregate_candidate_metrics,
    candidate_metrics,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def freeze_diagnostic(
    tasks_dir: Path,
    assisted_dir: Path,
    state_dir: Path,
    output_dir: Path,
    *,
    base_train: Path | None = None,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(
            f"diagnostic freeze output already exists: {output_dir}"
        )
    manifest_path = tasks_dir / "manifest.json"
    manifest = _read(manifest_path)
    if manifest.get("formal_p3") is not False:
        raise ValueError("input must explicitly be diagnostic-only")
    selections = manifest.get("selected", [])
    selected_juans = [int(row["juan"]) for row in selections]
    if len(selected_juans) != 5 or len(set(selected_juans)) != 5:
        raise ValueError("diagnostic freeze requires five distinct juans")
    examples = []
    inputs = {}
    per_juan = {}
    for selection in selections:
        juan = int(selection["juan"])
        if selection.get("mode") != "diagnostic_assisted":
            raise ValueError(f"juan {juan} is not diagnostic-assisted")
        task_path = tasks_dir / f"blind_juan_{juan:03d}.json"
        pack_path = assisted_dir / f"assisted_juan_{juan:03d}.json"
        state_path = state_dir / f"juan_{juan:03d}.json"
        task_sha256 = _sha256(task_path)
        pack_sha256 = _sha256(pack_path)
        if (
            task_sha256 != selection.get("task_sha256")
            or pack_sha256 != selection.get("pack_sha256")
        ):
            raise ValueError(f"frozen input hash differs for juan {juan}")
        task = _read(task_path)
        pack = _read(pack_path)
        state = _read(state_path)
        assisted = state.get("assisted", {})
        if not assisted.get("complete"):
            raise ValueError(f"diagnostic juan {juan} is incomplete")
        if assisted.get("pack_sha256") != pack_sha256:
            raise ValueError(
                f"diagnostic state pack binding differs for juan {juan}"
            )
        annotations = assisted["annotations"]
        examples.extend(build_examples(
            juan,
            task,
            {
                "role_audit": {
                    "complete": True,
                    "annotations": annotations,
                },
            },
            label_provenance="human_assisted_copilot_diagnostic",
        ))
        per_juan[str(juan)] = candidate_metrics(
            _spans(annotations), _spans(pack["candidates"])
        )
        inputs[str(juan)] = {
            "task_sha256": _sha256(task_path),
            "pack_sha256": _sha256(pack_path),
            "state_sha256": _sha256(state_path),
        }
    output_dir.mkdir(parents=True)
    _write_jsonl(output_dir / "train_diagnostic.jsonl", examples)
    report = {
        "schema_version": 1,
        "status": "frozen_diagnostic_assisted",
        "formal_p3": False,
        "eligible_for_sealed_metric": False,
        "label_provenance": "human_assisted_copilot_diagnostic",
        "examples": len(examples),
        "spans": sum(row["span_count"] for row in examples),
        "candidate_accuracy": aggregate_candidate_metrics(per_juan),
        "candidate_accuracy_by_juan": per_juan,
        "frozen_inputs": inputs,
        "source_manifest_sha256": _sha256(manifest_path),
    }
    if base_train is not None:
        combined = _read_jsonl(base_train) + examples
        identities = []
        for row in combined:
            juan = int(row["juan"])
            jie_index = int(row["jie_index"])
            expected_id = f"juan-{juan:03d}-jie-{jie_index:04d}"
            if row.get("id") != expected_id:
                raise ValueError(
                    f"training example has noncanonical id: {row.get('id')}"
                )
            identities.append((juan, jie_index))
        if len(identities) != len(set(identities)):
            raise ValueError(
                "combined training examples duplicate a juan/jie target"
            )
        _write_jsonl(output_dir / "train_combined.jsonl", combined)
        report["combined_train"] = {
            "base_train_sha256": _sha256(base_train),
            "diagnostic_train_sha256": _sha256(
                output_dir / "train_diagnostic.jsonl"
            ),
            "examples": len(combined),
            "spans": sum(int(row["span_count"]) for row in combined),
            "juans": sorted({int(row["juan"]) for row in combined}),
        }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze locked Copilot diagnostic corrections as BIO data."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--assisted", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-train", type=Path)
    args = parser.parse_args()
    report = freeze_diagnostic(
        args.tasks,
        args.assisted,
        args.state,
        args.output,
        base_train=args.base_train,
    )
    print(json.dumps({
        "examples": report["examples"],
        "spans": report["spans"],
        "formal_p3": report["formal_p3"],
        "combined_train": report.get("combined_train"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
