from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LABELS = ("O", "B-PER", "I-PER")
TRAIN_JUANS = (13, 27)
HOLDOUT_JUAN = 52
CHALLENGE_TERMS = ("可汗", "单于")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_examples(juan: int, task: dict, state: dict) -> list[dict]:
    audit = state.get("role_audit", {})
    if not audit.get("complete"):
        raise ValueError(f"juan {juan} role audit is incomplete")
    annotations = audit["annotations"]
    by_para: dict[int, list[dict]] = {}
    for row in annotations:
        by_para.setdefault(int(row["para_id"]), []).append(row)
    examples = []
    seen_annotations = 0
    for jie in task["jies"]:
        text = str(jie["text"])
        labels = ["O"] * len(text)
        span_count = 0
        segments = {
            int(segment["para_id"]): segment for segment in jie["segments"]
        }
        for para_id, segment in segments.items():
            assembled_start = int(segment["assembled_start"])
            assembled_end = int(segment["assembled_end"])
            paragraph = text[assembled_start:assembled_end]
            for row in by_para.get(para_id, []):
                start = int(row["start"])
                end = int(row["end"])
                if not 0 <= start < end <= len(paragraph):
                    raise ValueError(f"annotation outside paragraph: {row}")
                if paragraph[start:end] != row["surface"]:
                    raise ValueError(f"annotation surface mismatch: {row}")
                absolute_start = assembled_start + start
                absolute_end = assembled_start + end
                if any(label != "O" for label in labels[absolute_start:absolute_end]):
                    raise ValueError(f"overlapping annotation: {row}")
                labels[absolute_start] = "B-PER"
                labels[absolute_start + 1:absolute_end] = (
                    ["I-PER"] * (absolute_end - absolute_start - 1)
                )
                span_count += 1
                seen_annotations += 1
        examples.append({
            "id": f"juan-{juan:03d}-jie-{int(jie['jie_index']):04d}",
            "juan": juan,
            "jie_index": int(jie["jie_index"]),
            "jie_number": jie.get("jie_number"),
            "text": text,
            "labels": labels,
            "span_count": span_count,
            "label_provenance": "human_audited",
            "segments": [
                {
                    "para_id": int(segment["para_id"]),
                    "assembled_start": int(segment["assembled_start"]),
                    "assembled_end": int(segment["assembled_end"]),
                }
                for segment in jie["segments"]
            ],
        })
    if seen_annotations != len(annotations):
        raise ValueError(
            f"mapped {seen_annotations} of {len(annotations)} annotations"
        )
    return examples


def choose_contiguous_dev(
    examples: list[dict],
    target_fraction: float = 0.2,
) -> tuple[list[dict], list[dict]]:
    if not examples:
        raise ValueError("challenge examples are empty")
    total_spans = sum(row["span_count"] for row in examples)
    target = max(1, round(total_spans * target_fraction))
    best = None
    for start in range(len(examples)):
        spans = 0
        challenge = 0
        for end in range(start + 1, len(examples) + 1):
            row = examples[end - 1]
            spans += row["span_count"]
            challenge += sum(row["text"].count(term) for term in CHALLENGE_TERMS)
            key = (
                abs(spans - target),
                -challenge,
                end - start,
                start,
            )
            if best is None or key < best[0]:
                best = (key, start, end)
    assert best is not None
    _, start, end = best
    dev = examples[start:end]
    train = examples[:start] + examples[end:]
    if not train:
        raise ValueError("dev split consumed every challenge example")
    return train, dev


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _split_summary(rows: list[dict]) -> dict:
    return {
        "jies": len(rows),
        "characters": sum(len(row["text"]) for row in rows),
        "spans": sum(row["span_count"] for row in rows),
        "juans": sorted({row["juan"] for row in rows}),
    }


def export_dataset(
    blind_dir: Path,
    state_dir: Path,
    report_dir: Path,
    boundary_guide: Path,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = {}
    frozen_inputs = {}
    for juan in (*TRAIN_JUANS, HOLDOUT_JUAN):
        task_path = blind_dir / f"blind_juan_{juan:03d}.json"
        state_path = state_dir / f"juan_{juan:03d}.json"
        report_path = report_dir / f"juan_{juan:03d}.json"
        examples[juan] = build_examples(
            juan, _read(task_path), _read(state_path)
        )
        frozen_inputs[str(juan)] = {
            "blind_sha256": _sha256(task_path),
            "state_sha256": _sha256(state_path),
            "report_sha256": _sha256(report_path),
        }

    challenge_train, dev = choose_contiguous_dev(examples[27])
    train = sorted(
        examples[13] + challenge_train,
        key=lambda row: (row["juan"], row["jie_index"]),
    )
    holdout = examples[HOLDOUT_JUAN]
    for name, rows in (
        ("train", train),
        ("dev", dev),
        ("pilot_holdout", holdout),
    ):
        _write_jsonl(output_dir / f"{name}.jsonl", rows)

    manifest = {
        "schema_version": 1,
        "task": "character BIO person-span tagging",
        "labels": list(LABELS),
        "reference_status": (
            "provisionally trusted; delayed self-agreement explicitly waived"
        ),
        "claim_limit": "exploratory P1 only; pilot holdout is not a sealed test",
        "identity_or_evidence_fields_present": False,
        "split_policy": {
            "train": "all juan 13 plus juan 27 outside contiguous dev block",
            "dev": (
                "contiguous juan 27 block nearest 20% of spans; "
                "ties favor 可汗/单于 challenge density"
            ),
            "pilot_holdout": "all random pilot juan 52",
        },
        "splits": {
            "train": _split_summary(train),
            "dev": _split_summary(dev),
            "pilot_holdout": _split_summary(holdout),
        },
        "frozen_inputs": frozen_inputs,
        "boundary_guide_sha256": _sha256(boundary_guide),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze audited P0 references as a char-BIO P1 dataset."
    )
    parser.add_argument("--blind-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--boundary-guide", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_dataset(
        args.blind_dir,
        args.state_dir,
        args.report_dir,
        args.boundary_guide,
        args.output,
    )
    print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
