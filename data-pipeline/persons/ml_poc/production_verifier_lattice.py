from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tempfile
import unicodedata
from pathlib import Path

from p3_compact import _git_commit_clean
from production_train import SEEDS


MINIMUM_CONFIDENCE = 0.30
EXPECTED_JUANS = {105, 151, 278, 282}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _geometry(span: dict) -> tuple[int, int, int, str]:
    return (
        int(span["para_id"]),
        int(span["start"]),
        int(span["end"]),
        str(span["surface"]),
    )


def _source_surface(example: dict, geometry: tuple[int, int, int, str]) -> str:
    para_id, start, end, _ = geometry
    matches = [
        segment for segment in example["segments"]
        if int(segment["para_id"]) == para_id
    ]
    if len(matches) != 1:
        raise ValueError(f"candidate paragraph differs: {example['id']}")
    segment = matches[0]
    assembled_start = int(segment["assembled_start"]) + start
    assembled_end = int(segment["assembled_start"]) + end
    if (
        not int(segment["assembled_start"])
        <= assembled_start
        < assembled_end
        <= int(segment["assembled_end"])
    ):
        raise ValueError(f"candidate geometry differs: {example['id']}")
    return str(example["text"])[assembled_start:assembled_end]


def _intrinsic_vetoes(surface: str) -> list[str]:
    reasons = []
    if any(character == "\n" for character in surface):
        reasons.append("hard_separator")
    if any(unicodedata.category(character)[0] in {"N", "P", "S"} for character in surface):
        reasons.append("numeric_punctuation_or_symbol")
    return reasons


def build_lattice(
    prediction_root: Path,
    reference_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"verifier lattice output exists: {output_dir}")
    reference_manifest_path = reference_root / "manifest.json"
    reference_manifest = _read(reference_manifest_path)
    calibration_path = reference_root / "calibration.jsonl"
    examples = [
        json.loads(line)
        for line in calibration_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if (
        not isinstance(reference_manifest, dict)
        or reference_manifest.get("status")
        != "ml_production_precision_reference_ai_assisted"
        or reference_manifest.get("outputs", {}).get("calibration_sha256")
        != _sha256(calibration_path)
        or len(examples) != 45
        or {int(row["juan"]) for row in examples} != EXPECTED_JUANS
    ):
        raise ValueError("verifier reference binding differs")
    by_id = {str(row["id"]): row for row in examples}
    if len(by_id) != 45:
        raise ValueError("verifier reference inventory differs")

    predictions = {}
    inputs = {}
    for seed in SEEDS:
        root = prediction_root / (
            f"ml-production-precision-calibration-seed-{seed}-v1"
        )
        manifest_path = root / "manifest.json"
        prediction_path = root / "predictions.json"
        manifest = _read(manifest_path)
        rows = _read(prediction_path)
        if (
            not isinstance(manifest, dict)
            or not isinstance(rows, list)
            or manifest.get("status")
            != "ml_production_precision_confidence_predictions"
            or manifest.get("split") != "calibration"
            or manifest.get("seed") != seed
            or manifest.get("reference_manifest_sha256")
            != _sha256(reference_manifest_path)
            or manifest.get("predictions_sha256") != _sha256(prediction_path)
        ):
            raise ValueError(f"verifier prediction binding differs: {seed}")
        indexed = {str(row["id"]): row for row in rows}
        if set(indexed) != set(by_id):
            raise ValueError(f"verifier prediction inventory differs: {seed}")
        predictions[seed] = indexed
        inputs[str(seed)] = {
            "manifest_sha256": _sha256(manifest_path),
            "predictions_sha256": _sha256(prediction_path),
            "model_report_sha256": manifest["model_report_sha256"],
        }

    ordered_juans = sorted(
        EXPECTED_JUANS,
        key=lambda juan: hashlib.sha256(
            f"20260811:{juan}".encode("ascii")
        ).hexdigest(),
    )
    fold_by_juan = {juan: index + 1 for index, juan in enumerate(ordered_juans)}
    reference = set()
    candidate_rows = []
    covered = set()
    fold_counts = {
        str(fold): {"examples": 0, "candidates": 0, "positives": 0}
        for fold in range(1, 5)
    }
    for identity in sorted(by_id):
        example = by_id[identity]
        juan = int(example["juan"])
        fold = fold_by_juan[juan]
        fold_counts[str(fold)]["examples"] += 1
        expected_reference = {
            _geometry(span)
            for span in predictions[SEEDS[0]][identity]["reference_spans"]
        }
        if any(
            {
                _geometry(span)
                for span in predictions[seed][identity]["reference_spans"]
            }
            != expected_reference
            for seed in SEEDS[1:]
        ):
            raise ValueError(f"seed references differ: {identity}")
        reference.update((identity, geometry) for geometry in expected_reference)
        confidences_by_geometry = {}
        for seed in SEEDS:
            for span in predictions[seed][identity]["prediction_spans"]:
                confidence = float(span["confidence"])
                if confidence < MINIMUM_CONFIDENCE:
                    continue
                geometry = _geometry(span)
                if _source_surface(example, geometry) != geometry[3]:
                    raise ValueError(f"candidate surface differs: {identity}")
                confidences_by_geometry.setdefault(geometry, {})[seed] = confidence
        for geometry in sorted(confidences_by_geometry):
            confidences = confidences_by_geometry[geometry]
            positive = geometry in expected_reference
            if positive:
                covered.add((identity, geometry))
            row = {
                "id": identity,
                "juan": juan,
                "jie_index": int(example["jie_index"]),
                "fold": fold,
                "para_id": geometry[0],
                "start": geometry[1],
                "end": geometry[2],
                "surface": geometry[3],
                "label": int(positive),
                "support_count": len(confidences),
                "seed_confidences": {
                    str(seed): confidences.get(seed, 0.0) for seed in SEEDS
                },
                "intrinsic_hard_vetoes": _intrinsic_vetoes(geometry[3]),
            }
            candidate_rows.append(row)
            fold_counts[str(fold)]["candidates"] += 1
            fold_counts[str(fold)]["positives"] += int(positive)

    recall = len(covered) / len(reference)
    if recall < 0.98:
        raise RuntimeError(f"candidate lattice recall gate failed: {recall}")
    git_commit = _git_commit_clean()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        lattice_path = staging / "lattice.jsonl"
        lattice_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in candidate_rows
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "status": "ml_production_verifier_lattice_ai_assisted",
            "formal_grade": False,
            "eligible_for_production": False,
            "git_commit": git_commit,
            "minimum_seed_confidence": MINIMUM_CONFIDENCE,
            "candidate_rule": "one_or_more_of_three",
            "reference_manifest_sha256": _sha256(reference_manifest_path),
            "calibration_sha256": _sha256(calibration_path),
            "inputs": inputs,
            "fold_seed": 20260811,
            "fold_by_juan": {
                str(juan): fold for juan, fold in sorted(fold_by_juan.items())
            },
            "fold_counts": fold_counts,
            "counts": {
                "examples": len(by_id),
                "reference_spans": len(reference),
                "candidates": len(candidate_rows),
                "positive_candidates": len(covered),
                "negative_candidates": len(candidate_rows) - len(covered),
                "missed_reference_spans": len(reference) - len(covered),
            },
            "candidate_recall": recall,
            "candidate_recall_gate": 0.98,
            "lattice_sha256": _sha256(lattice_path),
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
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_lattice(
        args.prediction_root,
        args.reference,
        args.output,
    )
    print(json.dumps({
        "counts": manifest["counts"],
        "candidate_recall": manifest["candidate_recall"],
        "fold_counts": manifest["fold_counts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
