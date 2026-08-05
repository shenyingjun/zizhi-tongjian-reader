from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from p1_train import MODEL_REVISION
from p1_windows import (
    build_windows,
    constrain_predictions,
    labels_to_spans,
    merge_predictions,
)
from p2_context import add_soft_context
from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from production_train import SEEDS, _make_read_only


REFERENCE_STATUS = "ml_production_precision_reference_ai_assisted"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _geometry(span) -> tuple[int, int, int, str]:
    return int(span.para_id), int(span.start), int(span.end), str(span.surface)


def _span_confidence(
    example: dict,
    geometry: tuple[int, int, int, str],
    character_confidence: list[float],
) -> float:
    para_id, start, end, _ = geometry
    matches = [
        segment for segment in example["segments"]
        if int(segment["para_id"]) == para_id
    ]
    if len(matches) != 1:
        raise ValueError(f"prediction paragraph binding differs: {example['id']}")
    assembled_start = int(matches[0]["assembled_start"])
    probabilities = character_confidence[
        assembled_start + start : assembled_start + end
    ]
    if len(probabilities) != end - start or any(
        not 0.0 < probability <= 1.0 for probability in probabilities
    ):
        raise ValueError(f"prediction confidence differs: {example['id']}")
    return math.exp(sum(math.log(value) for value in probabilities) / len(probabilities))


def infer(
    model_root: Path,
    reference_root: Path,
    output_dir: Path,
    *,
    seed: int,
    split: str,
) -> dict:
    if seed not in SEEDS or split not in {"calibration", "confirmation"}:
        raise ValueError("unsupported precision inference control")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"precision inference output exists: {output_dir}")
    report_path = model_root / "report.json"
    report = _read(report_path)
    control = report.get("precision_control", {})
    reference_manifest_path = reference_root / "manifest.json"
    reference_manifest = _read(reference_manifest_path)
    input_path = reference_root / f"{split}.jsonl"
    if (
        control.get("seed") != seed
        or control.get("base_model_revision") != MODEL_REVISION
        or control.get("checkpoint_selection") != "fixed_epoch_5"
        or control.get("reference_manifest_sha256") != _sha256(reference_manifest_path)
        or _model_artifact(model_root / "model") != control.get("model_artifact")
        or reference_manifest.get("status") != REFERENCE_STATUS
        or reference_manifest.get("eligible_for_production_precision_claim") is not False
        or reference_manifest.get("outputs", {}).get(f"{split}_sha256")
        != _sha256(input_path)
    ):
        raise ValueError("precision inference binding differs")
    examples = _read_jsonl(input_path)
    if len(examples) != {"calibration": 45, "confirmation": 46}[split]:
        raise ValueError("precision inference split size differs")
    git_commit = _git_commit_clean()

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_root / "model", use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_root / "model")
    model.to(device)
    model.eval()
    prepared, context = add_soft_context(
        examples,
        tokenizer,
        mode="target_only",
        max_length=512,
    )
    rows = []
    with torch.inference_mode():
        for example in prepared:
            windows = build_windows(tokenizer, example, max_length=512, stride=128)
            prediction_ids = []
            probability_rows = []
            for window in windows:
                inputs = {
                    "input_ids": torch.tensor(
                        [window.input_ids], dtype=torch.long, device=device
                    ),
                    "attention_mask": torch.tensor(
                        [window.attention_mask], dtype=torch.long, device=device
                    ),
                }
                if window.token_type_ids is not None:
                    inputs["token_type_ids"] = torch.tensor(
                        [window.token_type_ids], dtype=torch.long, device=device
                    )
                probabilities = model(**inputs).logits.softmax(dim=-1)[0]
                ids = probabilities.argmax(dim=-1)
                prediction_ids.append(ids.cpu().tolist())
                probability_rows.append(
                    probabilities.gather(1, ids.unsqueeze(1)).squeeze(1).cpu().tolist()
                )
            pre_constraint, owned = merge_predictions(
                example["text"], windows, prediction_ids
            )
            character_confidence = [0.0] * len(example["text"])
            for window, probabilities in zip(windows, probability_rows):
                for token_index, ((start, end), probability) in enumerate(
                    zip(window.offsets, probabilities)
                ):
                    if not window.owned_tokens[token_index]:
                        continue
                    for character_index in range(start, end):
                        if character_confidence[character_index] != 0.0:
                            raise ValueError("character has multiple confidence owners")
                        character_confidence[character_index] = float(probability)
            constrained = constrain_predictions(
                example["text"], pre_constraint, owned
            )
            predictions = labels_to_spans(example, constrained, owned)
            references = labels_to_spans(
                example,
                example["labels"],
                [
                    is_target and character != "\n"
                    for character, is_target in zip(
                        example["text"],
                        example.get("target_mask", [True] * len(example["text"])),
                    )
                ],
            )
            prediction_rows = []
            for span in predictions:
                geometry = _geometry(span)
                prediction_rows.append({
                    "para_id": geometry[0],
                    "start": geometry[1],
                    "end": geometry[2],
                    "surface": geometry[3],
                    "confidence": _span_confidence(
                        example, geometry, character_confidence
                    ),
                })
            rows.append({
                "id": example["id"],
                "juan": int(example["juan"]),
                "jie_index": int(example["jie_index"]),
                "reference_spans": [
                    {
                        "para_id": geometry[0],
                        "start": geometry[1],
                        "end": geometry[2],
                        "surface": geometry[3],
                    }
                    for geometry in map(_geometry, references)
                ],
                "prediction_spans": prediction_rows,
            })

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        predictions_path = staging / "predictions.json"
        predictions_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "status": "ml_production_precision_confidence_predictions",
            "split": split,
            "seed": seed,
            "git_commit": git_commit,
            "formal_grade": False,
            "formal_evaluation": False,
            "eligible_for_production_precision_claim": False,
            "model_report_sha256": _sha256(report_path),
            "model_artifact": control["model_artifact"],
            "reference_manifest_sha256": _sha256(reference_manifest_path),
            "input_sha256": _sha256(input_path),
            "examples": len(rows),
            "context": context,
            "confidence": (
                "Geometric mean in log space of pre-constraint emitted-label "
                "probabilities over final span characters."
            ),
            "predictions_sha256": _sha256(predictions_path),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument(
        "--split", choices=("calibration", "confirmation"), required=True
    )
    args = parser.parse_args()
    manifest = infer(
        args.model,
        args.reference,
        args.output,
        seed=args.seed,
        split=args.split,
    )
    print(json.dumps({
        "seed": args.seed,
        "split": args.split,
        "examples": manifest["examples"],
        "predictions_sha256": manifest["predictions_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
