from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from p1_train import MODEL_REVISION
from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from production_precision_infer import run_confidence_inference
from production_precision_revision17_plan import PLAN_STATUS, _read, _sha256
from production_span_verifier import _assembled_bounds
from production_train import _make_read_only


REVISION = 17
CANDIDATE_STATUS = "ml_production_precision_revision17_candidates"
SEEDS = (20260727, 20260728, 20260729)
GENERATOR_BINDINGS = {
    20260727: {
        "report_sha256": (
            "db9c550c5b3b3df30382965a8878180a56ea360b5a337958f97cd4f2c293e4f9"
        ),
        "model_sha256": (
            "318a82482e95ce24e38d0ae6fe95cdc0a7bb6a7182a4e123800c1e8a426a7f4c"
        ),
    },
    20260728: {
        "report_sha256": (
            "3a03e1f6679501dd595b3cc493d945c3e3d591957595a0b082ee6aae1b9b7901"
        ),
        "model_sha256": (
            "38e55a232e5c3226cd316113628c2ec1f75d6bb5d772e3c82a320e3be458ba87"
        ),
    },
    20260729: {
        "report_sha256": (
            "e3cd798c6b5ae68858f8afa9526cd970a12c211f13f5aa499ef6e90e5cc134b8"
        ),
        "model_sha256": (
            "da06213353665ebbedc026c8bba0946409d749857b075146f355428772c6e261"
        ),
    },
}


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _geometry(row: dict) -> tuple[str, int, int, int]:
    return (
        str(row["id"]),
        int(row["para_id"]),
        int(row["start"]),
        int(row["end"]),
    )


def merge_predictions(
    examples: list[dict],
    predictions_by_seed: dict[int, list[dict]],
) -> list[dict]:
    examples_by_id = {str(row["id"]): row for row in examples}
    if len(examples_by_id) != len(examples):
        raise ValueError("Revision-17 mining examples contain duplicate IDs")

    candidates: dict[tuple[str, int, int, int], dict] = {}
    for seed in SEEDS:
        rows = predictions_by_seed.get(seed)
        if rows is None or len(rows) != len(examples):
            raise ValueError("Revision-17 generator prediction coverage differs")
        seen_examples = set()
        for prediction in rows:
            example_id = str(prediction["id"])
            example = examples_by_id.get(example_id)
            if example is None or example_id in seen_examples:
                raise ValueError("Revision-17 generator example binding differs")
            seen_examples.add(example_id)
            if (
                int(prediction["juan"]) != int(example["juan"])
                or int(prediction["jie_index"]) != int(example["jie_index"])
                or prediction.get("reference_spans") != []
            ):
                raise ValueError("Revision-17 generator prediction metadata differs")
            for span in prediction["prediction_spans"]:
                candidate = {
                    "id": example_id,
                    "juan": int(example["juan"]),
                    "jie_index": int(example["jie_index"]),
                    "para_id": int(span["para_id"]),
                    "start": int(span["start"]),
                    "end": int(span["end"]),
                    "surface": str(span["surface"]),
                }
                assembled_start, assembled_end, _segment = _assembled_bounds(
                    example, candidate
                )
                if (
                    assembled_start >= assembled_end
                    or str(example["text"])[assembled_start:assembled_end]
                    != candidate["surface"]
                ):
                    raise ValueError("Revision-17 candidate is not source exact")
                confidence = float(span["confidence"])
                if not 0.0 < confidence <= 1.0:
                    raise ValueError("Revision-17 generator confidence differs")
                key = _geometry(candidate)
                entry = candidates.setdefault(
                    key,
                    {
                        **candidate,
                        "generator_seeds": [],
                        "generator_confidences": [],
                    },
                )
                if entry["surface"] != candidate["surface"]:
                    raise ValueError("Revision-17 candidate surface collision")
                entry["generator_seeds"].append(seed)
                entry["generator_confidences"].append(confidence)
        if seen_examples != set(examples_by_id):
            raise ValueError("Revision-17 generator omitted a mining example")

    result = []
    for key in sorted(candidates):
        row = candidates[key]
        pairs = sorted(
            zip(row.pop("generator_seeds"), row.pop("generator_confidences"))
        )
        result.append({
            **row,
            "generator_support": len(pairs),
            "maximum_generator_confidence": max(value for _, value in pairs),
        })
    return result


def freeze_candidates(
    plan_root: Path,
    model_roots: dict[int, Path],
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Revision-17 candidates exist: {output_dir}")
    if set(model_roots) != set(SEEDS):
        raise ValueError("Revision-17 requires exactly three generator seeds")

    plan_manifest_path = plan_root / "manifest.json"
    mining_path = plan_root / "mining.jsonl"
    plan = _read(plan_manifest_path)
    if (
        plan.get("status") != PLAN_STATUS
        or plan.get("confirmation_read") is not False
        or plan.get("formal_reserve_text_read") is not False
        or int(plan.get("counts", {}).get("mining", -1)) != 1098
        or plan.get("outputs", {}).get("mining_sha256") != _sha256(mining_path)
    ):
        raise ValueError("Revision-17 plan binding differs")
    examples = _read_jsonl(mining_path)
    if len(examples) != 1098:
        raise ValueError("Revision-17 mining example count differs")

    model_bindings = {}
    predictions_by_seed = {}
    for seed in SEEDS:
        root = model_roots[seed]
        report_path = root / "report.json"
        model_dir = root / "model"
        report = _read(report_path)
        control = report.get("precision_control", {})
        artifact = _model_artifact(model_dir)
        expected = GENERATOR_BINDINGS[seed]
        if (
            _sha256(report_path) != expected["report_sha256"]
            or artifact.get("combined_sha256") != expected["model_sha256"]
            or control.get("seed") != seed
            or control.get("base_model_revision") != MODEL_REVISION
            or control.get("checkpoint_selection") != "fixed_epoch_5"
            or control.get("model_artifact") != artifact
        ):
            raise ValueError(f"Revision-17 generator binding differs: {seed}")
        predictions, context = run_confidence_inference(model_dir, examples)
        predictions_by_seed[seed] = predictions
        model_bindings[str(seed)] = {
            "report_sha256": _sha256(report_path),
            "model_artifact": artifact,
            "context": context,
        }

    candidates = merge_predictions(examples, predictions_by_seed)
    if not candidates:
        raise ValueError("Revision-17 generator union is empty")
    git_commit = _git_commit_clean()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        candidates_path = staging / "candidates.jsonl"
        _write_jsonl(candidates_path, candidates)
        manifest = {
            "schema_version": 1,
            "status": CANDIDATE_STATUS,
            "revision": REVISION,
            "fit_only": True,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "formal_reserve_text_read": False,
            "git_commit": git_commit,
            "bindings": {
                "plan_manifest_sha256": _sha256(plan_manifest_path),
                "mining_sha256": _sha256(mining_path),
                "generators": model_bindings,
            },
            "counts": {
                "examples": len(examples),
                "candidates": len(candidates),
                "support": {
                    str(support): sum(
                        row["generator_support"] == support for row in candidates
                    )
                    for support in (1, 2, 3)
                },
                "juans": len({int(row["juan"]) for row in candidates}),
            },
            "outputs": {"candidates_sha256": _sha256(candidates_path)},
            "claim_limit": (
                "Fit-only training-data discovery; generator predictions are not "
                "evaluation or deployment output."
            ),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the Revision-17 three-model candidate union."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-20260727", type=Path, required=True)
    parser.add_argument("--model-20260728", type=Path, required=True)
    parser.add_argument("--model-20260729", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_candidates(
        args.plan,
        {
            20260727: args.model_20260727,
            20260728: args.model_20260728,
            20260729: args.model_20260729,
        },
        args.output,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
