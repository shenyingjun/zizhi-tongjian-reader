from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from p1_train import MODEL_REVISION
from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from production_precision_infer import run_confidence_inference
from production_precision_mining_plan import FOLD_NUMBERS, MINING_ORDER_SEED
from production_train import SEEDS, _make_read_only


PLAN_STATUS = "ml_production_precision_mining_plan"


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


def infer_holdout(
    model_root: Path,
    plan_root: Path,
    output_dir: Path,
    *,
    fold: int,
    seed: int,
) -> dict:
    """Emit held-out span-confidence predictions for one mining model.

    The prediction rows use the exact same span-confidence semantics and
    provenance as ``production_precision_infer`` and are produced only on this
    fold's holdout jies. The bindings enforce that the model was trained on the
    other four folds, so no mining prediction comes from a model trained on its
    own jie or juan. Confirmation data is never opened.
    """
    if fold not in FOLD_NUMBERS:
        raise ValueError(f"mining fold is not one of 1..5: {fold}")
    if seed not in SEEDS:
        raise ValueError(f"mining seed is not predeclared: {seed}")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"mining inference output exists: {output_dir}")

    report_path = model_root / "report.json"
    report = _read(report_path)
    control = report.get("mining_control", {})
    plan_manifest_path = plan_root / "manifest.json"
    plan_manifest = _read(plan_manifest_path)
    fold_dir = plan_root / "folds" / f"fold-{fold}"
    holdout_path = fold_dir / "holdout.jsonl"
    fold_outputs = plan_manifest.get("outputs", {}).get(str(fold), {})
    fold_summary = plan_manifest.get("fold_summaries", {}).get(str(fold), {})
    if (
        control.get("fold") != fold
        or control.get("seed") != seed
        or control.get("base_model_revision") != MODEL_REVISION
        or control.get("checkpoint_selection") != "fixed_epoch_5"
        or control.get("mining_only") is not True
        or control.get("eligible_for_deployment") is not False
        or control.get("eligible_for_production") is not False
        or control.get("plan_manifest_sha256") != _sha256(plan_manifest_path)
        or control.get("holdout_sha256") != _sha256(holdout_path)
        or _model_artifact(model_root / "model") != control.get("model_artifact")
        or plan_manifest.get("status") != PLAN_STATUS
        or plan_manifest.get("mining_only") is not True
        or plan_manifest.get("order_seed") != MINING_ORDER_SEED
        or fold_outputs.get("holdout_sha256") != _sha256(holdout_path)
    ):
        raise ValueError("mining inference binding differs")

    examples = _read_jsonl(holdout_path)
    holdout_juans = {int(juan) for juan in fold_summary.get("holdout_juans", [])}
    example_juans = {int(example["juan"]) for example in examples}
    if (
        not examples
        or example_juans != holdout_juans
        or len(examples) != int(fold_summary.get("holdout_examples", -1))
    ):
        raise ValueError("mining inference holdout inventory differs")
    # A mining prediction must never come from a model trained on its own juan.
    train_juans = {
        int(juan)
        for juan, assigned in plan_manifest.get("fold_by_juan", {}).items()
        if int(assigned) != fold
    }
    if example_juans & train_juans:
        raise ValueError("held-out juan overlaps the model's training juans")
    git_commit = _git_commit_clean()

    rows, context = run_confidence_inference(model_root / "model", examples)

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
            "status": "ml_production_precision_mining_oof_predictions",
            "mining_only": True,
            "eligible_for_deployment": False,
            "eligible_for_production": False,
            "eligible_for_production_precision_claim": False,
            "formal_grade": False,
            "formal_evaluation": False,
            "fold": fold,
            "seed": seed,
            "git_commit": git_commit,
            "model_report_sha256": _sha256(report_path),
            "model_artifact": control["model_artifact"],
            "plan_manifest_sha256": _sha256(plan_manifest_path),
            "holdout_sha256": _sha256(holdout_path),
            "examples": len(rows),
            "holdout_juans": sorted(example_juans),
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
    parser = argparse.ArgumentParser(
        description=(
            "Infer the one-seed/0.30 lattice on a mining model's held-out fold."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=FOLD_NUMBERS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    manifest = infer_holdout(
        args.model,
        args.plan,
        args.output,
        fold=args.fold,
        seed=args.seed,
    )
    print(json.dumps({
        "fold": args.fold,
        "seed": args.seed,
        "examples": manifest["examples"],
        "predictions_sha256": manifest["predictions_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
