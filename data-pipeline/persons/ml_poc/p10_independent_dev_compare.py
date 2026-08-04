from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

from p2_round import _spans
from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from p6_locked_assisted_finalize import _read_jsonl
from p6_seed_ensemble import (
    _predict_examples,
    _prediction_metrics,
    vote_predictions,
)
from p6_seed_replication_train import (
    CUBLAS_WORKSPACE_CONFIG,
    REPLICATION_SEEDS,
    _configure_determinism,
)
from report import geometry_delta


EXPECTED_REFERENCE_REPORT_SHA256 = (
    "205f08d87aaabe39523030852bfa1e8357ab8fa060c10e816f0dcf32ec5cc817"
)
EXPECTED_REFERENCE_SHA256 = (
    "e3f9381c04d135e56ec6df84360d8fedbf7cd5aa5deded66f44faabb86b14a4a"
)
ENSEMBLE_THRESHOLD = 2
RECIPES = {
    "round6": {
        20260727: (
            "ml-poc-round6-controlled-model-v1",
            "e34278caa2f1e68888784cd6aa62a44f3c8fa4981992e478b7c3b6f38ba1c90e",
            "d62b795488292638dac2425f5f7b6798f39810d38746f508271ce174abe8d536",
        ),
        20260728: (
            "ml-poc-round6-attribution-seed-20260728-v1",
            "4af55a98b90714c132a5d34393a6c361803ff1ff4db4a8f8a62b3d811047d234",
            "3a0b200027891c5f3f9ed89460d960c59af620d33dd043bb6cd18f845342911e",
        ),
        20260729: (
            "ml-poc-round6-attribution-seed-20260729-v1",
            "14ea2913276d1a656656c665bbf66abe49295470998fdd2426d8287bbc58dab3",
            "5c875b98d454dbfcf5bc0a7c0e2f508d6d83c96e17e94a265e3d6653cb74fe71",
        ),
    },
    "round7": {
        20260727: (
            "ml-poc-round7-seed-20260727-v1",
            "0a75d04c04972bc2895992b240b914a1e2d0c485b9a520bdbd7d5fecd8e4d684",
            "c625bd79c7f3f3904bf6b9227e2881edb5abb9a1c356dc001326f8d86370852d",
        ),
        20260728: (
            "ml-poc-round7-seed-20260728-v1",
            "e605c9d6f1e44cd51ae5151054da2480bfe944dfaed1e81113bf8bd71ec1b62c",
            "06ec8460bf56fe36054324698e5f4a9acad3915e5c64687e954243d2db7de8d9",
        ),
        20260729: (
            "ml-poc-round7-seed-20260729-v1",
            "2ad035b62a733462e4817da2a1e4e53b971347416bc2c8c57ca5b9169420473a",
            "1870850db5753aaf7d12853b25b78f84e58750112e31a9f35223889ace25f475",
        ),
    },
    "round8": {
        20260727: (
            "ml-poc-round8-lr2e-5-seed-20260727-v1",
            "6eb0947e9522dddadd74248487470ca6350b0107aa58240377f2036b1d3477fe",
            "d11cd42385d843ec3539b5533a3db4334d92e1a252c78fe47e953bff9534e0a0",
        ),
        20260728: (
            "ml-poc-round8-lr2e-5-seed-20260728-v1",
            "e8bbc587d5e607ac4b19c66c5e906c6051d4f5798fbb074a6d4b814cf208f09e",
            "3de52d0373596b9e744b35a77a83e08cc5595d9a7b68328fd74345a6a8584233",
        ),
        20260729: (
            "ml-poc-round8-lr2e-5-seed-20260729-v1",
            "044754c96803c58dd5b8205039d5f8ab12bf47922995db20297d81b5837a0d1c",
            "2aa214066663ef1adc3763056856757b3c57c06c063cd46584a610f66011790b",
        ),
    },
    "round9": {
        20260727: (
            "ml-poc-round9-lr2.5e-5-seed-20260727-v1",
            "d2248961807226f7dc81bbeab519371ef7f237ec2c5180d49e0aa10d9b2f5d59",
            "165023b1cd68cc46a2f129d0a73c6e03cc207403dfb61cfbe29fc8db31b98907",
        ),
        20260728: (
            "ml-poc-round9-lr2.5e-5-seed-20260728-v1",
            "36dba248537251e7061c5b390783f2c7bfd9efb04a9071e7025f53b856eda76e",
            "76b1df31a85ae81e200d56a0df3afae4ce9dabf9362fdde44e7a93f9e4bcadac",
        ),
        20260729: (
            "ml-poc-round9-lr2.5e-5-seed-20260729-v1",
            "062aa2c0b1ecfab502dcf91a947590296559e31547da8221fc698556e1b92a38",
            "de37355882967ec599813622dc09af42d20c44196c3b062ef727e7a5c3c434fb",
        ),
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_recipe(metrics: dict[str, dict]) -> str:
    order = {name: index for index, name in enumerate(RECIPES)}
    return max(metrics, key=lambda name: (
        metrics[name]["f1"],
        metrics[name]["precision"],
        metrics[name]["recall"],
        -order[name],
    ))


def _compare(before: list[dict], after: list[dict]) -> tuple[dict, list[dict]]:
    before_by_id = {str(row["id"]): row for row in before}
    after_by_id = {str(row["id"]): row for row in after}
    if set(before_by_id) != set(after_by_id):
        raise ValueError("recipe prediction identities differ")
    totals = Counter()
    changes = []
    for identity in sorted(before_by_id):
        reference = set(_spans(before_by_id[identity]["reference_spans"]))
        if reference != set(_spans(after_by_id[identity]["reference_spans"])):
            raise ValueError(f"recipe reference differs: {identity}")
        before_spans = set(_spans(
            before_by_id[identity]["prediction_spans"]
        ))
        after_spans = set(_spans(
            after_by_id[identity]["prediction_spans"]
        ))
        additions = sorted(after_spans - before_spans)
        removals = sorted(before_spans - after_spans)
        delta = geometry_delta(list(before_spans), list(after_spans))
        totals.update({
            "raw_additions": len(additions),
            "removals": len(removals),
            "net_growth": len(additions) - len(removals),
            "reference_recoveries": len(set(additions) & reference),
            "reference_regressions": len(set(removals) & reference),
            "added_false_positives": len(set(additions) - reference),
            "removed_false_positives": len(set(removals) - reference),
            "geometry_replacements": delta["geometry_replacements"],
        })
        if additions or removals:
            changes.append({
                "id": identity,
                "additions": [row.__dict__ for row in additions],
                "removals": [row.__dict__ for row in removals],
                "geometry_replacements": delta["replacement_examples"],
            })
    totals["changed_jies"] = len(changes)
    return dict(totals), changes


def compare_recipes(
    reference_dir: Path,
    artifacts_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"independent dev comparison exists: {output_dir}")
    report_path = reference_dir / "report.json"
    reference_path = reference_dir / "development_reference.jsonl"
    reference_report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        _sha256(report_path) != EXPECTED_REFERENCE_REPORT_SHA256
        or _sha256(reference_path) != EXPECTED_REFERENCE_SHA256
        or reference_report.get("status")
        != "frozen_round10_independent_development_reference"
        or reference_report.get("development_only") is not True
        or reference_report.get("eligible_for_training") is not False
        or reference_report.get("eligible_for_promotion") is not False
        or reference_report.get("model_predictions_used_in_labeling") is not False
        or reference_report.get("outputs", {}).get(
            "development_reference_sha256"
        ) != EXPECTED_REFERENCE_SHA256
    ):
        raise ValueError("independent dev reference binding differs")
    examples = _read_jsonl(reference_path)
    if len(examples) != 20:
        raise ValueError("independent dev example inventory differs")
    _configure_determinism()

    predictions = {}
    metrics = {}
    inputs = {}
    for recipe, seeds in RECIPES.items():
        by_seed = {}
        inputs[recipe] = {}
        for seed in REPLICATION_SEEDS:
            name, expected_report, expected_model = seeds[seed]
            root = artifacts_root / name
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            artifact = _model_artifact(root / "model")["combined_sha256"]
            if (
                _sha256(root / "report.json") != expected_report
                or artifact != expected_model
                or report.get("config", {}).get("seed") != seed
            ):
                raise ValueError(f"{recipe} seed {seed} binding differs")
            by_seed[seed] = _predict_examples(root, examples)
            inputs[recipe][str(seed)] = {
                "artifact_name": name,
                "report_sha256": expected_report,
                "model_artifact_sha256": expected_model,
            }
        ensemble = vote_predictions(by_seed, ENSEMBLE_THRESHOLD)
        predictions[recipe] = {
            "by_seed": by_seed,
            "ensemble": ensemble,
        }
        metrics[recipe] = {
            "ensemble": _prediction_metrics(ensemble),
            "by_seed": {
                str(seed): _prediction_metrics(by_seed[seed])
                for seed in REPLICATION_SEEDS
            },
        }
    selected = select_recipe({
        recipe: row["ensemble"] for recipe, row in metrics.items()
    })
    comparisons = {}
    geometry_changes = {}
    for recipe in RECIPES:
        comparison, changes = _compare(
            predictions["round6"]["ensemble"],
            predictions[recipe]["ensemble"],
        )
        comparisons[recipe] = comparison
        geometry_changes[recipe] = changes

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        predictions_path = staging / "predictions.json"
        changes_path = staging / "geometry_changes_from_round6.json"
        predictions_path.write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        changes_path.write_text(
            json.dumps(geometry_changes, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report = {
            "schema_version": 1,
            "status": "round10_independent_dev_recipe_selection",
            "development_only": True,
            "formal_evaluation": False,
            "eligible_for_training": False,
            "eligible_for_promotion": False,
            "selection_metric": (
                "2_of_3_exact_geometry_ensemble_f1_then_precision_then_recall"
                "_then_earlier_recipe"
            ),
            "ensemble_threshold": ENSEMBLE_THRESHOLD,
            "determinism": {
                "torch_deterministic_algorithms": True,
                "warn_only": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
            },
            "selected_recipe": selected,
            "fresh_promotion_evaluation_consumed": False,
            "decision": f"{selected}_may_enter_fresh_promotion_evaluation",
            "metrics": metrics,
            "comparisons_from_round6": comparisons,
            "inputs": {
                "reference_report_sha256": EXPECTED_REFERENCE_REPORT_SHA256,
                "reference_sha256": EXPECTED_REFERENCE_SHA256,
                "models": inputs,
            },
            "git_commit": _git_commit_clean(),
            "outputs": {
                "predictions_sha256": _sha256(predictions_path),
                "geometry_changes_sha256": _sha256(changes_path),
            },
        }
        (staging / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in staging.iterdir():
            path.chmod(0o444)
        staging.chmod(0o555)
        staging.replace(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select a recipe on the independent Round 10 dev reference."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_recipes(args.reference, args.artifacts_root, args.output)
    print(json.dumps({
        "selected_recipe": report["selected_recipe"],
        "metrics": {
            name: row["ensemble"] for name, row in report["metrics"].items()
        },
        "comparisons_from_round6": report["comparisons_from_round6"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
