from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import stat
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from core import assemble_jies
from p3_compact import (
    MAX_CHARS,
    MIN_CHARS,
    RANDOM_JIES,
    FOREIGN_JIES,
    ROLE_JIES,
    _git_commit_clean,
    select_compact,
)
from p3_compact_evaluate import _model_artifact
from p4_fresh_sealed import EXCLUDED_JUANS as HISTORICAL_EXCLUDED_JUANS
from pilot import TEXT, _load


EXPECTED_INPUTS = {
    "round6_dataset_manifest": (
        "7788c8d3341e56a17eaa89fd582df6fcc930d9f5c723d1bd5a0c78b61f3852a1"
    ),
    "prior_fresh_manifest": (
        "4f29d59904e2da297c56221d1796bbbed6b8a25032d82cb58c9f6bc916c760da"
    ),
    "round5_tasks_manifest": (
        "de185c0d993db27e0d703a2962e5477fa1ee6bcc0b6d447b23a8f980c085021d"
    ),
    "round3_report": (
        "b65bccba68bb37dea23da7a57b84336b906a32769145f0d33b9f679d57d73353"
    ),
    "round6_report": (
        "072090a323146cfa8f83da3e86098dc3d042f9b4ab28aaa2f78f86b9956e1afa"
    ),
    "boundary_guide": (
        "f7c1dff1e97c41fc37846374867840ebde81b9b4fa601e3adbde2db9f1de8247"
    ),
    "spec": "ae0c837353adcdd88d0977acfea2836956ed377accc556ff02e21948923ea8f6",
    "spec_zh": "33f13ddefbb4fd46e5842abd1f476be291abf39140f286181287d28d2240fc66",
}
EXPECTED_MODELS = {
    "round3": "fb611bee98cdc672e19163ca84aef49612815cab285889c7458cff07ba098cbd",
    "round6_seed_20260727": (
        "d62b795488292638dac2425f5f7b6798f39810d38746f508271ce174abe8d536"
    ),
}
POLICY_PATHS = {
    "boundary_guide": Path(__file__).with_name("BOUNDARY_GUIDE.md"),
    "spec": Path(__file__).with_name("SPEC.md"),
    "spec_zh": Path(__file__).with_name("SPEC.zh.md"),
}
PROMOTION_ELIGIBILITY_REASON = (
    "predeclared_low_power_diagnostic_without_promotion_gate"
)


def evaluation_claim_metadata() -> dict:
    return {
        "formal_evaluation": False,
        "eligible_for_promotion": False,
        "promotion_eligibility_reason": PROMOTION_ELIGIBILITY_REASON,
        "copilot_assistance_is_disqualifying": False,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _eligible_jies(
    sources: dict[int, dict],
    excluded_juans: set[int],
) -> list[dict]:
    rows = []
    for juan in sorted(set(sources) - excluded_juans):
        for jie in assemble_jies(sources[juan]["paragraphs"]):
            length = len(jie.text)
            if jie.number is None or not MIN_CHARS <= length <= MAX_CHARS:
                continue
            rows.append({
                "juan": juan,
                "jie_index": int(jie.index),
                "jie_number": jie.number,
                "text": jie.text,
                "segments": [asdict(segment) for segment in jie.segments],
                "characters": length,
            })
    return rows


def _validate_candidate(model_root: Path, label: str) -> dict:
    report_path = model_root / "report.json"
    if _sha256(report_path) != EXPECTED_INPUTS[f"{label.split('_seed_')[0]}_report"]:
        raise ValueError(f"{label} report hash differs")
    report = _load(report_path)
    artifact = _model_artifact(model_root / "model")
    if artifact.get("combined_sha256") != EXPECTED_MODELS[label]:
        raise ValueError(f"{label} model artifact differs")
    if label == "round3":
        control = report.get("round3_control", {})
        expected_artifact = control.get("model_artifact")
        valid = (
            control.get("formal_evaluation") is False
            and control.get("eligible_for_promotion_without_fresh_sealed_set")
            is False
        )
    else:
        control = report.get("seed_replication_control", {})
        expected_artifact = control.get("model_artifact")
        valid = (
            control.get("dataset_kind") == "round6"
            and control.get("full_control", {}).get("seed") == 20260727
            and control.get("determinism", {}).get(
                "torch_deterministic_algorithms"
            ) is True
            and control.get("formal_evaluation") is False
            and control.get("eligible_for_promotion") is False
        )
    if artifact != expected_artifact or not valid:
        raise ValueError(f"{label} candidate provenance differs")
    return {
        "label": label,
        "artifact_sha256": artifact["combined_sha256"],
        "report_sha256": _sha256(report_path),
        "selection_basis": (
            "predeclared Round 3 active model"
            if label == "round3"
            else "predeclared seed 20260727 selected by Juan 27 dev only"
        ),
    }


def _manifest_juans(manifest: dict) -> set[int]:
    result = set()
    for split in manifest.get("splits", {}).values():
        result.update(int(juan) for juan in split.get("juans", []))
    result.update(
        int(row["juan"]) for row in manifest.get("selected", [])
        if "juan" in row
    )
    result.update(
        int(row["juan"]) for row in manifest.get("selected_jies", [])
        if "juan" in row
    )
    result.update(
        int(row["juan"]) for row in manifest.get("private_selected_jies", [])
        if "juan" in row
    )
    return result


def prepare_locked_assisted_tasks(
    output_dir: Path,
    round6_dataset_manifest: Path,
    prior_fresh_manifest: Path,
    round5_tasks_manifest: Path,
    round3_model: Path,
    round6_model: Path,
    *,
    seed: int | None = None,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"locked assisted tasks exist: {output_dir}")
    input_paths = {
        "round6_dataset_manifest": round6_dataset_manifest,
        "prior_fresh_manifest": prior_fresh_manifest,
        "round5_tasks_manifest": round5_tasks_manifest,
        **POLICY_PATHS,
    }
    input_hashes = {name: _sha256(path) for name, path in input_paths.items()}
    if input_hashes != {
        key: value for key, value in EXPECTED_INPUTS.items()
        if key not in {"round3_report", "round6_report"}
    }:
        raise ValueError("locked task input hashes differ")
    manifests = {
        name: _load(path)
        for name, path in input_paths.items()
        if name.endswith("manifest")
    }
    if (
        manifests["round6_dataset_manifest"].get("status")
        != "round6_controlled_training_dataset"
        or manifests["prior_fresh_manifest"].get("status")
        != "fresh_sealed_before_annotation"
        or manifests["round5_tasks_manifest"].get("status")
        != "copilot_double_pass_tasks_before_labeling"
    ):
        raise ValueError("historical exclusion manifest status differs")
    candidates = [
        _validate_candidate(round3_model, "round3"),
        _validate_candidate(round6_model, "round6_seed_20260727"),
    ]
    excluded_juans = set(HISTORICAL_EXCLUDED_JUANS)
    for manifest in manifests.values():
        excluded_juans.update(_manifest_juans(manifest))

    sources = {}
    source_paths = {}
    for juan in range(1, 295):
        path = TEXT / f"juan_{juan:03d}.json"
        if path.is_file() and juan not in excluded_juans:
            sources[juan] = _load(path)
            source_paths[juan] = path
    frame = _eligible_jies(sources, excluded_juans)
    selection_seed = seed if seed is not None else secrets.randbits(128)
    selected_jies = select_compact(frame, seed=selection_seed)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in selected_jies:
        grouped[int(row["juan"])].append(row)

    manifest = {
        "schema_version": 1,
        "status": "candidate_blind_copilot_double_pass_tasks_before_labeling",
        **evaluation_claim_metadata(),
        "candidate_model_blind": True,
        "provenance": "copilot_double_pass_blind_diagnostic",
        "model_predictions_generated": False,
        "rules_generated": False,
        "v1_loaded": False,
        "selection_seed": selection_seed,
        "selection_policy": {
            "frame": (
                f"numbered jies with {MIN_CHARS}-{MAX_CHARS} Unicode codepoints "
                "from whole juans absent from every bound historical artifact"
            ),
            "probability_random": f"{RANDOM_JIES} uniform jie draws",
            "role_appellation_challenge": (
                f"{ROLE_JIES} private draws from the frozen top-40 cohort"
            ),
            "foreign_title_challenge": (
                f"{FOREIGN_JIES} private draws from the frozen top-40 cohort"
            ),
        },
        "labeling_protocol": {
            "pass_a": "independent_recall_first",
            "pass_b": "independent_exact_boundary_first",
            "pass_visibility": "mutually_hidden_candidate_free_raw_text_only",
            "auto_accept": "exact_A_B_consensus_without_explicit_low",
            "user_review": (
                "disagreement_explicit_low_and_predeclared_consensus_audit"
            ),
            "lock_before_candidate_inference": True,
        },
        "inputs": input_hashes,
        "frozen_candidates": candidates,
        "excluded_juans": sorted(excluded_juans),
        "sampling_frame": {
            "eligible_jies": len(frame),
            "eligible_juans": len({int(row["juan"]) for row in frame}),
            "min_characters": MIN_CHARS,
            "max_characters": MAX_CHARS,
        },
        "git_commit": _git_commit_clean(),
        "selected": [],
        "private_selected_jies": [
            {
                key: value
                for key, value in row.items()
                if key not in {"text", "segments"}
            }
            for row in selected_jies
        ],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        tasks_dir = staging / "tasks"
        tasks_dir.mkdir()
        for juan, rows in sorted(grouped.items()):
            task = {
                "schema_version": 1,
                "phase": "candidate_blind_copilot_pass",
                "diagnostic_only": True,
                "juan": juan,
                "instructions": (
                    "Independently mark every main-text person span. Apply the "
                    "frozen BOUNDARY_GUIDE.md using target-jie evidence only. "
                    "Do not read v1, rules, any model prediction, another pass, "
                    "notes, translations, identity data, selection roles, or "
                    "omitted text."
                ),
                "jies": [{
                    "jie_index": row["jie_index"],
                    "jie_number": row["jie_number"],
                    "text": row["text"],
                    "segments": row["segments"],
                    "annotations": [],
                } for row in sorted(rows, key=lambda item: item["jie_index"])],
            }
            task_path = tasks_dir / f"blind_juan_{juan:03d}.json"
            task_path.write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest["selected"].append({
                "juan": juan,
                "task": str(Path("tasks") / task_path.name),
                "task_sha256": _sha256(task_path),
                "source_sha256": _sha256(source_paths[juan]),
                "sampled_jies": len(rows),
            })
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in tasks_dir.iterdir():
            path.chmod(stat.S_IREAD)
        manifest_path.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze candidate-blind Copilot A/B diagnostic tasks."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round6-dataset-manifest", type=Path, required=True)
    parser.add_argument("--prior-fresh-manifest", type=Path, required=True)
    parser.add_argument("--round5-tasks-manifest", type=Path, required=True)
    parser.add_argument("--round3-model", type=Path, required=True)
    parser.add_argument("--round6-model", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_locked_assisted_tasks(
        args.output,
        args.round6_dataset_manifest,
        args.prior_fresh_manifest,
        args.round5_tasks_manifest,
        args.round3_model,
        args.round6_model,
    )
    print(json.dumps({
        "sampled_jies": len(manifest["private_selected_jies"]),
        "characters": sum(
            row["characters"] for row in manifest["private_selected_jies"]
        ),
        "task_juans": len(manifest["selected"]),
        "eligible_juans": manifest["sampling_frame"]["eligible_juans"],
        "excluded_juans": len(manifest["excluded_juans"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
