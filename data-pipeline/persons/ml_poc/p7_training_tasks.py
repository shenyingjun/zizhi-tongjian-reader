from __future__ import annotations

import argparse
import hashlib
import json
import random
import secrets
import stat
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from core import assemble_jies
from p3_compact import MAX_CHARS, MIN_CHARS, _git_commit_clean
from p3_compact_evaluate import _model_artifact
from p6_locked_assisted_tasks import _manifest_juans
from pilot import TEXT, _load


SAMPLED_JIES = 40
EXPECTED_INPUTS = {
    "historical_exclusion_manifest": (
        "b020541f12207244d110f38133e0c2dd1a43a554c18c952a1644a6cef45ec401"
    ),
    "round6_dataset_manifest": (
        "7788c8d3341e56a17eaa89fd582df6fcc930d9f5c723d1bd5a0c78b61f3852a1"
    ),
    "boundary_guide": (
        "f7c1dff1e97c41fc37846374867840ebde81b9b4fa601e3adbde2db9f1de8247"
    ),
    "spec": "ae0c837353adcdd88d0977acfea2836956ed377accc556ff02e21948923ea8f6",
    "spec_zh": "33f13ddefbb4fd46e5842abd1f476be291abf39140f286181287d28d2240fc66",
    "round3_report": (
        "b65bccba68bb37dea23da7a57b84336b906a32769145f0d33b9f679d57d73353"
    ),
    "round3_artifact": (
        "fb611bee98cdc672e19163ca84aef49612815cab285889c7458cff07ba098cbd"
    ),
}
POLICY_PATHS = {
    "boundary_guide": Path(__file__).with_name("BOUNDARY_GUIDE.md"),
    "spec": Path(__file__).with_name("SPEC.md"),
    "spec_zh": Path(__file__).with_name("SPEC.zh.md"),
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


def select_unique_juan_rows(
    rows: list[dict],
    *,
    seed: int,
    count: int = SAMPLED_JIES,
) -> list[dict]:
    by_juan: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_juan[int(row["juan"])].append(row)
    if len(by_juan) < count:
        raise ValueError(f"only {len(by_juan)} unique eligible juans")
    rng = random.Random(seed)
    juans = rng.sample(sorted(by_juan), count)
    return [rng.choice(by_juan[juan]) for juan in juans]


def prepare_round7_tasks(
    output_dir: Path,
    historical_exclusion_manifest: Path,
    round6_dataset_manifest: Path,
    round3_model_root: Path,
    *,
    seed: int | None = None,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Round 7 task output exists: {output_dir}")
    input_paths = {
        "historical_exclusion_manifest": historical_exclusion_manifest,
        "round6_dataset_manifest": round6_dataset_manifest,
        **POLICY_PATHS,
    }
    input_hashes = {name: _sha256(path) for name, path in input_paths.items()}
    if input_hashes != {
        key: value for key, value in EXPECTED_INPUTS.items()
        if key not in {"round3_report", "round3_artifact"}
    }:
        raise ValueError("Round 7 task input hashes differ")
    exclusion_manifest = _load(historical_exclusion_manifest)
    dataset_manifest = _load(round6_dataset_manifest)
    if (
        exclusion_manifest.get("status")
        != "candidate_blind_copilot_double_pass_tasks_before_labeling"
        or dataset_manifest.get("status")
        != "round6_controlled_training_dataset"
    ):
        raise ValueError("Round 7 historical manifest status differs")
    report_path = round3_model_root / "report.json"
    artifact = _model_artifact(round3_model_root / "model")
    if (
        _sha256(report_path) != EXPECTED_INPUTS["round3_report"]
        or artifact["combined_sha256"] != EXPECTED_INPUTS["round3_artifact"]
    ):
        raise ValueError("Round 7 omission model binding differs")

    excluded_juans = set(int(row) for row in exclusion_manifest["excluded_juans"])
    excluded_juans.update(_manifest_juans(exclusion_manifest))
    excluded_juans.update(_manifest_juans(dataset_manifest))
    sources = {}
    source_paths = {}
    for juan in range(1, 295):
        path = TEXT / f"juan_{juan:03d}.json"
        if path.is_file() and juan not in excluded_juans:
            sources[juan] = _load(path)
            source_paths[juan] = path
    frame = _eligible_jies(sources, excluded_juans)
    selection_seed = seed if seed is not None else secrets.randbits(128)
    selected = select_unique_juan_rows(frame, seed=selection_seed)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in selected:
        grouped[int(row["juan"])].append(row)

    manifest = {
        "schema_version": 1,
        "status": "round7_copilot_double_pass_training_tasks_before_labeling",
        "training_only": True,
        "formal_evaluation": False,
        "eligible_for_training_after_focused_review": True,
        "eligible_for_evaluation": False,
        "evaluation_ineligibility_reason": "sampled_for_round7_training",
        "selection_seed": selection_seed,
        "selection_policy": (
            f"{SAMPLED_JIES} wholly unused eligible juans sampled uniformly "
            "without replacement, then one eligible numbered jie sampled "
            "uniformly within each selected juan"
        ),
        "evaluation_reserve_policy": (
            "sample only 40 training juans so at least 50 wholly unused juans "
            "remain available before text-length eligibility filtering"
        ),
        "labeling_protocol": {
            "pass_a": "independent_recall_first",
            "pass_b": "independent_exact_boundary_first",
            "pass_visibility": "mutually_hidden_raw_text_only",
            "auto_accept": "exact_A_B_consensus_without_explicit_low",
            "focused_review": (
                "teacher_disagreement_explicit_low_and_round3_model_only"
            ),
            "locked_p3_errors_used": False,
        },
        "sampling_frame": {
            "eligible_jies": len(frame),
            "eligible_juans": len({int(row["juan"]) for row in frame}),
            "min_characters": MIN_CHARS,
            "max_characters": MAX_CHARS,
        },
        "excluded_juans": sorted(excluded_juans),
        "inputs": {
            **input_hashes,
            "round3_report": _sha256(report_path),
            "round3_artifact": artifact["combined_sha256"],
        },
        "selected_model": {
            "purpose": "post_teacher_omission_candidates_only",
            "label": "active_round3",
            "artifact_sha256": artifact["combined_sha256"],
            "report_sha256": _sha256(report_path),
        },
        "v1_used_for_selection": False,
        "rules_used_for_selection": False,
        "model_predictions_used_for_selection": False,
        "locked_p3_outputs_used_for_selection": False,
        "model_predictions_generated": False,
        "git_commit": _git_commit_clean(),
        "selected": [],
        "selected_jies": [
            {
                key: value for key, value in row.items()
                if key not in {"text", "segments"}
            }
            for row in selected
        ],
    }
    remaining_eligible_juans = (
        manifest["sampling_frame"]["eligible_juans"] - SAMPLED_JIES
    )
    if remaining_eligible_juans < 50:
        raise ValueError(
            f"Round 7 would leave only {remaining_eligible_juans} eligible juans"
        )
    manifest["sampling_frame"]["remaining_eligible_juans"] = (
        remaining_eligible_juans
    )

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
                "phase": "copilot_double_pass",
                "training_only": True,
                "juan": juan,
                "instructions": (
                    "Independently mark every main-text person span in the "
                    "sampled jie. Apply BOUNDARY_GUIDE.md using target-jie "
                    "evidence only. Do not read another pass, any model or rule "
                    "prediction, v1, translations, notes, identity data, locked "
                    "P3 output, or text outside this task."
                ),
                "jies": [{
                    "jie_index": row["jie_index"],
                    "jie_number": row["jie_number"],
                    "text": row["text"],
                    "segments": row["segments"],
                    "annotations": [],
                } for row in rows],
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
        description="Freeze fresh uniform Round 7 training-only tasks."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--historical-exclusion-manifest", type=Path, required=True
    )
    parser.add_argument("--round6-dataset-manifest", type=Path, required=True)
    parser.add_argument("--round3-model-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_round7_tasks(
        args.output,
        args.historical_exclusion_manifest,
        args.round6_dataset_manifest,
        args.round3_model_root,
    )
    print(json.dumps({
        "sampled_jies": len(manifest["selected_jies"]),
        "characters": sum(
            row["characters"] for row in manifest["selected_jies"]
        ),
        "eligible_juans": manifest["sampling_frame"]["eligible_juans"],
        "remaining_eligible_juans": (
            manifest["sampling_frame"]["remaining_eligible_juans"]
        ),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
