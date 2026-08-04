from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from core import assemble_jies
from p3_compact import (
    CHALLENGE_COHORT,
    FOREIGN_TERMS,
    MAX_CHARS,
    MIN_CHARS,
    ROLE_TERMS,
    _git_commit_clean,
)
from p6_locked_assisted_tasks import POLICY_PATHS
from pilot import TEXT, _load


SELECTION_SEED = 20260806
ROLE_JIES = 4
FOREIGN_JIES = 4
EXPECTED_INPUTS = {
    "round11_task_manifest": (
        "ebe06c87674e49eee014624291553cac620b14b42a5ee15ae519c69c16a2d39a"
    ),
    "round11_reference_report": (
        "25f8788b1d49c22d133a4eca0851e746cb905c15f0d30b836bda27213e15e9be"
    ),
    "round12_adoption_report": (
        "539144dc11f936c6da81095459472032a21baf1ae54c1397ca4dd938f9437b38"
    ),
    "boundary_guide": (
        "f7c1dff1e97c41fc37846374867840ebde81b9b4fa601e3adbde2db9f1de8247"
    ),
    "spec": "ae0c837353adcdd88d0977acfea2836956ed377accc556ff02e21948923ea8f6",
    "spec_zh": "33f13ddefbb4fd46e5842abd1f476be291abf39140f286181287d28d2240fc66",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _term_counts(text: str, terms: tuple[str, ...]) -> dict[str, int]:
    return {term: text.count(term) for term in terms if text.count(term)}


def select_challenges(frame: list[dict], seed: int) -> list[dict]:
    available = {
        (int(row["juan"]), int(row["jie_index"])): row for row in frame
    }
    if len(available) != len(frame):
        raise ValueError("challenge frame has duplicate jie identities")
    rng = random.Random(seed)

    def cohort(terms: tuple[str, ...]) -> list[tuple[int, int]]:
        ranked = sorted(
            available,
            key=lambda key: (
                sum(available[key]["text"].count(term) for term in terms),
                available[key]["characters"],
                -key[0],
                -key[1],
            ),
            reverse=True,
        )
        return [
            key for key in ranked[:CHALLENGE_COHORT]
            if any(available[key]["text"].count(term) for term in terms)
        ]

    role_cohort = cohort(ROLE_TERMS)
    foreign_cohort = cohort(FOREIGN_TERMS)
    selected = []
    for role, terms, count, keys in (
        ("role_appellation_challenge", ROLE_TERMS, ROLE_JIES, role_cohort),
        ("foreign_title_challenge", FOREIGN_TERMS, FOREIGN_JIES, foreign_cohort),
    ):
        choices = [key for key in keys if key in available]
        if len(choices) < count:
            raise ValueError(f"not enough {role} jies")
        for key in rng.sample(choices, count):
            row = available.pop(key)
            counts = _term_counts(row["text"], terms)
            selected.append({
                "role": role,
                "term_count": sum(counts.values()),
                "term_counts": counts,
                **row,
            })
    return selected


def prepare_compact_challenge_tasks(
    output_dir: Path,
    round11_task_manifest: Path,
    round11_reference_report: Path,
    round12_adoption_report: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"compact challenge tasks exist: {output_dir}")
    paths = {
        "round11_task_manifest": round11_task_manifest,
        "round11_reference_report": round11_reference_report,
        "round12_adoption_report": round12_adoption_report,
        **POLICY_PATHS,
    }
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if hashes != EXPECTED_INPUTS:
        raise ValueError("compact challenge task inputs differ")
    round11_tasks = _load(round11_task_manifest)
    round11_reference = _load(round11_reference_report)
    adoption = _load(round12_adoption_report)
    if (
        round11_tasks.get("status")
        != "round11_fresh_promotion_tasks_before_labeling"
        or round11_reference.get("status")
        != "frozen_round11_fresh_promotion_reference"
        or adoption.get("status")
        != "round11_spec_aligned_adoption_statistics"
        or adoption.get("production_adoption_authorized") is not False
        or adoption.get("adoption_gates", {}).get(
            "predeclared_challenge_strata_available"
        ) is not False
    ):
        raise ValueError("compact challenge source status differs")
    source_juans = {int(row["juan"]) for row in round11_tasks["selected"]}
    expected_source_hashes = {
        int(row["juan"]): row["source_sha256"]
        for row in round11_tasks["selected"]
    }
    used_jies = {
        (int(row["juan"]), int(row["jie_index"]))
        for row in round11_tasks["selected_jies"]
    }
    if len(source_juans) != 37 or len(used_jies) != 37:
        raise ValueError("Round 11 source inventory differs")
    frame = []
    source_hashes = {}
    for juan in sorted(source_juans):
        path = TEXT / f"juan_{juan:03d}.json"
        source_sha256 = _sha256(path)
        if source_sha256 != expected_source_hashes[juan]:
            raise ValueError(f"Round 11 source changed for juan {juan}")
        source = _load(path)
        source_hashes[str(juan)] = source_sha256
        for jie in assemble_jies(source["paragraphs"]):
            identity = (juan, int(jie.index))
            if (
                identity in used_jies
                or jie.number is None
                or not MIN_CHARS <= len(jie.text) <= MAX_CHARS
            ):
                continue
            frame.append({
                "juan": juan,
                "jie_index": int(jie.index),
                "jie_number": jie.number,
                "text": jie.text,
                "segments": [asdict(segment) for segment in jie.segments],
                "characters": len(jie.text),
            })
    selected = select_challenges(frame, SELECTION_SEED)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in selected:
        grouped[int(row["juan"])].append(row)
    manifest = {
        "schema_version": 1,
        "status": "round13_compact_challenge_tasks_before_labeling",
        "supplementary_challenge_evidence_only": True,
        "formal_probability_metric": False,
        "eligible_for_training": False,
        "eligible_to_reverse_failed_precision_gate": False,
        "candidate_model_blind": True,
        "challenger_locked": "round7_2_of_3_exact_geometry_ensemble",
        "selection_seed": SELECTION_SEED,
        "selection_policy": {
            "frame": (
                "previously unused numbered jies of 20-600 characters from the "
                "37 juans that were wholly unused before Round 11"
            ),
            "role_appellation_challenge": (
                "four fixed-seed draws from the raw-text top-40 cohort"
            ),
            "foreign_title_challenge": (
                "four fixed-seed draws from the raw-text top-40 cohort"
            ),
        },
        "labeling_protocol": {
            "pass_a": "independent_recall_first",
            "pass_b": "independent_exact_boundary_first",
            "pass_visibility": "mutually_hidden_candidate_free_raw_text_only",
            "focused_review": "A_B_disagreement_and_explicit_low_only",
            "model_omission_candidates": False,
        },
        "sampling_frame": {
            "eligible_jies": len(frame),
            "source_juans": len(source_juans),
            "previously_used_jies_excluded": len(used_jies),
            "min_characters": MIN_CHARS,
            "max_characters": MAX_CHARS,
        },
        "inputs": hashes,
        "source_sha256_by_juan": source_hashes,
        "v1_used_for_selection": False,
        "rules_used_for_selection": False,
        "model_predictions_used_for_selection": False,
        "model_predictions_generated": False,
        "git_commit": _git_commit_clean(),
        "selected": [],
        "private_selected_jies": [{
            key: value for key, value in row.items()
            if key not in {"text", "segments"}
        } for row in selected],
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
                "phase": "copilot_double_pass",
                "supplementary_challenge_evidence_only": True,
                "candidate_model_blind": True,
                "juan": juan,
                "instructions": (
                    "Independently mark every main-text person span in the sampled "
                    "jie. Apply BOUNDARY_GUIDE.md using target-jie evidence only. "
                    "Do not read selection roles, another pass, any model/rule/v1 "
                    "prediction, translations, notes, identity data, historical "
                    "evaluation, or text outside this task."
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
                "source_sha256": source_hashes[str(juan)],
                "sampled_jies": len(rows),
            })
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in tasks_dir.iterdir():
            path.chmod(0o444)
        tasks_dir.chmod(0o555)
        (staging / "manifest.json").chmod(0o444)
        staging.chmod(0o555)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze supplementary compact-P3 challenge tasks."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round11-task-manifest", type=Path, required=True)
    parser.add_argument("--round11-reference-report", type=Path, required=True)
    parser.add_argument("--round12-adoption-report", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_compact_challenge_tasks(
        args.output,
        args.round11_task_manifest,
        args.round11_reference_report,
        args.round12_adoption_report,
    )
    print(json.dumps({
        "selected_jies": len(manifest["private_selected_jies"]),
        "task_juans": len(manifest["selected"]),
        "characters": sum(
            row["characters"] for row in manifest["private_selected_jies"]
        ),
        "roles": {
            role: sum(row["role"] == role for row in manifest["private_selected_jies"])
            for role in (
                "role_appellation_challenge",
                "foreign_title_challenge",
            )
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
