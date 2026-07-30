from __future__ import annotations

import argparse
import hashlib
import json
import random
import secrets
import stat
import tempfile
from collections import defaultdict
from pathlib import Path

from p3_compact import MAX_CHARS, MIN_CHARS, _git_commit_clean
from p4_fresh_sealed import (
    EXCLUDED_JUANS,
    EXPECTED_DATASET_MANIFEST_SHA256,
    _validate_candidate,
    eligible_jies,
)
from pilot import TEXT, _load


PREVIOUS_ASSISTED_JUANS = {
    5, 28, 92, 97, 101, 107, 134, 149, 158, 163,
    165, 166, 169, 173, 188, 226, 260, 267, 269, 270,
}
SAMPLED_JIES = 60


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def prepare_next_assisted(
    output_dir: Path,
    model_root: Path,
    report_path: Path,
    *,
    seed: int | None = None,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"next assisted output exists: {output_dir}")
    artifact = _validate_candidate(model_root, report_path)
    git_commit = _git_commit_clean()
    sources = {}
    source_paths = {}
    excluded = EXCLUDED_JUANS | PREVIOUS_ASSISTED_JUANS
    for juan in range(1, 295):
        path = TEXT / f"juan_{juan:03d}.json"
        if path.is_file() and juan not in excluded:
            sources[juan] = _load(path)
            source_paths[juan] = path
    frame = eligible_jies(sources)
    selection_seed = seed if seed is not None else secrets.randbits(128)
    selected = select_unique_juan_rows(frame, seed=selection_seed)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in selected:
        grouped[int(row["juan"])].append(row)
    manifest = {
        "schema_version": 1,
        "status": "copilot_double_pass_tasks_before_labeling",
        "formal_evaluation": False,
        "eligible_for_training_after_focused_review": True,
        "labeling_protocol": {
            "pass_a": "independent_recall_first",
            "pass_b": "independent_exact_boundary_first",
            "auto_accept": "exact_A_B_consensus_without_explicit_low",
            "human_review": "disagreement_explicit_low_and_model_only",
            "model_role": "omission_candidates_only",
        },
        "selection_seed": selection_seed,
        "selection_policy": (
            f"{SAMPLED_JIES} previously unused eligible juans sampled uniformly "
            "without replacement, then one eligible numbered jie sampled "
            "uniformly within each selected juan"
        ),
        "sampling_frame": {
            "eligible_jies": len(frame),
            "eligible_juans": len({int(row["juan"]) for row in frame}),
            "min_characters": MIN_CHARS,
            "max_characters": MAX_CHARS,
        },
        "excluded_juans": sorted(excluded),
        "v1_used_for_selection": False,
        "rules_used_for_selection": False,
        "model_predictions_used_for_selection": False,
        "selected_model": {
            "purpose": "post_teacher_omission_candidates_only",
            "artifact_sha256": artifact["combined_sha256"],
            "report_sha256": _sha256(report_path),
            "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
            "selected_epoch": 4,
        },
        "git_commit": git_commit,
        "selected": [],
        "selected_jies": [
            {
                key: value
                for key, value in row.items()
                if key not in {"text", "segments"}
            }
            for row in selected
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
                "phase": "copilot_double_pass",
                "diagnostic_only": True,
                "juan": juan,
                "instructions": (
                    "Independently mark every main-text person span in the "
                    "sampled jie. Apply BOUNDARY_GUIDE.md using target-jie "
                    "evidence only. Do not read another teacher pass, model "
                    "predictions, rules, v1, translations, notes, or identity data."
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
        description="Freeze the next 60-jie double-Copilot assisted batch."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_next_assisted(
        args.output, args.model_root, args.report
    )
    print(json.dumps({
        "sampled_jies": len(report["selected_jies"]),
        "characters": sum(
            row["characters"] for row in report["selected_jies"]
        ),
        "eligible_juans": report["sampling_frame"]["eligible_juans"],
        "model_artifact_sha256": (
            report["selected_model"]["artifact_sha256"]
        ),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
