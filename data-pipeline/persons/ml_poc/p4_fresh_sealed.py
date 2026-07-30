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
from pilot import TEXT, _load


EXPECTED_MODEL_ARTIFACT_SHA256 = (
    "fb611bee98cdc672e19163ca84aef49612815cab285889c7458cff07ba098cbd"
)
EXPECTED_MODEL_REPORT_SHA256 = (
    "b65bccba68bb37dea23da7a57b84336b906a32769145f0d33b9f679d57d73353"
)
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "d60b8f73f5fc16e091f0aa7747ebb692620126e5f6c2c7dd1cbd4579d08084da"
)

# Every juan previously used for training, development, evaluation, challenge
# selection, or exposed diagnostic work before the Round 3 candidate was frozen.
EXCLUDED_JUANS = {
    3, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24, 27, 29, 32, 33,
    34, 36, 37, 41, 44, 45, 46, 47, 48, 50, 51, 52, 53, 57, 60, 62,
    69, 76, 78, 80, 83, 86, 87, 90, 95, 96, 103, 106, 108, 109,
    112, 113, 120, 126, 129, 130, 136, 138, 141, 145, 150, 155, 156,
    168, 172, 174, 176, 177, 180, 185, 193, 194, 195, 198, 200, 201,
    202, 203, 204, 205, 206, 207, 212, 216, 220, 222, 225, 227, 235,
    237, 244, 245, 246, 248, 249, 250, 251, 264, 271, 279, 280, 283,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eligible_jies(sources: dict[int, dict]) -> list[dict]:
    rows = []
    for juan in sorted(set(sources) - EXCLUDED_JUANS):
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


def _validate_candidate(model_root: Path, report_path: Path) -> dict:
    if _sha256(report_path) != EXPECTED_MODEL_REPORT_SHA256:
        raise ValueError("Round 3 model report hash differs")
    report = _load(report_path)
    artifact = _model_artifact(model_root / "model")
    if (
        artifact.get("combined_sha256") != EXPECTED_MODEL_ARTIFACT_SHA256
        or artifact != report.get("round3_control", {}).get("model_artifact")
        or report.get("round3_control", {}).get("dataset_manifest_sha256")
        != EXPECTED_DATASET_MANIFEST_SHA256
        or report.get("config", {}).get("selected_epoch") != 4
    ):
        raise ValueError("model is not the frozen Round 3 candidate")
    return artifact


def prepare_fresh_sealed(
    output_dir: Path,
    model_root: Path,
    report_path: Path,
    *,
    seed: int | None = None,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"fresh sealed output exists: {output_dir}")
    artifact = _validate_candidate(model_root, report_path)
    git_commit = _git_commit_clean()
    sources = {}
    source_paths = {}
    for juan in range(1, 295):
        path = TEXT / f"juan_{juan:03d}.json"
        if path.is_file():
            sources[juan] = _load(path)
            source_paths[juan] = path
    frame = eligible_jies(sources)
    selection_seed = seed if seed is not None else secrets.randbits(128)
    selected_jies = select_compact(frame, seed=selection_seed)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in selected_jies:
        grouped[int(row["juan"])].append(row)

    manifest = {
        "schema_version": 1,
        "status": "fresh_sealed_before_annotation",
        "formal_evaluation": True,
        "candidate_blind": True,
        "selection_seed": selection_seed,
        "selection_policy": {
            "frame": (
                f"numbered jies with {MIN_CHARS}-{MAX_CHARS} Unicode "
                "codepoints from 196 juans never previously consumed"
            ),
            "probability_random": (
                f"{RANDOM_JIES} jies sampled uniformly without replacement"
            ),
            "role_appellation_challenge": (
                f"{ROLE_JIES} private-seed draws from the frozen top-40 cohort"
            ),
            "foreign_title_challenge": (
                f"{FOREIGN_JIES} private-seed draws from the frozen top-40 cohort"
            ),
        },
        "sampling_frame": {
            "eligible_jies": len(frame),
            "eligible_juans": len({row["juan"] for row in frame}),
            "min_characters": MIN_CHARS,
            "max_characters": MAX_CHARS,
        },
        "excluded_juans": sorted(EXCLUDED_JUANS),
        "v1_used_for_selection": False,
        "rules_used_for_selection": False,
        "model_predictions_generated": False,
        "selected_model": {
            "artifact_sha256": artifact["combined_sha256"],
            "report_sha256": _sha256(report_path),
            "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
            "selected_epoch": 4,
            "selection_basis": "highest Juan 27 challenge-dev exact F1",
        },
        "git_commit": git_commit,
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
        for juan, juan_rows in grouped.items():
            task = {
                "schema_version": 1,
                "phase": "blind",
                "juan": juan,
                "instructions": (
                    "Mark every main-text person span in these sampled jies. "
                    "Apply the frozen boundary guide without consulting candidates, "
                    "models, rules, translations, identity data, or omitted text."
                ),
                "jies": [
                    {
                        "jie_index": row["jie_index"],
                        "jie_number": row["jie_number"],
                        "text": row["text"],
                        "segments": row["segments"],
                        "annotations": [],
                    }
                    for row in sorted(
                        juan_rows, key=lambda item: item["jie_index"]
                    )
                ],
            }
            task_path = staging / f"blind_juan_{juan:03d}.json"
            task_path.write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest["selected"].append({
                "juan": juan,
                "mode": "fresh_sealed_blind",
                "source_sha256": _sha256(source_paths[juan]),
                "task_sha256": _sha256(task_path),
                "task": task_path.name,
                "sampled_jies": len(juan_rows),
            })
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in staging.iterdir():
            path.chmod(stat.S_IREAD)
            if path.stat().st_mode & (
                stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
            ):
                raise PermissionError(f"failed to make sealed file read-only: {path}")
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a fresh candidate-blind Round 3 evaluation set."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_fresh_sealed(
        args.output, args.model_root, args.report
    )
    print(json.dumps({
        "sampled_jies": len(manifest["private_selected_jies"]),
        "characters": sum(
            row["characters"] for row in manifest["private_selected_jies"]
        ),
        "ui_tasks": len(manifest["selected"]),
        "eligible_juans": manifest["sampling_frame"]["eligible_juans"],
        "model_artifact_sha256": (
            manifest["selected_model"]["artifact_sha256"]
        ),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
