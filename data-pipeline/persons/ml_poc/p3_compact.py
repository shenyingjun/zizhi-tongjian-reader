from __future__ import annotations

import argparse
import hashlib
import json
import random
import secrets
import stat
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from core import assemble_jies
from pilot import TEXT, _load


REPO_ROOT = Path(__file__).resolve().parents[3]
EXCLUDED_JUANS = {
    12, 13, 21, 24, 27, 37, 44, 46, 52, 76, 177, 201, 204, 205,
    207, 225, 248, 251,
}
MIN_CHARS = 20
MAX_CHARS = 600
RANDOM_JIES = 12
ROLE_JIES = 4
FOREIGN_JIES = 4
CHALLENGE_COHORT = 40
EXPECTED_MODEL_SHA256 = (
    "2149e9283f239a02969b6d7663d64faf2dbb193832fe5e1bcd7a3c623aa7f90c"
)
ROLE_TERMS = (
    "太后", "太子", "皇后", "皇帝", "丞相", "大将军", "皇太后",
    "使君", "贵人", "夫人",
)
FOREIGN_TERMS = (
    "单于", "可汗", "谷蠡王", "左贤王", "右贤王", "达干", "叶护",
    "俟斤", "特勒", "大莫弗", "昆弥", "阏氏",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _term_counts(text: str, terms: tuple[str, ...]) -> dict[str, int]:
    return {term: text.count(term) for term in terms if text.count(term)}


def _git_commit_clean() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    ).stdout
    if status:
        raise RuntimeError(
            "refusing to generate a formal sealed set from a dirty worktree"
        )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    ).stdout.strip()


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


def select_compact(jies: list[dict], *, seed: int) -> list[dict]:
    if len(jies) < RANDOM_JIES + ROLE_JIES + FOREIGN_JIES:
        raise ValueError("compact sampling frame is too small")
    rng = random.Random(seed)
    available = {
        (int(row["juan"]), int(row["jie_index"])): row for row in jies
    }
    if len(available) != len(jies):
        raise ValueError("sampling frame has duplicate juan/jie rows")

    def challenge_cohort(terms: tuple[str, ...]) -> list[tuple[int, int]]:
        ranked = sorted(
            available.items(),
            key=lambda item: (
                sum(item[1]["text"].count(term) for term in terms),
                item[1]["characters"],
                -item[0][0],
                -item[0][1],
            ),
            reverse=True,
        )
        cohort = [
            key
            for key, row in ranked[:CHALLENGE_COHORT]
            if any(row["text"].count(term) for term in terms)
        ]
        return cohort

    role_cohort = challenge_cohort(ROLE_TERMS)
    foreign_cohort = challenge_cohort(FOREIGN_TERMS)
    random_keys = rng.sample(sorted(available), RANDOM_JIES)
    selected = [
        {
            "role": "probability_random",
            **available.pop(key),
        }
        for key in random_keys
    ]

    def draw_challenge(
        role: str,
        terms: tuple[str, ...],
        count: int,
        cohort_keys: list[tuple[int, int]],
    ) -> None:
        cohort = [
            (key, available[key])
            for key in cohort_keys
            if key in available
        ]
        if len(cohort) < count:
            raise ValueError(f"not enough {role} challenge jies")
        chosen = rng.sample(cohort, count)
        for key, row in chosen:
            counts = _term_counts(row["text"], terms)
            selected.append({
                "role": role,
                "term_count": sum(counts.values()),
                "term_counts": counts,
                **available.pop(key),
            })

    draw_challenge(
        "role_appellation_challenge", ROLE_TERMS, ROLE_JIES, role_cohort
    )
    draw_challenge(
        "foreign_title_challenge",
        FOREIGN_TERMS,
        FOREIGN_JIES,
        foreign_cohort,
    )
    if len(selected) != RANDOM_JIES + ROLE_JIES + FOREIGN_JIES:
        raise AssertionError("compact selection count differs")
    return selected


def prepare_compact(
    output_dir: Path,
    model_dir: Path,
    selected_model_path: Path,
    *,
    seed: int | None = None,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"compact sealed output exists: {output_dir}")
    selected_model = _load(selected_model_path)
    model_path = model_dir / "model.safetensors"
    model_sha256 = _sha256(model_path)
    if model_sha256 != EXPECTED_MODEL_SHA256:
        raise ValueError("model.safetensors is not the frozen Round 2 candidate")
    if model_sha256 != selected_model.get("model_sha256"):
        raise ValueError("selected model hash does not match model.safetensors")
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
    public_juans = list(grouped)
    secrets.SystemRandom().shuffle(public_juans)

    manifest = {
        "schema_version": 1,
        "status": "compact_sealed_before_annotation",
        "formal_p3": True,
        "selection_seed": selection_seed,
        "selection_policy": {
            "frame": (
                f"numbered jies with {MIN_CHARS}-{MAX_CHARS} Unicode "
                "codepoints from previously unused juans"
            ),
            "probability_random": (
                f"{RANDOM_JIES} jies sampled uniformly without replacement"
            ),
            "role_appellation_challenge": (
                f"private-seed draw of {ROLE_JIES} from the top "
                f"{CHALLENGE_COHORT} predeclared-term jies, frozen before "
                "any draw and excluding already selected jies"
            ),
            "foreign_title_challenge": (
                f"private-seed draw of {FOREIGN_JIES} from the top "
                f"{CHALLENGE_COHORT} predeclared-term jies, frozen before "
                "any draw and excluding already selected jies"
            ),
        },
        "sampling_frame": {
            "eligible_jies": len(frame),
            "eligible_juans": len({int(row["juan"]) for row in frame}),
            "min_characters": MIN_CHARS,
            "max_characters": MAX_CHARS,
        },
        "excluded_juans": sorted(EXCLUDED_JUANS),
        "candidate_blind": True,
        "v1_used_for_selection": False,
        "rules_used_for_selection": False,
        "model_predictions_generated": False,
        "selected_model": {
            "sha256": model_sha256,
            "selection_manifest_sha256": _sha256(selected_model_path),
            "selection_basis": selected_model["selection_basis"],
            "selected_mode": selected_model["selected_mode"],
            "selected_epoch": selected_model["selected_epoch"],
        },
        "git_commit": git_commit,
        "selected": [],
        "private_selected_jies": [],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        for juan in public_juans:
            rows = sorted(grouped[juan], key=lambda row: row["jie_index"])
            task = {
                "schema_version": 1,
                "phase": "blind",
                "juan": juan,
                "instructions": (
                    "Mark every main-text person span in these sampled jies. "
                    "Do not consult candidates, models, rules, translations, "
                    "notes, identity data, or omitted parts of the juan."
                ),
                "jies": [
                    {
                        "jie_index": row["jie_index"],
                        "jie_number": row["jie_number"],
                        "text": row["text"],
                        "segments": row["segments"],
                        "annotations": [],
                    }
                    for row in rows
                ],
            }
            task_path = staging / f"blind_juan_{juan:03d}.json"
            task_path.write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest["selected"].append({
                "juan": juan,
                "role": "compact_sealed",
                "mode": "sealed_blind",
                "source_sha256": _sha256(source_paths[juan]),
                "task_sha256": _sha256(task_path),
                "task": task_path.name,
                "sampled_jies": len(rows),
            })
        manifest["private_selected_jies"] = [
            {
                key: value
                for key, value in row.items()
                if key not in {"text", "segments"}
            }
            for row in selected_jies
        ]
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in staging.iterdir():
            if path.is_file():
                path.chmod(stat.S_IREAD)
                if path.stat().st_mode & (
                    stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
                ):
                    raise PermissionError(
                        f"failed to make compact task read-only: {path}"
                    )
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a compact candidate-blind P3 jie sample."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--selected-model", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_compact(
        args.output, args.model, args.selected_model
    )
    counts = defaultdict(int)
    characters = 0
    for row in manifest["private_selected_jies"]:
        counts[row["role"]] += 1
        characters += int(row["characters"])
    print(json.dumps({
        "sampled_jies": sum(counts.values()),
        "probability_random": counts["probability_random"],
        "role_challenge": counts["role_appellation_challenge"],
        "foreign_challenge": counts["foreign_title_challenge"],
        "characters": characters,
        "ui_tasks": len(manifest["selected"]),
        "model_sha256": manifest["selected_model"]["sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
