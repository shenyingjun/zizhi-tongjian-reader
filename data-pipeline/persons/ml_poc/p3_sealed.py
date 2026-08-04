from __future__ import annotations

import argparse
import hashlib
import json
import random
import secrets
import stat
import subprocess
import tempfile
from pathlib import Path

from pilot import TEXT, _load, build_blind_task


EXCLUDED_JUANS = {
    12, 13, 21, 27, 44, 52, 76, 201, 204, 207, 225, 248, 251,
}
RANDOM_JUANS = 3
CHALLENGE_COHORT = 5
ROLE_TERMS = (
    "太后", "太子", "皇后", "皇帝", "丞相", "大将军", "皇太后",
)
FOREIGN_TERMS = (
    "单于", "可汗", "谷蠡王", "左贤王", "右贤王", "达干", "叶护",
    "俟斤", "特勒", "大莫弗",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(source: dict) -> str:
    return "".join(
        str(paragraph.get("main", "") or "")
        for paragraph in source["paragraphs"]
    )


def _term_counts(text: str, terms: tuple[str, ...]) -> dict[str, int]:
    return {term: text.count(term) for term in terms if text.count(term)}


def select_sealed(
    sources: dict[int, dict],
    *,
    seed: int,
) -> list[dict]:
    available = sorted(set(sources) - EXCLUDED_JUANS)
    if len(available) < RANDOM_JUANS + 2:
        raise ValueError("at least five unconsumed juans are required")
    rng = random.Random(seed)
    random_juans = rng.sample(available, RANDOM_JUANS)
    remaining = [juan for juan in available if juan not in random_juans]
    texts = {juan: _text(sources[juan]) for juan in remaining}

    def take_challenge(role: str, terms: tuple[str, ...]) -> dict:
        cohort = sorted(
            remaining,
            key=lambda value: (
                sum(texts[value].count(term) for term in terms),
                len(texts[value]),
                -value,
            ),
            reverse=True,
        )[:CHALLENGE_COHORT]
        juan = rng.choice(cohort)
        remaining.remove(juan)
        counts = _term_counts(texts[juan], terms)
        return {
            "role": role,
            "juan": juan,
            "term_count": sum(counts.values()),
            "term_counts": counts,
        }

    selected = [
        {"role": "probability_random", "juan": juan}
        for juan in random_juans
    ]
    selected.append(take_challenge("role_appellation_challenge", ROLE_TERMS))
    selected.append(take_challenge("foreign_title_challenge", FOREIGN_TERMS))
    secrets.SystemRandom().shuffle(selected)
    return selected


def prepare_sealed(
    output_dir: Path,
    model_dir: Path,
    selected_model_path: Path,
    *,
    seed: int | None = None,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"sealed output already exists: {output_dir}")
    selected_model = _load(selected_model_path)
    model_path = model_dir / "model.safetensors"
    model_sha256 = _sha256(model_path)
    if model_sha256 != selected_model.get("model_sha256"):
        raise ValueError("selected model hash does not match model.safetensors")
    sources = {}
    source_paths = {}
    for juan in range(1, 295):
        path = TEXT / f"juan_{juan:03d}.json"
        if path.is_file():
            sources[juan] = _load(path)
            source_paths[juan] = path
    selection_seed = seed if seed is not None else secrets.randbits(128)
    selected = select_sealed(sources, seed=selection_seed)

    manifest = {
        "schema_version": 1,
        "status": "sealed_before_annotation",
        "selection_seed": selection_seed,
        "selection_policy": {
            "probability_random": (
                "three juans sampled uniformly without replacement from all "
                "available unconsumed juans"
            ),
            "role_appellation_challenge": (
                "private-seed draw from the five highest raw-text counts of "
                "the predeclared role terms"
            ),
            "foreign_title_challenge": (
                "private-seed draw from the five highest raw-text counts of "
                "the predeclared foreign-title terms"
            ),
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
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "selected": [],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        for selection in selected:
            juan = int(selection["juan"])
            task = build_blind_task(juan, sources[juan], selection["role"])
            task["instructions"] = (
                "Mark all main-text person spans. This is a sealed "
                "candidate-blind evaluation."
            )
            task_path = staging / f"blind_juan_{juan:03d}.json"
            task_path.write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest["selected"].append({
                **selection,
                "mode": "sealed_blind",
                "source_sha256": _sha256(source_paths[juan]),
                "task_sha256": _sha256(task_path),
                "task": task_path.name,
            })
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in staging.iterdir():
            if path.is_file():
                path.chmod(stat.S_IREAD)
                writable = path.stat().st_mode & (
                    stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
                )
                if writable:
                    raise PermissionError(
                        f"failed to make sealed task read-only: {path}"
                    )
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze candidate-blind P3 probability and challenge tasks."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--selected-model", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_sealed(
        args.output, args.model, args.selected_model
    )
    print(json.dumps({
        "tasks": len(manifest["selected"]),
        "random_juans": sum(
            row["role"] == "probability_random"
            for row in manifest["selected"]
        ),
        "model_sha256": manifest["selected_model"]["sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
