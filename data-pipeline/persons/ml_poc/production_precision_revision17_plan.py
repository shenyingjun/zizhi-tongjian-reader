from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from production_program import (
    BOUNDARY_ANAPHORA_TERMS,
    CHALLENGE_COHORT,
    FOREIGN_TERMS,
    ROLE_TERMS,
    TEXT,
    _term_score,
    eligible_jies,
    load_exact_exclusions,
)
from production_train import _make_read_only


REVISION = 17
SEED = 20260822
FORMAL_COUNTS = {
    "foreign_title": 20,
    "role_appellation": 20,
    "boundary_anaphora": 20,
    "uniform_random": 100,
}
EXPECTED_BASE_EXCLUSIONS = 19079
EXPECTED_ROUND2_SELECTED = 180
EXPECTED_FRAME = 1258
EXPECTED_JUANS = 37
EXPECTED_MINING = 1098
PLAN_STATUS = "ml_production_precision_revision17_plan"
ROUND2_TASK_STATUS = "ml_production_round_tasks_before_labeling"
PRIVATE_STATUS = "ml_production_private_task_roles"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _key(row: dict) -> tuple[int, int]:
    return int(row["juan"]), int(row["jie_index"])


def _challenge_cohort(
    frame: list[dict], terms: tuple[str, ...]
) -> list[tuple[tuple[int, int], dict, int]]:
    ranked = sorted(
        (
            (_key(row), row, _term_score(row, terms))
            for row in frame
        ),
        key=lambda item: (
            item[2],
            int(item[1]["characters"]),
            -item[0][0],
            -item[0][1],
        ),
        reverse=True,
    )
    return [item for item in ranked[:CHALLENGE_COHORT] if item[2] > 0]


def select_formal_reserve(
    frame: list[dict],
    *,
    seed: int = SEED,
    counts: dict[str, int] = FORMAL_COUNTS,
) -> list[dict]:
    available = {_key(row): row for row in frame}
    if len(available) != len(frame):
        raise ValueError("Revision-17 frame contains duplicate jies")

    selected: list[dict] = []

    def reserve(key: tuple[int, int], row: dict, stratum: str, score: int | None):
        if key not in available:
            raise ValueError("Revision-17 formal reserve repeats a jie")
        available.pop(key)
        selected.append({
            "stratum": stratum,
            "term_score": score,
            **row,
        })

    foreign = [
        (key, row, _term_score(row, FOREIGN_TERMS))
        for key, row in sorted(available.items())
        if _term_score(row, FOREIGN_TERMS) > 0
    ]
    if len(foreign) != counts["foreign_title"]:
        raise ValueError(
            "Revision-17 foreign-title reserve does not exactly exhaust its cohort"
        )
    for key, row, score in foreign:
        reserve(key, row, "foreign_title", score)

    for stream, (stratum, terms) in enumerate(
        (
            ("role_appellation", ROLE_TERMS),
            ("boundary_anaphora", BOUNDARY_ANAPHORA_TERMS),
        ),
        1,
    ):
        cohort = [
            item for item in _challenge_cohort(frame, terms)
            if item[0] in available
        ]
        count = counts[stratum]
        if len(cohort) < count:
            raise ValueError(f"Revision-17 lacks {stratum} formal reserve")
        ordered_cohort = sorted(cohort, key=lambda item: item[0])
        for key, row, score in random.Random(seed + stream).sample(
            ordered_cohort, count
        ):
            reserve(key, row, stratum, score)

    uniform_count = counts["uniform_random"]
    if len(available) < uniform_count:
        raise ValueError("Revision-17 lacks uniform formal reserve")
    uniform_keys = random.Random(seed + 3).sample(
        sorted(available), uniform_count
    )
    for key in uniform_keys:
        reserve(key, available[key], "uniform_random", None)

    expected = sum(counts.values())
    if len(selected) != expected or len({_key(row) for row in selected}) != expected:
        raise ValueError("Revision-17 formal reserve cardinality differs")
    return sorted(selected, key=lambda row: (row["stratum"], *_key(row)))


def _mining_example(row: dict) -> dict:
    text = str(row["text"])
    return {
        "id": f"juan-{int(row['juan']):03d}-jie-{int(row['jie_index']):04d}",
        "juan": int(row["juan"]),
        "jie_index": int(row["jie_index"]),
        "text": text,
        "segments": list(row["segments"]),
        "labels": [0] * len(text),
        "target_mask": [True] * len(text),
    }


def freeze_plan(
    exclusion_path: Path,
    round2_root: Path,
    output_dir: Path,
    *,
    source_dir: Path = TEXT,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Revision-17 plan exists: {output_dir}")

    excluded, exclusion_manifest = load_exact_exclusions(exclusion_path)
    round2_manifest_path = round2_root / "manifest.json"
    private_path = round2_root / "private" / "selection.json"
    round2_manifest = _read(round2_manifest_path)
    private = _read(private_path)
    selected_rows = private.get("selected_jies", [])
    round2_selected = {
        (int(row["juan"]), int(row["jie_index"])) for row in selected_rows
    }
    if (
        len(excluded) != EXPECTED_BASE_EXCLUSIONS
        or exclusion_manifest.get("program_round") != 2
        or exclusion_manifest.get("replacement_round_authorized") is not True
        or round2_manifest.get("status") != ROUND2_TASK_STATUS
        or round2_manifest.get("replacement_round") is not True
        or round2_manifest.get("exclusion_manifest_sha256")
        != _sha256(exclusion_path)
        or round2_manifest.get("private_selection_sha256") != _sha256(private_path)
        or private.get("status") != PRIVATE_STATUS
        or len(selected_rows) != EXPECTED_ROUND2_SELECTED
        or len(round2_selected) != EXPECTED_ROUND2_SELECTED
        or excluded & round2_selected
    ):
        raise ValueError("Revision-17 Round-2 exclusion provenance differs")

    frame, source_paths = eligible_jies(source_dir, excluded | round2_selected)
    if (
        len(frame) != EXPECTED_FRAME
        or len({_key(row) for row in frame}) != EXPECTED_FRAME
        or len({int(row["juan"]) for row in frame}) != EXPECTED_JUANS
    ):
        raise ValueError("Revision-17 post-Round-2 frame differs")

    reserve = select_formal_reserve(frame)
    reserve_keys = {_key(row) for row in reserve}
    mining = [row for row in frame if _key(row) not in reserve_keys]
    if (
        len(reserve) != sum(FORMAL_COUNTS.values())
        or len(mining) != EXPECTED_MINING
        or reserve_keys & {_key(row) for row in mining}
        or reserve_keys | {_key(row) for row in mining}
        != {_key(row) for row in frame}
    ):
        raise ValueError("Revision-17 reserve/mining partition differs")

    git_commit = _git_commit_clean()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        sealed = staging / "sealed"
        sealed.mkdir()

        mining_path = staging / "mining.jsonl"
        _write_jsonl(
            mining_path,
            [_mining_example(row) for row in sorted(mining, key=_key)],
        )
        reserve_path = sealed / "formal-reserve.jsonl"
        _write_jsonl(
            reserve_path,
            [
                {
                    "juan": int(row["juan"]),
                    "jie_index": int(row["jie_index"]),
                    "jie_number": int(row["jie_number"]),
                    "stratum": str(row["stratum"]),
                    "term_score": row["term_score"],
                    "source_sha256": _sha256(source_paths[int(row["juan"])]),
                }
                for row in reserve
            ],
        )
        mining_geometry_path = sealed / "mining-geometry.jsonl"
        _write_jsonl(
            mining_geometry_path,
            [
                {
                    "juan": int(row["juan"]),
                    "jie_index": int(row["jie_index"]),
                    "jie_number": int(row["jie_number"]),
                    "source_sha256": _sha256(source_paths[int(row["juan"])]),
                }
                for row in sorted(mining, key=_key)
            ],
        )
        manifest = {
            "schema_version": 1,
            "status": PLAN_STATUS,
            "revision": REVISION,
            "fit_only": True,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "formal_reserve_text_read": False,
            "candidate_inference_performed": False,
            "seed": SEED,
            "git_commit": git_commit,
            "bindings": {
                "exclusion_manifest_sha256": _sha256(exclusion_path),
                "round2_manifest_sha256": _sha256(round2_manifest_path),
                "round2_private_selection_sha256": _sha256(private_path),
            },
            "counts": {
                "base_exclusions": len(excluded),
                "round2_selected": len(round2_selected),
                "eligible_frame": len(frame),
                "eligible_juans": len({int(row["juan"]) for row in frame}),
                "formal_reserve": len(reserve),
                "mining": len(mining),
                "formal_strata": {
                    name: sum(row["stratum"] == name for row in reserve)
                    for name in FORMAL_COUNTS
                },
            },
            "outputs": {
                "mining_sha256": _sha256(mining_path),
                "formal_reserve_sha256": _sha256(reserve_path),
                "mining_geometry_sha256": _sha256(mining_geometry_path),
            },
            "claim_limit": (
                "Candidate-blind fit-only mining plan. The sealed formal reserve "
                "is excluded before inference and remains unread."
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
        description="Freeze the Revision-17 formal reserve and mining frame."
    )
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--round2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=TEXT)
    args = parser.parse_args()
    manifest = freeze_plan(
        args.exclusions,
        args.round2,
        args.output,
        source_dir=args.source_dir,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
