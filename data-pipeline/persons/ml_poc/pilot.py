from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from core import assemble_jies


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TEXT = REPO / "web" / "public" / "text"
V1 = TEXT / "persons" / "mentions"
RULES = TEXT / "persons-v2" / "agent1"
DEFAULT_SEED = 20260726
CHALLENGE_TERMS = ("可汗", "单于")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v1_geometries(document: dict) -> set[tuple[int, int, int]]:
    return {
        (int(row["pid"]), int(row["start"]), int(row["end"]))
        for row in document.get("mentions", [])
        if row.get("source", "main") == "main"
    }


def _rule_geometries(document: dict) -> set[tuple[int, int, int]]:
    return {
        (int(row["para_id"]), int(row["start"]), int(row["end"]))
        for row in document.get("occurrences", [])
        if row.get("field", "main") == "main"
    }


def score_juan(source: dict, v1: dict, rules: dict) -> dict:
    v1_geometry = _v1_geometries(v1)
    rule_geometry = _rule_geometries(rules)
    union = v1_geometry | rule_geometry
    disagreement = v1_geometry ^ rule_geometry
    main_mentions = [
        row for row in v1.get("mentions", [])
        if row.get("source", "main") == "main"
    ]
    all_text = "".join(
        str(row.get("main", "") or "") for row in source["paragraphs"]
    )
    feng = sum(row.get("kind") == "feng" for row in main_mentions)
    single_anaphora = sum(
        row.get("kind") == "anaphora"
        and int(row["end"]) - int(row["start"]) == 1
        for row in main_mentions
    )
    foreign_titles = sum(all_text.count(term) for term in CHALLENGE_TERMS)
    challenge_score = (
        20 * feng + 5 * foreign_titles + min(single_anaphora, 20)
    )
    return {
        "v1_spans": len(v1_geometry),
        "rule_spans": len(rule_geometry),
        "exact_agreements": len(v1_geometry & rule_geometry),
        "exact_disagreements": len(disagreement),
        "disagreement_rate": len(disagreement) / len(union) if union else 0.0,
        "feng": feng,
        "foreign_titles": foreign_titles,
        "single_char_anaphora": single_anaphora,
        "challenge_score": challenge_score,
    }


def select_pilot(
    scores: dict[int, dict],
    *,
    seed: int = DEFAULT_SEED,
) -> list[tuple[str, int]]:
    if len(scores) < 3:
        raise ValueError("at least three juans are required")
    juans = sorted(scores)
    random_juan = random.Random(seed).choice(juans)
    remaining = [juan for juan in juans if juan != random_juan]
    disagreement_juan = max(
        remaining,
        key=lambda juan: (
            scores[juan]["disagreement_rate"],
            scores[juan]["exact_disagreements"],
            -juan,
        ),
    )
    remaining.remove(disagreement_juan)
    challenge_juan = max(
        remaining,
        key=lambda juan: (
            scores[juan]["feng"] + scores[juan]["foreign_titles"],
            scores[juan]["challenge_score"],
            scores[juan]["feng"],
            scores[juan]["foreign_titles"],
            -juan,
        ),
    )
    return [
        ("random", random_juan),
        ("rules_v1_disagreement", disagreement_juan),
        ("rare_pattern_challenge", challenge_juan),
    ]


def build_blind_task(juan: int, source: dict, role: str) -> dict:
    return {
        "schema_version": 1,
        "phase": "blind",
        "juan": juan,
        "instructions": (
            "Mark main-text person spans without consulting v1, rules, "
            "translation, notes, or identity data."
        ),
        "jies": [
            {
                "jie_index": jie.index,
                "jie_number": jie.number,
                "text": jie.text,
                "segments": [asdict(segment) for segment in jie.segments],
                "annotations": [],
            }
            for jie in assemble_jies(source["paragraphs"])
        ],
    }


def prepare(output_dir: Path, juans: Iterable[int] = range(1, 295)) -> dict:
    scores: dict[int, dict] = {}
    paths: dict[int, tuple[Path, Path, Path]] = {}
    for juan in juans:
        source_path = TEXT / f"juan_{juan:03d}.json"
        v1_path = V1 / f"juan_{juan:03d}.json"
        rules_path = RULES / f"juan_{juan:03d}.json"
        for path in (source_path, v1_path, rules_path):
            if not path.is_file():
                raise FileNotFoundError(f"required pilot input is missing: {path}")
        paths[juan] = (source_path, v1_path, rules_path)
        scores[juan] = score_juan(
            _load(source_path), _load(v1_path), _load(rules_path)
        )

    selected = select_pilot(scores)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for role, juan in selected:
        source_path, v1_path, rules_path = paths[juan]
        task = build_blind_task(juan, _load(source_path), role)
        task_path = output_dir / f"blind_juan_{juan:03d}.json"
        task_path.write_text(
            json.dumps(task, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                "role": role,
                "juan": juan,
                "scores": scores[juan],
                "source_sha256": _sha256(source_path),
                "v1_sha256": _sha256(v1_path),
                "rules_sha256": _sha256(rules_path),
                "blind_task": task_path.name,
            }
        )

    manifest = {
        "schema_version": 1,
        "selection_seed": DEFAULT_SEED,
        "selection_policy": {
            "random": "uniform over available juans with fixed seed",
            "rules_v1_disagreement": "highest exact symmetric-difference rate",
            "rare_pattern_challenge": (
                "highest feng + 可汗/单于 count, then "
                "20*feng + 5*(可汗/单于 count) "
                "+ min(single-character anaphora, 20)"
            ),
        },
        "selected": manifest_rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select P0 pilot juans and emit candidate-blind annotation tasks."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New or existing directory for the pilot manifest and blind tasks.",
    )
    args = parser.parse_args()
    manifest = prepare(args.output)
    print(json.dumps(manifest["selected"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
