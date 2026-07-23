"""Audited exclusions from the production-v1 compatibility reference."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PERS = HERE.parent
REPO = PERS.parents[1]
TEXT = REPO / "web" / "public" / "text"
V1_MENTIONS = TEXT / "persons" / "mentions"
EXCLUSIONS_PATH = HERE / "benchmark-reference-exclusions.jsonl"


def geometry_key(mention: dict) -> tuple[int, int, int]:
    return mention["pid"], mention["start"], mention["end"]


def exclusions_sha256(path: Path = EXCLUSIONS_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_exclusions(
    path: Path = EXCLUSIONS_PATH,
    *,
    validate: bool = True,
) -> dict[int, dict[tuple[int, int, int], dict]]:
    exclusions: dict[int, dict[tuple[int, int, int], dict]] = (
        collections.defaultdict(dict)
    )
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        row = json.loads(raw)
        required = {
            "juan", "pid", "start", "end", "surface", "reason", "review"
        }
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"{path}:{line_number}: missing fields {sorted(missing)}"
            )
        key = (row["pid"], row["start"], row["end"])
        if key in exclusions[row["juan"]]:
            raise ValueError(
                f"{path}:{line_number}: duplicate geometry "
                f"{row['juan']}:{row['pid']}:{row['start']}:{row['end']}"
            )
        exclusions[row["juan"]][key] = row
    if validate:
        validate_exclusions(exclusions, path)
    return dict(exclusions)


def validate_exclusions(
    exclusions: dict[int, dict[tuple[int, int, int], dict]],
    path: Path = EXCLUSIONS_PATH,
) -> None:
    for juan, rows in exclusions.items():
        text_document = json.loads(
            (TEXT / f"juan_{juan:03d}.json").read_text(encoding="utf-8")
        )
        paragraph_text = {
            paragraph["id"]: paragraph.get("main", "")
            for paragraph in text_document["paragraphs"]
        }
        v1_document = json.loads(
            (V1_MENTIONS / f"juan_{juan:03d}.json").read_text(encoding="utf-8")
        )
        v1_geometries = {
            geometry_key(mention)
            for mention in v1_document.get("mentions", ())
            if mention.get("source", "main") == "main"
        }
        for key, row in rows.items():
            pid, start, end = key
            text = paragraph_text.get(pid)
            if text is None:
                raise ValueError(f"{path}: missing paragraph {juan}:{pid}")
            if text[start:end] != row["surface"]:
                raise ValueError(
                    f"{path}: stale surface at {juan}:{pid}:{start}:{end}; "
                    f"expected {row['surface']!r}, found {text[start:end]!r}"
                )
            if key not in v1_geometries:
                raise ValueError(
                    f"{path}: geometry is absent from production v1: "
                    f"{juan}:{pid}:{start}:{end}"
                )


def is_excluded(
    exclusions: dict[int, dict[tuple[int, int, int], dict]],
    juan: int,
    mention: dict,
) -> bool:
    return geometry_key(mention) in exclusions.get(juan, {})


def exclusion_summary(
    exclusions: dict[int, dict[tuple[int, int, int], dict]],
) -> dict:
    rows = [row for juan_rows in exclusions.values() for row in juan_rows.values()]
    reasons = collections.Counter(row["reason"] for row in rows)
    return {
        "count": len(rows),
        "by_reason": dict(sorted(reasons.items())),
    }
