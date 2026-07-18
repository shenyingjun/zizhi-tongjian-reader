"""Build time-scoped administrative-origin evidence for Agent 1.

The generated lexicon has two independent evidence layers:

1. Complete high-confidence POS Geo entities, retained for corpus audit.
2. Administrative-origin surfaces that precede independently known, POS-proven
   full names for at least two distinct people in the same chronicle period.

Only main text is mined. Commentary frequently discusses administrative history
from several eras and therefore cannot inherit the event paragraph's year.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PERS = HERE.parent
REPO = PERS.parents[1]
TEXT = REPO / "web" / "public" / "text"
POS_DIR = TEXT / "persons" / "pos_giv"
PEOPLE = TEXT / "persons" / "people.json"
DEFAULT_OUTPUT = HERE / "admin-places.json"

sys.path.insert(0, str(PERS))
import pos_giv  # noqa: E402


GEO_SCORE = 0.9
SURNAME_SCORE = 0.9
GIVEN_SCORE = 0.7
ADMIN_TAILS = set("郡州县國国邑")
MIN_DISTINCT_PEOPLE = 2
MAX_ADMIN_SURFACE = 4


def _known_names() -> set[str]:
    data = json.loads(PEOPLE.read_text(encoding="utf-8"))
    people = data["people"] if isinstance(data, dict) else data
    names = set()
    for person in people:
        candidates = [person.get("canonical_name", "")]
        candidates.extend(
            name.get("text", "") if isinstance(name, dict) else name
            for name in person.get("names", [])
        )
        names.update(name for name in candidates if 2 <= len(name) <= 4)
    return names


def _complete_tokens(evidence, start: int, end: int):
    tokens = []
    cursor = start
    while cursor < end:
        token = evidence.token_at(cursor)
        if token is None or token.start != cursor or token.end > end:
            return ()
        tokens.append(token)
        cursor = token.end
    return tuple(tokens) if cursor == end else ()


def _complete_geo_entities(evidence):
    tokens = evidence.tokens
    index = 0
    while index < len(tokens):
        token = tokens[index]
        is_geo = "Case=Loc" in token.tag and "NameType=Geo" in token.tag
        if not is_geo or token.score is None or token.score < GEO_SCORE:
            index += 1
            continue
        group = [token]
        if token.bio == "B":
            cursor = index + 1
            while cursor < len(tokens):
                following = tokens[cursor]
                if (
                    following.bio != "I"
                    or following.start != group[-1].end
                    or "Case=Loc" not in following.tag
                    or "NameType=Geo" not in following.tag
                    or following.score is None
                    or following.score < GEO_SCORE
                ):
                    break
                group.append(following)
                cursor += 1
            index = cursor
        else:
            index += 1
        yield "".join(part.text for part in group), min(part.score for part in group)


def _admin_candidates(text: str, name_start: int, evidence):
    for length in range(2, min(MAX_ADMIN_SURFACE, name_start) + 1):
        start = name_start - length
        surface = text[start:name_start]
        if not surface or surface[-1] not in ADMIN_TAILS:
            continue
        tokens = _complete_tokens(evidence, start, name_start)
        if not tokens:
            continue
        # Rescue model-confused proper names such as 陈国 (Sur+Giv). Independent
        # occurrences often appear as 陈=Nat/Loc + 国=NOUN, which is also useful
        # evidence. Ordinary office nouns such as 柱国 match neither shape.
        confused_proper_name = all(
            token.pos == "PROPN" and "NameType=" in token.tag
            for token in tokens
        )
        locality_plus_admin_tail = (
            tokens[-1].text in ADMIN_TAILS
            and tokens[-1].pos == "NOUN"
            and all(
                token.pos == "PROPN"
                and (
                    "Case=Loc" in token.tag
                    or "NameType=Geo" in token.tag
                    or "NameType=Nat" in token.tag
                )
                for token in tokens[:-1]
            )
        )
        if not (confused_proper_name or locality_plus_admin_tail):
            continue
        yield surface
        return


def build() -> dict:
    known_names = _known_names()
    geo = collections.defaultdict(
        lambda: {
            "occurrences": 0,
            "juans": set(),
            "years": set(),
            "dynasties": set(),
            "min_score": 1.0,
            "max_score": 0.0,
        }
    )
    origins = collections.defaultdict(
        lambda: {
            "juans": set(),
            "years": set(),
            "fullnames": set(),
            "occurrences": 0,
            "examples": [],
        }
    )

    text_paths = sorted(TEXT.glob("juan_*.json"))
    for text_path in text_paths:
        document = json.loads(text_path.read_text(encoding="utf-8"))
        juan = int(document["juan_no"])
        dynasty = document.get("dynasty") or ""
        paragraphs = document["paragraphs"]
        evidence_by_pid = pos_giv.giv_for_juan(juan, paragraphs, POS_DIR)
        for paragraph in paragraphs:
            pid = paragraph["id"]
            year = paragraph.get("ce_year")
            text = paragraph.get("main", "") or ""
            evidence = evidence_by_pid.get(pid)
            if evidence is None:
                continue

            for surface, score in _complete_geo_entities(evidence):
                row = geo[surface]
                row["occurrences"] += 1
                row["juans"].add(juan)
                if year is not None:
                    row["years"].add(int(year))
                if dynasty:
                    row["dynasties"].add(dynasty)
                row["min_score"] = min(row["min_score"], score)
                row["max_score"] = max(row["max_score"], score)

            for surname in evidence.tokens:
                if (
                    surname.tag != "PROPN|NameType=Sur"
                    or surname.score is None
                    or surname.score < SURNAME_SCORE
                    or surname.end - surname.start != 1
                ):
                    continue
                given_end = next(
                    (
                        end
                        for start, end in evidence.spans
                        if start == surname.end and 1 <= end - start <= 2
                    ),
                    None,
                )
                if given_end is None:
                    continue
                given_tokens = _complete_tokens(evidence, surname.end, given_end)
                if not given_tokens or any(
                    token.score is None or token.score < GIVEN_SCORE
                    for token in given_tokens
                ):
                    continue
                fullname = text[surname.start:given_end]
                if fullname not in known_names:
                    continue
                for admin_surface in _admin_candidates(text, surname.start, evidence):
                    key = (admin_surface, dynasty)
                    row = origins[key]
                    row["juans"].add(juan)
                    if year is not None:
                        row["years"].add(int(year))
                    row["fullnames"].add(fullname)
                    row["occurrences"] += 1
                    if len(row["examples"]) < 12:
                        row["examples"].append({
                            "juan": juan,
                            "pid": pid,
                            "ce_year": year,
                            "fullname": fullname,
                            "context": text[max(0, surname.start - 8):given_end + 8],
                        })

    geo_rows = []
    for surface, row in geo.items():
        geo_rows.append({
            "surface": surface,
            "occurrences": row["occurrences"],
            "juans": sorted(row["juans"]),
            "years": sorted(row["years"]),
            "dynasties": sorted(row["dynasties"]),
            "min_score": round(row["min_score"], 6),
            "max_score": round(row["max_score"], 6),
        })
    geo_rows.sort(key=lambda row: row["surface"])

    origin_rows = []
    for (surface, dynasty), row in origins.items():
        if len(row["fullnames"]) < MIN_DISTINCT_PEOPLE or not row["years"]:
            continue
        origin_rows.append({
            "surface": surface,
            "dynasty": dynasty,
            "attested_years": sorted(row["years"]),
            "juans": sorted(row["juans"]),
            "distinct_fullnames": sorted(row["fullnames"]),
            "evidence_count": row["occurrences"],
            "examples": row["examples"],
        })
    origin_rows.sort(key=lambda row: (row["surface"], row["dynasty"]))

    return {
        "version": 1,
        "scope": "main text only",
        "temporal_policy": (
            "Fallback membership is valid only in an exactly attested CE year; "
            "dynasty is retained as provenance and no continuous lifetime is inferred."
        ),
        "thresholds": {
            "geo_score": GEO_SCORE,
            "surname_score": SURNAME_SCORE,
            "given_score": GIVEN_SCORE,
            "minimum_distinct_fullnames_per_dynasty": MIN_DISTINCT_PEOPLE,
        },
        "counts": {
            "high_confidence_geo_entities": len(geo_rows),
            "fallback_admin_periods": len(origin_rows),
            "fallback_admin_surfaces": len({
                row["surface"] for row in origin_rows
            }),
        },
        "high_confidence_geo_entities": geo_rows,
        "fallback_admin_origins": origin_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["counts"], ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
