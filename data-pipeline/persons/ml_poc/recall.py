from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core import sanitize_note_mentions
from pilot import REPO, RULES, TEXT, V1, _load


TWOSTAGE = Path(__file__).resolve().parent.parent / "twostage"
sys.path.insert(0, str(TWOSTAGE))
import translation_evidence as TE  # noqa: E402


TRANSLATION = TWOSTAGE / "translation" / "evidence"


def _add_candidate(
    grouped: dict[tuple[int, int, int], dict],
    *,
    para_id: int,
    start: int,
    end: int,
    surface: str,
    channel: str,
) -> None:
    key = (para_id, start, end)
    row = grouped.setdefault(
        key,
        {
            "id": f"{para_id}:{start}:{end}",
            "para_id": para_id,
            "start": start,
            "end": end,
            "surface": surface,
            "channels": [],
        },
    )
    if row["surface"] != surface:
        raise ValueError(f"candidate surface conflict at {key}")
    if channel not in row["channels"]:
        row["channels"].append(channel)


def build_recall_pack(juan: int) -> dict:
    source = _load(TEXT / f"juan_{juan:03d}.json")
    paragraphs = source["paragraphs"]
    text_by_pid = {
        int(paragraph["id"]): str(paragraph.get("main", "") or "")
        for paragraph in paragraphs
    }
    v1 = _load(V1 / f"juan_{juan:03d}.json")
    rules = _load(RULES / f"juan_{juan:03d}.json")
    grouped: dict[tuple[int, int, int], dict] = {}

    for row in v1.get("mentions", []):
        if row.get("source", "main") != "main":
            continue
        _add_candidate(
            grouped,
            para_id=int(row["pid"]),
            start=int(row["start"]),
            end=int(row["end"]),
            surface=str(row["surface"]),
            channel="v1",
        )

    for row in rules.get("occurrences", []):
        if row.get("field", "main") != "main":
            continue
        _add_candidate(
            grouped,
            para_id=int(row["para_id"]),
            start=int(row["start"]),
            end=int(row["end"]),
            surface=str(row["surface"]),
            channel="rules",
        )

    translated = TE.load_juan(TRANSLATION, juan, paragraphs)
    for para_id, identities in translated.items():
        for identity in identities:
            for row in identity.get("candidates", []):
                if not row.get("eligible", False):
                    continue
                _add_candidate(
                    grouped,
                    para_id=para_id,
                    start=int(row["start"]),
                    end=int(row["end"]),
                    surface=str(row["surface"]),
                    channel="translation",
                )

    note_evidence = sanitize_note_mentions(
        paragraphs, v1.get("mentions", [])
    )
    for row in grouped.values():
        text = text_by_pid[row["para_id"]]
        if text[row["start"]:row["end"]] != row["surface"]:
            raise ValueError(f"candidate does not match main text: {row}")
        row["channels"].sort()

    return {
        "schema_version": 1,
        "phase": "recall",
        "juan": juan,
        "candidates": sorted(
            grouped.values(),
            key=lambda row: (row["para_id"], row["start"], row["end"]),
        ),
        "note_evidence": note_evidence,
        "provenance_contract": {
            "translation_scope": "original paragraph only",
            "note_scope": "original jie only; anchor evidence, not automatic span",
            "identity_fields_present": False,
        },
    }


def prepare(blind_dir: Path, output_dir: Path) -> list[Path]:
    manifest = _load(blind_dir / "manifest.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for selection in manifest["selected"]:
        juan = int(selection["juan"])
        output = output_dir / f"recall_juan_{juan:03d}.json"
        output.write_text(
            json.dumps(
                build_recall_pack(juan), ensure_ascii=False, indent=2
            ) + "\n",
            encoding="utf-8",
        )
        written.append(output)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build phase-separated recall packs for selected pilot juans."
    )
    parser.add_argument("--blind-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in prepare(args.blind_dir, args.output):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
