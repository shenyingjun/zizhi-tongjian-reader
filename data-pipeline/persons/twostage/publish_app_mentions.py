"""Publish safe Agent-1 geometry replacements to the experimental app dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TEXT = REPO / "web" / "public" / "text"
DEFAULT_OUTPUT = TEXT / "persons-v2" / "mentions"

SAFE_REPLACEMENT_RULES = frozenset({
    "jue_name",
    "posthumous_emperor_title",
})


def replace_contained_mentions(
    occurrences: list[dict],
    mentions: list[dict],
    paragraphs: list[dict],
) -> list[dict]:
    """Expand one existing bound mention to a rule-proven containing geometry."""
    text_by_pid = {paragraph["id"]: paragraph["main"] for paragraph in paragraphs}
    result = [dict(mention) for mention in mentions]

    for occurrence in occurrences:
        if (
            occurrence.get("field") != "main"
            or occurrence.get("rule") not in SAFE_REPLACEMENT_RULES
        ):
            continue
        pid = occurrence["para_id"]
        start, end = occurrence["start"], occurrence["end"]
        text = text_by_pid.get(pid)
        if text is None or text[start:end] != occurrence["surface"]:
            raise ValueError(
                f"occurrence text mismatch at paragraph {pid} [{start},{end})"
            )

        overlaps = [
            (index, mention)
            for index, mention in enumerate(result)
            if (
                mention.get("pid") == pid
                and mention.get("source") == "main"
                and mention["start"] < end
                and mention["end"] > start
            )
        ]
        if any(
            mention["start"] == start and mention["end"] == end
            for _, mention in overlaps
        ):
            continue
        contained = [
            (index, mention)
            for index, mention in overlaps
            if start <= mention["start"] and mention["end"] <= end
        ]
        if not contained:
            continue
        if len(overlaps) != 1 or len(contained) != 1:
            raise ValueError(
                f"unsafe overlapping mentions at paragraph {pid} [{start},{end})"
            )

        index, mention = contained[0]
        if text[mention["start"]:mention["end"]] != mention["surface"]:
            raise ValueError(
                f"published mention text mismatch at paragraph {pid} "
                f"[{mention['start']},{mention['end']})"
            )
        result[index] = {
            **mention,
            "start": start,
            "end": end,
            "surface": occurrence["surface"],
        }

    return result


def publish(
    occurrence_dir: Path,
    juans: list[int],
    output_dir: Path = DEFAULT_OUTPUT,
) -> list[dict]:
    manifest = json.loads(
        (occurrence_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("translation_evidence") is None:
        raise ValueError("publication requires approved translation-assisted output")
    if manifest.get("scope") != "numbered-jie":
        raise ValueError("publication requires numbered-jie Agent-1 output")

    summaries = []
    for juan in sorted(set(juans)):
        if juan not in manifest.get("juans", []):
            raise ValueError(f"juan {juan} is absent from the occurrence manifest")
        occurrence_doc = json.loads(
            (occurrence_dir / f"juan_{juan:03d}.json").read_text(encoding="utf-8")
        )
        app_path = output_dir / f"juan_{juan:03d}.json"
        trailing_newline = app_path.read_bytes().endswith(b"\n")
        app_doc = json.loads(app_path.read_text(encoding="utf-8"))
        text_doc = json.loads(
            (TEXT / f"juan_{juan:03d}.json").read_text(encoding="utf-8")
        )
        before = {
            (m["pid"], m["start"], m["end"], m["surface"])
            for m in app_doc["mentions"]
        }
        app_doc["mentions"] = replace_contained_mentions(
            occurrence_doc["occurrences"],
            app_doc["mentions"],
            text_doc["paragraphs"],
        )
        after = {
            (m["pid"], m["start"], m["end"], m["surface"])
            for m in app_doc["mentions"]
        }
        app_path.write_text(
            json.dumps(app_doc, ensure_ascii=False, indent=1)
            + ("\n" if trailing_newline else ""),
            encoding="utf-8",
        )
        summaries.append({
            "juan": juan,
            "additions": sorted(after - before),
            "removals": sorted(before - after),
            "net": len(after) - len(before),
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--occurrence-dir", type=Path, required=True)
    parser.add_argument("--juans", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for summary in publish(args.occurrence_dir, args.juans, args.output_dir):
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
