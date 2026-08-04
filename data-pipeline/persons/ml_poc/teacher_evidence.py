from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from pilot import TEXT, _load


TWOSTAGE = Path(__file__).resolve().parent.parent / "twostage"
sys.path.insert(0, str(TWOSTAGE))
import recover_translation_mapping as RTM  # noqa: E402


TRANSLATION_SCOPE = (
    TWOSTAGE / "translation" / "agent1_translation_scope.json"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _approved_source(scope: dict, juan: int) -> dict:
    rows = [
        row for row in scope.get("sources", [])
        if int(row["juan"]) == juan
    ]
    if len(rows) != 1:
        raise ValueError(f"juan {juan} must have one approved translation source")
    return rows[0]


def bounded_evidence(
    juan: int,
    jie_index: int,
    source: dict,
    translation_pairs,
    *,
    source_url: str,
    source_sha256: str,
    approved_pair_jies: dict[int, set[int]] | None = None,
) -> dict:
    jies = RTM._jies(source["paragraphs"])
    selected = next(
        (jie for jie in jies if jie.index == jie_index),
        None,
    )
    if selected is None:
        raise ValueError(f"juan {juan} has no jie index {jie_index}")
    para_ids = {int(row["id"]) for row in selected.paragraphs}
    notes = [
        {
            "para_id": int(paragraph["id"]),
            "note_index": note_index,
            "after": int(note["after"]),
            "text": str(note["text"]),
        }
        for paragraph in selected.paragraphs
        for note_index, note in enumerate(paragraph.get("notes", []))
    ]
    translations = []
    for pair in translation_pairs:
        if approved_pair_jies is not None:
            aligned_indexes = approved_pair_jies.get(int(pair.index), set())
        else:
            aligned_indexes = {
                aligned.index
                for aligned in RTM.aligned_jies(pair.original, jies)
            }
        if aligned_indexes != {jie_index}:
            continue
        translations.append({
            "pair_index": int(pair.index),
            "translation": str(pair.translation),
        })
    return {
        "schema_version": 1,
        "juan": juan,
        "jie_index": jie_index,
        "juan_context": [
            {
                "jie_index": int(jie.index),
                "jie_number": jie.number,
                "text": jie.text,
                "authorization": (
                    "target"
                    if jie.index == jie_index
                    else "context_only"
                ),
            }
            for jie in jies
        ],
        "main_text": selected.text,
        "paragraph_ids": sorted(para_ids),
        "hu_sansheng_notes": notes,
        "translations": translations,
        "scope_contract": {
            "main_text_output_only": True,
            "full_juan_context_visible": True,
            "other_jies_are_non_authorizing": True,
            "target_jie_authorization_only": True,
            "translation_unique_alignment_required": True,
            "identity_fields_present": False,
            "cross_jie_surface_reuse": False,
        },
        "source": {
            "url": source_url,
            "sha256": source_sha256,
            "translation_prose_persist": False,
        },
    }


def load_live_evidence(juan: int, jie_index: int) -> dict:
    scope = _load(TRANSLATION_SCOPE)
    if scope.get("identity_fields_present") is not False:
        raise ValueError("translation scope must explicitly exclude identity fields")
    approved = _approved_source(scope, juan)
    source_url, source_bytes = RTM.fetch_source(juan)
    actual_hash = _sha256(source_bytes)
    if source_url != approved["source_page"]:
        raise ValueError("translation source URL differs from approved mapping")
    if actual_hash != approved["source_sha256"]:
        raise ValueError("translation source hash differs from approved mapping")
    translation_page = approved.get("translation_page")
    if translation_page is None:
        pairs = RTM.parse_source(source_bytes)
    else:
        translation_bytes = RTM._fetch(str(translation_page))
        translation_hash = _sha256(translation_bytes)
        if translation_hash != approved.get("translation_page_sha256"):
            raise ValueError(
                "separate translation page hash differs from approved mapping"
            )
        pairs = RTM._whole_page_pair(source_bytes, translation_bytes)
    source = _load(TEXT / f"juan_{juan:03d}.json")
    approved_pair_jies = {
        int(pair_index): {int(value) for value in jie_indexes}
        for pair_index, jie_indexes in scope.get(
            "pair_jies", {}
        ).get(str(juan), {}).items()
    }
    evidence = bounded_evidence(
        juan,
        jie_index,
        source,
        pairs,
        source_url=source_url,
        source_sha256=actual_hash,
        approved_pair_jies=approved_pair_jies,
    )
    if translation_page is not None:
        evidence["source"]["translation_url"] = str(translation_page)
        evidence["source"]["translation_sha256"] = translation_hash
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stream current-jie note and translation prose to a Copilot teacher. "
            "The evidence is printed only and is not persisted."
        )
    )
    parser.add_argument("--juan", type=int, required=True)
    parser.add_argument("--jie-index", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(
        load_live_evidence(args.juan, args.jie_index),
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
