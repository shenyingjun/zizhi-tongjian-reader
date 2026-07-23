"""Retag the corpus with the current Agent-1 rules.

The output contains identity-free occurrence cards. It does not overwrite the shipped
production-v1 mentions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PERS = HERE.parent
REPO = PERS.parents[1]
TEXT = REPO / "web" / "public" / "text"
POS_DIR = TEXT / "persons" / "pos_giv"
ADMIN_PLACES = HERE / "admin-places.json"

sys.path[:0] = [str(PERS), str(HERE)]
import build_admin_places  # noqa: E402
import pos_giv  # noqa: E402
import rules as R  # noqa: E402
import translation_evidence as TE  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_admin_places() -> None:
    result = build_admin_places.build()
    ADMIN_PLACES.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(
    juans: list[int],
    output_dir: Path,
    rebuild_admin_places: bool = True,
    translation_evidence_dir: Path | None = None,
) -> dict:
    if not juans or any(juan < 1 or juan > 294 for juan in juans):
        raise ValueError("juans must contain values from 1 through 294")
    if rebuild_admin_places:
        _write_admin_places()

    corpus = R.load_corpus()
    output_dir.mkdir(parents=True, exist_ok=True)
    rules_hash = R.rules_bundle_sha256()
    admin_hash = _sha256(ADMIN_PLACES)
    translation_manifest = (
        translation_evidence_dir / "manifest.json"
        if translation_evidence_dir is not None
        else None
    )
    translation_manifest_hash = (
        _sha256(translation_manifest)
        if translation_manifest is not None
        else None
    )
    counts = {}

    for juan in sorted(set(juans)):
        text_path = TEXT / f"juan_{juan:03d}.json"
        document = json.loads(text_path.read_text(encoding="utf-8"))
        paragraphs = document["paragraphs"]
        evidence = pos_giv.giv_for_juan(juan, paragraphs, POS_DIR)
        translated = (
            TE.load_juan(translation_evidence_dir, juan, paragraphs)
            if translation_evidence_dir is not None
            else None
        )
        cards = R.detect_juan(
            juan,
            paragraphs,
            evidence,
            corpus,
            enabled=R.PRESET_RECALL,
            scan_notes=False,
            translation_evidence=translated,
        )
        occurrences = [
            {
                key: card[key]
                for key in (
                    "juan",
                    "para_id",
                    "start",
                    "end",
                    "surface",
                    "chunk_type",
                    "rule",
                    "scope",
                    "ce_year",
                    "field",
                )
            } | {
                key: card[key]
                for key in (
                    "evidence_policy",
                    "evidence_families",
                    "evidence_signals",
                    "evidence_witnesses",
                    "evidence_missing",
                    "evidence_soft_conflicts",
                )
                if key in card
            }
            for card in cards
            if card.get("field") == "main"
        ]
        occurrences.sort(
            key=lambda card: (
                card["para_id"],
                card["start"],
                card["end"],
                card["surface"],
                card["rule"],
            )
        )
        payload = {
            "schema_version": 1,
            "juan": juan,
            "rules_sha256": rules_hash,
            "admin_places_sha256": admin_hash,
            "translation_evidence_manifest_sha256": translation_manifest_hash,
            "occurrences": occurrences,
        }
        (output_dir / f"juan_{juan:03d}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        counts[str(juan)] = len(occurrences)

    manifest = {
        "schema_version": 1,
        "preset": "PRESET_RECALL",
        "scope": "numbered-jie",
        "source": "main text only",
        "rules_sha256": rules_hash,
        "admin_places_sha256": admin_hash,
        "translation_evidence": (
            {
                "directory": str(translation_evidence_dir),
                "manifest_sha256": translation_manifest_hash,
            }
            if translation_evidence_dir is not None
            else None
        ),
        "juans": sorted(set(juans)),
        "occurrences_by_juan": counts,
        "total_occurrences": sum(counts.values()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="write identity-free Agent-1 occurrence cards here",
    )
    parser.add_argument(
        "--juans",
        nargs="*",
        type=int,
        default=list(range(1, 295)),
        help="optional subset; defaults to all 294",
    )
    parser.add_argument(
        "--skip-admin-rebuild",
        action="store_true",
        help="reuse the existing admin-places.json instead of rebuilding it",
    )
    parser.add_argument(
        "--translation-evidence-dir",
        type=Path,
        help=(
            "optional per-juan translation identity evidence; omitted by default "
            "so production behavior is unchanged"
        ),
    )
    args = parser.parse_args()
    result = run(
        args.juans,
        args.output_dir,
        rebuild_admin_places=not args.skip_admin_rebuild,
        translation_evidence_dir=args.translation_evidence_dir,
    )
    print(
        f"wrote {result['total_occurrences']} occurrences for "
        f"{len(result['juans'])} juans to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
