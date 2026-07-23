"""Audit admitted and rejected combined-evidence candidates against production v1."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PERS = HERE.parent
REPO = PERS.parents[1]
TEXT = REPO / "web" / "public" / "text"
V1_MENTIONS = TEXT / "persons" / "mentions"
POS_DIR = TEXT / "persons" / "pos_giv"

sys.path[:0] = [str(PERS), str(HERE)]
import pos_giv  # noqa: E402
import rules as R  # noqa: E402
import translation_evidence as TE  # noqa: E402
import benchmark_reference as BR  # noqa: E402


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _signature(row: dict) -> tuple:
    nearest = row.get("nearest_policy") or {}
    return (
        tuple(row["evidence_families"]),
        tuple(row["evidence_signals"]),
        tuple(row["evidence_soft_conflicts"]),
        tuple(row["evidence_vetoes"]),
        nearest.get("policy"),
        tuple(nearest.get("missing_required", ())),
        tuple(nearest.get("missing_prerequisites", ())),
        tuple(nearest.get("unallowed_soft_conflicts", ())),
    )


def _write_signatures(
    path: Path,
    signatures: dict,
    *,
    no_veto_only: bool = False,
) -> None:
    rows = sorted(
        (
            row
            for row in signatures.values()
            if not no_veto_only or not row["vetoes"]
        ),
        key=lambda row: (
            -len(row["residual_ids"]),
            -row["candidate_count"],
            row["families"],
        ),
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "candidate_count",
                "residual_spans",
                "v1_overlap_candidates",
                "family_count",
                "support_family_count",
                "families",
                "support_families",
                "signals",
                "soft_conflicts",
                "vetoes",
                "nearest_policy",
                "missing_required",
                "missing_prerequisites",
                "unallowed_soft_conflicts",
                "example_surfaces",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: value
                for key, value in row.items()
                if key not in {"residual_ids", "surfaces"}
            } | {
                "residual_spans": len(row["residual_ids"]),
                "example_surfaces": " ".join(
                    surface
                    for surface, _ in row["surfaces"].most_common(12)
                ),
            })


def run(
    juans: list[int],
    output_dir: Path,
    translation_evidence_dir: Path | None = None,
) -> dict:
    corpus = R.load_corpus()
    selected_juans = sorted(set(juans))
    all_exclusions = BR.load_exclusions()
    exclusions = {
        juan: all_exclusions[juan]
        for juan in selected_juans
        if juan in all_exclusions
    }
    exclusion_summary = BR.exclusion_summary(exclusions)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidates.jsonl"
    funnel = collections.Counter()
    residual_total = 0
    residual_hits = collections.defaultdict(set)
    signatures = {}

    with candidate_path.open("w", encoding="utf-8", newline="\n") as candidate_file:
        for juan in selected_juans:
            document = json.loads(
                (TEXT / f"juan_{juan:03d}.json").read_text(encoding="utf-8")
            )
            paragraphs = document["paragraphs"]
            paragraph_text = {
                paragraph["id"]: paragraph.get("main", "")
                for paragraph in paragraphs
            }
            giv = pos_giv.giv_for_juan(juan, paragraphs, POS_DIR)
            translated = (
                TE.load_juan(translation_evidence_dir, juan, paragraphs)
                if translation_evidence_dir is not None
                else None
            )
            audit_rows = []
            cards = R.detect_juan(
                juan,
                paragraphs,
                giv,
                corpus,
                enabled=R.PRESET_RECALL,
                scan_notes=False,
                translation_evidence=translated,
                evidence_audit=audit_rows,
            )
            final_spans = collections.defaultdict(list)
            for card in cards:
                if card.get("field") == "main":
                    final_spans[card["para_id"]].append(
                        (card["start"], card["end"])
                    )

            v1_document = json.loads(
                (V1_MENTIONS / f"juan_{juan:03d}.json").read_text(
                    encoding="utf-8"
                )
            )
            v1_by_paragraph = collections.defaultdict(list)
            residual_by_paragraph = collections.defaultdict(list)
            for index, mention in enumerate(v1_document.get("mentions", ())):
                if mention.get("source", "main") != "main":
                    continue
                if BR.is_excluded(exclusions, juan, mention):
                    continue
                paragraph_id = mention["pid"]
                mention_span = (mention["start"], mention["end"])
                mention_id = (
                    f"{juan}:{paragraph_id}:{mention['start']}:"
                    f"{mention['end']}:{index}"
                )
                row = (mention_span, mention_id)
                v1_by_paragraph[paragraph_id].append(row)
                if not any(
                    _overlap(mention_span, span)
                    for span in final_spans.get(paragraph_id, ())
                ):
                    residual_by_paragraph[paragraph_id].append(row)
                    residual_total += 1

            for row in audit_rows:
                paragraph_id = row["para_id"]
                span = (row["start"], row["end"])
                residual_ids = [
                    mention_id
                    for mention_span, mention_id
                    in residual_by_paragraph.get(paragraph_id, ())
                    if _overlap(span, mention_span)
                ]
                exact_residual_ids = [
                    mention_id
                    for mention_span, mention_id
                    in residual_by_paragraph.get(paragraph_id, ())
                    if span == mention_span
                ]
                overlaps_v1 = any(
                    _overlap(span, mention_span)
                    for mention_span, _ in v1_by_paragraph.get(paragraph_id, ())
                )
                row["family_count"] = len(row["evidence_families"])
                row["support_family_count"] = len(
                    row["evidence_support_families"]
                )
                row["overlaps_v1"] = overlaps_v1
                row["residual_ids"] = residual_ids
                row["exact_residual_ids"] = exact_residual_ids
                text = paragraph_text.get(paragraph_id, "")
                row["context"] = text[
                    max(0, row["start"] - 18):min(len(text), row["end"] + 18)
                ]
                candidate_file.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )

                funnel["candidates"] += 1
                if row["admitted"]:
                    funnel["admitted"] += 1
                    continue
                funnel["rejected"] += 1
                if "model_ner_witness" in row["evidence_signals"]:
                    funnel["rejected_with_exact_model_witness"] += 1
                if row["evidence_vetoes"]:
                    funnel["rejected_with_veto"] += 1
                else:
                    funnel["rejected_without_veto"] += 1
                    for minimum in (2, 3, 4, 5):
                        if row["support_family_count"] >= minimum:
                            funnel[f"rejected_without_veto_{minimum}plus_families"] += 1
                            residual_hits[
                                f"no_veto_{minimum}plus_families"
                            ].update(residual_ids)

                key = _signature(row)
                signature = signatures.setdefault(key, {
                    "candidate_count": 0,
                    "v1_overlap_candidates": 0,
                    "family_count": row["family_count"],
                    "support_family_count": row["support_family_count"],
                    "families": "|".join(row["evidence_families"]),
                    "support_families": "|".join(
                        row["evidence_support_families"]
                    ),
                    "signals": "|".join(row["evidence_signals"]),
                    "soft_conflicts": "|".join(
                        row["evidence_soft_conflicts"]
                    ),
                    "vetoes": "|".join(row["evidence_vetoes"]),
                    "nearest_policy": (
                        row["nearest_policy"]["policy"]
                        if row["nearest_policy"] else ""
                    ),
                    "missing_required": "|".join(
                        (row["nearest_policy"] or {}).get(
                            "missing_required", ()
                        )
                    ),
                    "missing_prerequisites": "|".join(
                        (row["nearest_policy"] or {}).get(
                            "missing_prerequisites", ()
                        )
                    ),
                    "unallowed_soft_conflicts": "|".join(
                        (row["nearest_policy"] or {}).get(
                            "unallowed_soft_conflicts", ()
                        )
                    ),
                    "residual_ids": set(),
                    "surfaces": collections.Counter(),
                })
                signature["candidate_count"] += 1
                signature["v1_overlap_candidates"] += int(overlaps_v1)
                signature["residual_ids"].update(residual_ids)
                signature["surfaces"][row["surface"]] += 1

    summary = {
        "schema_version": 1,
        "rules_sha256": R.rules_bundle_sha256(),
        "reference_exclusions_sha256": BR.exclusions_sha256(),
        "reference_exclusions": exclusion_summary["count"],
        "translation_evidence": (
            str(translation_evidence_dir)
            if translation_evidence_dir is not None
            else None
        ),
        "juans": len(selected_juans),
        "v1_residual_spans": residual_total,
        "funnel": dict(sorted(funnel.items())),
        "residual_candidate_coverage": {
            name: len(ids)
            for name, ids in sorted(residual_hits.items())
        },
        "signature_count": len(signatures),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_signatures(output_dir / "signatures.csv", signatures)
    _write_signatures(
        output_dir / "signatures-no-veto.csv",
        signatures,
        no_veto_only=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--juans",
        nargs="*",
        type=int,
        default=list(range(1, 295)),
    )
    parser.add_argument("--translation-evidence-dir", type=Path)
    args = parser.parse_args()
    result = run(
        args.juans,
        args.output_dir,
        args.translation_evidence_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
