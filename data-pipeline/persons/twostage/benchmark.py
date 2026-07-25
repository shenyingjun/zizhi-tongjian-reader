"""Benchmark the current Agent-1 rules against shipped production-v1 spans.

This is a compatibility benchmark, not an independent accuracy benchmark: v1 contains
both true mentions and known false positives / cross-jie bindings. Metrics therefore use
the terms `v1_coverage` and `v1_overlap_proxy`, never recall/precision without a qualifier.

Run from the repository root:

    data-pipeline\.venv-ner\Scripts\python.exe -X utf8 \
      data-pipeline\persons\twostage\benchmark.py \
      --json data-pipeline\persons\twostage\benchmark-latest.json
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import sys
import time
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


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def run(
    juans: list[int],
    translation_evidence_dir: Path | None = None,
    workers: int = 1,
) -> dict:
    started = time.perf_counter()
    if workers and workers > 1:
        R.DETECT_WORKERS = workers
    corpus = R.load_corpus()
    all_exclusions = BR.load_exclusions()
    exclusions = {
        juan: all_exclusions[juan]
        for juan in juans
        if juan in all_exclusions
    }
    exclusion_summary = BR.exclusion_summary(exclusions)
    reference_by_kind = collections.defaultdict(lambda: [0, 0])
    raw_reference_by_kind = collections.defaultdict(lambda: [0, 0])
    tagged_by_chunk = collections.defaultdict(lambda: [0, 0])
    raw_v1_total = raw_v1_covered = 0
    v1_total = v1_covered = agent_total = agent_overlapped = 0
    agent_overlapped_raw = 0

    for juan in juans:
        text_path = TEXT / f"juan_{juan:03d}.json"
        v1_path = V1_MENTIONS / f"juan_{juan:03d}.json"
        for required in (text_path, v1_path):
            if not required.is_file():
                raise FileNotFoundError(f"required benchmark input is missing: {required}")
        paras = json.loads(text_path.read_text(encoding="utf-8"))["paragraphs"]
        giv = pos_giv.giv_for_juan(juan, paras, POS_DIR)
        translated = (
            TE.load_juan(translation_evidence_dir, juan, paras)
            if translation_evidence_dir is not None
            else None
        )
        cards = R.detect_juan(
            juan,
            paras,
            giv,
            corpus,
            enabled=R.PRESET_RECALL,
            scan_notes=False,
            translation_evidence=translated,
        )
        agent = collections.defaultdict(list)
        for card in cards:
            if card.get("field") != "main":
                continue
            agent[card["para_id"]].append(card)

        v1_doc = json.loads(v1_path.read_text(encoding="utf-8"))
        v1 = collections.defaultdict(list)
        raw_v1 = collections.defaultdict(list)
        for mention in v1_doc.get("mentions", []):
            if mention.get("source", "main") == "main":
                raw_v1[mention["pid"]].append(mention)
                if not BR.is_excluded(exclusions, juan, mention):
                    v1[mention["pid"]].append(mention)

        for pid, mentions in raw_v1.items():
            spans = [(c["start"], c["end"]) for c in agent.get(pid, ())]
            for mention in mentions:
                kind = mention.get("kind", "?")
                raw_reference_by_kind[kind][0] += 1
                raw_v1_total += 1
                span = (mention["start"], mention["end"])
                if any(_overlap(span, candidate) for candidate in spans):
                    raw_reference_by_kind[kind][1] += 1
                    raw_v1_covered += 1

        for pid, mentions in v1.items():
            spans = [(c["start"], c["end"]) for c in agent.get(pid, ())]
            for mention in mentions:
                kind = mention.get("kind", "?")
                reference_by_kind[kind][0] += 1
                v1_total += 1
                span = (mention["start"], mention["end"])
                if any(_overlap(span, candidate) for candidate in spans):
                    reference_by_kind[kind][1] += 1
                    v1_covered += 1

        for pid, cards_for_para in agent.items():
            spans = [(m["start"], m["end"]) for m in v1.get(pid, ())]
            raw_spans = [
                (m["start"], m["end"]) for m in raw_v1.get(pid, ())
            ]
            for card in cards_for_para:
                chunk = card["chunk_type"]
                tagged_by_chunk[chunk][0] += 1
                agent_total += 1
                span = (card["start"], card["end"])
                if any(_overlap(span, reference) for reference in spans):
                    tagged_by_chunk[chunk][1] += 1
                    agent_overlapped += 1
                if any(_overlap(span, reference) for reference in raw_spans):
                    agent_overlapped_raw += 1

    admin_places_path = HERE / "admin-places.json"
    return {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "python": sys.version.split()[0],
        "rules_sha256": R.rules_bundle_sha256(),
        "admin_places_sha256": hashlib.sha256(
            admin_places_path.read_bytes()
        ).hexdigest(),
        "preset": "PRESET_RECALL",
        "scope": "numbered-jie",
        "reference": (
            "audited production v1 main-source geometries; exact reviewed "
            "non-person/bad-partial exclusions applied"
        ),
        "reference_exclusions": {
            "path": str(BR.EXCLUSIONS_PATH.relative_to(REPO)),
            "sha256": BR.exclusions_sha256(),
            **exclusion_summary,
        },
        "matching": "same juan + paragraph id + overlapping [start,end)",
        "translation_evidence": (
            str(translation_evidence_dir)
            if translation_evidence_dir is not None
            else None
        ),
        "juans": len(juans),
        "summary": {
            "v1_spans_raw": raw_v1_total,
            "v1_covered_raw": raw_v1_covered,
            "v1_missed_raw": raw_v1_total - raw_v1_covered,
            "v1_coverage_raw_pct": round(
                _pct(raw_v1_covered, raw_v1_total), 3
            ),
            "v1_excluded": exclusion_summary["count"],
            "v1_spans": v1_total,
            "v1_covered": v1_covered,
            "v1_missed": v1_total - v1_covered,
            "v1_coverage_pct": round(_pct(v1_covered, v1_total), 3),
            "agent1_spans": agent_total,
            "agent1_overlapping_v1_raw": agent_overlapped_raw,
            "agent1_overlapping_v1": agent_overlapped,
            "agent1_nonoverlapping_v1": agent_total - agent_overlapped,
            "v1_overlap_proxy_pct": round(_pct(agent_overlapped, agent_total), 3),
        },
        "v1_coverage_by_kind": {
            kind: {
                "total": total,
                "covered": hit,
                "missed": total - hit,
                "coverage_pct": round(_pct(hit, total), 3),
            }
            for kind, (total, hit) in sorted(reference_by_kind.items())
        },
        "v1_raw_coverage_by_kind": {
            kind: {
                "total": total,
                "covered": hit,
                "missed": total - hit,
                "coverage_pct": round(_pct(hit, total), 3),
            }
            for kind, (total, hit) in sorted(raw_reference_by_kind.items())
        },
        "v1_overlap_by_chunk_type": {
            chunk: {
                "agent1_spans": total,
                "overlapping_v1": hit,
                "nonoverlapping_v1": total - hit,
                "overlap_proxy_pct": round(_pct(hit, total), 3),
            }
            for chunk, (total, hit) in sorted(
                tagged_by_chunk.items(), key=lambda item: (-item[1][0], item[0])
            )
        },
    }


def print_report(result: dict) -> None:
    summary = result["summary"]
    print("Agent 1 vs audited production v1 (compatibility benchmark)")
    print(f"juans={result['juans']}  runtime={result['runtime_seconds']}s")
    print(
        f"v1 coverage: {summary['v1_covered']}/{summary['v1_spans']} "
        f"= {summary['v1_coverage_pct']:.3f}%  missed={summary['v1_missed']}"
    )
    print(
        f"raw v1: {summary['v1_covered_raw']}/{summary['v1_spans_raw']}  "
        f"excluded={summary['v1_excluded']}"
    )
    print(
        f"Agent1 spans: {summary['agent1_spans']}  "
        f"overlap-v1={summary['agent1_overlapping_v1']}  "
        f"nonoverlap-v1={summary['agent1_nonoverlapping_v1']}  "
        f"overlap proxy={summary['v1_overlap_proxy_pct']:.3f}%"
    )
    print("\nV1 coverage by kind")
    for kind, row in result["v1_coverage_by_kind"].items():
        print(
            f"  {kind:10s} {row['covered']:7d}/{row['total']:7d} "
            f"{row['coverage_pct']:7.3f}%  missed={row['missed']}"
        )
    print("\nAgent1 overlap by chunk type")
    for chunk, row in result["v1_overlap_by_chunk_type"].items():
        print(
            f"  {chunk:12s} {row['overlapping_v1']:7d}/{row['agent1_spans']:7d} "
            f"{row['overlap_proxy_pct']:7.3f}%  "
            f"nonoverlap={row['nonoverlapping_v1']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="write machine-readable result")
    parser.add_argument(
        "--juans",
        nargs="*",
        type=int,
        default=list(range(1, 295)),
        help="optional subset; defaults to all 294",
    )
    parser.add_argument(
        "--translation-evidence-dir",
        type=Path,
        help="optional paragraph-scoped translation identity evidence",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "split each juan's jie-blocks across this many worker processes "
            "(default 1 = serial). Metrics are identical regardless of value; "
            "4 is a good default on multi-core machines."
        ),
    )
    args = parser.parse_args()
    result = run(args.juans, args.translation_evidence_dir, workers=args.workers)
    print_report(result)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
