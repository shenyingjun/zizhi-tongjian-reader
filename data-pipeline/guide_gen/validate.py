"""Validate authored 白话导读 guide files against the source corpus.

Automatic gates (mirrors the plan's stage 5):
  1. schema        — version:1, summaries[], required fields, types
  2. anchoring     — anchor_pid is a real year paragraph_id; source_range in-bounds
  3. simplified    — no traditional-Chinese codepoints (OpenCC if available; else
                     a curated traditional-only blocklist as a fallback)
  4. grounding     — every key_people[].name occurs literally in its source span
  5. non-原文      — longest classical n-gram shared with the source span < N
                     (rejects verbatim 文言 paste)
  6. anti-spoiler  — no future CE years referenced; lexical "later/afterwards"
                     blocklist flagged
  7. shape         — one_liner length, required fields per content tier

Usage:
    python validate.py 41 67 179
    python validate.py --all
Exit code is non-zero if any ERROR-level gate fails. WARN items are advisory.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                 # data-pipeline/
GUIDE_SRC = ROOT / "guide"
TEXT_DIR = ROOT.parent / "web" / "public" / "text"
YEARS_JSONL = ROOT / "guide_gen" / "out" / "years.jsonl"

ONE_LINER_MAX = 60
TITLE_MAX = 16
NGRAM_MAX = 8  # longest allowed verbatim classical run shared with source

SPOILER_LEXEMES = ["后来", "此后", "日后", "其后", "不久之后", "将来", "后世"]
IMPORTANT_SALIENCE_REASONS = {
    "state_ending",
    "title_change",
    "regicide_deposition_succession",
    "major_battle_territory",
    "diplomacy_alliance",
}

try:  # OpenCC is the authoritative simplified check; fall back if absent.
    from opencc import OpenCC  # type: ignore

    _CC = OpenCC("t2s")

    def to_simplified(s: str) -> str:
        return _CC.convert(s)

    _HAS_OPENCC = True
except Exception:  # pragma: no cover - fallback path
    _CC = None
    _HAS_OPENCC = False
    # Small blocklist of common traditional-only forms as a backstop.
    _TRAD_ONLY = set("讓國說劉漢魏隋將軍劉孫權關張趙馬黃龐諸葛說書聲東邊樂業歲")

    def to_simplified(s: str) -> str:  # identity; real check done via blocklist
        return s


def load_juan(no: int) -> dict:
    with (TEXT_DIR / f"juan_{no:03d}.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def year_anchor_ids(juan: dict) -> set[int]:
    return {y["paragraph_id"] for y in juan["years"]}


def span_text(juan: dict, start_pid: int, end_pid: int) -> str:
    parts = []
    for p in juan["paragraphs"]:
        if start_pid <= p["id"] <= end_pid:
            if p.get("main"):
                parts.append(p["main"])
            for n in p.get("notes", []):
                if n.get("text"):
                    parts.append(n["text"])
    return "\n".join(parts)


def max_shared_ngram(a: str, b: str) -> int:
    """Longest common contiguous CJK run between summary `a` and source `b`.

    Cheap suffix-style scan over a (summaries are short). Punctuation/space are
    stripped so we only measure shared *content* runs.
    """
    clean = lambda s: re.sub(r"[\s，。、；：「」『』（）()\u3000·]", "", s)
    a, b = clean(a), clean(b)
    if not a:
        return 0
    best = 0
    for i in range(len(a)):
        j = i + best + 1
        while j <= len(a) and a[i:j] in b:
            best = j - i
            j += 1
    return best


def check_summary(s: dict, juan: dict, anchors: set[int], errors, warns) -> None:
    sid = s.get("id", "<no-id>")

    def err(msg):
        errors.append(f"[{sid}] {msg}")

    def warn(msg):
        warns.append(f"[{sid}] {msg}")

    # 1/2. schema + anchoring
    for field in ("id", "juan_no", "anchor_pid", "one_liner"):
        if field not in s:
            err(f"missing required field '{field}'")
    anchor = s.get("anchor_pid")
    if isinstance(anchor, int):
        if anchor not in anchors:
            err(f"anchor_pid {anchor} is not a year paragraph_id")
    elif anchor is not None:
        err(f"anchor_pid must be int, got {type(anchor).__name__}")

    rng = s.get("source_range")
    span = ""
    if rng:
        sp, ep = rng.get("start_pid"), rng.get("end_pid", rng.get("start_pid"))
        if not isinstance(sp, int):
            err("source_range.start_pid must be int")
        else:
            span = span_text(juan, sp, ep if isinstance(ep, int) else sp)
            if not span:
                warn(f"source_range {sp}-{ep} matched no paragraphs")

    # collect all authored prose for simplified + non-原文 checks
    prose_fields = ["title", "one_liner", "what_happened", "why_it_matters", "background"]
    prose = " ".join(str(s.get(f, "")) for f in prose_fields)
    for kp in s.get("key_people", []) or []:
        prose += " " + str(kp.get("role", ""))

    # 3. simplified
    if _HAS_OPENCC:
        if to_simplified(prose) != prose:
            diff = [c for c in prose if to_simplified(c) != c]
            err(f"contains traditional-Chinese chars: {''.join(sorted(set(diff)))}")
    else:
        bad = [c for c in prose if c in _TRAD_ONLY]
        if bad:
            err(f"contains traditional-Chinese chars (fallback list): {''.join(sorted(set(bad)))}")

    # 4. grounding — key people must appear in the source span
    if span:
        for kp in s.get("key_people", []) or []:
            name = kp.get("name", "")
            if name and name not in span:
                err(f"key person '{name}' not found in source span (possible hallucination)")

    # 5. non-原文 — reject long verbatim classical runs
    if span:
        for f in ("one_liner", "what_happened", "why_it_matters"):
            v = s.get(f)
            if v:
                run = max_shared_ngram(v, span)
                if run >= NGRAM_MAX:
                    err(f"field '{f}' shares a {run}-char verbatim run with source (非原文 violation)")

    # 6. anti-spoiler
    this_year = s.get("ce_year")
    if isinstance(this_year, int):
        for m in re.finditer(r"(?:公元前\s*(\d+)|前\s*(\d+)|(\d{2,4})\s*年)", prose):
            g = m.group(1) or m.group(2) or m.group(3)
            if not g:
                continue
            val = int(g)
            year = -val if (m.group(1) or m.group(2)) else val
            # Allow referencing the current year; flag clearly-future ones.
            if year > this_year and abs(year - this_year) <= 1200:
                warn(f"references year {year} (> current {this_year}) — check for spoiler")
    for lex in SPOILER_LEXEMES:
        if lex in prose:
            warn(f"contains forward-looking lexeme '{lex}' — verify no spoiler")

    # 7. shape
    ol = s.get("one_liner", "")
    if len(ol) > ONE_LINER_MAX:
        warn(f"one_liner is {len(ol)} chars (> {ONE_LINER_MAX})")
    if len(s.get("title", "")) > TITLE_MAX:
        warn(f"title is {len(s['title'])} chars (> {TITLE_MAX})")


def coverage_check(no: int, summaries: list[dict], warns, coverage_summaries) -> None:
    if not YEARS_JSONL.exists():
        warns.append(f"[coverage j{no:03d}] {YEARS_JSONL} not found; skipped coverage check")
        coverage_summaries.append(f"coverage j{no:03d}: skipped (years.jsonl missing)")
        return

    authored_anchors = {s.get("anchor_pid") for s in summaries if isinstance(s.get("anchor_pid"), int)}
    commentary_gaps = 0
    dense_gaps = 0
    salience_gap_anchors: set[int] = set()

    with YEARS_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("juan_no") != no:
                continue
            anchor = rec.get("anchor_pid")
            if anchor in authored_anchors:
                continue
            year = rec.get("ce_year")
            prefix = f"[coverage j{no:03d} {year}]"

            if rec.get("has_commentary") is True:
                commentary_gaps += 1
                warns.append(f"{prefix} has_commentary=true but no summary.")
            if isinstance(rec.get("event_paras"), int) and rec["event_paras"] >= 4:
                dense_gaps += 1
                warns.append(f"{prefix} event_paras>=4 but no summary.")

            sal = rec.get("salience") if isinstance(rec.get("salience"), dict) else {}
            score = sal.get("score")
            reasons = set(sal.get("reasons") or [])
            salience_gap = False
            if isinstance(score, int) and score >= 5:
                salience_gap = True
                warns.append(f"{prefix} salience.score>=5 but no summary.")
            hit_reasons = reasons & IMPORTANT_SALIENCE_REASONS
            if hit_reasons:
                salience_gap = True
                warns.append(
                    f"{prefix} salience.reasons intersects {sorted(hit_reasons)} but no summary."
                )
            if salience_gap and isinstance(anchor, int):
                salience_gap_anchors.add(anchor)

    coverage_summaries.append(
        f"coverage j{no:03d}: {commentary_gaps} commentary gaps, "
        f"{dense_gaps} dense-year gaps, {len(salience_gap_anchors)} salience gaps"
    )


def validate_file(no: int, errors, warns, coverage_summaries) -> None:
    path = GUIDE_SRC / f"juan_{no:03d}.json"
    if not path.exists():
        errors.append(f"[juan {no}] guide file not found: {path}")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        errors.append(f"[juan {no}] version must be 1")
    if not isinstance(data.get("summaries"), list):
        errors.append(f"[juan {no}] summaries[] missing")
        return
    juan = load_juan(no)
    anchors = year_anchor_ids(juan)
    for s in data["summaries"]:
        check_summary(s, juan, anchors, errors, warns)
    coverage_check(no, data["summaries"], warns, coverage_summaries)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    if argv[0] == "--all":
        nos = [int(p.stem.split("_")[1]) for p in sorted(GUIDE_SRC.glob("juan_*.json"))]
    else:
        nos = [int(a) for a in argv]

    errors: list[str] = []
    warns: list[str] = []
    coverage_summaries: list[str] = []
    for no in nos:
        validate_file(no, errors, warns, coverage_summaries)

    for w in warns:
        print(f"WARN  {w}")
    for c in coverage_summaries:
        print(c)
    for e in errors:
        print(f"ERROR {e}")
    backend = "OpenCC" if _HAS_OPENCC else "fallback-blocklist"
    print(
        f"\nvalidated {len(nos)} 卷 — {len(errors)} error(s), {len(warns)} warning(s) "
        f"[simplified check: {backend}]"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
