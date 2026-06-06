"""
Year segmentation for parsed 资治通鉴 卷.

For each parsed (Traditional or Simplified) 卷, classify paragraphs and emit
a year TOC. Operates on the simplified output so the reader can use it directly.

Heuristics (operating on Simplified text):
  - emperor paragraph: very short (<= 6 chars) ending in 帝|王|后|公 or starting
    with a known 谥号 marker, AND followed soon by a year-of-reign paragraph.
  - year paragraph: matches ^(元|[一二三四五六七八九十百]+)年(春|夏|秋|冬)?$
    or starts with that pattern + a 干支 in parens.
  - season paragraph: ^(春|夏|秋|冬)(正月|二月|...|十二月)?$  (rare standalone)

Also parse the 卷 title's CE range (起戊寅（前403）尽壬子（前369）) to anchor
the first year-in-reign to a CE year, then walk forward year-by-year using the
干支 cycle.

Output: rewrites cache/simplified/juan_NNN.json adding:
  - paragraphs[i].type = 'emperor' | 'reign' | 'year' | 'event' | 'commentary' | 'season'
  - paragraphs[i].ce_year = int | null
  - top-level "years": [ { ce: 403, label: '威烈王二十三年', paragraph_id: 1, ... }, ... ]
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SIMPLIFIED_DIR = ROOT / "cache" / "simplified"

CN_DIGITS = "零一二三四五六七八九十百"
GANZHI_STEMS = "甲乙丙丁戊己庚辛壬癸"
GANZHI_BRANCHES = "子丑寅卯辰巳午未申酉戌亥"

YEAR_RE = re.compile(rf"^(元|[{CN_DIGITS}]+)年(春|夏|秋|冬)?$")
# Year-with-ganzhi-and-month, e.g. "元年春王正月" "二年冬十月"
YEAR_GANZHI_RE = re.compile(rf"^(元|[{CN_DIGITS}]+)年[，。]?\s*(春|夏|秋|冬)?")
EMPEROR_TAIL_RE = re.compile(r"[帝王后公]$")
SEASON_RE = re.compile(r"^(春|夏|秋|冬)(正月|[一二三四五六七八九十]+月)?$")

# Match the CE start anchor in 卷 title: "起戊寅（前403）" or "起癸亥(3)"
CE_RANGE_RE = re.compile(r"起(\S{2})[（(](前)?(\d+)[）)]")


def cn_to_int(s: str) -> int | None:
    if s == "元":
        return 1
    # Simple Chinese numeral parser (1-100 range).
    units = {c: i for i, c in enumerate(CN_DIGITS)}
    if "十" not in s:
        # single digit
        return units.get(s) if s in units else None
    # "十" alone = 10; "十N" = 10 + N; "N十" = N*10; "N十M" = N*10 + M
    if s == "十":
        return 10
    if s.startswith("十"):
        rest = s[1:]
        if len(rest) == 1 and rest in units:
            return 10 + units[rest]
        return None
    if "十" in s:
        a, _, b = s.partition("十")
        if a not in units:
            return None
        n = units[a] * 10
        if b == "":
            return n
        if len(b) == 1 and b in units:
            return n + units[b]
    return None


def ganzhi_index(gz: str) -> int | None:
    """Return 0-59 index in the sexagenary cycle for a 干支 string."""
    if len(gz) != 2:
        return None
    s = GANZHI_STEMS.find(gz[0])
    b = GANZHI_BRANCHES.find(gz[1])
    if s < 0 or b < 0:
        return None
    # Stems repeat every 10, branches every 12. The combined cycle:
    # advance by 1 each year, both wheels turn together; both end at 60.
    # Find the smallest n s.t. n%10==s and n%12==b.
    for n in range(60):
        if n % 10 == s and n % 12 == b:
            return n
    return None


def classify(text: str) -> str:
    if YEAR_RE.match(text):
        return "year"
    if SEASON_RE.match(text):
        return "season"
    # Emperor: very short, ends with 帝|王|后|公, no digits.
    t = text.strip()
    if 1 <= len(t) <= 6 and EMPEROR_TAIL_RE.search(t) and not any(c.isdigit() for c in t):
        return "emperor"
    if t.startswith("臣光曰"):
        return "commentary"
    if YEAR_GANZHI_RE.match(t):
        return "year"
    return "event"


def parse_ce_start(title: str) -> tuple[int, int] | None:
    """Return (ce_year, ganzhi_idx) for the first year of the 卷."""
    m = CE_RANGE_RE.search(title)
    if not m:
        return None
    gz = m.group(1)
    sign = -1 if m.group(2) == "前" else 1
    ce = sign * int(m.group(3))
    gi = ganzhi_index(gz)
    if gi is None:
        return None
    return ce, gi


def annotate_juan(data: dict) -> dict:
    """Mutates data: adds 'type' and 'ce_year' to each paragraph, plus 'years' list."""
    start = parse_ce_start(data.get("title", ""))
    cur_ce = start[0] if start else None
    cur_gi = start[1] if start else None

    years: list[dict] = []
    cur_emperor = ""
    prev_year_in_reign: int | None = None
    prev_emperor: str | None = None
    first_year_seen = False

    for p in data["paragraphs"]:
        typ = classify(p["main"])
        p["type"] = typ
        if typ == "emperor":
            cur_emperor = p["main"].strip()
        if typ == "year":
            m = YEAR_RE.match(p["main"]) or YEAR_GANZHI_RE.match(p["main"])
            yr_in_reign = cn_to_int(m.group(1)) if m else None
            # Compute CE based on (emperor, year-in-reign) deltas.
            if cur_ce is not None and yr_in_reign is not None:
                if not first_year_seen:
                    # First year-marker in 卷 — already anchored at cur_ce from title.
                    first_year_seen = True
                else:
                    if prev_emperor == cur_emperor and prev_year_in_reign is not None:
                        delta = yr_in_reign - prev_year_in_reign
                    else:
                        # Emperor transition: new emperor's 元年 immediately follows
                        # predecessor's last year, so CE += 1.
                        delta = 1
                    if delta > 0:
                        cur_ce += delta
                        if cur_gi is not None:
                            cur_gi = (cur_gi + delta) % 60
                prev_emperor = cur_emperor
                prev_year_in_reign = yr_in_reign
            p["ce_year"] = cur_ce
            years.append({
                "ce_year": cur_ce,
                "ganzhi_idx": cur_gi,
                "label": "".join(s for s in [cur_emperor, p["main"]] if s),
                "paragraph_id": p["id"],
            })
        else:
            p["ce_year"] = cur_ce

    data["years"] = years
    return data


def main() -> int:
    files = sorted(SIMPLIFIED_DIR.glob("juan_*.json"))
    for path in tqdm(files, desc="year-seg", unit="卷"):
        data = json.loads(path.read_text(encoding="utf-8"))
        annotate_juan(data)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    print(f"annotated {len(files)} 卷")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
