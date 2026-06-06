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

# Year paragraph: optional era-name prefix (年号, typically 2 chars) +
# (元 | Chinese digits) + 年/载 + optional season.
# Tongjian uses 载 instead of 年 during 玄宗 天宝 — 肃宗 至德 era (744-758),
# e.g. "天宝六载", "至德元载".
# Examples that must match: "二十三年", "元年", "兴元元年", "永徽六年",
# "元初三年", "广顺元年春", "永宁元年", "永和十一年", "天宝六载", "至德元载".
# The era class excludes CN digits and 年/载 themselves so the (digits|元)+年/载
# tail can always be peeled off the right of the string.
_ERA_CLASS = r"[^\d\s年载，。；：、！？,.;:?!\-\(\)（）「」『』《》〈〉" + CN_DIGITS + "]"
YEAR_RE = re.compile(
    rf"^({_ERA_CLASS}{{0,4}})(元|[{CN_DIGITS}]+)[年载](春|夏|秋|冬)?$"
)
# Same pattern but allowing trailing month/punct.
YEAR_GANZHI_RE = re.compile(
    rf"^({_ERA_CLASS}{{0,4}})(元|[{CN_DIGITS}]+)[年载][，。]?\s*(春|夏|秋|冬)?"
)

# Emperor paragraph: typically a 谥号 ending in 帝|王|后|公, possibly followed by
# positional markers 上|中|下|之, sequence number (一二三四五六七八九十),
# or 干支 stems (甲乙丙丁戊己庚辛壬癸) used as sequence markers in some 卷.
# Examples to match:
#   "威烈王", "孝安皇帝中", "高祖武皇帝六", "孝宗穆皇帝中之下",
#   "宪宗昭文章武大圣至神孝皇帝中之下", "世宗睿武孝文皇帝下",
#   "安皇帝己", "赧王下"
# Must not match:
#   "唐纪七十六" (dynasty/juan label — no 帝/王/后/公)
#   long event sentences (use ^...$ + length cap + no punctuation)
_EMP_SUFFIX_CLASS = r"[上中下之" + GANZHI_STEMS + r"一二三四五六七八九十○]"
EMPEROR_RE = re.compile(
    rf"^[^\s\d，。；：、！？,.;:?!]{{1,15}}(帝|王|后|公){_EMP_SUFFIX_CLASS}{{0,5}}$"
)

SEASON_RE = re.compile(r"^(春|夏|秋|冬)(正月|[一二三四五六七八九十]+月)?$")

# Match the CE start anchor in 卷 title. Tongjian uses two formats:
#   1) "起戊寅（前403）" or "起癸亥(3)"  — main format
#   2) "丙申(936)一年"                  — sometimes no 起 prefix (single-year 卷)
# We take the FIRST 干支+CE-with-parens combo in the title.
CE_RANGE_RE = re.compile(
    rf"(?:起)?([{GANZHI_STEMS}][{GANZHI_BRANCHES}])[（(]\s*(前)?(\d+)\s*[）)]"
)


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


def _emperor_key(text: str) -> str:
    """Normalize emperor text by stripping trailing positional/sequence markers.
    'Same emperor' across paragraphs uses this key."""
    return re.sub(_EMP_SUFFIX_CLASS + r"+$", "", text.strip())


# Historian-commentary openings quoted by 司马光 throughout 通鉴. These run
# from the era under discussion: 太史公 (Sima Qian), 班固/班彪, 范晔, 陈寿,
# 习凿齿, 裴松之/裴子野, 干宝, 孙盛, 袁宏, 沈约, 萧子显, 魏徵, 欧阳修, etc.
# 臣光曰 is 司马光's own commentary; "史臣曰"/"论曰" are the official-history
# editor formulae. The opening is always "<2-4 char name>曰" followed by 「/︰/：.
_COMMENTARY_OPENERS = (
    "臣光曰", "太史公曰", "史臣曰", "论曰", "赞曰",
)
_COMMENTARY_RE = re.compile(
    r"^("
    + r"|".join(_COMMENTARY_OPENERS)
    + r"|[\u4e00-\u9fff]{2,4}曰[：︰「『])"
)


def classify(text: str) -> str:
    t = text.strip()
    if not t:
        return "event"
    if YEAR_RE.match(t):
        return "year"
    if SEASON_RE.match(t):
        return "season"
    if EMPEROR_RE.match(t):
        return "emperor"
    if _COMMENTARY_RE.match(t):
        return "commentary"
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
    prev_emperor_key: str | None = None
    prev_era: str | None = None
    first_year_seen = False

    for p in data["paragraphs"]:
        typ = classify(p["main"])
        p["type"] = typ
        if typ == "emperor":
            cur_emperor = p["main"].strip()
        if typ == "year":
            m = YEAR_RE.match(p["main"])
            era = (m.group(1) or "").strip() if m else ""
            yr_in_reign = cn_to_int(m.group(2)) if m else None
            cur_emperor_key = _emperor_key(cur_emperor) if cur_emperor else ""
            # Compute CE based on (emperor, era, year-in-reign) deltas.
            if cur_ce is not None and yr_in_reign is not None:
                if not first_year_seen:
                    # First year-marker in 卷 — already anchored at cur_ce from title.
                    first_year_seen = True
                else:
                    same_context = (
                        prev_emperor_key == cur_emperor_key
                        and prev_era == era
                        and prev_year_in_reign is not None
                    )
                    if same_context:
                        delta = yr_in_reign - prev_year_in_reign  # type: ignore[operator]
                    else:
                        # Emperor or era transition (改元 / 新君即位): the new
                        # period's 元年 immediately follows the previous year, so +1.
                        delta = 1
                    if delta > 0:
                        cur_ce += delta
                        if cur_gi is not None:
                            cur_gi = (cur_gi + delta) % 60
                prev_emperor_key = cur_emperor_key
                prev_era = era
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
