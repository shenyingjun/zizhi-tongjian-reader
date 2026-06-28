"""Wave 5 P3 — curated reign tables for role appellations (语境称谓 / role pseudo-alias).

A *role appellation* such as 吴主 / 魏主 / 齐主 is a pseudo-alias: the same surface
points to DIFFERENT people depending on the narrative year — whoever sat on that
throne at the time. The auto-seeder previously collapsed every 吴主 across 50 years
into a single conglomerate card (孙权…孙皓 jumbled together). This module replaces
that with position-aware resolution: each role occurrence is bound to the monarch
reigning in that paragraph's `ce_year`, emitted as a kind:"role" mention.

REIGNS[role] = [(monarch_canonical_name, start_ce, end_ce), ...]
    Inclusive CE years (BCE negative). A mention at year Y resolves to the entry
    whose [start, end] contains Y; on a boundary the latest-starting reign wins.
    Years outside every window → suppress (no card invented).

Only the high-frequency, well-bounded roles are curated. Tables follow 资治通鉴
narrative usage: a state's ruler is called 主 only while that state is NON-正统;
the 正统 line of any given year is narrated as 帝 / 上 and is intentionally absent
here (those are handled by ordinary person cards, not role resolution).

Monarch identities are book-anchored locators (dynasty + 庙号/谥号 + reign span +
首见卷) — the same category as the auto-seed brief, never a Wikipedia bio.
"""

# Surfaces removed from the alias/RULES path and resolved here instead. 长公主 is a
# specific-person title (馆陶长公主), NOT a reigning role — left untouched.
ROLE_APPELLATIONS = {
    "魏主", "齐主", "吴主", "周主", "蜀主", "汉主", "唐主", "隋主",
    "梁主", "夏主", "陈主", "闽主", "北汉主", "契丹主",
}

REIGNS = {
    # 孙吴 222–280 ; 杨吴 902–937
    "吴主": [
        ("孙权", 222, 252), ("孙亮", 252, 258), ("孙休", 258, 264), ("孙皓", 264, 280),
        ("杨行密", 902, 905), ("杨渥", 905, 908), ("杨隆演", 908, 920), ("杨溥", 920, 937),
    ],
    # 北魏 386–530（530 后分裂为东/西魏，另称 东魏主/西魏主，此处止于孝庄帝）
    "魏主": [
        ("拓跋珪", 386, 409), ("拓跋嗣", 409, 423), ("拓跋焘", 423, 452),
        ("拓跋濬", 452, 465), ("拓跋弘", 465, 471), ("元宏", 471, 499),
        ("元恪", 499, 515), ("元诩", 515, 528), ("元子攸", 528, 530),
    ],
    # 北齐 550–577
    "齐主": [
        ("高洋", 550, 559), ("高殷", 559, 560), ("高演", 560, 561),
        ("高湛", 561, 565), ("高纬", 565, 577), ("高恒", 577, 577),
    ],
    # 北周 557–581
    "周主": [
        ("宇文觉", 557, 557), ("宇文毓", 557, 560), ("宇文邕", 560, 578),
        ("宇文赟", 578, 579), ("宇文阐", 579, 581),
    ],
    # 蜀汉 221–263
    "汉主": [
        ("刘备", 221, 223), ("刘禅", 223, 263),
    ],
    # 前蜀 907–925 ; 后蜀 934–965
    "蜀主": [
        ("王建", 907, 918), ("王衍", 918, 925),
        ("孟知祥", 934, 934), ("孟昶", 934, 965),
    ],
    # 南唐 937–975（937–958，《通鉴》止于 959）
    "唐主": [
        ("李昪", 937, 943), ("李璟", 943, 961),
    ],
    # 隋 581–618
    "隋主": [
        ("杨坚", 581, 604), ("杨广", 604, 618),
    ],
    # 西梁（后梁·萧詧系）555–587 ; 朱梁（后梁）907–923
    "梁主": [
        ("萧詧", 555, 562), ("萧岿", 562, 585), ("萧琮", 585, 587),
        ("朱温", 907, 912), ("朱友珪", 912, 913), ("朱友贞", 913, 923),
    ],
    # 赫连夏 407–431
    "夏主": [
        ("赫连勃勃", 407, 425), ("赫连昌", 425, 428), ("赫连定", 428, 431),
    ],
    # 陈 557–589
    "陈主": [
        ("陈霸先", 557, 559), ("陈蒨", 559, 566), ("陈伯宗", 566, 568),
        ("陈顼", 569, 582), ("陈叔宝", 582, 589),
    ],
    # 闽 909–945
    "闽主": [
        ("王审知", 909, 925), ("王延翰", 925, 926), ("王延钧", 926, 935),
        ("王继鹏", 935, 939), ("王延羲", 939, 944), ("王延政", 944, 945),
    ],
    # 北汉 951–979（《通鉴》止于 959）
    "北汉主": [
        ("刘旻", 951, 954), ("刘承钧", 954, 968),
    ],
    # 契丹/辽 907–
    "契丹主": [
        ("耶律阿保机", 907, 926), ("耶律德光", 926, 947),
        ("耶律阮", 947, 951), ("耶律璟", 951, 969),
    ],
}

# Book-anchored identification for monarch cards that have to be CREATED (their
# personal name rarely surfaces as a clean token — they appear almost only via the
# role appellation). name -> (dynasty, 庙号/谥号 descriptor). Monarchs already in
# the people set (by full name, e.g. 孙权 / 杨坚 / 元宏 / 拓跋嗣 / 刘备 / 刘禅) are
# reused as-is and need no entry here.
MONARCH_META = {
    "孙休":   ("吴", "景帝"),
    "孙皓":   ("吴", "末帝"),
    "杨行密": ("吴", "太祖（杨吴奠基者）"),
    "杨渥":   ("吴", "烈祖"),
    "杨隆演": ("吴", "高祖"),
    "杨溥":   ("吴", "睿帝"),
    "拓跋焘": ("北魏", "太武帝"),
    "拓跋濬": ("北魏", "文成帝"),
    "拓跋弘": ("北魏", "献文帝"),
    "元恪":   ("北魏", "宣武帝"),
    "元诩":   ("北魏", "孝明帝"),
    "元子攸": ("北魏", "孝庄帝"),
    "高殷":   ("北齐", "废帝"),
    "高演":   ("北齐", "孝昭帝"),
    "高湛":   ("北齐", "武成帝"),
    "高恒":   ("北齐", "幼主"),
    "宇文觉": ("北周", "孝闵帝"),
    "宇文毓": ("北周", "明帝"),
    "宇文赟": ("北周", "宣帝"),
    "宇文阐": ("北周", "静帝"),
    "王衍":   ("前蜀", "后主"),
    "孟知祥": ("后蜀", "高祖"),
    "孟昶":   ("后蜀", "后主"),
    "李昪":   ("南唐", "烈祖"),
    "李璟":   ("南唐", "元宗（中主）"),
    "杨广":   ("隋", "炀帝"),
    "萧詧":   ("后梁（西梁）", "宣帝"),
    "萧岿":   ("后梁（西梁）", "明帝"),
    "萧琮":   ("后梁（西梁）", "后主"),
    "朱温":   ("后梁", "太祖"),
    "朱友珪": ("后梁", "废帝"),
    "朱友贞": ("后梁", "末帝"),
    "赫连勃勃": ("夏", "武烈帝"),
    "赫连昌": ("夏", "废帝"),
    "赫连定": ("夏", "末主"),
    "陈霸先": ("陈", "武帝"),
    "陈蒨":   ("陈", "文帝"),
    "陈伯宗": ("陈", "废帝"),
    "陈顼":   ("陈", "宣帝"),
    "陈叔宝": ("陈", "后主"),
    "王审知": ("闽", "太祖"),
    "王延翰": ("闽", "嗣王"),
    "王延钧": ("闽", "惠宗"),
    "王继鹏": ("闽", "康宗"),
    "王延羲": ("闽", "景宗"),
    "王延政": ("闽", "（殷）天德帝"),
    "刘旻":   ("北汉", "世祖"),
    "刘承钧": ("北汉", "睿宗"),
    "耶律阿保机": ("契丹", "太祖"),
    "耶律德光": ("契丹", "太宗"),
    "耶律阮": ("契丹", "世宗"),
    "耶律璟": ("契丹", "穆宗"),
}


def resolve_role(role: str, ce):
    """Return the monarch canonical name reigning under `role` at year `ce`,
    or None when ce is unknown or falls outside every curated reign window."""
    if ce is None:
        return None
    table = REIGNS.get(role)
    if not table:
        return None
    best, best_start = None, None
    for name, start, end in table:
        if start <= ce <= end and (best_start is None or start > best_start):
            best, best_start = name, start
    return best


def role_candidates(role: str, ce):
    """Monarch names whose reign window contains `ce`, sorted by accession year.
    Length 1 in a normal year; length 2 across a same-year succession (the first
    is the outgoing ruler, the last the incoming) — the caller splits those by
    reading position around the accession cue."""
    if ce is None:
        return []
    return [nm for (nm, s, e) in sorted(REIGNS.get(role, []), key=lambda x: x[1])
            if s <= ce <= e]


def reign_span(name: str):
    """[first_start, last_end] across every role table this monarch appears in."""
    starts, ends = [], []
    for table in REIGNS.values():
        for nm, s, e in table:
            if nm == name:
                starts.append(s); ends.append(e)
    if not starts:
        return None
    return [min(starts), max(ends)]


# Clan/surname prefixes stripped to derive a monarch's given-name tail for
# title-glue detection (契丹主德光 → 德光, 魏主嗣 → 嗣). Compound clans first.
_CLAN_PREFIXES = ("耶律", "拓跋", "宇文", "赫连", "慕容")


def _given_tail(name: str) -> str:
    for pre in _CLAN_PREFIXES:
        if name.startswith(pre):
            return name[len(pre):]
    return name[1:] if len(name) >= 2 else name


def monarch_tails(role: str):
    """[(tail, canonical_name)] for every monarch under `role`, longest tail first.
    Used for title-glue precedence: when a role surface is immediately followed by a
    monarch's given-name tail (契丹主德光), bind to that named monarch regardless of
    year. Only tails of length >=2 are returned — single-char tails (魏主嗣) are left
    to the curated seed aliases and would risk false glue (汉主禅位 → 刘禅)."""
    seen, out = set(), []
    for nm, _s, _e in REIGNS.get(role, []):
        if nm in seen:
            continue
        seen.add(nm)
        tail = _given_tail(nm)
        if len(tail) >= 2:
            out.append((tail, nm))
    out.sort(key=lambda t: len(t[0]), reverse=True)
    return out

