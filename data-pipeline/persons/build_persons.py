"""Build the person knowledge base + per-卷 mention sidecars for all 294 卷.

Deterministic, *reviewed* extractor (P0, Option 1). The hand cast lives in
cast.py (confidence 'reviewed'); seed.py layers a deterministic auto-seed
('high') under it, growing hand 卷 ranges across batches and adding new figures
from the 白话导读 key_people. This driver matches each 卷's text against the
collision-free surface rule table, then emits:

  web/public/text/persons/people.json            merged KB (only shipped people)
  web/public/text/persons/mentions/juan_NNN.json one per 卷 with matches
  web/public/text/persons/appearances.json       person_id -> cross-卷 appearances
  web/public/text/persons/manifest.json          coverage summary

Matching is conservative: longest surface first, consumed char ranges block
shorter/overlapping matches (so 燕王 never matches inside 燕王喜, 王翦 never split),
and a surface is only applied in the 卷 listed in its person's `juans`.

Run:  python build_persons.py
"""
from __future__ import annotations
import json, datetime, hashlib, re, collections
from pathlib import Path

from cast import PEOPLE
import seed as seed_mod
import reigns as reigns_mod

JUANS = list(range(1, 295))  # all 294 卷 (hand 'reviewed' + auto seed layer)
REPO = Path(__file__).resolve().parents[2]
TEXT = REPO / "web" / "public" / "text"
OUT = TEXT / "persons"


def _write_text_retry(path, text, tries=8, delay=0.5):
    """Windows write with retry — an external scanner (Defender real-time, search
    indexer) intermittently holds a freshly-written JSON file, surfacing as
    OSError [Errno 22] on the next open. Retry a few times before giving up."""
    import time
    for k in range(tries):
        try:
            path.write_text(text, encoding="utf-8")
            return
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(delay)


# Merge the hand-curated cast with the deterministic auto-seed layer. PEOPLE is
# the combined people list (hand juans grown across batches, auto names split by
# contiguity); RULES is the collision-free {juan: {surface: person_id}} table.
# ANAPHORA_RULES is the Wave 5 single-char 省称 anchor table {juan: {char: pid}}.
PEOPLE_MERGED, RULES, ANAPHORA_RULES, SEED_STATS = seed_mod.build_seed(PEOPLE, JUANS)


# ── Wave 5 P3 — role appellation resolver (语境称谓) ──────────────────────────
# Replace the conglomerate role pseudo-cards (one 吴主 card jumbling 孙权…孙皓) with
# per-year resolution to the actual reigning monarch. Three steps, run once at
# import so main()'s by_id / surfaces_for see the corrected card set:
#   1. drop the conglomerate role cards + strip their surfaces from RULES,
#   2. reuse existing monarch cards / mint book-anchored ones for the absent rulers
#      (拓跋焘 等几乎只以「魏主」称谓出现，本名罕作干净 token),
#   3. expose ROLE_NAME_TO_PID so the per-paragraph role pass can bind each role
#      occurrence (resolved by reigns.resolve_role) to a real monarch person_id.
ROLE_SURFACES: list[str] = sorted(reigns_mod.ROLE_APPELLATIONS, key=len, reverse=True)
ROLE_NAME_TO_PID: dict[str, str] = {}


def _install_roles():
    # 1. Drop conglomerate role cards + their surfaces.
    role_ids = {p["id"] for p in PEOPLE_MERGED
                if p["canonical_name"] in reigns_mod.ROLE_APPELLATIONS}
    PEOPLE_MERGED[:] = [p for p in PEOPLE_MERGED if p["id"] not in role_ids]
    for juan_no, surf_map in RULES.items():
        for surf in list(surf_map):
            if surf in reigns_mod.ROLE_APPELLATIONS or surf_map[surf] in role_ids:
                del surf_map[surf]

    # 2. Map every monarch named in the reign tables to a person_id, reusing an
    #    existing card when the personal name already ships, minting one otherwise.
    by_name: dict[str, list] = {}
    for p in PEOPLE_MERGED:
        by_name.setdefault(p["canonical_name"], []).append(p)
    wanted = {nm for table in reigns_mod.REIGNS.values() for (nm, _, _) in table}
    for nm in sorted(wanted):
        cards = by_name.get(nm)
        if cards:
            # Prefer a reviewed card, else the first window.
            pick = next((c for c in cards if c.get("confidence") == "reviewed"), cards[0])
            ROLE_NAME_TO_PID[nm] = pick["id"]
            continue
        meta = reigns_mod.MONARCH_META.get(nm)
        if not meta:
            # No existing card and no curated identity — cannot mint safely; the
            # role pass will suppress occurrences that resolve to this name.
            continue
        dyn, hao = meta
        span = reigns_mod.reign_span(nm) or [None, None]
        s, e = span
        yr = f"（在位{s}–{e}年）" if s is not None else ""
        brief = f"{dyn}{hao}{yr}。"
        pid = "r:" + hashlib.md5(nm.encode("utf-8")).hexdigest()[:8]
        card = {
            "id": pid, "canonical_name": nm, "names": [],
            "dynasty": dyn, "era_hint": hao,
            "floruit": span, "brief": brief, "identity": brief,
            "match": [], "juans": [], "confidence": "reviewed",
        }
        PEOPLE_MERGED.append(card)
        by_name.setdefault(nm, []).append(card)
        ROLE_NAME_TO_PID[nm] = pid


_install_roles()


# RC-5 precision slice — drop 谥号/封号/官名 truncation FRAGMENT cards. NER sometimes
# emits a 2-char auto card that is really the first two chars of a longer title term
# (武灵<赵武灵王, 悼惠<齐悼惠王, 衞思<衞思后, 葛仆<葛…仆射). The principled, homograph-safe
# signal: scan the whole corpus and drop a 2-char AUTO card only when EVERY occurrence
# of its surface is immediately glued to a 封号/谥号/官名 tail char AND it NEVER stands
# alone. This spares real persons that share the prefix (曹参→曹参曰, 魏尚→魏尚书 but also
# 魏尚 standalone) because they have ≥1 standalone occurrence. Pure data-driven; no
# hand-list, so it self-adjusts if the seed/NER layer changes.
_FRAG_TAIL = set("王公侯君妃主后帝太嫔军书射同丽")


def _drop_fragment_cards():
    cand = {p["canonical_name"]: p["id"] for p in PEOPLE_MERGED
            if len(p["canonical_name"]) == 2 and str(p["id"]).startswith("a:")}
    if not cand:
        return 0
    glued = {s: 0 for s in cand}
    standalone = {s: 0 for s in cand}
    for j in JUANS:
        jf = TEXT / f"juan_{j:03d}.json"
        if not jf.exists():
            continue
        juan = json.loads(jf.read_text(encoding="utf-8"))
        for para in juan["paragraphs"]:
            chunks = [para.get("main", "")]
            chunks += [n.get("text", "") for n in para.get("notes", [])]
            for t in chunks:
                n = len(t)
                for i in range(n - 1):
                    s = t[i:i + 2]
                    if s in cand:
                        nxt = t[i + 2] if i + 2 < n else "。"
                        if nxt in _FRAG_TAIL:
                            glued[s] += 1
                        else:
                            standalone[s] += 1
    drop_surf = {s for s in cand
                 if glued[s] > 0 and standalone[s] == 0}
    drop_ids = {cand[s] for s in drop_surf}
    if not drop_ids:
        return 0
    PEOPLE_MERGED[:] = [p for p in PEOPLE_MERGED if p["id"] not in drop_ids]
    for surf_map in RULES.values():
        for surf in list(surf_map):
            if surf in drop_surf or surf_map[surf] in drop_ids:
                del surf_map[surf]
    return len(drop_ids)


_FRAG_DROPPED = _drop_fragment_cards()


def extract_roles(text, ce, consumed, by_id, para_idx=0, cue_idx=None):
    """Position-aware role pass: each role surface (吴主/魏主…) → the monarch
    reigning at `ce`. Skips chars already consumed by the alias/anaphora passes
    (so 魏主嗣→拓跋嗣 title-glue wins over bare 魏主). Suppresses when the year is
    unknown, falls outside every reign window, or the monarch has no shipped card.

    Title-glue precedence: when a role surface is immediately followed by a monarch's
    given-name tail (契丹主德光 / 契丹主阿保机), it binds to that NAMED monarch
    regardless of year — an explicit name always beats year-based resolution.

    Same-year succession (两位君主同年在位) is split by reading position: the
    paragraph that narrates the handover (`cue_idx[(role, ce)]`) and everything
    before it bind to the OUTGOING ruler; paragraphs after it bind to the INCOMING
    one. With no detected cue we fall back to the incoming ruler (latest accession),
    so resolution is never worse than a plain year lookup.
    Returns [(start, end, person_id, surface)]."""
    hits = []
    for surf in ROLE_SURFACES:  # longest first: 北汉主 before 汉主
        cand = reigns_mod.role_candidates(surf, ce)
        if len(cand) == 1:
            year_nm = cand[0]
        elif len(cand) >= 2:
            outgoing, incoming = cand[0], cand[-1]
            ci = (cue_idx or {}).get((surf, ce))
            year_nm = outgoing if (ci is not None and para_idx <= ci) else incoming
        else:
            year_nm = None
        tails = reigns_mod.monarch_tails(surf)
        for start in find_all(text, surf):
            end = start + len(surf)
            if any(consumed[start:end]):
                continue
            # Title-glue: explicit monarch name immediately after the role surface.
            glue_nm, glue_len = None, 0
            for tail, canon in tails:
                if text[end:end + len(tail)] == tail:
                    glue_nm, glue_len = canon, len(tail)
                    break
            nm = glue_nm or year_nm
            if not nm:
                continue
            pid_ = ROLE_NAME_TO_PID.get(nm)
            if not pid_ or pid_ not in by_id:
                continue
            full_end = end + glue_len
            if glue_len and any(consumed[end:full_end]):
                full_end = end  # name already taken by another pass; emit bare role
                glue_len = 0
            for k in range(start, full_end):
                consumed[k] = True
            hits.append((start, full_end, pid_, surf + (text[end:full_end] if glue_len else "")))
    hits.sort(key=lambda h: h[0])
    return hits


# Throne-handover cues in the 原文 used to split a same-year succession. A cue is
# tied to a specific role so it attributes the switch to the right state in a
# multi-state 卷, not whatever accession happens to occur. We prefer the handover
# of the OUTGOING ruler (deposition or terminal illness/death) since that is the
# unambiguous signal; an incoming accession is a tightly-bound fallback. The death
# marker need only co-occur with the role surface — emperor-grade terms (殂/崩/遗诏/
# 大渐/得疾…) rarely apply to anyone but the reigning monarch in a 主-paragraph.
_TERMINAL = ("得疾|寝疾|遇疾|不豫|疾笃|疾甚|疾亟|大渐|属纩|殂|崩|遗诏|遗制|遗令|"
             "山陵|坠马|落马|坠地|绝肋|视疾|大行")
_TERMINAL_RE = re.compile(_TERMINAL)
# Another subject's accession inside a 主-paragraph (契丹主闻帝即位) — never the role's
# own handover. Used to veto a false accession cue.
_OTHER_ACC_RE = re.compile("(?:帝|上|太子|太孙|太后|王|公)(?:即位|即皇帝位|即帝位|践阼|践祚)")
_DEATH = "殂|崩|薨|弑|卒|殒"
_ACCESSION = ("即皇帝位|即帝位|即位|践阼|践祚|受禅|禅位|传位|嗣位|入纂大统|入承大统|"
              "纂大统|统兹大宝|统大宝|立.{0,6}为帝|立.{0,6}为皇帝|迎立|自立为帝|自立为王|"
              "称帝|僭即皇帝位|僭称帝")
_ACC_RE = re.compile(_ACCESSION)



def build_role_cue_index(paragraphs):
    """{(role, ce_year): para_index} — for each same-year succession a role's state
    goes through inside this 卷, the paragraph that narrates the handover. We take
    the FIRST (reading order) of: the outgoing ruler being deposed (废{role}), the
    outgoing ruler's terminal illness/death co-located with the role surface (or the
    outgoing name), or — as a tightly-bound fallback — the incoming ruler acceding
    (accession verb adjacent to the role surface or the incoming name, never another
    subject's 帝/太子即位). Paragraphs at/before it bind to the outgoing ruler, after
    it to the incoming — so the deposition/death paragraph itself stays outgoing."""
    out: dict[tuple, int] = {}
    for idx, para in enumerate(paragraphs):
        ce = para.get("ce_year")
        if ce is None:
            continue
        txt = para.get("main", "")
        if not txt:
            continue
        for surf in ROLE_SURFACES:
            if (surf, ce) in out:
                continue
            cand = reigns_mod.role_candidates(surf, ce)
            if len(cand) < 2:
                continue
            outgoing, incoming = cand[0], cand[-1]
            depose = f"废{surf}" in txt
            death = (surf in txt or outgoing in txt) and _TERMINAL_RE.search(txt)
            accede = False
            if _ACC_RE.search(txt) and not _OTHER_ACC_RE.search(txt):
                in_tail = reigns_mod._given_tail(incoming)
                accede = (re.search(f"{surf}(?:{_ACCESSION})", txt)
                          or (in_tail and re.search(f"{re.escape(in_tail)}.{{0,3}}(?:{_ACCESSION})", txt)))
            if depose or death or accede:
                out[(surf, ce)] = idx
    return out


# Wave 5 P2 — common-word bigrams that a bare given-char forms in 文言; an anchor
# char inside one of these is the ordinary word, not a 省称 (坚守/谦让/翻译…). The
# guard rejects a single-char match whose left or right neighbor completes such a
# word. Kept curated + conservative; the audit prunes/extends it.
COMMON_BIGRAMS = {
    # 坚 (杨坚)
    "坚守", "坚壁", "坚固", "坚请", "坚执", "坚卧", "坚城", "坚甲", "坚锐",
    "坚冰", "坚营", "坚陈", "坚距", "坚白", "坚劲", "坚厚", "坚密", "深坚",
    # 谦 (王谦)
    "谦恭", "谦让", "谦虚", "谦逊", "谦谨", "谦冲", "谦退", "谦下", "谦谦",
    # 译 (郑译)
    "译语", "翻译", "译者", "重译", "译人", "译鞮", "译史",
    # 招 (宇文招)
    "招募", "招降", "招纳", "招集", "招抚", "招诱", "招怀", "招还", "招致",
    "招延", "招携", "招谕", "招辑", "招引", "招军",
    # 赞 / misc common
    "赞拜", "赞成", "称赞", "赞礼", "赞导", "赞曰",
    "邕邕",
}

# Preceding chars that turn a following anchor char into an adjective/verb reading
# (不坚 / 愈坚 / 自谦), never a name.
_ANAPHORA_MODS = set("不无甚愈益弥颇至自相見见尤极最稍渐")

# Fixed specialized terms (官名/官署) whose INTERNAL characters are split-prone: a bare
# given-char anaphora that lands inside one of these is the title fragment, not a 省称.
# The dominant case is the 唐五代 使职 family — e.g. 节度使 made bare 度 (裴度) fire 302×
# corpus-wide. A position covered by any of these terms is suppressed in extract_anaphora.
# These are all unambiguous offices (none doubles as a surname); the broader 官名/复姓
# reference list lives in the Phase-2 research notes and can be extended here as needed.
FIXED_NONNAME_TERMS: tuple = (
    # 地方军政使职 (highest split-risk — 内含 度/察/略/抚/运/练/御/讨/田/访)
    "节度副使", "节度判官", "节度留后", "节度使", "观察使", "经略使", "安抚使",
    "转运使", "团练使", "防御使", "招讨使", "营田使", "按察使", "采访使",
    "度支使", "支度使", "宣慰使", "宣抚使", "黜陟使",
    # 中枢使职
    "枢密副使", "枢密使", "三司使", "都指挥使",
    # 近侍/台省官 with name-like internal chars
    "散骑常侍", "御史中丞", "御史大夫", "监察御史", "殿中侍御史", "侍御史",
    "司隶校尉", "给事中", "光禄大夫", "太中大夫", "谏议大夫",
)


# ── Lifespan gate (Phase 3) ──────────────────────────────────────────────────
# A bare single-char 省称 (anaphora) refers to someone ACTIVE on stage, so it must
# fall inside the antecedent's lifetime. When a stale char_anchor carries a
# homograph given-char across a long 卷, the bare char binds to the wrong-era person
# (bare 征 in 738 → 魏征 d.643; bare 晏 in 417 → 王晏 active 489+). This gate drops
# such bindings: precision-first, a missing underline beats a wrong one (no re-bind
# here — that is rc3 single-surname disambiguation, a later Phase-3 step).
#
# NOTE — why this is applied to the `anaphora` kind ONLY. A person legitimately
# appears OUTSIDE their own lifespan in exactly two ways, and BOTH use full names,
# not bare 省称, so neither reaches this gate:
#   (a) someone is talking ABOUT them — a citation / 史论 / quotation (卷018 汉武帝
#       discussing 李斯·蒙恬·董仲舒). These surface as `alias` full-name matches.
#   (b) kinship — a descendant carries the name forward, glossed as 「X，<祖>之孙也」
#       (卷205 …韦孝宽之孙, 113y after 孝宽). These surface as `gloss` mentions.
# alias/gloss/role are therefore never gated. The corpus probe confirmed all gated
# rows are genuine stale-anchor errors; the only near-misses (张昭 active 195 but
# floruit records only 229–236; 段韶) sit at gap≈33 and are spared by the margin.
_LIFESPAN_MARGIN = 50    # floruit spans record peak years, not birth→death; be generous
_LIFESPAN_MIN_SPAN = 5   # ignore single-卷-year auto floruit (span 0–4) — not a real life


def _lifespan_outside(card, ce):
    """True when ce is confidently outside card's lifetime → gate the anaphora.
    False (keep) when the year is inside, the margin covers it, the floruit is too
    narrow to trust, or either bound / the year is unknown."""
    if ce is None:
        return False
    fl = card.get("floruit") or [None, None]
    s, e = (fl[0], fl[1]) if isinstance(fl, (list, tuple)) else (fl.get("start"), fl.get("end"))
    if s is None or e is None or (e - s) < _LIFESPAN_MIN_SPAN:
        return False
    return ce < s - _LIFESPAN_MARGIN or ce > e + _LIFESPAN_MARGIN


def find_all(hay: str, needle: str):
    i, out = 0, []
    while True:
        j = hay.find(needle, i)
        if j < 0:
            break
        out.append(j)
        i = j + len(needle)
    return out


def surfaces_for(juan: int):
    """[(surface, person_id)] applicable to this 卷, longest-first."""
    out = list(RULES.get(juan, {}).items())  # surface -> pid, collision-free
    out.sort(key=lambda t: len(t[0]), reverse=True)
    return out


def extract(text: str, surfaces):
    """Return [(start, end, person_id, surface)] for one text blob.

    Longest surface first; consumed character ranges block shorter/overlapping
    matches."""
    consumed = [False] * len(text)
    hits = []
    for surface, pid in surfaces:
        for start in find_all(text, surface):
            end = start + len(surface)
            if any(consumed[start:end]):
                continue
            for k in range(start, end):
                consumed[k] = True
            hits.append((start, end, pid, surface))
    hits.sort(key=lambda h: h[0])
    return hits


def _han(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


# Wave 5 P2 — context that marks a bare given-char as a person, not a word.
# RIGHT: the char is an agent/subject followed by a verb/particle (坚遣, 迥据, 谦薨).
_ANAPHORA_RIGHT = set(
    "遣将帅率曰谓言以据举攻击伐讨破降奔走卒薨死请欲乃遂使命还入出收"
    "徇略守拔围斩杀立得至引进退知闻怒喜患惧及等之也矣与为不复既又亦即因故大兵兼)")
# LEFT: the char is the object of a verb that takes a person (杀坚, 废坚, 立谦).
_ANAPHORA_LEFT = set(
    "杀执废立遣召讨击破降诛斩围获擒释赦贬黜任用见责让劝说谓命拜封逐囚弑代")


def extract_anaphora(text, admitted, consumed, char_anchor, anchor_events, ce, by_id):
    """Wave 5 P2 — single-char 省称 matches in one main-text blob, each bound to its
    NEAREST preceding full-name antecedent (the person most recently named whose
    given char == this char). Gates:
      * the char is an admitted candidate for this 卷 (admitted set),
      * an antecedent for that char already exists in char_anchor (else suppress —
        precision-first, never guess without an anchor),
      * the position is not consumed by a longer alias match,
      * a common-word bigram / modifier / clean-surname guard (坚守, 不坚, 杨惠 …),
      * a positive person-context gate (agent before a verb, or object after a
        person-taking verb).
    char_anchor is mutated in place (carried across paragraphs within a 卷);
    anchor_events is [(end_pos, given_char, person_id)] for this para's full-name
    hits, applied in reading order so an antecedent earlier in the SAME paragraph
    is visible to a later bare char. Returns [(start, end, person_id, surface)]."""
    out = []
    n = len(text)
    # Mask positions covered by a fixed specialized term (节度使 → 节/度/使 are the title,
    # never a 省称). Overlapping search so adjacent terms are all covered.
    fixed_mask = bytearray(n)
    for term in FIXED_NONNAME_TERMS:
        start = text.find(term)
        while start >= 0:
            for k in range(start, start + len(term)):
                fixed_mask[k] = 1
            start = text.find(term, start + 1)
    kin2_y_starts = set()
    for m in _KIN2_RE.finditer(text):
        if m.start(1) > 0 and text[m.start(1) - 1] not in _GLOSS_BOUNDARY \
                and text[m.start(1) - 1] not in "「『（":
            continue
        if _gloss_subject_ok(m.group(1)) and len(m.group(1)) == 1:
            kin2_y_starts.add(m.start(3))
    ev_i, nev = 0, len(anchor_events)
    for i, ch in enumerate(text):
        # establish every full-name antecedent that ends at or before this char
        while ev_i < nev and anchor_events[ev_i][0] <= i:
            _, gc, pidp = anchor_events[ev_i]
            char_anchor[gc] = pidp
            ev_i += 1
        if ch not in admitted or consumed[i]:
            continue
        if fixed_mask[i]:       # inside a fixed 官名 (节度使…) → title fragment, not a 省称
            continue
        pid_ = char_anchor.get(ch)
        if pid_ is None:        # no antecedent seen yet → suppress
            continue
        if pid_ not in by_id:   # xref-merged card id; RULES now points at the survivor
            continue
        left = text[i - 1] if i > 0 else ""
        right = text[i + 1] if i + 1 < n else ""
        # a bare 省称 drops the surname — if the left neighbor IS a clean surname
        # char, this is a surname+given full name the alias pass missed (杨惠, 窦毅),
        # not an anaphor. CLEAN_SURNAMES excludes 于/方/白… which double as common
        # words/prepositions (私于译曰 — 于 is "to", not the 于 surname).
        if left in seed_mod.CLEAN_SURNAMES:
            continue
        if (left and left + ch in COMMON_BIGRAMS) or (right and ch + right in COMMON_BIGRAMS):
            continue
        if left in _ANAPHORA_MODS:
            continue
        if not (right in _ANAPHORA_RIGHT or left in _ANAPHORA_LEFT or i in kin2_y_starts):
            continue
        if _lifespan_outside(by_id[pid_], ce):  # bare 省称 to a wrong-era antecedent
            continue
        out.append((i, i + 1, pid_, ch))
        char_anchor[ch] = pid_   # a resolved 省称 refreshes recency for this char
    return out


# ── Wave 5 P5 · RC-2b 「X，Y之Z也」 genealogical gloss ─────────────────────────
# At a clause boundary 资治通鉴 glosses a just-named person: 「温裕，戣之兄子也」 =
# X(温裕) is Y(戣)'s Z(兄子). Two recoveries: (a) Y is written 省姓 and shares X's
# surname (戣 → 孔戣) — reconstruct 姓+Y and bind that mention, recall a surname-gate
# / model NER cannot reach; (b) a kinship edge X——Z——>Y.
#
# PATRILINEAL terms only: for 子/孙/兄/弟/父/兄子/从子/族子… X and Y share a surname,
# so 姓(X)+Y is sound. 婿/甥/外孙/妻/母/舅 (different surname) are deliberately
# EXCLUDED. The hard precision backstop is the existence gate: 姓+Y is bound ONLY
# when it already matches a card reachable near this 卷, so a wrong surname guess
# (董卓+布 → 董布) finds no card and is silently dropped.
_KINSHIP_PATRILINEAL = [
    "曾孙", "玄孙", "兄子", "从子", "族子", "犹子", "从孙", "兄孙",
    "从弟", "从兄", "母弟", "母兄", "季父", "叔父", "伯父",
    "子", "孙", "弟", "兄", "父",
]
_KIN_ALT = (r"[一二三四五六七八九十]+世孙|世孙|"
            + "|".join(sorted(_KINSHIP_PATRILINEAL, key=len, reverse=True)))
_GLOSS_RE = re.compile(
    r"([\u4e00-\u9fff]{1,3})，([\u4e00-\u9fff]{1,4})之(" + _KIN_ALT + r")(?:也|[。；，、])"
)
_GLOSS_BOUNDARY = set("。；，！？、：")
# compact X-kin-Y (no 之/也): 「滈父绹」 X=given, kin, Y=known relative → mint 姓(Y)+X.
_KIN2_RE = re.compile(r"([\u4e00-\u9fff]{1,2})(" + _KIN_ALT + r")([\u4e00-\u9fff]{1,3})")


def _juan_gap(juans, j):
    return min((abs(a - j) for a in juans), default=10 ** 6)


# ── rc4 封号-glue (Phase 3) ───────────────────────────────────────────────────
# 北朝/南朝 宗室 are named 「(封号地名)(爵)(given)」: 任城王澄=元澄, 咸阳王禧=元禧,
# 高阳王雍=元雍, 湘东王宝晊=萧宝晊. The alias matcher glues 爵+given into a false
# 王-surname surface (王澄/王禧/王雍/王宝). rc4 reads the 封号 frame and binds the given
# to the reigning clan (国姓), then suppresses the overlapping false glue. A precision
# fix (kills 王X) AND recall (adds 元X). Tight gates keep it safe:
#   * clan candidates come from the 国姓 of dynasties whose reign covers the para year
#     (a textbook dynasty-surname table, NOT person data) — never a free guess,
#   * 姓+given must already be a known card reachable near this 卷 (existence gate),
#   * exactly one clan may resolve, else ambiguous → skip,
#   * lifespan-gate on the resolved person,
#   * the 爵 must follow a real 2-char place whose tail is not an office cue — excludes
#     郡望「太原王神念」 and 官名「尚书令王亮 / 中书监王莹」 (which also fail existence).
# 国姓 table: (clan, start, end). Overlaps (南北朝) are resolved by the unique gate.
_DYNASTY_CLANS = [
    ("刘", -206, 9), ("刘", 25, 220),                       # 两汉
    ("曹", 196, 265), ("刘", 221, 263), ("孙", 222, 280),     # 三国
    ("司马", 265, 420),                                      # 两晋
    ("刘", 420, 479), ("萧", 479, 502), ("萧", 502, 557), ("陈", 557, 589),  # 南朝
    ("拓跋", 386, 499), ("元", 471, 557), ("高", 534, 577), ("宇文", 535, 581),  # 北朝
    ("杨", 581, 618), ("李", 618, 907),                      # 隋唐
    ("朱", 907, 923), ("李", 923, 936), ("石", 936, 947), ("刘", 947, 951), ("郭", 951, 960),  # 五代
    ("慕容", 337, 410), ("苻", 351, 394), ("姚", 384, 417), ("赫连", 407, 431),  # 主要十六国
]
_FENG_RANKS = "王公"
_FENG_OFFICE_CUE = set("令监军尉史书丞卿傅师保都督刺守内将使后")  # place tail that is really an 官名
_FENG_GIVEN_BLOCK = set("公王侯主子后帝氏第官人臣")  # 王公/侯王… → a title/common word, not a given
# leading verb/particle wrongly captured as the place's 1st char — trim from the span.
_FENG_TRIM = set("讨遣为以与命于须诏诣从闻进牧尹瘗逼随乘数使立故封拜征诛杀废黜遂及又置"
                 "军是附让告助迎畏葬约见请史今知据兼率收徙拒留唯谓谮恶受皆次遣")
_FENG_RE = re.compile(r"([\u4e00-\u9fff]{2})([王公])([\u4e00-\u9fff]{1,2})")


def _clans_at(ce):
    if ce is None:
        return ()
    return tuple({c for (c, s, e) in _DYNASTY_CLANS if s <= ce <= e})


def extract_titleglue(text, ce, consumed, rule_map, by_id, alias_map=None):
    """rc4. Bind 「封号+given」 to its 宗室 person and mark the 封号 span consumed so the
    caller can drop the false 王X alias glue. Returns [(start, end, pid, surface)] with
    the WHOLE appellation (任城王澄) as the surface, mapped to the clan person (元澄).

    Precision anchor: the clan+given must be a surface in THIS 卷's collision-free map
    (rule_map) — i.e. the person is co-referenced by full name somewhere in the same 卷.
    That ties the 封号 to a real local person and kills cross-dynasty homograph picks
    (会稽王昱 is 司马昱, never 后燕 慕容昱; the latter is absent from a 东晋 卷's map)."""
    clans = _clans_at(ce)
    if not clans or not rule_map:
        return []
    out = []
    for m in _FENG_RE.finditer(text):
        place, given = m.group(1), m.group(3)
        if place[-1] in _FENG_OFFICE_CUE:        # 节度使王重 / 节度留后王重 → not a 封号
            continue
        rank_pos = m.start(2)
        # if 爵+2~3 chars is itself a recognized name (王重荣 / 王神念), the 爵 is a real
        # surname here, not a 封号 鞍 — leave it for the alias pass. (A single-char 王雍-type
        # glue is NOT a name and stays overridable below.)
        if any(text[rank_pos:rank_pos + L] in rule_map for L in (4, 3)):
            continue
        gs = m.start(3)
        bound = None
        for glen in (2, 1):                       # longer given first: 萧宝晊 before 萧宝
            if glen > len(given):
                continue
            g = given[:glen]
            if g in _FENG_GIVEN_BLOCK:            # 王公 / 侯王 → noble title, not a person
                continue
            hits = {rule_map.get(clan + g) for clan in clans}
            hits.discard(None)
            hits = {pid_c for pid_c in hits if not _lifespan_outside(by_id[pid_c], ce)}
            if len(hits) == 1:
                bound = (gs + glen, next(iter(hits)))
                break
        if not bound:
            # Precision-only slice: the 封号 frame is valid (place not an 官名, clan reigns
            # this year) but no clan+given card exists to bind/mint. If the alias pass would
            # otherwise glue 「爵+given」 (王骏/公弥) onto a WRONG-ERA homograph (汉 王骏 in a
            # 晋 卷), reserve that span so the false alias is dropped. No mention emitted —
            # a missing underline beats a wrong one; the right-era person is minted later.
            for glen in (2, 1):
                if glen > len(given):
                    continue
                g = given[:glen]
                if g in _FENG_GIVEN_BLOCK:
                    continue
                false_s, false_e = rank_pos, gs + glen
                fsurf = text[false_s:false_e]
                pid_w = rule_map.get(fsurf) or (alias_map.get(fsurf) if alias_map else None)
                if not pid_w or any(consumed[false_s:false_e]):
                    continue
                fl = by_id[pid_w].get("floruit") or [None, None]
                undated = fl[0] is None and fl[1] is None
                if _lifespan_outside(by_id[pid_w], ce) or undated:
                    for k in range(false_s, false_e):
                        consumed[k] = True
                    break
            continue
        ge, pid_c = bound
        start = m.start(1)
        if text[start] in _FENG_TRIM:            # drop a leading verb/particle (讨赵王伦→赵王伦)
            start += 1
        if any(consumed[start:ge]):              # never override a longer real match
            continue
        for k in range(start, ge):
            consumed[k] = True
        out.append((start, ge, pid_c, text[start:ge]))
    return out


def extract_gloss(text, rule_map, canon_to_pids, juan_no, by_id, consumed):
    """RC-2b. Return (mentions, relations) for one main-text blob:
      mentions = [(start, end, person_id, surface)] — Y bound to its full 姓名 card,
      relations = [(subject_pid, kin, object_pid, x_surface, y_surface)] meaning
                  subject IS object's <kin>.
    Precision-first: X must resolve through the 卷's collision-free surface map,
    its surname must be derivable, and 姓+Y must already be a known card."""
    mentions, relations = [], []
    for m in _GLOSS_RE.finditer(text):
        # require a real clause boundary just before X (start, or punctuation)
        if m.start(1) > 0 and text[m.start(1) - 1] not in _GLOSS_BOUNDARY:
            continue
        x_surf, y_surf, z = m.group(1), m.group(2), m.group(3)
        if not _gloss_y_ok(y_surf):
            continue
        pid_x = rule_map.get(x_surf)
        if not pid_x:
            continue
        cx = by_id[pid_x]["canonical_name"]
        surname = seed_mod._surname_of(cx)
        if not surname:
            continue
        # Y written 省姓 (戣) → 姓+Y; or already a full 姓名 present as a card.
        if seed_mod._surname_of(y_surf) and y_surf in canon_to_pids:
            y_full = y_surf
        else:
            y_full = surname + y_surf
        cands = canon_to_pids.get(y_full)
        if not cands:
            continue
        pid_y = min(cands, key=lambda pc: _juan_gap(pc[1], juan_no))[0]
        if pid_y == pid_x:
            continue
        ys, ye = m.start(2), m.end(2)
        if not any(consumed[ys:ye]):
            for k in range(ys, ye):
                consumed[k] = True
            mentions.append((ys, ye, pid_y, y_surf))
        relations.append((pid_x, z, pid_y, x_surf, y_surf))
    return mentions, relations


_CN_DIGIT = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
             "八": 8, "九": 9, "十": 10, "百": 100, "零": 0, "〇": 0}

# Y morphemes that are pronouns / 尊号 / 庙号, never a personal given name — a gloss
# 「X，上之弟也」 / 「X，太祖之孙也」 refers to the reigning ruler or a temple name, so
# 姓+上 / 姓+太祖 must NOT be reconstructed as a person (萧上 / 萧太祖 are bogus).
_GLOSS_Y_BLOCK = set("上帝后主君王公侯太子孙父母弟兄妻女甥婿妃嫔")


def _gloss_y_ok(y: str) -> bool:
    if not y or y in _GLOSS_Y_BLOCK:
        return False
    if len(y) <= 2 and y[-1] in "祖宗":   # 庙号: 太祖/高祖/世祖/太宗/高宗
        return False
    return True


def _cn2int(s: str):
    """Parse a small Chinese numeral (≤ 三百, the 资治通鉴 卷 range) to int. Handles
    百/十 positional forms: 二百四十→240, 三十→30, 百→100. None if unparseable."""
    if not s:
        return None
    total, section, last = 0, 0, 0
    for ch in s:
        if ch not in _CN_DIGIT:
            return None
        v = _CN_DIGIT[ch]
        if v == 100:
            section = (section + (last or 1)) * 100
            total += section
            section, last = 0, 0
        elif v == 10:
            section += (last or 1) * 10
            last = 0
        else:
            last = v
    return total + section + last or None


def _hu_xref_juan(notes, y_full, allowed):
    """A 胡注 「{Y}见…二百四十卷…」 cross-ref 卷 for a glossed person, when present
    and within range. Anchors a generatively-created card to where Y actually is."""
    for note in notes or ():
        t = note.get("text", "")
        idx = t.find(y_full)
        if idx < 0:
            continue
        m = re.search(r"([一二三四五六七八九十百零〇]+)\s*卷", t[idx:idx + 24])
        if m:
            j = _cn2int(m.group(1))
            if j and j in allowed:
                return j
    return None


_XREF_NUM_RE = re.compile(r"见([一二三四五六七八九十百零〇]+)\s*卷")


def _merge_xref_card(keep, drop, rules, meta=None):
    """Fold the later split-window card `drop` into the earlier `keep`: union 卷 /
    aliases, re-point every RULES surface, extend floruit end. The earliest-anchored
    brief on `keep` is left as-is (it already reports the true first appearance)."""
    keep["juans"] = sorted(set(keep.get("juans", [])) | set(drop.get("juans", [])))
    for fld in ("names", "match"):
        have = {(n["text"] if isinstance(n, dict) else n)
                for n in keep.get(fld, [])}
        for n in drop.get(fld, []):
            t = n["text"] if isinstance(n, dict) else n
            if t not in have:
                keep.setdefault(fld, []).append(n)
                have.add(t)
    # NOTE: do NOT extend floruit from the dropped window. Most xref-merges re-unite
    # a RETROSPECTIVE ALLUSION (e.g. 卷148 崔光「引汉光武崩赵熹…故事」 cites the 东汉 赵熹
    # centuries later); pulling floruit end to the citing 卷's era would invent a
    # multi-century lifespan and poison the lifespan-gate disambiguation signal. Leave
    # floruit anchored to the earliest window — consistent with the (unchanged) brief.
    kf = keep.get("floruit") or [None, None]
    df = drop.get("floruit") or [None, None]
    if kf[0] is None and df[0] is not None:
        kf[0] = df[0]
    if kf[1] is None and df[1] is not None and (kf[0] is None or df[1] >= kf[0]):
        kf[1] = df[1]
    keep["floruit"] = kf
    drop_id = drop["id"]
    keep_id = keep["id"]
    for smap in rules.values():
        for s, sp in list(smap.items()):
            if sp == drop_id:
                smap[s] = keep_id


def merge_xref_windows(by_id, people_list, rules, juans, text_dir, meta=None):
    """Wave B — re-unite an auto person that split_windows split across a large 卷
    gap, when the book's OWN 胡注 「见{N}卷」 cross-reference points from a later window
    back into an earlier window of the SAME canonical surface (李洧 卷227+卷250 → one
    card; 张建封 卷225–235+卷250 → one card). The 注 is the book's same-person signal.

    Precision-first: only AUTO cards sharing an identical canonical name; merge a later
    window into the earliest only when some 卷X in that later window has a paragraph
    where the surface occurs AND a 注 on that paragraph cites 见N卷 with N inside the
    earliest window ±1 (the ±1 absorbs the book's imprecise 卷/年 citing). Because the
    membership test is against THIS surface's own window, two people quoted in one
    paragraph self-disambiguate — each only matches the xref that lands in its window."""
    groups: dict[str, list] = collections.defaultdict(list)
    for pid_, p in by_id.items():
        if str(pid_).startswith("a:"):
            groups[p["canonical_name"]].append(p)
    juan_cache: dict[int, list] = {}

    def juan_paras(j):
        if j not in juan_cache:
            jf = text_dir / f"juan_{j:03d}.json"
            juan_cache[j] = (json.loads(jf.read_text(encoding="utf-8"))["paragraphs"]
                             if jf.exists() else [])
        return juan_cache[j]

    merged = 0
    dropped_ids: set = set()
    for name, cards in groups.items():
        if len(cards) < 2 or len(name) < 2:
            continue
        cards.sort(key=lambda c: min(c["juans"]) if c.get("juans") else 10 ** 9)
        keep = cards[0]
        kjs = [j for j in keep.get("juans", []) if j in juans]
        if not kjs:
            continue
        lo, hi = min(kjs) - 1, max(kjs) + 1
        for C in cards[1:]:
            hit = False
            for jx in sorted(set(C.get("juans", []))):
                if jx not in juans:
                    continue
                for para in juan_paras(jx):
                    main = para.get("main", "")
                    if name not in main:
                        continue
                    # positions where the surface occurs in main
                    occ = []
                    p0 = main.find(name)
                    while p0 >= 0:
                        occ.append(p0)
                        p0 = main.find(name, p0 + 1)
                    for note in para.get("notes", []) or ():
                        for m in _XREF_NUM_RE.finditer(note.get("text", "")):
                            n = _cn2int(m.group(1))
                            if not (n and lo <= n <= hi):
                                continue
                            # PROXIMITY: the citation must annotate THIS surface — its
                            # attach offset must follow an occurrence of `name` within a
                            # short window (李洧自归，见227卷). This disambiguates two
                            # people cited in one paragraph (李洧 vs 张建封) and blocks a
                            # coincidental 见N卷 about an unrelated person from merging a
                            # cross-era homograph (王郎 = 汉王昌 vs 魏王朗).
                            aft = note.get("after")
                            if aft is None:
                                continue
                            if any(0 <= aft - (o + len(name)) <= 16 for o in occ):
                                hit = True
                                break
                        if hit:
                            break
                    if hit:
                        break
                if hit:
                    break
            if not hit:
                continue
            _merge_xref_card(keep, C, rules, meta)
            dropped_ids.add(C["id"])
            del by_id[C["id"]]
            merged += 1
    if dropped_ids:
        people_list[:] = [p for p in people_list if p["id"] not in dropped_ids]
    return merged, dropped_ids
# OBJECT Y is the KNOWN ancestor (谱，珪之六世孙也 → 谱 = 王珪's 6th-gen descendant).
# The forward path mints the relative Y from a known X; this inverse path mints the
# subject X from a known ancestor Y, inheriting Y's surname (patrilineal). The
# ancestor's surname is recovered from the paragraph 胡注 when Y is written 省姓
# (「王珪事太宗…」 disambiguates 珪 = 王珪, not the 李珪 elsewhere in the 卷).
def _gloss_subject_ok(x: str) -> bool:
    return bool(x) and 1 <= len(x) <= 2 and x not in _GLOSS_Y_BLOCK \
        and not (len(x) <= 2 and x[-1] in "祖宗")


def _ancestor_surname(y_surf, canon_to_pids, notes, by_id=None, gindex=None, juan_no=None):
    """Surname of the gloss ancestor Y. A full 姓名 Y yields its own 姓; a 省姓
    single-char Y is resolved through the paragraph 胡注 「{姓}{Y}…」 (uniquely), or
    via a 2-hop 「{Y}父{父名}」 chain when the father resolves to a card (宰→智兴→王).
    A 省姓 2-char given (崇文 of 高崇文) resolves to a carded surname; when several
    surnames pair with it (高崇文 vs 王崇文), the ancestor nearest this 卷 wins."""
    sn = seed_mod._surname_of(y_surf)
    if sn and len(y_surf) >= 2 and y_surf in canon_to_pids:
        return sn
    if len(y_surf) >= 2 and gindex:
        cands = gindex.get(y_surf, set())
        if len(cands) == 1:
            return next(iter(cands))
        if len(cands) > 1 and juan_no is not None:
            best, bg = None, 10 ** 6
            for s in cands:
                for _pid, pj in canon_to_pids.get(s + y_surf, []):
                    g = _juan_gap(pj, juan_no)
                    if g < bg:
                        best, bg = s, g
            if best:
                return best
    if len(y_surf) != 1:
        return None
    found = set()
    for note in notes or ():
        t = note.get("text", "")
        idx = t.find(y_surf)
        while idx > 0:
            if idx >= 2 and t[idx - 2:idx] in seed_mod.COMPOUND:
                found.add(t[idx - 2:idx])
            elif t[idx - 1] in seed_mod.CLEAN_SURNAMES:
                found.add(t[idx - 1])
            idx = t.find(y_surf, idx + 1)
    if not found and by_id is not None:
        # 2-hop chain: 「宰父智兴」 — Y's father resolves to a card; Y inherits that
        # surname (智兴→王智兴→王). Father written 省姓 (alias) or full 姓名.
        for note in notes or ():
            for fm in re.finditer(y_surf + r"父([\u4e00-\u9fff]{2})", note.get("text", "")):
                f = fm.group(1)
                names = {c + f for c in seed_mod.CLEAN_SURNAMES} | {f}
                fc = {seed_mod._surname_of(p["canonical_name"]) for p in (by_id or {}).values()
                      if p.get("canonical_name") in names}
                fc.discard(None)
                if len(fc) == 1:
                    found.add(next(iter(fc)))
    return next(iter(found)) if len(found) == 1 else None


def _resolve_y_surname(y, by_id, gindex):
    """Surname of relative Y: a 2-3 char card yields its own 姓; a 省姓 single char
    resolves only if exactly one carded surname pairs with that given (绹→令狐)."""
    sn = seed_mod._surname_of(y)
    if sn and len(y) >= 2 and any(p.get("canonical_name") == y for p in by_id.values()):
        return sn
    if len(y) == 1:
        cands = gindex.get(y, set())
        if len(cands) == 1:
            return next(iter(cands))
    return None


def build_gloss_cards(juans, text_dir, rules, canon_to_pids, by_id,
                      people_merged, meta, allowed):
    """RC-2b pre-pass — generative recall. When a 「X，Y之Z也」 gloss reconstructs
    Y = 姓(X)+Y to a name that has NO card (孔戣, only ever written 省姓 as 戣), mint
    one, anchored to the gloss 卷 (and the 胡注 cross-ref 卷 when given). Strong
    precision gates: X must resolve to a real person in this 卷, the kin term is
    patrilineal (shared surname), and Y+姓 must not be a common non-person word.
    Returns the number of cards created; mutates by_id/canon_to_pids/people_merged."""
    created: dict[str, dict] = {}
    anaphora: dict[str, tuple] = {}   # name -> (given_char) for inverse-minted subjects
    carded_anaphora: dict[str, tuple[set[int], str]] = {}  # existing pid -> ({juans}, given_char)
    # given-char → set of carded surnames, for 省姓 relative resolution (绹→令狐绹→令狐)
    gindex: dict[str, set] = {}
    for p in by_id.values():
        cn = p.get("canonical_name", "")
        # a CARDED name proves its head is a real surname, so accept ambiguous-clan
        # heads (高/严/任/康/万…) here even though _surname_of refuses raw candidates —
        # the ancestor-gloss recall (崇文→高 for 高骈) needs them; precision stays
        # because minting requires the resolved full name to be an existing card.
        sn = seed_mod._surname_of(cn)
        if not sn:
            if len(cn) >= 3 and cn[:2] in seed_mod.COMPOUND:
                sn = cn[:2]
            elif len(cn) >= 2 and cn[0] in seed_mod.SURNAMES:
                sn = cn[0]
        if sn and len(cn) >= 2:
            gindex.setdefault(cn[len(sn):], set()).add(sn)
    office_glue_cues: tuple[str, ...] = tuple(sorted({
        "判度支河南", "判度支", "判支", "平章事", "侍郎", "将军", "尚书", "刺史",
        "节度使", "观察使", "校理", "拾遗", "中丞",
    }, key=len, reverse=True))
    office_glue_given_block = set("王公侯主子后帝氏第官人臣军使州府司部省道不")
    office_glue_right_ok = set("同充为迁拜罢谏曰薨卒贬除加兼领权知，。；、：︰")
    compounds = sorted(seed_mod.COMPOUND, key=len, reverse=True)
    surnames = "".join(sorted(re.escape(ch) for ch in seed_mod.CLEAN_SURNAMES))
    cue_alt = "|".join(re.escape(c) for c in office_glue_cues)
    compound_alt = "|".join(re.escape(c) for c in compounds)
    office_glue_re = re.compile(
        rf"(?:{cue_alt})({compound_alt}|[{surnames}])([\u4e00-\u9fff])")
    office_glue_hits: dict[str, dict] = collections.defaultdict(
        lambda: {"n": 0, "juans": set()})
    for juan_no in juans:
        jf = text_dir / f"juan_{juan_no:03d}.json"
        if not jf.exists():
            continue
        juan = json.loads(jf.read_text(encoding="utf-8"))
        for para in juan["paragraphs"]:
            scan = [para.get("main", "")] + [
                nt.get("text", "") for nt in para.get("notes", [])]
            for mtext in scan:
                for m in office_glue_re.finditer(mtext):
                    new_full = m.group(1) + m.group(2)
                    if m.group(2) in office_glue_given_block:
                        continue
                    right = mtext[m.end():m.end() + 1]
                    if right and (right in seed_mod.CLEAN_SURNAMES or
                                  right not in office_glue_right_ok):
                        continue
                    if new_full in canon_to_pids:
                        continue
                    if seed_mod.bad_auto_surface(new_full) or \
                            new_full in seed_mod.COMMON_WORD_NONPERSON or \
                            new_full in seed_mod.COMPOUND:
                        continue
                    office_glue_hits[new_full]["n"] += 1
                    office_glue_hits[new_full]["juans"].add(juan_no)

    for new_full, info in office_glue_hits.items():
        if info["n"] < 2 or new_full in created:
            continue
        juans_c = sorted(info["juans"])
        j0 = min(juans_c)
        dyn = (meta.get(j0, {}).get("dynasty") or "").strip()
        cs = meta.get(j0, {}).get("ce_start")
        card = {
            "id": seed_mod._auto_id(new_full, 9000 + len(created)),
            "canonical_name": new_full,
            "names": [],
            "dynasty": dyn or "—",
            "era_hint": f"{dyn}人物" if dyn else "人物",
            "floruit": [cs, cs] if cs else [None, None],
            "brief": f"{dyn + '·' if dyn else ''}见于卷{j0:03d}（官衔连写）。",
            "identity": f"{dyn + '·' if dyn else ''}见于卷{j0:03d}（官衔连写）。",
            "match": [new_full],
            "juans": juans_c,
            "confidence": "high",
        }
        created[new_full] = card
        for j in juans_c:
            rules.setdefault(j, {}).setdefault(new_full, card["id"])

    for juan_no in juans:
        jf = text_dir / f"juan_{juan_no:03d}.json"
        if not jf.exists():
            continue
        rule_map = rules.get(juan_no, {})
        if not rule_map:
            continue
        juan = json.loads(jf.read_text(encoding="utf-8"))
        for para in juan["paragraphs"]:
            mt = para.get("main", "")
            scan = [mt] + [nt.get("text", "") for nt in para.get("notes", [])]
            for mtext in scan:
              for m in _GLOSS_RE.finditer(mtext):
                if m.start(1) > 0 and mtext[m.start(1) - 1] not in _GLOSS_BOUNDARY:
                    continue
                x_surf, y_surf = m.group(1), m.group(2)
                pid_x = rule_map.get(x_surf)
                inverse = False
                if pid_x:
                    # FORWARD — X known, mint the 省姓 relative Y = 姓(X)+Y.
                    if not _gloss_y_ok(y_surf):
                        continue
                    surname = seed_mod._surname_of(by_id[pid_x]["canonical_name"])
                    if not surname:
                        continue
                    if seed_mod._surname_of(y_surf) and y_surf in canon_to_pids:
                        continue  # already a full-name card → handled by binding pass
                    new_full, anchor_given = surname + y_surf, None
                else:
                    # INVERSE — X is the new descendant, Y the KNOWN ancestor; mint
                    # X = 姓(Y)+X with the ancestor's surname (recovered from 胡注).
                    if not _gloss_subject_ok(x_surf):
                        continue
                    surname = _ancestor_surname(y_surf, canon_to_pids, para.get("notes"), by_id, gindex, juan_no)
                    if not surname:
                        continue
                    new_full = surname + x_surf
                    anchor_given = x_surf if len(x_surf) == 1 else None
                    inverse = True
                if new_full in canon_to_pids or len(new_full) < 2 or len(new_full) > 3:
                    # ancestor gloss resolved a card that already exists (高骈，崇文之孙
                    # 也): register its 省称 given (骈) so the section-local anaphora pass
                    # binds bare mentions even when the surname (高) is ambiguous.
                    if inverse and anchor_given and new_full in canon_to_pids:
                        for pcid, _pj in canon_to_pids[new_full]:
                            if anchor_given not in seed_mod.ANAPHORA_CHAR_EXCLUDE:
                                carded_anaphora.setdefault(pcid, (set(), anchor_given))[0].add(juan_no)
                    continue
                if seed_mod.bad_auto_surface(new_full) or \
                        new_full in seed_mod.COMMON_WORD_NONPERSON:
                    continue
                xref = _hu_xref_juan(para.get("notes"), new_full, allowed)
                if new_full in created:
                    for j in {juan_no, xref}:
                        if j and j not in created[new_full]["juans"]:
                            created[new_full]["juans"].append(j)
                    if inverse:
                        rule_map.setdefault(new_full, created[new_full]["id"])
                    continue
                dyn = (meta.get(juan_no, {}).get("dynasty") or "").strip()
                cs = meta.get(juan_no, {}).get("ce_start")
                juans_c = sorted({j for j in (juan_no, xref) if j})
                j0 = min(juans_c)
                card = {
                    "id": seed_mod._auto_id(new_full, 9000 + len(created)),
                    "canonical_name": new_full,
                    "names": [],
                    "dynasty": dyn or "—",
                    "era_hint": f"{dyn}人物" if dyn else "人物",
                    "floruit": [cs, cs] if cs else [None, None],
                    "brief": f"{dyn + '·' if dyn else ''}见于卷{j0:03d}（家世胡注）。",
                    "identity": f"{dyn + '·' if dyn else ''}见于卷{j0:03d}（家世胡注）。",
                    "match": [new_full],
                    "juans": juans_c,
                    "confidence": "high",
                }
                created[new_full] = card
                if inverse:
                    # register the full surface so the alias pass binds 王谱; record
                    # the 省姓 given char so the anaphora pass binds the bare 谱.
                    rule_map.setdefault(new_full, card["id"])
                    if anchor_given:
                        anaphora[new_full] = anchor_given
              for m in _KIN2_RE.finditer(mtext):
                x, kin, y = m.group(1), m.group(2), m.group(3)
                if m.start(1) > 0 and mtext[m.start(1) - 1] not in _GLOSS_BOUNDARY \
                        and mtext[m.start(1) - 1] not in "「『（":
                    continue
                if not _gloss_subject_ok(x) or len(x) != 1:
                    continue
                surname = None
                y_pid = None
                for yl in (2, 1, 3):
                    if yl <= len(y):
                        y_surf = y[:yl]
                        surname = _resolve_y_surname(y_surf, by_id, gindex)
                        if surname:
                            y_full = y_surf if seed_mod._surname_of(y_surf) and len(y_surf) >= 2 \
                                else surname + y_surf
                            cands = canon_to_pids.get(y_full) or []
                            if cands:
                                y_pid = min(cands, key=lambda pc: _juan_gap(pc[1], juan_no))[0]
                            break
                if not surname:
                    continue
                if y_pid:
                    y_given = seed_mod._given_single(by_id[y_pid]["canonical_name"])
                    if y_given and y_given not in seed_mod.ANAPHORA_CHAR_EXCLUDE:
                        carded_anaphora.setdefault(y_pid, (set(), y_given))[0].add(juan_no)
                nf = surname + x
                if nf in canon_to_pids or len(nf) < 2 or len(nf) > 3 or nf in created:
                    continue
                if seed_mod.bad_auto_surface(nf) or nf in seed_mod.COMMON_WORD_NONPERSON:
                    continue
                dyn = (meta.get(juan_no, {}).get("dynasty") or "").strip()
                cs = meta.get(juan_no, {}).get("ce_start")
                created[nf] = {
                    "id": seed_mod._auto_id(nf, 9000 + len(created)),
                    "canonical_name": nf, "names": [], "dynasty": dyn or "—",
                    "era_hint": f"{dyn}人物" if dyn else "人物",
                    "floruit": [cs, cs] if cs else [None, None],
                    "brief": f"{dyn + '·' if dyn else ''}见于卷{juan_no:03d}（家世胡注）。",
                    "identity": f"{dyn + '·' if dyn else ''}见于卷{juan_no:03d}（家世胡注）。",
                    "match": [nf], "juans": [juan_no], "confidence": "high"}
                rule_map.setdefault(nf, created[nf]["id"])
                anaphora[nf] = x
    for new_full, card in created.items():
        people_merged.append(card)
        by_id[card["id"]] = card
        canon_to_pids.setdefault(new_full, []).append(
            (card["id"], set(card["juans"])))
    inv_anaphora = [(created[n]["id"], created[n]["juans"], g)
                    for n, g in anaphora.items()]
    inv_anaphora.extend((pid, sorted(juans), g)
                        for pid, (juans, g) in carded_anaphora.items())
    return len(created), inv_anaphora


# RC book-enrich — first-mention courtesy-name apposition 「{姓名}，字{X}」. The 字
# (表字) is the single most reliable book-derived identity fact: it follows a rigid
# punctuation-bounded form and uniquely tags the biographical subject. We mine it
# corpus-wide and attach by 卷-proximity so a same-name figure from another era never
# inherits the wrong 字 (汉 张温 字惠恕 ≠ 陈 张温). Book-derived only — never Wikipedia.
_ZI_PAIR_RE = re.compile(
    r"([\u4e00-\u9fff]{2,4})，字([\u4e00-\u9fff]{1,2})(?=[，。、；！？])")


def enrich_briefs(juans, text_dir, people_merged):
    """Prepend 「字X。」 to the brief/identity of auto/gloss cards whose canonical name
    is introduced with a courtesy name in the text. Returns count enriched."""
    by_name: dict[str, list] = {}
    for p in people_merged:
        if p.get("confidence") == "reviewed":
            continue  # never touch hand-curated cards
        by_name.setdefault(p["canonical_name"], []).append(p)
    if not by_name:
        return 0
    found: list[tuple[str, str, int]] = []  # (name, zi, juan) in reading order
    for juan_no in juans:
        jf = text_dir / f"juan_{juan_no:03d}.json"
        if not jf.exists():
            continue
        juan = json.loads(jf.read_text(encoding="utf-8"))
        for para in juan["paragraphs"]:
            chunks = [para.get("main", "")]
            chunks += [n.get("text", "") for n in (para.get("notes") or [])]
            for ch in chunks:
                for m in _ZI_PAIR_RE.finditer(ch):
                    if m.group(1) in by_name:
                        found.append((m.group(1), m.group(2), juan_no))
    enriched: set[str] = set()
    for nm, zi, juan_no in found:
        cands = by_name[nm]
        best = next((p for p in cands if juan_no in (p.get("juans") or [])), None)
        if best is None:
            best = min(cands, key=lambda p: _juan_gap(p.get("juans") or [10 ** 6], juan_no))
        if best["id"] in enriched:
            continue
        clause = f"字{zi}。"
        if not best["brief"].startswith("字"):
            best["brief"] = clause + best["brief"]
        if not best["identity"].startswith("字"):
            best["identity"] = clause + best["identity"]
        enriched.add(best["id"])
    return len(enriched)


def main():
    by_id = {p["id"]: p for p in PEOPLE_MERGED}
    # Duplicate-id guard — a copy/paste slip would silently drop a person.
    if len(by_id) != len(PEOPLE_MERGED):
        seen, dups = set(), set()
        for p in PEOPLE_MERGED:
            if p["id"] in seen:
                dups.add(p["id"])
            seen.add(p["id"])
        raise SystemExit(f"duplicate person ids: {sorted(dups)}")

    # Wave 5 P2 — each person's single-char given name (元胄→胄, 杨坚→坚), used to
    # build per-paragraph anaphora antecedents from full-name alias hits.
    given_of = {}
    for pid_, p in by_id.items():
        g = seed_mod._given_single(p["canonical_name"])
        if g and g not in seed_mod.ANAPHORA_CHAR_EXCLUDE:
            given_of[pid_] = g

    # RC-2b — canonical 姓名 → [(pid, {juans})] for gloss Y reconstruction (孔戣).
    canon_to_pids: dict[str, list] = {}
    for pid_, p in by_id.items():
        canon_to_pids.setdefault(p["canonical_name"], []).append(
            (pid_, set(p.get("juans", []))))

    # RC-2b pre-pass — mint cards for pure-new-recall glossed persons (孔戣, only
    # ever written 省姓) so the binding pass below can resolve their 戣 mentions.
    # Inverse glosses (谱，珪之六世孙也 → 王谱) also register their full surface in
    # RULES and return the 省姓 given char so the anaphora pass binds the bare 谱.
    gloss_meta = seed_mod.juan_meta()
    gloss_new_cards, gloss_inv_anaphora = build_gloss_cards(
        JUANS, TEXT, RULES, canon_to_pids, by_id,
        PEOPLE_MERGED, gloss_meta, set(JUANS))
    minted_admit: dict[int, set] = {}   # juan -> {given_char} for inverse-minted/carded gloss subjects
    minted_anchor_candidates: dict[int, dict[str, set[str]]] = {}
    for cid, cjuans, gch in gloss_inv_anaphora:
        if gch and gch not in seed_mod.ANAPHORA_CHAR_EXCLUDE:
            given_of[cid] = gch
            for j in cjuans:
                minted_admit.setdefault(j, set()).add(gch)
                minted_anchor_candidates.setdefault(j, {}).setdefault(gch, set()).add(cid)
    minted_anchor = {
        j: {gch: next(iter(cids)) for gch, cids in cmap.items() if len(cids) == 1}
        for j, cmap in minted_anchor_candidates.items()
    }

    # book-enrich — courtesy-name apposition into briefs (after gloss cards exist so
    # newly-minted 家世 cards also get their 字 when the text introduces one).
    briefs_enriched = enrich_briefs(JUANS, TEXT, PEOPLE_MERGED)

    # ── Wave B: 胡注 见N卷 xref window-merge ──
    # split_windows separates one NER surface into per-window cards across a large 卷
    # gap. The book's own 胡注 「见{N}卷」 cross-reference is the same-person signal —
    # re-unite李洧 (卷227+卷250) / 张建封 (卷225–235+卷250) before the lookback/mention
    # passes so their RULES + mentions aggregate onto one card.
    xref_merged, xref_dropped = merge_xref_windows(
        by_id, PEOPLE_MERGED, RULES, set(JUANS), TEXT, gloss_meta)
    for did in xref_dropped:
        given_of.pop(did, None)
    if xref_dropped:
        for nm_, lst in list(canon_to_pids.items()):
            canon_to_pids[nm_] = [t for t in lst if t[0] not in xref_dropped]

    # ── lookback pass — recover 官衔-glued earlier appearances ──
    # An auto-card's 卷 set comes from NER, which can't split a name glued to a long
    # 官衔 first-introduction (御史中丞兼尚书右丞夏侯孜), so the card anchors to a LATER
    # 卷 where the name stands bare (夏侯孜曰) and the genuine first appearance is lost
    # — the person card then shows only the later 卷. Fix: for each card scan the 1–2
    # 卷 immediately before its earliest 卷 for a literal occurrence of a DISTINCTIVE
    # full-name surface (姓+名, ≥3 chars). If the name is there and no other person
    # claims it in that 卷, register the surface so the mention pass picks it up and
    # the card aggregates its true first appearance. Bounded to 2 卷 and gated on
    # «unclaimed» to stay precision-first (single-char 单名 2-char surfaces are left
    # out — adjacent-卷 homograph risk outweighs the recall).
    _juan_text_cache: dict[int, str] = {}

    def _juan_text(j):
        if j not in _juan_text_cache:
            jf2 = TEXT / f"juan_{j:03d}.json"
            _juan_text_cache[j] = ("\n".join(
                pp.get("main", "") for pp in json.loads(
                    jf2.read_text(encoding="utf-8"))["paragraphs"])
                if jf2.exists() else "")
        return _juan_text_cache[j]

    juan_set = set(JUANS)
    pid_surfaces: dict[str, set] = {}  # pid -> seed-accepted full-name surfaces
    for jn, smap in RULES.items():
        for s, sp in smap.items():
            if len(s) >= 3 and seed_mod._surname_of(s):
                pid_surfaces.setdefault(sp, set()).add(s)

    lookback_added = 0
    lookback_rebriefed = 0
    for pid_, p in list(by_id.items()):
        js = [j for j in p.get("juans", []) if j in juan_set]
        surfs = pid_surfaces.get(pid_)
        if not js or not surfs:
            continue
        j0 = min(js)
        earliest_added = None
        for jb in (j0 - 1, j0 - 2):
            if jb not in juan_set or jb in p["juans"]:
                continue
            txt = _juan_text(jb)
            for s in surfs:
                if RULES.get(jb, {}).get(s) is not None:
                    continue  # claimed by another person here → homograph, skip
                if s in txt:
                    RULES.setdefault(jb, {})[s] = pid_
                    if jb not in p["juans"]:
                        p["juans"].append(jb)
                    lookback_added += 1
                    if earliest_added is None or jb < earliest_added:
                        earliest_added = jb
        # A2: the seed-time brief/floruit anchor to the OLD first 卷 (j0). When lookback
        # proves an earlier appearance, re-anchor the auto card's 见于卷 locator + floruit
        # start to that earlier 卷 so the card reports the person's true first appearance
        # (夏侯孜: 卷250/860 → 卷249/859). Only auto cards whose brief is the program-minted
        # 见于卷NNN locator are touched — never reviewed/gloss/wiki-enriched briefs.
        if earliest_added is not None and earliest_added < j0 \
                and str(pid_).startswith("a:"):
            jm = gloss_meta.get(earliest_added, {})
            cs = jm.get("ce_start")
            m = re.search(r"见于卷(\d{3})", p.get("brief", ""))
            if m:
                ystr = ""
                if cs is not None:
                    ystr = f"前{-cs}年" if cs < 0 else f"{cs}年"
                new_loc = f"见于卷{earliest_added:03d}" + (f"（{ystr}）" if ystr else "")
                old_loc = re.search(r"见于卷\d{3}（[^）]*）|见于卷\d{3}", p["brief"])
                if old_loc:
                    p["brief"] = p["brief"].replace(old_loc.group(0), new_loc, 1)
                    if p.get("identity") and "见于卷" in p["identity"]:
                        old_loc_i = re.search(
                            r"见于卷\d{3}（[^）]*）|见于卷\d{3}", p["identity"])
                        if old_loc_i:
                            p["identity"] = p["identity"].replace(
                                old_loc_i.group(0), new_loc, 1)
                    if cs is not None:
                        fl = p.get("floruit") or [None, None]
                        if fl[0] is None or cs < fl[0]:
                            fl[0] = cs
                        p["floruit"] = fl
                    lookback_rebriefed += 1

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mentions").mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    # appearances[pid] = { (juan, pid_para): {juan, pid, ce_year, source} }
    appearances: dict[str, dict[tuple, dict]] = {}
    per_juan_counts: dict[int, tuple[int, int]] = {}
    anaphora_emitted = 0
    role_emitted = 0
    gloss_emitted = 0
    feng_emitted = 0
    relations_all: list = []

    for juan_no in JUANS:
        jf = TEXT / f"juan_{juan_no:03d}.json"
        if not jf.exists():
            continue
        juan = json.loads(jf.read_text(encoding="utf-8"))
        surfaces = surfaces_for(juan_no)
        admitted = ANAPHORA_RULES.get(juan_no, set()) | minted_admit.get(juan_no, set())
        char_anchor: dict[str, str] = {
            gch: pid_ for gch, pid_ in minted_anchor.get(juan_no, {}).items()
            if pid_ in by_id
        }  # given-char -> nearest antecedent person_id (卷-local)
        cue_idx = build_role_cue_index(juan["paragraphs"])  # P3 succession split
        mentions = []
        for p_idx, para in enumerate(juan["paragraphs"]):
            pid = para["id"]
            ce = para.get("ce_year")
            main_text = para.get("main", "")
            consumed = [False] * len(main_text)
            # Section-local anaphora gate: a numbered section (①②…⑳) resets the
            # 省称 anchor table so a stale full name in an earlier section can't bind
            # a bare given char (云 in 云大破蛮 ≠ 张云). Full names recur per section.
            if main_text[:1] and "\u2460" <= main_text[0] <= "\u2473":
                char_anchor.clear()
            # rc4 封号-glue runs FIRST: reserve 「封号+given」 spans (任城王澄→元澄) so the
            # alias matcher cannot form the false 王X glue over the same characters.
            feng_hits = extract_titleglue(main_text, ce, consumed,
                                          RULES.get(juan_no, {}), by_id, dict(surfaces))
            for (s, e, pid_, surf) in feng_hits:
                mentions.append({"pid": pid, "ce_year": ce, "source": "main",
                                 "start": s, "end": e, "surface": surf,
                                 "person_id": pid_, "kind": "feng",
                                 "confidence": by_id[pid_].get("confidence", "reviewed")})
                seen_ids.add(pid_)
                feng_emitted += 1
            # alias matches that overlap a reserved 封号 span are the suppressed glue.
            alias_hits = [h for h in extract(main_text, surfaces)
                          if not any(consumed[h[0]:h[1]])]
            for (s, e, pid_, surf) in alias_hits:
                for k in range(s, e):
                    consumed[k] = True
                mentions.append({"pid": pid, "ce_year": ce, "source": "main",
                                 "start": s, "end": e, "surface": surf,
                                 "person_id": pid_, "kind": "alias",
                                 "confidence": by_id[pid_].get("confidence", "reviewed")})
                seen_ids.add(pid_)
            # Wave 5 P3 — role appellation (吴主/魏主…) → reigning monarch by ce_year,
            # same-year successions split by reading position around the accession cue.
            for (s, e, pid_, surf) in extract_roles(main_text, ce, consumed, by_id,
                                                    p_idx, cue_idx):
                mentions.append({"pid": pid, "ce_year": ce, "source": "main",
                                 "start": s, "end": e, "surface": surf,
                                 "person_id": pid_, "kind": "role",
                                 "confidence": by_id[pid_].get("confidence", "reviewed")})
                seen_ids.add(pid_)
                role_emitted += 1
            # Wave 5 P2 — single-char 省称 anaphora on the 原文, each bound to its
            # nearest preceding full-name antecedent (disambiguates 元胄 vs 宇文胄).
            if admitted:
                anchor_events = []
                for (s, e, pid_, surf) in alias_hits:
                    g = given_of.get(pid_)
                    if g:
                        anchor_events.append((e, g, pid_))
                for (s, e, pid_, surf) in feng_hits:   # 任城王澄 anchors 澄→元澄
                    g = given_of.get(pid_)
                    if g:
                        anchor_events.append((e, g, pid_))
                anchor_events.sort()
                for (s, e, pid_, surf) in extract_anaphora(
                        main_text, admitted, consumed, char_anchor, anchor_events, ce, by_id):
                    mentions.append({"pid": pid, "ce_year": ce, "source": "main",
                                     "start": s, "end": e, "surface": surf,
                                     "person_id": pid_, "kind": "anaphora",
                                     "confidence": by_id[pid_].get("confidence", "reviewed")})
                    seen_ids.add(pid_)
                    anaphora_emitted += 1
            # Wave 5 P5 — RC-2b 「X，Y之Z也」 gloss: recover Y=姓+Y (戣→孔戣) and a
            # kinship edge. Runs on 原文 after alias/role/anaphora consumption.
            g_ments, g_rels = extract_gloss(main_text, RULES.get(juan_no, {}),
                                            canon_to_pids, juan_no, by_id, consumed)
            for (s, e, pid_, surf) in g_ments:
                mentions.append({"pid": pid, "ce_year": ce, "source": "main",
                                 "start": s, "end": e, "surface": surf,
                                 "person_id": pid_, "kind": "gloss",
                                 "confidence": by_id[pid_].get("confidence", "reviewed")})
                seen_ids.add(pid_)
                gloss_emitted += 1
            for (px, z, py, xs, ys) in g_rels:
                seen_ids.add(px)
                seen_ids.add(py)
                relations_all.append({"subject": px, "kin": z, "object": py,
                                      "juan": juan_no, "para": pid,
                                      "x": xs, "y": ys})
            for ni, note in enumerate(para.get("notes", [])):
                for (s, e, pid_, surf) in extract(note.get("text", ""), surfaces):
                    mentions.append({"pid": pid, "ce_year": ce, "source": "hu",
                                     "note_index": ni, "start": s, "end": e,
                                     "surface": surf, "person_id": pid_, "kind": "alias",
                                     "confidence": by_id[pid_].get("confidence", "reviewed")})
                    seen_ids.add(pid_)

        # Per-(person, paragraph) appearance row, 原文 preferred over 胡注.
        for m in mentions:
            key = (juan_no, m["pid"])
            slot = appearances.setdefault(m["person_id"], {})
            cur = slot.get(key)
            if cur is None or (cur["source"] == "hu" and m["source"] == "main"):
                slot[key] = {"juan": juan_no, "pid": m["pid"],
                             "ce_year": m["ce_year"], "source": m["source"]}

        para_ids = {m["pid"] for m in mentions}
        per_juan_counts[juan_no] = (len(mentions), len(para_ids))
        _write_text_retry(OUT / "mentions" / f"juan_{juan_no:03d}.json",
            json.dumps({"juan_no": juan_no, "version": 1, "mentions": mentions},
                       ensure_ascii=False, indent=2))

    # ── people.json — only people actually matched somewhere ──
    missing_brief = [pid for pid in seen_ids if not by_id[pid].get("brief")]
    if missing_brief:
        raise SystemExit(f"BRIEF missing for shipped people (would leak spoilers): {sorted(missing_brief)}")

    people_out = []
    for p in PEOPLE_MERGED:
        if p["id"] not in seen_ids:
            continue
        fl = p["floruit"]
        people_out.append({
            "id": p["id"], "canonical_name": p["canonical_name"],
            "names": [{"text": p["canonical_name"], "type": "name"}]
                     + [{"text": n, "type": "alias"} for n in p["names"]],
            "dynasty": p["dynasty"], "era_hint": p["era_hint"],
            "floruit": {"start": fl[0], "end": fl[1]},
            "brief": p["brief"], "identity": p["identity"],
            "confidence": p.get("confidence", "reviewed"),
        })

    # ── appearances.json — cross-卷 index in reading order ──
    appearances_out = {}
    for pid, slot in appearances.items():
        rows = sorted(slot.values(), key=lambda r: (r["juan"], r["pid"]))
        appearances_out[pid] = rows

    _write_text_retry(OUT / "people.json",
        json.dumps({"version": 1, "people": people_out}, ensure_ascii=False, indent=2))
    _write_text_retry(OUT / "appearances.json",
        json.dumps({"version": 1, "appearances": appearances_out},
                   ensure_ascii=False, indent=2))

    # ── relations.json — RC-2b kinship edges (subject IS object's <kin>) ──
    shipped_ids = {p["id"] for p in people_out}
    relations_out = [r for r in relations_all
                     if r["subject"] in shipped_ids and r["object"] in shipped_ids]
    _write_text_retry(OUT / "relations.json",
        json.dumps({"version": 1, "relations": relations_out},
                   ensure_ascii=False, indent=2))

    covered = [j for j in JUANS if j in per_juan_counts]
    total_mentions = sum(c[0] for c in per_juan_counts.values())
    _write_text_retry(OUT / "manifest.json",
        json.dumps({"version": 1,
                    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "juans": covered, "people_count": len(people_out),
                    "mention_count": total_mentions}, ensure_ascii=False, indent=2))

    # ── report ──
    n_reviewed = sum(1 for p in people_out if p["confidence"] == "reviewed")
    n_auto = len(people_out) - n_reviewed
    print(f"people shipped: {len(people_out)} (reviewed {n_reviewed} / auto {n_auto})"
          f"   total mentions: {total_mentions}   ambiguous surfaces dropped: {SEED_STATS['ambiguous_dropped']}")
    print(f"卷 covered: {len(covered)} / {len(JUANS)}")
    print(f"single-char 省称 anaphora emitted: {anaphora_emitted}"
          f"   (candidate chars admitted: {SEED_STATS['anaphora_char_admitted']})")
    print(f"role appellation (语境称谓) emitted: {role_emitted}"
          f"   (monarchs mapped: {len(ROLE_NAME_TO_PID)})")
    print(f"RC-2b gloss recall emitted: {gloss_emitted}"
          f"   kinship relations: {len(relations_out)}"
          f"   new cards minted: {gloss_new_cards}")
    print(f"rc4 封号-glue (titleglue) emitted: {feng_emitted}")
    print(f"rc5 谥号/封号 fragment cards dropped: {_FRAG_DROPPED}")
    print(f"lookback 卷 surfaces registered: {lookback_added}")
    print(f"lookback brief/floruit re-anchored: {lookback_rebriefed}")
    print(f"xref window-merge (见N卷): {xref_merged} later windows folded")
    print(f"book-enrich 字 briefs: {briefs_enriched}")
    print(f"title-glue aliases bound: {SEED_STATS['glue_bound']}"
          f"   missing canonical (cast to add): {SEED_STATS['glue_missing']}")
    hand_ids = {p["id"] for p in PEOPLE_MERGED if p.get("confidence") == "reviewed"}
    unmatched = sorted(hand_ids - seen_ids)
    if unmatched:
        print(f"  reviewed-but-unmatched ({len(unmatched)}):",
              ", ".join(by_id[u]["canonical_name"] for u in unmatched))


if __name__ == "__main__":
    main()
