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
import json, datetime, hashlib, re
from pathlib import Path

from cast import PEOPLE
import seed as seed_mod
import reigns as reigns_mod

JUANS = list(range(1, 295))  # all 294 卷 (hand 'reviewed' + auto seed layer)
REPO = Path(__file__).resolve().parents[2]
TEXT = REPO / "web" / "public" / "text"
OUT = TEXT / "persons"

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


def extract_anaphora(text, admitted, consumed, char_anchor, anchor_events):
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
    ev_i, nev = 0, len(anchor_events)
    for i, ch in enumerate(text):
        # establish every full-name antecedent that ends at or before this char
        while ev_i < nev and anchor_events[ev_i][0] <= i:
            _, gc, pidp = anchor_events[ev_i]
            char_anchor[gc] = pidp
            ev_i += 1
        if ch not in admitted or consumed[i]:
            continue
        pid_ = char_anchor.get(ch)
        if pid_ is None:        # no antecedent seen yet → suppress
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
        if not (right in _ANAPHORA_RIGHT or left in _ANAPHORA_LEFT):
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
_KIN_ALT = "|".join(sorted(_KINSHIP_PATRILINEAL, key=len, reverse=True))
_GLOSS_RE = re.compile(
    r"([\u4e00-\u9fff]{1,3})，([\u4e00-\u9fff]{1,4})之(" + _KIN_ALT + r")也"
)
_GLOSS_BOUNDARY = set("。；，！？、：")


def _juan_gap(juans, j):
    return min((abs(a - j) for a in juans), default=10 ** 6)


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


def build_gloss_cards(juans, text_dir, rules, canon_to_pids, by_id,
                      people_merged, meta, allowed):
    """RC-2b pre-pass — generative recall. When a 「X，Y之Z也」 gloss reconstructs
    Y = 姓(X)+Y to a name that has NO card (孔戣, only ever written 省姓 as 戣), mint
    one, anchored to the gloss 卷 (and the 胡注 cross-ref 卷 when given). Strong
    precision gates: X must resolve to a real person in this 卷, the kin term is
    patrilineal (shared surname), and Y+姓 must not be a common non-person word.
    Returns the number of cards created; mutates by_id/canon_to_pids/people_merged."""
    created: dict[str, dict] = {}
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
            for m in _GLOSS_RE.finditer(mt):
                if m.start(1) > 0 and mt[m.start(1) - 1] not in _GLOSS_BOUNDARY:
                    continue
                x_surf, y_surf = m.group(1), m.group(2)
                if not _gloss_y_ok(y_surf):
                    continue
                pid_x = rule_map.get(x_surf)
                if not pid_x:
                    continue
                surname = seed_mod._surname_of(by_id[pid_x]["canonical_name"])
                if not surname:
                    continue
                if seed_mod._surname_of(y_surf) and y_surf in canon_to_pids:
                    continue  # already a full-name card → handled by binding pass
                y_full = surname + y_surf
                if y_full in canon_to_pids or len(y_full) < 2 or len(y_full) > 4:
                    continue
                if seed_mod.bad_auto_surface(y_full) or \
                        y_full in seed_mod.COMMON_WORD_NONPERSON:
                    continue
                xref = _hu_xref_juan(para.get("notes"), y_full, allowed)
                if y_full in created:
                    for j in {juan_no, xref}:
                        if j and j not in created[y_full]["juans"]:
                            created[y_full]["juans"].append(j)
                    continue
                dyn = (meta.get(juan_no, {}).get("dynasty") or "").strip()
                cs = meta.get(juan_no, {}).get("ce_start")
                juans_c = sorted({j for j in (juan_no, xref) if j})
                j0 = min(juans_c)
                card = {
                    "id": seed_mod._auto_id(y_full, 9000 + len(created)),
                    "canonical_name": y_full,
                    "names": [],
                    "dynasty": dyn or "—",
                    "era_hint": f"{dyn}人物" if dyn else "人物",
                    "floruit": [cs, cs] if cs else [None, None],
                    "brief": f"{dyn + '·' if dyn else ''}见于卷{j0:03d}（家世胡注）。",
                    "identity": f"{dyn + '·' if dyn else ''}见于卷{j0:03d}（家世胡注）。",
                    "match": [y_full],
                    "juans": juans_c,
                    "confidence": "high",
                }
                created[y_full] = card
    for y_full, card in created.items():
        people_merged.append(card)
        by_id[card["id"]] = card
        canon_to_pids.setdefault(y_full, []).append(
            (card["id"], set(card["juans"])))
    return len(created)


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
    gloss_meta = seed_mod.juan_meta()
    gloss_new_cards = build_gloss_cards(JUANS, TEXT, RULES, canon_to_pids, by_id,
                                        PEOPLE_MERGED, gloss_meta, set(JUANS))

    # book-enrich — courtesy-name apposition into briefs (after gloss cards exist so
    # newly-minted 家世 cards also get their 字 when the text introduces one).
    briefs_enriched = enrich_briefs(JUANS, TEXT, PEOPLE_MERGED)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mentions").mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    # appearances[pid] = { (juan, pid_para): {juan, pid, ce_year, source} }
    appearances: dict[str, dict[tuple, dict]] = {}
    per_juan_counts: dict[int, tuple[int, int]] = {}
    anaphora_emitted = 0
    role_emitted = 0
    gloss_emitted = 0
    relations_all: list = []

    for juan_no in JUANS:
        jf = TEXT / f"juan_{juan_no:03d}.json"
        if not jf.exists():
            continue
        juan = json.loads(jf.read_text(encoding="utf-8"))
        surfaces = surfaces_for(juan_no)
        admitted = ANAPHORA_RULES.get(juan_no, set())
        char_anchor: dict[str, str] = {}  # given-char -> nearest antecedent person_id (卷-local)
        cue_idx = build_role_cue_index(juan["paragraphs"])  # P3 succession split
        mentions = []
        for p_idx, para in enumerate(juan["paragraphs"]):
            pid = para["id"]
            ce = para.get("ce_year")
            main_text = para.get("main", "")
            alias_hits = extract(main_text, surfaces)
            consumed = [False] * len(main_text)
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
                anchor_events.sort()
                for (s, e, pid_, surf) in extract_anaphora(
                        main_text, admitted, consumed, char_anchor, anchor_events):
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
        (OUT / "mentions" / f"juan_{juan_no:03d}.json").write_text(
            json.dumps({"juan_no": juan_no, "version": 1, "mentions": mentions},
                       ensure_ascii=False, indent=2), encoding="utf-8")

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

    (OUT / "people.json").write_text(
        json.dumps({"version": 1, "people": people_out}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (OUT / "appearances.json").write_text(
        json.dumps({"version": 1, "appearances": appearances_out},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # ── relations.json — RC-2b kinship edges (subject IS object's <kin>) ──
    shipped_ids = {p["id"] for p in people_out}
    relations_out = [r for r in relations_all
                     if r["subject"] in shipped_ids and r["object"] in shipped_ids]
    (OUT / "relations.json").write_text(
        json.dumps({"version": 1, "relations": relations_out},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    covered = [j for j in JUANS if j in per_juan_counts]
    total_mentions = sum(c[0] for c in per_juan_counts.values())
    (OUT / "manifest.json").write_text(
        json.dumps({"version": 1,
                    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "juans": covered, "people_count": len(people_out),
                    "mention_count": total_mentions}, ensure_ascii=False, indent=2),
        encoding="utf-8")

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
