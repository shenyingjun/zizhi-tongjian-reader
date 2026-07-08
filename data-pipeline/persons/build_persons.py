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

import os

from cast import PEOPLE
import seed as seed_mod
import reigns as reigns_mod
import pos_giv

JUANS = list(range(1, 295))  # all 294 卷 (hand 'reviewed' + auto seed layer)
# Iteration hook: ZTJ_ONLY_JUANS="251,252" limits the build to a few 卷 for fast
# validation. Leave UNSET for a real full build (it also shrinks people.json).
_only = os.environ.get("ZTJ_ONLY_JUANS")
if _only:
    JUANS = [int(x) for x in _only.replace(",", " ").split() if x.strip()]
REPO = Path(__file__).resolve().parents[2]
TEXT = REPO / "web" / "public" / "text"
OUT = TEXT / "persons"
LLM_ANN = Path(__file__).resolve().parent / "llm_annotations"
# Real surnames that are absent from seed.SURNAMES (kept out of the general alias
# pass to avoid FP-storms — 云/凌/洪 also mean say/encroach/flood, 许 = "to allow"),
# but safe to admit in the LLM tier, which re-verifies text-occurrence and mints the
# specific full name.
LLM_EXTRA_SURNAMES = set("元凌楼洪云柳鲁穆归楚许")
_BIND_STATS = {"bound": 0, "minted": 0, "para_scoped": 0, "anaphora": 0}
# Populated by the binding tier (build_gloss_cards) for context-dependent 封号 that are
# reassigned within a single 卷 (胶东王=刘雄渠 in the 七国之乱 paras, 刘彻/刘寄 in the
# accession paras). Maps juan_no -> [(lo, hi, surface, person_id)] with inclusive
# paragraph-id range [lo,hi]; consumed by the emission loop as a paragraph-scoped
# overlay so a rotating 封号 never mislinks outside its window (precision-first).
_PARA_BINDINGS: dict[int, list] = {}
# Populated by the binding tier for SINGLE-CHAR 省称 bindings (bare given char 卬→刘卬,
# 省姓 single 戣→孔戣). List of (juan_no, char, person_id). These are NOT registered as
# whole-卷 alias surfaces (the alias pass skips <2-char and a bare char like 遂/农 is a
# homograph storm); instead they feed the GATED single-char anaphora pass via
# minted_admit/minted_anchor — the LLM supplies the antecedent identity while the
# deterministic context / COMMON_BIGRAMS / lifespan / section-reset gates still decide
# which occurrences actually bind (precision-first).
_BIND_ANAPHORA: list = []


def _load_llm_ann(juan_no):
    """Load per-卷 LLM annotations. Preferred format is JSONL (juan_NNN.jsonl), one
    person per line: {"name", "aliases"[], "role", "confidence", "evidence"}. Falls
    back to the legacy TSV (name<TAB>confidence<TAB>evidence) when no JSONL exists.
    Returns a list of normalized dicts with keys name/aliases/role/confidence."""
    jl = LLM_ANN / f"juan_{juan_no:03d}.jsonl"
    if jl.exists():
        out = []
        for raw in jl.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = (rec.get("name") or "").strip()
            if not name:
                continue
            out.append({
                "name": name,
                "aliases": [a.strip() for a in rec.get("aliases", []) if a and a.strip()],
                "role": (rec.get("role") or "").strip(),
                "confidence": (rec.get("confidence") or "high").strip(),
            })
        return out
    tsv = LLM_ANN / f"juan_{juan_no:03d}.tsv"
    if tsv.exists():
        out = []
        for raw in tsv.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            name = cols[0].strip()
            if not name:
                continue
            conf = cols[1].strip() if len(cols) > 1 else "high"
            out.append({"name": name, "aliases": [], "role": "", "confidence": conf})
        return out
    return []


def _load_llm_binding(juan_no):
    """Load per-卷 recall BINDING records from llm_annotations/juan_NNN.jsonl.

    A binding record ({"type":"binding","surface":"吴王","canonical":"刘濞",
    "dynasty":"汉","role":"吴王，七国之乱首","para_range":[lo,hi]?,"evidence":...})
    tells the build to register the SURFACE (a 封号/官职/省称 the deterministic scanner
    misses) as pointing at CANONICAL — the person's real 姓名 — in THIS 卷 only. This
    is what recovers 封号-only persons (吴王→刘濞, where 刘濞 never appears literally) and
    context 省称 (文泰→曲文泰). Offsets stay deterministic: the pipeline scans `surface`
    in the 卷 text and re-verifies every hit; the LLM only supplies the surface→person
    identity. `para_range` (inclusive [lo,hi] paragraph ids) scopes context-dependent
    封号 (广川王=刘越 in one 段, 刘彭祖 elsewhere); omit for a whole-卷-stable binding.

    Returns a list of normalized dicts. The `name`-tier `_load_llm_ann` loader ignores
    these lines (no `name`), so persons + bindings + vetoes coexist in one v4 file."""
    jl = LLM_ANN / f"juan_{juan_no:03d}.jsonl"
    if not jl.exists():
        return []
    out = []
    for raw in jl.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") != "binding":
            continue
        surf = (rec.get("surface") or "").strip()
        canon = (rec.get("canonical") or "").strip()
        if not surf or not canon:
            continue
        pr = rec.get("para_range")
        if isinstance(pr, list) and len(pr) == 2:
            pr = (int(pr[0]), int(pr[1]))
        else:
            pr = None
        out.append({"surface": surf, "canonical": canon,
                    "dynasty": (rec.get("dynasty") or "").strip(),
                    "role": (rec.get("role") or "").strip(),
                    "para_range": pr})
    return out


def _load_llm_veto(juan_no):
    """Load per-卷 precision VETO surfaces from llm_annotations/juan_NNN.jsonl.

    A veto record ({"type":"veto","surface":"胡可","reason":...}) tells the build to
    SUPPRESS every mention of that surface in THIS 卷 — used to kill audit-confirmed
    non-person spans (文言 phrases like 胡可/王必欲, title/verb-boundary garbage) and
    context-local mis-binds. Veto is delete-only and 卷-scoped, so it can never remove
    a surface from a 卷 where the LLM did not audit it — the precision-first guarantee
    (a missing underline beats a wrong one) is preserved by construction.

    The regular `_load_llm_ann` minting loader ignores these lines (they carry no
    `name`), so a v4 full-cast file mixing person + veto records stays drop-in
    compatible. Returns a set of vetoed surface strings for this 卷."""
    jl = LLM_ANN / f"juan_{juan_no:03d}.jsonl"
    if not jl.exists():
        return set()
    out = set()
    for raw in jl.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") == "veto" or rec.get("veto"):
            surf = (rec.get("surface") or rec.get("name") or "").strip()
            if surf:
                out.add(surf)
    return out


def _load_llm_card(juan_no):
    """Load per-卷 CARD-curation records from llm_annotations/juan_NNN.jsonl (v4).

    A card record ({"type":"card","canonical":"慕容德","dynasty":"后燕","brief":"...",
    "merge_into":"慕容盛"?,"evidence":...}) carries metadata-only fixes for the audit's
    card-quality gaps: a **dynasty** relabel (十六国 actors first seen in a 晋 卷 are
    mislabeled 晋), a **book-only one-line brief** (replaces the placeholder 见于卷NNN),
    and an evidence-gated **merge_into** (fold a duplicate/省称 card into the survivor
    named by `merge_into`). Card records touch ONLY card fields — never spans/offsets —
    so the precision-first offset guarantee is untouched by construction. Keyed by the
    person's `canonical` name (resolved to the nearest in-卷 card at apply time).

    The minting loader `_load_llm_ann` ignores these lines (no `name`), so person / veto
    / binding / card records coexist in one v4 file. Returns a list of normalized dicts."""
    jl = LLM_ANN / f"juan_{juan_no:03d}.jsonl"
    if not jl.exists():
        return []
    out = []
    for raw in jl.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") != "card":
            continue
        canon = (rec.get("canonical") or "").strip()
        if not canon:
            continue
        out.append({
            "canonical": canon,
            "dynasty": (rec.get("dynasty") or "").strip(),
            "brief": (rec.get("brief") or "").strip(),
            "merge_into": (rec.get("merge_into") or "").strip(),
            "juan": juan_no,
        })
    return out


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


# Title-vs-surname precision slice — drop 王/公/侯/君/主 TITLE-MISREAD auto cards.
# 王公侯君主 double as princely titles, so NER segments 「{封地}王{名}」 (河南王炽磐,
# 临川王义庆, 南阳王宝炬) into a bogus surname-王 card (王炽磐=乞伏炽磐, 王义庆=刘义庆,
# 王宝炬=元宝炬). This is a corpus-wide, source-agnostic centralization of the per-pass
# title guards: scan every occurrence of a title-headed AUTO card's canonical surface and
# classify the char immediately to its left. A surname START is legitimate only after a
# clause boundary, an appointment/action cue (以/为/拜/除/封/遣/诏…), or an office-name
# final char (尚书令王导, 大将军王敦 → 令/军). Drop the card only when it NEVER appears in
# such a clean context AND appears ≥1 time inside a fief/title context. Real 王/公/侯
# surnames (王导/王敦/王猛/王景/刘义庆) always have ≥1 clean start, so they are spared;
# precision-first means a princely-title misread is dropped (missing < wrong), not
# re-attributed. Pure data-driven, no hand-list, mirrors _drop_fragment_cards.
_TITLE_SURNAME_CHARS = set("王公侯君主")
_TITLE_CLEAN_BOUNDARY = set("。！？；、，：「」『』（）〈〉《》【】　 \n\t")
_TITLE_CLEAN_CUE = set("以为與与及拜除封署領领迁徙赠谥遣诏召命立使遣徵征辟用授进进黜废")
# office-name FINAL chars: a surname after 尚书令/大将军/太常博士/金紫光禄大夫/右补阙
# is a real name start (令/军/史/士/夫/阙…), never a fief title.
_TITLE_CLEAN_OFFICE = set("令史书郎军尉丞卿牧守相傅师保监将中仆射镇空徒马尹卫士夫阙议詹")
_TITLE_CLEAN_LEFT = _TITLE_CLEAN_BOUNDARY | _TITLE_CLEAN_CUE | _TITLE_CLEAN_OFFICE
_TITLE_BANNED: set = set()  # title-headed surfaces proven to be princely-title misreads


def _drop_title_misreads():
    # Corpus-wide enumeration: every [王/公/侯/君/主]+given surface (len 2–3) that
    # occurs in the text. Classify the char immediately to the left of each occurrence.
    # A surname START is legitimate only after a clause boundary, an appointment/action
    # cue, an office-name final char, or at chunk start. A surface that NEVER starts a
    # name that way (always 「{封地}王{名}」: 河南王炽磐, 平阿侯仁, 邵陵王纶) is a
    # princely-title misread → ban it everywhere. This is source-agnostic: it drops the
    # seed/NER cards AND blocks the gloss pass from re-minting the same surface in main().
    HANZI_LO, HANZI_HI = "\u4e00", "\u9fff"
    clean: dict = collections.defaultdict(int)
    title: dict = collections.defaultdict(int)
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
                for i in range(n):
                    if t[i] not in _TITLE_SURNAME_CHARS:
                        continue
                    left = t[i - 1] if i > 0 else ""
                    is_clean = (left == "" or left in _TITLE_CLEAN_LEFT)
                    for L in (2, 3):
                        if i + L <= n:
                            s = t[i:i + L]
                            if all(HANZI_LO <= c <= HANZI_HI for c in s):
                                (clean if is_clean else title)[s] += 1
    banned = {s for s in title if clean[s] == 0 and title[s] > 0}
    # title-char + a kinship word (王弟/公兄 = "the king's younger brother") is never a
    # given name. 子/孙 are excluded: 王孙/公孙 are real 复姓, 公子/王子 real names.
    for s in list(clean) + list(title):
        if len(s) == 2 and s[1] in "弟兄":
            banned.add(s)
    _TITLE_BANNED.update(banned)
    # drop the seed/NER cards whose canonical is a banned surface
    drop_ids = {p["id"] for p in PEOPLE_MERGED
                if str(p["id"]).startswith("a:") and p["canonical_name"] in banned}
    if not drop_ids:
        return 0
    PEOPLE_MERGED[:] = [p for p in PEOPLE_MERGED if p["id"] not in drop_ids]
    for surf_map in RULES.values():
        for surf in list(surf_map):
            if surf_map[surf] in drop_ids:
                del surf_map[surf]
    return len(drop_ids)


_FRAG_DROPPED = _drop_fragment_cards()
_TITLE_DROPPED = _drop_title_misreads()


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
    # 足 (王足/马足) — 何足 = 成语「何足…」(how is it worth), never 马足(=horse-foot)+anaphora.
    "何足",
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


def extract_anaphora(text, admitted, consumed, char_anchor, sec_owners, anchor_events, ce, by_id):
    """Wave 5 P2 — single-char 省称 matches in one main-text blob, each bound to its
    NEAREST preceding full-name antecedent (the person most recently named whose
    given char == this char). Gates:
      * the char is an admitted candidate for this 卷 (admitted set),
      * an antecedent for that char already exists in char_anchor (else suppress —
        precision-first, never guess without an anchor),
      * NODE-LOCAL COLLISION GUARD: ≥2 DISTINCT people have been named in full for
        this char earlier in the current 节 (sec_owners[ch]) → ambiguous, drop. This
        replaced a blunt whole-卷 endswith filter (see seed.build_anaphora_rules); it
        only fires where the ambiguity is actually local, so a char with one owner in
        this 节 (even if a same-tail person exists in another 节) still binds.
      * the position is not consumed by a longer alias match,
      * a common-word bigram / modifier / clean-surname guard (坚守, 不坚, 杨惠 …),
      * a positive person-context gate (agent before a verb, or object after a
        person-taking verb).
    char_anchor is mutated in place (carried across paragraphs within a 卷); sec_owners
    is likewise carried across paragraphs but RESET at each 节 lead by the caller.
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
            sec_owners.setdefault(gc, set()).add(pidp)  # 节-local distinct owner tally
            ev_i += 1
        if ch not in admitted or consumed[i]:
            continue
        if fixed_mask[i]:       # inside a fixed 官名 (节度使…) → title fragment, not a 省称
            continue
        pid_ = char_anchor.get(ch)
        if pid_ is None:        # no antecedent seen yet → suppress
            continue
        if len(sec_owners.get(ch, ())) >= 2:  # ≥2 full-named owners in THIS 节 → ambiguous
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


# ── Wave 6 · POS-gated structural 省称 resolver ──────────────────────────────
# A recall-forward second 省称 pass that complements the curated extract_anaphora
# above. For EVERY person whose full name appears in a 卷 it binds their bare given
# to the closest antecedent, sidestepping disambiguation structurally:
#   * antecedent = closest full name in the SAME paragraph (同段), else the UNIQUE
#     owner of that given within the 节 (跨段, single-owner only);
#   * never crosses a 节 (a ①..⑳ numbered paragraph opens a fresh 节);
#   * a single-char given (元胄→胄) is admitted ONLY where the Classical-Chinese
#     UPOS tagger labels PROPN|NameType=Giv at that offset (giv_map) — this is what
#     separates the real 省称 from homograph function words / verbs;
#   * a clean 双名 given (行密→行密, no 封号 tail) is admitted without POS.
# It is purely additive: every span already emitted by the alias/feng/curated-
# anaphora/gloss passes (exist_span) is skipped, so it never mis-binds an existing
# mention — it only fills the recall gap the curated allow-list left behind.
_TITLE_END_CHARS = set("王侯公伯子男君后主帝卿相将")


def _sec_num(mt: str):
    """Section (节) number 1..N from a paragraph lead, or None if not a section head.

    资治通鉴 numbers 节 within each year. The digitiser uses circled glyphs where
    Unicode provides them — ①..⑳ (U+2460..2473), ㉑..㉟ (U+3251..325F), ㊱..㊿
    (U+32B1..32BF), i.e. 1..50 — and falls back to plain ASCII digits for 51+
    (a single year can reach 节90), written as the number immediately followed by a
    Chinese character (e.g. 「51陇西处士王嘉…」). The old resolver saw only ①..⑳, so
    everything from ㉑ on — and every ASCII 51+ 节 — collapsed into the preceding
    section, producing incorrect cross-节 省称 references."""
    if not mt:
        return None
    o = ord(mt[0])
    if 0x2460 <= o <= 0x2473:
        return o - 0x2460 + 1        # ①..⑳
    if 0x3251 <= o <= 0x325F:
        return o - 0x3251 + 21       # ㉑..㉟
    if 0x32B1 <= o <= 0x32BF:
        return o - 0x32B1 + 36       # ㊱..㊿
    m = _SEC_ASCII_RE.match(mt)      # 51.. — ASCII digits directly before a 汉字
    if m:
        return int(m.group(1))
    return None


_SEC_ASCII_RE = re.compile(r"^([0-9]{1,3})[\u4e00-\u9fff]")


def _is_sec_lead(mt: str) -> bool:
    """True if the paragraph text opens a fresh 节 (see _sec_num)."""
    return _sec_num(mt) is not None

# Bare appellations that are titles, not given names. A person carded under a
# royal/posthumous style (窦太后, 魏惠王, 秦武王, 成侯) has a surname-stripped
# "given" that is really a title — binding it as a bare 省称 is both ambiguous and
# rejected by validate_persons. Kept in sync with that validator's BANNED set.
_BANNED_APPELLATIONS = {
    "王", "公", "侯", "君", "太子", "公子", "孝公", "惠王", "武王", "文王",
    "威王", "成侯", "太后", "大王", "上", "帝",
}


def resolve_anaphora_pos(paras, present_pids, by_id, giv_map, exist_span, full_anchor=None):
    """Return [(para_id, ce_year, start, end, person_id, given)] new 省称 hits.

    present_pids — person_ids whose full name matched in this 卷.
    giv_map      — {para_id: set(offsets)} POS·Giv positions (single-char gate).
    exist_span   — {para_id: set((start,end))} spans already emitted (dedup).
    full_anchor  — {para_id: [(start,end,person_id,surface)]} already-tagged
                   full / title forms (e.g. 封号 后秦王苌) of present people. These
                   seed a 节-local antecedent even when the bare canonical (姚苌)
                   is absent from the section, so a 节 re-introduced only by a
                   title+given form still binds its bare givens. Stays 节-local."""
    full_anchor = full_anchor or {}
    pid2 = {}                                   # pid -> (canon, given, cls)
    given_to_pids = collections.defaultdict(set)
    two_char_givens = set()
    for pid in present_pids:
        card = by_id.get(pid)
        canon = card.get("canonical_name") if card else None
        if not canon or not (2 <= len(canon) <= 4):
            continue
        sn = seed_mod._surname_of_known(canon)
        if not sn:
            continue
        g = canon[len(sn):]
        if len(g) not in (1, 2):
            continue
        if g in _BANNED_APPELLATIONS:            # 太后/惠王/武王… are titles, not 省称
            continue
        # 封号 guard for the newly-enabled ambiguous-surname path: 平阳君 (阳君) is a
        # fief-lord title, not a 姓名 — its "given" is a 封号 tail, never a 省称. Only
        # applies to single-char ambiguous 姓 (clean-surname behaviour unchanged).
        if len(sn) == 1 and sn not in seed_mod.CLEAN_SURNAMES and g[-1] in "君王公侯":
            continue
        cls = "s1" if len(g) == 1 else ("title" if g[-1] in _TITLE_END_CHARS else "clean2")
        pid2[pid] = (canon, g, cls)
        given_to_pids[g].add(pid)
        if len(g) == 2:
            two_char_givens.add(g)
    if not pid2:
        return []

    # split into 节 at numbered paragraphs ①..㊿ and ASCII 51.. (see _sec_num).
    # A bare given never binds an antecedent in a different 节 — cross-节 reference
    # is treated as incorrect.
    sections, cur = [], []
    for p in paras:
        mt = p.get("main", "")
        if _is_sec_lead(mt) and cur:
            sections.append(cur)
            cur = []
        cur.append(p)
    if cur:
        sections.append(cur)

    cur_year = None
    year_of = {}
    for p in paras:
        if p.get("ce_year") is not None:
            cur_year = p["ce_year"]
        year_of[p["id"]] = cur_year

    out = []
    for sec in sections:
        sec_full = {}                           # given -> most-recent full-name pid in 节
        sec_owners = collections.defaultdict(set)
        for p in sec:
            mt = p.get("main", "")
            if not mt:
                continue
            pid_para = p["id"]
            raw_ce = p.get("ce_year")
            eff_ce = year_of.get(pid_para)
            events, fullspans = [], collections.defaultdict(list)
            for pid, (canon, g, cls) in pid2.items():
                for s in find_all(mt, canon):
                    events.append((s, 0, pid, g))
                    fullspans[pid].append((s, s + len(canon)))
            # Seed anchors from already-tagged title/full forms (后秦王苌 …) whose
            # surface ends in the person's given, so the 节 they appear in gains a
            # 苌->pid antecedent even without the bare canonical 姚苌 present. Their
            # span is registered in fullspans so the given INSIDE the title is not
            # itself re-emitted as a bare 省称.
            for (a_s, a_e, a_pid, a_surf) in full_anchor.get(pid_para, ()):
                info = pid2.get(a_pid)
                if not info or not a_surf.endswith(info[1]):
                    continue
                events.append((a_s, 0, a_pid, info[1]))
                fullspans[a_pid].append((a_s, a_e))
            for g, pids in given_to_pids.items():
                for s in find_all(mt, g):
                    e = s + len(g)
                    if any(a <= s and e <= b for pid in pids for (a, b) in fullspans.get(pid, [])):
                        continue
                    events.append((s, 1, None, g))
            events.sort(key=lambda t: (t[0], t[1]))
            para_full = {}
            done = set(exist_span.get(pid_para, ()))
            giv_here = giv_map.get(pid_para, ())
            for (s, typ, pid, g) in events:
                if typ == 0:
                    para_full[g] = pid
                    sec_full[g] = pid
                    sec_owners[g].add(pid)
                    continue
                e = s + len(g)
                if g in para_full:
                    rpid = para_full[g]
                elif g in sec_full and len(sec_owners[g]) == 1:
                    rpid = sec_full[g]
                else:
                    continue
                if (s, e) in done or any(a <= s and e <= b for (a, b) in done):
                    continue
                if pid2[rpid][2] == "s1":
                    if mt[s:s + 2] in two_char_givens:   # 胄 that opens a 双名 given → defer
                        continue
                    if s not in giv_here:                 # POS·Giv gate
                        continue
                if _lifespan_outside(by_id[rpid], eff_ce):
                    continue
                out.append((pid_para, raw_ce, s, e, rpid, g))
                done.add((s, e))
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
                # 袁公弁 looks like 封号(会稽王昱-style)+given, but here the char just before the
                # 爵 (公) is the surname 袁 and 袁公弁 (姓+爵medial+given) IS this person's canonical
                # — a 公/王-in-given misread, not a wrong-era homograph. Don't reserve; let the
                # alias pass tag the real person. (会稽王昱: char before 爵 is 稽, a 封号 char, so
                # 稽王昱 ≠ 王昱 → still reserved, and the 司马昱 occurrence is never mis-tagged.)
                cn = by_id[pid_w].get("canonical_name")
                if cn and rank_pos >= 1 and text[rank_pos - 1:false_e] == cn and rule_map.get(cn) == pid_w:
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
        x_surf, y_surf, z = m.group(1), m.group(2), m.group(3)
        if not _gloss_y_ok(y_surf):
            continue
        pid_x = rule_map.get(x_surf)
        # X must start a fresh clause — UNLESS it already resolves to a carded
        # multi-char full name. The boundary guard exists only to reject a mid-word
        # fragment of an UNKNOWN X; 「观察使崔彦曾，愼由之从子也」 sits mid-clause after an
        # office title yet 崔彦曾 is unambiguously a known person, so demanding a
        # boundary there silently drops the 愼由→崔愼由 glue.
        x_carded = pid_x in by_id and len(x_surf) >= 2
        if not x_carded and m.start(1) > 0 and text[m.start(1) - 1] not in _GLOSS_BOUNDARY:
            continue
        x_emit = None
        if not pid_x or pid_x not in by_id:
            # subject unmapped (省称 not NER'd). Governance rule: honor a LOCALLY-attested
            # full name for X before any cross-卷 reconstruction. Resolve X's surname from
            # the NEAREST preceding 姓+X / 复姓+X in THIS paragraph — the just-named
            # antecedent (魏宗正卿元树来奔。树，翼之弟也 → 元) — never a distant 同名 phantom
            # (刘树). The patrilineal ancestor Y=姓+Y is then reconstructed downstream; if X
            # and Y differ in surname, Y appears 姓名-in-full and takes its own 姓 (1132).
            surname = None
            pre_txt = text[:m.start(1)]
            idx = pre_txt.rfind(x_surf)
            while idx > 0:
                if idx >= 2 and pre_txt[idx - 2:idx] in seed_mod.COMPOUND:
                    surname = pre_txt[idx - 2:idx]
                    break
                if pre_txt[idx - 1] in seed_mod.SURNAMES:
                    cand = pre_txt[idx - 1]
                    # 王/公/侯/君/主 double as princely titles: 「彭城王雄」 is prince 雄
                    # (姓 元/司马…), not surname 王 — accept only at a fresh name boundary.
                    if cand in "王公侯君主" and not (
                            idx == 1 or pre_txt[idx - 2] in _GLOSS_BOUNDARY
                            or pre_txt[idx - 2] in "「『（〈《【　 "):
                        idx = pre_txt.rfind(x_surf, 0, idx)
                        continue
                    # 孙 after a 宗室 grandson prefix (皇孙/太孙/曾孙/从孙/嫡孙…) is the
                    # relation word, not surname 孙 — 皇孙暕 is 杨暕, never 孙暕.
                    if cand == "孙" and idx >= 2 and pre_txt[idx - 2] in "皇太世曾玄从嫡冢":
                        idx = pre_txt.rfind(x_surf, 0, idx)
                        continue
                    surname = cand
                    break
                idx = pre_txt.rfind(x_surf, 0, idx)
            if surname and (surname + x_surf) in canon_to_pids:
                xc = canon_to_pids[surname + x_surf]
                pid_x = min(xc, key=lambda pc: _juan_gap(pc[1], juan_no))[0]
                x_emit = (m.start(1), m.end(1))
            else:
                # fall back: both subject 姓+X and ancestor 姓+Y carded under exactly one
                # shared surname, the patrilineal rule binds both.
                surname = None
                sub_ok = []
                for s in seed_mod.SURNAMES:
                    xf, yf = s + x_surf, s + y_surf
                    if xf in canon_to_pids and yf in canon_to_pids:
                        sub_ok.append((s, canon_to_pids[xf], canon_to_pids[yf]))
                if len(sub_ok) != 1:
                    continue
                s, xc, yc = sub_ok[0]
                pid_x = min(xc, key=lambda pc: _juan_gap(pc[1], juan_no))[0]
                x_emit = (m.start(1), m.end(1))
                surname = s
        else:
            cx = by_id[pid_x]["canonical_name"]
            surname = seed_mod._surname_of(cx)
            if not surname and len(cx) >= 2 and cx[0] in seed_mod.SURNAMES:
                surname = cx[0]  # carded name proves an ambiguous head (严/高) is a real 姓
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
        if x_emit:  # subject 省称 recovered (瑑→刘瑑): bind it too if span free
            xs2, xe2 = x_emit
            if not any(consumed[xs2:xe2]):
                for k in range(xs2, xe2):
                    consumed[k] = True
                mentions.append((xs2, xe2, pid_x, x_surf))
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
    # 后妃 honorific tails: 姓+Y where Y ends in a consort title (皇后/太后/淑妃/贵妃…)
    # is never a patrilineal given name to reconstruct — block so a gloss like
    # 「温，韦皇后之兄也」 never rebuilds 王皇后 from a 王-surnamed subject.
    if len(y) >= 2 and y[-1] in "后妃嫔":
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


# 异体/繁简 character equivalences: a rare variant char → its standard-form
# representative. Deliberately CURATED and tiny — each entry was corpus-verified to
# denote ONE historical person split into two AUTO cards whose canonicals differ only
# by this glyph: 高騈/高骈 (Gao Pian), 张须陀/张须陁 (Zhang Xutuo), 晁错/鼌错 (Chao Cuo),
# 李谷/李榖 (Li Gu). NOT a general trad→simp table: a blanket same-length merge would
# wrongly fold 封号/官衔 glue (太傅懿≈司马懿, 夏王勃勃≈赫连勃勃 — real dup people but era-
# gated, deferred) and outright homograph traps (何承天 vs 何承之 — different people;
# 李严→李平 — a rename). Add a glyph here only after confirming the two cards are the
# same person and differ solely by an orthographic variant.
_VARIANT_CHAR = {"騈": "骈", "陁": "陀", "鼌": "晁", "榖": "谷"}


def _variant_key(s: str) -> str:
    return "".join(_VARIANT_CHAR.get(c, c) for c in s)


def merge_variant_cards(by_id, people_list, rules, juans, text_dir, meta=None):
    """Fold AUTO cards whose canonicals are equal under _variant_key (same person,
    orthographic-variant glyph) into one, reusing _merge_xref_card so RULES surfaces
    re-point and the dropped canonical survives as a resolvable name. Keeper = the
    variant the MAIN TEXT actually writes most (that is the glyph readers see
    underlined — 高騈 over 高骈, 鼌错 over 晁错, 李谷 over 李榖), then widest 卷 coverage."""
    juan_cache: dict[int, str] = {}

    def main_of(j):
        if j not in juan_cache:
            jf = text_dir / f"juan_{j:03d}.json"
            paras = (json.loads(jf.read_text(encoding="utf-8"))["paragraphs"]
                     if jf.exists() else [])
            juan_cache[j] = "".join(p.get("main", "") for p in paras)
        return juan_cache[j]

    def text_freq(card):
        cn = card["canonical_name"]
        return sum(main_of(j).count(cn) for j in (card.get("juans") or [])
                   if j in juans)

    groups: dict[str, list] = collections.defaultdict(list)
    for pid_, p in by_id.items():
        if str(pid_).startswith("a:"):
            groups[_variant_key(p["canonical_name"])].append(p)
    merged = 0
    dropped_ids: set = set()
    for key, cards in groups.items():
        if len(cards) < 2 or len(key) < 2:
            continue
        if len({c["canonical_name"] for c in cards}) < 2:
            continue  # all identical — a same-canonical case, not a variant split
        cards.sort(key=lambda c: (-text_freq(c), -len(c.get("juans") or []),
                                  min(c["juans"]) if c.get("juans") else 10 ** 9,
                                  str(c["id"])))
        keep = cards[0]
        for C in cards[1:]:
            if C["canonical_name"] == keep["canonical_name"]:
                continue  # sibling instance of the keeper's own form — leave it
            _merge_xref_card(keep, C, rules, meta)
            have = {(n["text"] if isinstance(n, dict) else n)
                    for n in keep.get("names", [])}
            if C["canonical_name"] not in have:
                keep.setdefault("names", []).append(C["canonical_name"])
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


def _ancestor_surname(y_surf, canon_to_pids, notes, by_id=None, gindex=None):
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

    # ── strong-office recall (Wave 48) — an office/身份 title glued directly to a
    # 姓名 is a high-precision anchor even for hapax names the NER model missed
    # (都将王仲甫, 观察使李丛, 都押牙李雅, 押牙张琯…). Length rule, precision-first:
    # if the char after 姓+g1 is a hard terminator (punct or a pure verb) the name is
    # 2-char (观察使李丛移 → 李丛; 太守韩歆降 → 韩歆); a 3-char name is minted ONLY when
    # 姓+g1+g2 is immediately followed by PUNCTUATION (都将王仲甫， → 王仲甫), never on a
    # verb — this avoids truncating homograph-surname middles (陈少游, 李彝殷, 姚彦章) or
    # 之-names (沈攸之) into wrong 2-char underlines. n≥1 is safe: office + surname +
    # hard boundary + non-person guards beat a bare NER hit for personhood.
    strong_office_cues = tuple(sorted({
        "观察使", "节度使", "团练使", "防御使", "经略使", "招讨使", "都统",
        "刺史", "太守", "都督", "都将", "大将", "牙将", "衙将", "其将",
        "都押牙", "押牙", "都虞候", "虞候", "都知兵马使", "兵马使",
        "镇遏使", "团练判官", "节度判官", "行军司马", "都指挥使",
        "宣徽使", "枢密使",
    }, key=len, reverse=True))
    strong_cue_alt = "|".join(re.escape(c) for c in strong_office_cues)
    # office cue disambiguates AMBIGUOUS surnames (高/严/任/武…) → add the clearly-real
    # ones back for this pass so 「刺史高锡望」 (高 is dropped from CLEAN_SURNAMES) is caught.
    strong_sur_set = seed_mod.CLEAN_SURNAMES | set("高严任武史田文安万华成牛丁乐金时后")
    strong_sur = "".join(sorted(re.escape(c) for c in strong_sur_set))
    _strong_ban = {"来受祸", "来难信", "习武事", "来降"}  # proven verb phrases, not names
    _punct = set("，。；、：︰！？「」『』（）〈〉《》【】　 ")
    _hard_stop = _punct | set(
        "移迁拜除薨卒贬曰奏遣督统据击拒追讨入放患破攻杀败走叛请将为知以"
        "有奉素设代镇率兼领加复夺弃斩执获召徙"
        "降反怒惧坐死书离战等陷溃遁逃奔亡诏贪饮言谋营也")
    _g2_block = office_glue_given_block | set("中外内东西南北城郡县也")
    # A handful of _g2_block chars double as real given-name characters (公锷/公著/
    # 公弼); when preceded by a strong office cue AND POS·Giv-confirmed on BOTH the
    # blocked char and the next, 姓+公+X is a genuine 2-char given (押牙田公锷), not
    # 封号/office glue (王公, 都督使). Everything else in _g2_block stays hard-blocked.
    _G2_BLOCK_GIV_OK = set("公")
    strong_re = re.compile(
        rf"(?:{strong_cue_alt})({compound_alt}|[{strong_sur}])([\u4e00-\u9fff])")
    strong_hits: dict[str, set] = collections.defaultdict(set)

    def _strong_ok(chosen):
        return not (chosen in canon_to_pids or chosen in created
                    or chosen in _strong_ban
                    or seed_mod.bad_auto_surface(chosen)
                    or chosen in seed_mod.COMMON_WORD_NONPERSON
                    or chosen in seed_mod.COMPOUND or chosen in _TITLE_BANNED)

    for juan_no in juans:
        jf = text_dir / f"juan_{juan_no:03d}.json"
        if not jf.exists():
            continue
        juan = json.loads(jf.read_text(encoding="utf-8"))
        juan_text = "".join(p.get("main", "") for p in juan["paragraphs"])
        giv_map = None   # POS·Giv, lazy per juan (warm cache = cheap JSON read)
        for para in juan["paragraphs"]:
            scan = [para.get("main", "")] + [
                nt.get("text", "") for nt in para.get("notes", [])]
            for si, mtext in enumerate(scan):
                is_main = si == 0
                for m in strong_re.finditer(mtext):
                    sur, g1 = m.group(1), m.group(2)
                    p2 = m.end()                       # index just past g1
                    g2 = mtext[p2:p2 + 1]
                    r3 = mtext[p2 + 1:p2 + 2]
                    if is_main and giv_map is None:
                        giv_map = pos_giv.giv_for_juan(
                            juan_no, juan["paragraphs"], OUT / "pos_giv")
                    giv_pos = giv_map.get(para["id"], ()) if (is_main and giv_map) else ()
                    rescue3 = None
                    if g1 in _g2_block:
                        # #6 押牙田公锷 / 经略使严公素: rescue a 封号-homograph given (公)
                        # ONLY when a strong cue precedes and POS·Giv confirms BOTH 公 and
                        # the next char → a real 2-char given. Mint the 3-char directly so
                        # a hard_stop g2 (素 ∈ _hard_stop) can't truncate 严公素 → 严公.
                        if g1 in _G2_BLOCK_GIV_OK and (p2 - 1) in giv_pos \
                                and g2 and _han(g2) and g2 not in _g2_block \
                                and p2 in giv_pos:
                            rescue3 = sur + g1 + g2
                        else:
                            continue
                    chosen = None
                    if rescue3:
                        chosen = rescue3
                    elif g2 in _hard_stop:
                        chosen = sur + g1              # 姓 + 1 given
                    elif g2 and _han(g2) and g2 not in _g2_block:
                        cand3 = sur + g1 + g2
                        cand2 = sur + g1
                        if r3 in _punct:
                            # clean right boundary → default 3-char, truncate to 2-char
                            # only on strong evidence (g2 not Giv AND 姓+g1 recurs bare):
                            # 「都虞候郭昢家。」— 家 is a NOUN and 郭昢 recurs → 郭昢.
                            keep3 = True
                            if is_main:
                                g2_is_giv = p2 in giv_pos
                                rec3 = juan_text.count(cand3)
                                rec2_only = juan_text.count(cand2) - rec3
                                if (not g2_is_giv) and rec2_only >= 1 \
                                        and _strong_ok(cand2):
                                    keep3 = False
                            chosen = cand3 if keep3 else cand2
                        elif p2 in giv_pos:
                            # #3/#1 verb/word-terminated 3-char (大将高文集守朔州 →
                            # 高文集; 宣徽使李顺融为枢密使 → 李顺融): the r3-in-punct gate
                            # used to drop these. Mint the 3-char name ONLY when g2 is
                            # POS·Giv, so a hapax 2-char name + non-stop verb (张镒表)
                            # is not wrongly glued into a 3-char surface.
                            chosen = cand3
                    if chosen and _strong_ok(chosen):
                        strong_hits[chosen].add(juan_no)

    for new_full, jset in strong_hits.items():
        if new_full in created:
            continue
        juans_c = sorted(jset)
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

    # ── LLM-annotation recall tier (durable cache; hybrid with pipeline) ──
    # Consumes precomputed per-卷 annotations from llm_annotations/juan_NNN.jsonl
    # (preferred) or legacy juan_NNN.tsv. Each record: {name, aliases[], role,
    # confidence}. Precision-first: a name is minted ONLY when it literally occurs in
    # that 卷 and clears every non-person guard. Declared aliases (省称/别名) are registered
    # in THIS 卷's RULES only (per-卷 scope) so 道伟→康道伟 resolves deterministically and the
    # alias pass never fires ahead of the full name. Offsets/enumeration stay
    # deterministic (the mention scan later); the LLM contributes detection only.
    _llm_surname = seed_mod.SURNAMES | seed_mod.AMBIGUOUS_SURNAMES | LLM_EXTRA_SURNAMES
    for juan_no in juans:
        recs = _load_llm_ann(juan_no)
        if not recs:
            continue
        jf = text_dir / f"juan_{juan_no:03d}.json"
        if not jf.exists():
            continue
        juan = json.loads(jf.read_text(encoding="utf-8"))
        blob = []
        for para in juan["paragraphs"]:
            blob.append(para.get("main", ""))
            for nt in para.get("notes", []):
                blob.append(nt.get("text", ""))
        fulltext = "\n".join(blob)
        for rec in recs:
            surface = rec["name"]
            if rec.get("confidence", "high") != "high":
                continue                       # only high-confidence names are minted
            if not (2 <= len(surface) <= 4) or not all(_han(c) for c in surface):
                continue
            if surface[:1] not in _llm_surname and surface[:2] not in seed_mod.COMPOUND:
                continue                       # must start with a known surname / 复姓
            card = created.get(surface)
            if card is None:
                if not _strong_ok(surface):
                    continue                   # already carded or a banned non-person
                if surface not in fulltext:
                    continue                   # precision-first: must occur in this 卷
                dyn = (meta.get(juan_no, {}).get("dynasty") or "").strip()
                cs = meta.get(juan_no, {}).get("ce_start")
                role = rec.get("role", "")
                tag = (f"{dyn + '·' if dyn else ''}"
                       f"{role + '，' if role else ''}"
                       f"见于卷{juan_no:03d}（LLM 校补）。")
                card = {
                    "id": seed_mod._auto_id(surface, 9500 + len(created)),
                    "canonical_name": surface,
                    "names": [],
                    "dynasty": dyn or "—",
                    "era_hint": f"{dyn}人物" if dyn else "人物",
                    "floruit": [cs, cs] if cs else [None, None],
                    "brief": tag,
                    "identity": tag,
                    "match": [surface],
                    "juans": [juan_no],
                    "confidence": "high",
                }
                created[surface] = card
            rules.setdefault(juan_no, {}).setdefault(surface, card["id"])
            # per-卷 aliases (省称/别名): register ONLY in this 卷's RULES. setdefault keeps
            # the table collision-free — an alias already owned by another surface is not
            # overridden. The ≥2-char alias pass will then tag e.g. 道伟 as 康道伟 here.
            for alias in rec.get("aliases", []):
                if not alias or alias == surface or not all(_han(c) for c in alias):
                    continue
                if alias not in fulltext:
                    continue                   # must occur in this 卷
                if alias in canon_to_pids:
                    continue                   # a seed/earlier card owns this name
                other = created.get(alias)
                if other is not None and other is not card:
                    continue                   # another minted card owns it
                if alias in seed_mod.COMMON_WORD_NONPERSON:
                    continue
                rules.setdefault(juan_no, {}).setdefault(alias, card["id"])

    # ── LLM binding tier (recall): 封号/官职/省称 surface → canonical person ──
    # Registers an explicit 卷-local RULES surface for a form the deterministic scan
    # misses (吴王→刘濞, 辽西王农→慕容农, 文泰→曲文泰). The person's real 姓名 (canonical) is
    # resolved to an existing card / name-tier mint, or a fresh card is minted. The 封号
    # surface then binds to it in THIS 卷 only. Offsets stay deterministic (extract()
    # longest-match + re-verify in the emission loop); the LLM supplies only the
    # surface→identity mapping. Precision gates: the surface must occur in the 卷; a
    # surface already owned in this 卷's RULES is never overridden (setdefault); a minted
    # canonical must head with a real 姓/复姓. A record with `para_range` [lo,hi] binds
    # the surface only within those paragraph ids (rotating 封号) — registered into
    # _PARA_BINDINGS for the emission-time overlay instead of the whole-卷 RULES.
    _bind_surname = seed_mod.SURNAMES | seed_mod.AMBIGUOUS_SURNAMES | LLM_EXTRA_SURNAMES
    for juan_no in juans:
        recs = _load_llm_binding(juan_no)
        if not recs:
            continue
        jf = text_dir / f"juan_{juan_no:03d}.json"
        if not jf.exists():
            continue
        juan = json.loads(jf.read_text(encoding="utf-8"))
        blob = []
        for para in juan["paragraphs"]:
            blob.append(para.get("main", ""))
            for nt in para.get("notes", []):
                blob.append(nt.get("text", ""))
        fulltext = "\n".join(blob)
        rule_map = rules.setdefault(juan_no, {})
        for rec in recs:
            surf, canon, pr = rec["surface"], rec["canonical"], rec["para_range"]
            if surf not in fulltext:
                continue                       # precision-first: surface must occur
            if pr is None and surf in rule_map:
                continue                       # this 卷 already binds the surface whole-卷
            # resolve canonical → pid: seed card, name-tier mint, or a fresh mint
            pid = None
            cands = canon_to_pids.get(canon)
            if cands:
                pid = min(cands, key=lambda pc: _juan_gap(pc[1], juan_no))[0]
            elif canon in created:
                pid = created[canon]["id"]
                if juan_no not in created[canon]["juans"]:
                    created[canon]["juans"].append(juan_no)
            else:
                if not (2 <= len(canon) <= 4) or not all(_han(c) for c in canon):
                    continue
                if canon[:1] not in _bind_surname and canon[:2] not in seed_mod.COMPOUND:
                    continue                   # canonical must head with a real 姓/复姓
                dyn = rec["dynasty"] or (meta.get(juan_no, {}).get("dynasty") or "").strip()
                cs = meta.get(juan_no, {}).get("ce_start")
                role = rec["role"]
                tag = (f"{dyn + '·' if dyn else ''}"
                       f"{role + '，' if role else ''}"
                       f"见于卷{juan_no:03d}（LLM 校补）。")
                card = {
                    "id": seed_mod._auto_id(canon, 9700 + len(created)),
                    "canonical_name": canon,
                    "names": [surf] if surf != canon else [],
                    "dynasty": dyn or "—",
                    "era_hint": f"{dyn}人物" if dyn else "人物",
                    "floruit": [cs, cs] if cs else [None, None],
                    "brief": tag, "identity": tag,
                    "match": [canon], "juans": [juan_no], "confidence": "high",
                }
                created[canon] = card
                pid = card["id"]
                _BIND_STATS["minted"] += 1
            if len(surf) == 1 and _han(surf):
                # single-char 省称 → gated anaphora channel (never a blanket alias)
                _BIND_ANAPHORA.append((juan_no, surf, pid))
                _BIND_STATS["anaphora"] += 1
            elif pr is None:
                rule_map.setdefault(surf, pid)
                # Also expose the canonical as a 卷-surface so rc4 titleglue can bind
                # 封号+given (范阳王德→慕容德) via its primary clan+given path — otherwise
                # an undated target card makes titleglue reserve-and-drop the glued span.
                rule_map.setdefault(canon, pid)
                _BIND_STATS["bound"] += 1
            else:
                _PARA_BINDINGS.setdefault(juan_no, []).append((pr[0], pr[1], surf, pid))
                _BIND_STATS["para_scoped"] += 1

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
                if (not pid_x or pid_x not in by_id) and len(x_surf) == 1 \
                        and _gloss_subject_ok(x_surf):
                    # X written 省称 (bare given 收): resolve its surname from the NEAREST
                    # preceding 姓+X / 复姓+X mention in THIS paragraph — the just-named
                    # antecedent (…杨收同平章事。收，发之弟也) — NOT a 卷-wide card, which can
                    # be a different 同名 person (姚详 vs the 省称 慕容详). When that full
                    # name is carded, X is a KNOWN person, so take the FORWARD path and
                    # mint the patrilineal relative Y (杨发); the old code dropped to the
                    # INVERSE branch, which only re-mints the carded subject and loses Y.
                    pre_txt = mtext[:m.start(1)]
                    sname = None
                    for i in range(len(pre_txt) - 1, 0, -1):
                        if pre_txt[i] != x_surf:
                            continue
                        if i >= 2 and pre_txt[i - 2:i] in seed_mod.COMPOUND:
                            sname = pre_txt[i - 2:i]
                            break
                        if pre_txt[i - 1] in seed_mod.SURNAMES:
                            cand = pre_txt[i - 1]
                            # 王/公/侯/君/主 double as princely titles: 「彭城王雄」 is the
                            # prince named 雄 (姓 元/司马…), NOT surname 王. Accept these
                            # only when the char starts a fresh name (preceded by a clause
                            # boundary), so a fief+王 title (阳平王/襄阳王/高凉王) is rejected
                            # and we never mint a bogus 王释 / 王玮 for 元释 / 司马玮.
                            if cand in "王公侯君主" and not (
                                    i == 1 or pre_txt[i - 2] in _GLOSS_BOUNDARY
                                    or pre_txt[i - 2] in "「『（〈《【　 "):
                                break
                            sname = cand
                            break
                    if sname:
                        cands = canon_to_pids.get(sname + x_surf)
                        if cands:
                            pid_x = min(cands, key=lambda pc: _juan_gap(pc[1], juan_no))[0]
                            if x_surf not in seed_mod.ANAPHORA_CHAR_EXCLUDE:
                                carded_anaphora.setdefault(pid_x, (set(), x_surf))[0].add(juan_no)
                            if len(y_surf) == 1 and y_surf not in seed_mod.ANAPHORA_CHAR_EXCLUDE:
                                for apid, _aj in canon_to_pids.get(sname + y_surf, []):
                                    carded_anaphora.setdefault(apid, (set(), y_surf))[0].add(juan_no)
                inverse = False
                if pid_x and pid_x in by_id:
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
                    # INVERSE — X is the descendant. By the patrilineal rule X shares
                    # the surname of whatever full 姓名 ending in X is already carded in
                    # THIS 卷 (高騈 in section → bare 騈 = 高騈); the ancestor merely
                    # confirms personhood. Fall back to ancestor-胡注 only to MINT a new
                    # card when no such subject card exists.
                    if not _gloss_subject_ok(x_surf):
                        continue
                    surname = None
                    if len(x_surf) == 1:
                        subj = {k[0] for k in rule_map if len(k) == 2 and k[1] == x_surf}
                        if len(subj) == 1:
                            surname = next(iter(subj))
                        if not surname:
                            # shared-surname from the subject's OWN full mention in this
                            # 卷 (严譔为镇南… → 譔，震之从孙: 严譔 confirms 严) — mint even
                            # when NER missed it and the ancestor's clan is ambiguous.
                            txt = " ".join(scan)
                            pre = {txt[k - 1] for k in range(1, len(txt))
                                   if txt[k] == x_surf and txt[k - 1] in seed_mod.SURNAMES}
                            if len(pre) == 1:
                                surname = next(iter(pre))
                    if not surname:
                        surname = _ancestor_surname(y_surf, canon_to_pids, para.get("notes"), by_id, gindex)
                    if not surname:
                        continue
                    new_full = surname + x_surf
                    anchor_given = x_surf if len(x_surf) == 1 else None
                    inverse = True
                    # ancestor 省称: shared surname makes 震 = 严震 (carded) — bind the
                    # bare ancestor mention too (譔，震之从孙也 → 震 underlines 严震). The
                    # gloss confirms personhood, so register a 卷-alias directly.
                    if len(y_surf) == 1 and y_surf not in seed_mod.ANAPHORA_CHAR_EXCLUDE:
                        for apid, _aj in canon_to_pids.get(surname + y_surf, []):
                            carded_anaphora.setdefault(apid, (set(), y_surf))[0].add(juan_no)
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
                        new_full in seed_mod.COMMON_WORD_NONPERSON or \
                        new_full in _TITLE_BANNED:
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
                    # the 省姓 given char so the anaphora pass binds the bare 谱; and
                    # register the bare given so the FORWARD gloss binds the ancestor
                    # (譔→严譔 lets 震 resolve to 严震 via 「譔，震之从孙也」).
                    rule_map.setdefault(new_full, card["id"])
                    if anchor_given:
                        anaphora[new_full] = anchor_given
                        rule_map.setdefault(anchor_given, card["id"])
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


# ── Narrow R1 — truncation-repair discovery ───────────────────────────────────
# The NER/seed layer sometimes truncates a 姓+双名 (张禹谟) to a carded 姓+单名
# (张禹) plus a stranded tail (禹谟 — occasionally itself minted as a phantom
# auto-card). This pre-pass re-mints the full 3-char person, registers 张禹谟 in the
# 卷's RULES (longest-match then beats the 2-char truncation inside extract()), binds
# the recurring bare tail, and folds any phantom tail card into the new one. Validated
# corpus-wide at 13 hits / ~92% precision (ref session files/narrow_r1.py). The
# POS Giv+Giv gate reuses the pos_giv cache, so a warm build stays torch-free.
_R1_KIN_G2 = set("子孙女弟兄父母妻姊妹")
_R1_STOPC = set("之而其以为于与也者乎则乃因故不无有莫皆各此是所曰云谓将欲今日时"
                "上下中大天地国家军兵民人道州县城门内外东西南北左右前后等已来")
_R1_OFFICE_TAIL = set("尉令史使夫将军卿相守牧刺郎丞尹帅监正长牧簿")
_R1_VERBS = set("为以拜授除召见斩杀立命封进用擢领判集遣纳戮诛贬黜召署补迁")
_R1_KIN = "子孙弟兄父祖甥侄婿"
# 爵号/官职-glue guard heads: chars that are BOTH a rare surname AND a common title/
# office head (王=爵号, 牧=州牧, 守=太守, 令=县令, 尉=都尉, 尹=京兆尹, 史=刺史). When such a
# head is glued after a place/noun (凉州牧·轲弹) the token is office+name, not 姓+名, so we
# reject unless the head is introduced by an office tail / appointment verb / punctuation.
_R1_GUARD_HEADS = set("王牧守令尉尹史")


def _r1_clean_given2(g):
    return (len(g) == 2 and all("\u4e00" <= c <= "\u9fff" for c in g)
            and not any(c in _R1_STOPC for c in g))


def _r1_pos_giv_ok(tail, surname, sec_pids, ptext, giv_map):
    """POS gate: at the FIRST bare occurrence of `tail` (a T not preceded by its
    surname) within the section, both chars must be tagged PROPN|Giv per the cache."""
    for pid in sec_pids:
        txt = ptext.get(pid, "")
        j = txt.find(tail)
        while j >= 0:
            prev = txt[j - 1] if j > 0 else ""
            if prev != surname:  # a bare occurrence
                gset = giv_map.get(pid, set())
                return (j in gset) and (j + 1 in gset)
            j = txt.find(tail, j + 1)
    return False


def build_truncation_cards(juans, text_dir, rules, canon_to_pids, by_id,
                           people_merged, meta, pos_dir):
    """Mint 姓+双名 cards for NER truncations; returns (n_created, n_phantoms_merged)."""
    carded: set[str] = set()
    owned_tails: set[str] = set()  # trailing given of any carded person (王勃勃=赫连勃勃)
    for p in by_id.values():
        nms = [p.get("canonical_name", "")]
        nms += [(n if isinstance(n, str) else n.get("text", ""))
                for n in p.get("names", [])]
        nms += list(p.get("match", []))
        for nm in nms:
            if not nm:
                continue
            carded.add(nm)
            sn = seed_mod._surname_of(nm)
            if sn and len(nm) - len(sn) == 2:
                owned_tails.add(nm[len(sn):])
            if len(nm) >= 3:
                owned_tails.add(nm[-2:])
    created: dict[tuple, dict] = {}
    phantoms_merged = 0
    for juan_no in juans:
        jf = text_dir / f"juan_{juan_no:03d}.json"
        if not jf.exists():
            continue
        rule_map = rules.get(juan_no)
        if not rule_map:
            continue
        paras = json.loads(jf.read_text(encoding="utf-8"))["paragraphs"]
        ptext = {p["id"]: p.get("main", "") for p in paras}
        cur = 0
        sec_of: dict = {}
        for p in paras:
            m0 = p.get("main", "")
            n0 = _sec_num(m0)
            if n0 is not None:
                cur = n0
            sec_of[p["id"]] = cur
        sec_pids: dict = {}
        sec_text: dict = {}
        for p in paras:
            s = sec_of[p["id"]]
            sec_pids.setdefault(s, []).append(p["id"])
            sec_text[s] = sec_text.get(s, "") + p.get("main", "")
        # anchors: carded 2-char 姓+单名 surfaces registered in this 卷
        anchors = []
        for surf in rule_map:
            if len(surf) != 2:
                continue
            sn = seed_mod._surname_of(surf)
            if sn and len(sn) == 1 and surf.startswith(sn):
                anchors.append(surf)
        if not anchors:
            continue
        giv_map = None  # POS cache — loaded lazily only when a candidate reaches the gate
        seen: set = set()
        for surf in anchors:
            for p in paras:
                txt = ptext[p["id"]]
                pos = txt.find(surf)
                while pos >= 0:
                    end = pos + 2
                    g2 = txt[end] if end < len(txt) else ""
                    if not ("\u4e00" <= g2 <= "\u9fff") or g2 in _R1_KIN_G2:
                        pos = txt.find(surf, pos + 1)
                        continue
                    full3 = surf + g2
                    T = surf[1] + g2
                    if (not _r1_clean_given2(T) or full3 in carded
                            or T in owned_tails or (juan_no, full3) in seen):
                        pos = txt.find(surf, pos + 1)
                        continue
                    if surf[0] in _R1_GUARD_HEADS:  # 爵号/官职-glue guard (楚王景驹, 凉州牧轲弹)
                        prev = txt[pos - 1] if pos > 0 else ""
                        if (prev and "\u4e00" <= prev <= "\u9fff"
                                and prev not in _R1_OFFICE_TAIL
                                and prev not in _R1_VERBS):
                            pos = txt.find(surf, pos + 1)
                            continue
                    sec = sec_of[p["id"]]
                    stext = sec_text[sec]
                    bare = 0
                    idx = stext.find(T)
                    while idx >= 0:
                        if (stext[idx - 1] if idx > 0 else "") != surf[0]:
                            bare += 1
                        idx = stext.find(T, idx + 1)
                    gloss = bool(re.search(
                        "，?" + re.escape(T) + r"，[\u4e00-\u9fff]{1,6}之["
                        + _R1_KIN + r"]也?", stext)) or bool(re.search(
                        re.escape(T) + r"，[\u4e00-\u9fff]{1,6}之[" + _R1_KIN + r"]",
                        stext))
                    if bare < 1 and not gloss:
                        pos = txt.find(surf, pos + 1)
                        continue
                    if giv_map is None:
                        giv_map = pos_giv.giv_for_juan(juan_no, paras, pos_dir)
                    if not _r1_pos_giv_ok(T, surf[0], sec_pids[sec], ptext, giv_map):
                        pos = txt.find(surf, pos + 1)
                        continue
                    seen.add((juan_no, full3))
                    _r1_mint(full3, T, juan_no, meta, created, rules, canon_to_pids,
                             by_id, people_merged, carded)
                    pos = txt.find(surf, pos + 1)
    # Fold phantom tail cards (canonical == a minted T, confined to that 卷) into the
    # new 3-char card so bare-T mentions aggregate onto one person (禹谟 → 张禹谟).
    for (juan_no, full3), card in created.items():
        T = card["_r1_tail"]
        for pid_, jset in list(canon_to_pids.get(T, [])):
            ph = by_id.get(pid_)
            if (ph and pid_.startswith("a:") and set(jset) <= {juan_no}
                    and ph is not card):
                _merge_xref_card(card, ph, rules, meta)
                by_id.pop(pid_, None)
                people_merged[:] = [q for q in people_merged if q["id"] != pid_]
                canon_to_pids[T] = [t for t in canon_to_pids.get(T, [])
                                    if t[0] != pid_]
                phantoms_merged += 1
    for card in created.values():
        card.pop("_r1_tail", None)
    return len(created), phantoms_merged


def _r1_mint(full3, tail, juan_no, meta, created, rules, canon_to_pids, by_id,
             people_merged, carded):
    dyn = (meta.get(juan_no, {}).get("dynasty") or "").strip()
    cs = meta.get(juan_no, {}).get("ce_start")
    card = {
        "id": seed_mod._auto_id(full3, 7000 + len(created)),
        "canonical_name": full3,
        "names": [tail],
        "dynasty": dyn or "—",
        "era_hint": f"{dyn}人物" if dyn else "人物",
        "floruit": [cs, cs] if cs else [None, None],
        "brief": f"{dyn + '·' if dyn else ''}见于卷{juan_no:03d}（断名修复）。",
        "identity": f"{dyn + '·' if dyn else ''}见于卷{juan_no:03d}（断名修复）。",
        "match": [full3, tail],
        "juans": [juan_no],
        "confidence": "high",
        "_r1_tail": tail,
    }
    created[(juan_no, full3)] = card
    people_merged.append(card)
    by_id[card["id"]] = card
    carded.add(full3)
    carded.add(tail)
    canon_to_pids.setdefault(full3, []).append((card["id"], {juan_no}))
    rules.setdefault(juan_no, {}).setdefault(full3, card["id"])
    rules.setdefault(juan_no, {}).setdefault(tail, card["id"])


# ── R2 — gloss subject (X) discovery ──────────────────────────────────────────
# 「X，Y之{子孙弟…}也」 names a NEW person X whose ancestor Y is already carded. When
# the NER/seed layer missed X (宫苑使李瑑，西平王晟之孙 → 李瑑 dropped because the greedy
# 3-char grab 「使李瑑」 fails the clause-boundary guard), mint X here. Four boundary
# fixes over the raw gloss: (1) peel a 官职/爵号 prefix by taking the surname-anchored
# suffix before the comma (使李瑑→李瑑); (2) reject an 氏-tail (clan, not a person);
# (3) resolve Y by BORROWING X's surname (西平王·晟 → 李晟) and require that
# reconstruction to be an existing CANONICAL card — shared-surname patrilineal proof,
# which drops echo-Y false positives (王绰→义庆=刘义庆, 审晖→审琦 alias-of-王审琦);
# (4) POS-gate X's given (瑑 must be NameType=Giv) to drop non-name fragments
# (来赴难→赴难, 吴攻具→攻具). Validated against session files/r2_probe.py.
_R2_COMMA_KIN = re.compile(r"，([\u4e00-\u9fff]{1,4})之([" + _R1_KIN + r"])也?")
# relation nouns that legitimately introduce a person by kinship right before a name
# (兄子壻王磐) plus office-institution chars (三司王颁) — an escape for the 爵号-glue
# guard so real 王-姓 first-mentions survive. FPs are glued after fief chars (临川/邵).
_R2_REL_NOUNS = set("壻婿子弟兄父孙甥侄妻女母祖司")


def _r2_peel_x(mt, c):
    """Surname-anchored X ending right before comma `c`: try 3- then 2-char suffix,
    return (full, surname, given) for the longest that is a clean 姓+名, else None."""
    for L in (3, 2):
        if c - L < 0:
            continue
        cand = mt[c - L:c]
        if not all("\u4e00" <= ch <= "\u9fff" for ch in cand):
            continue
        sn = seed_mod._surname_of(cand)
        if sn and cand.startswith(sn):
            g = cand[len(sn):]
            if 1 <= len(g) <= 2 and not any(ch in _R1_STOPC for ch in g):
                return cand, sn, g
    return None


def build_gloss_subject_cards(juans, text_dir, rules, canon_to_pids, by_id,
                              people_merged, meta, pos_dir):
    """Mint NEW gloss-subject cards (李瑑). Returns (n_created, subj_inv_anaphora)
    where subj_inv_anaphora = [(card_id, [juan], given_char)] for bare-given binding."""
    canon_names = {p.get("canonical_name", "") for p in by_id.values()}
    canon_names.discard("")
    carded = set(canon_names)
    for p in by_id.values():
        for n in p.get("names", []):
            carded.add(n if isinstance(n, str) else n.get("text", ""))
        for mm in p.get("match", []):
            carded.add(mm)
    created: dict[tuple, dict] = {}
    inv: list = []
    for juan_no in juans:
        jf = text_dir / f"juan_{juan_no:03d}.json"
        if not jf.exists():
            continue
        paras = json.loads(jf.read_text(encoding="utf-8"))["paragraphs"]
        giv_map = None
        for p in paras:
            mt = p.get("main", "")
            for m in _R2_COMMA_KIN.finditer(mt):
                c = m.start()
                yseg = m.group(1)
                xr = _r2_peel_x(mt, c)
                if not xr:
                    continue
                x_full, xsn, xg = xr
                if x_full in carded or x_full[-1] == "氏":
                    continue
                if (juan_no, x_full) in created:
                    continue
                if any(ch in _GLOSS_Y_BLOCK for ch in xg):
                    continue
                # 爵号-glue guard: 「临川王绰」/「邵王友诲」 peel 王 as surname, but 王 is a
                # 封号 (real 姓 is the ancestor's). Reject when a GUARD_HEAD surname is
                # glued after a place/fief char — legit 王-姓 first-mentions are
                # introduced by an office tail, an appointment verb, or a relation noun.
                if xsn in _R1_GUARD_HEADS:
                    sp = c - len(x_full)  # offset of the surname char
                    prev = mt[sp - 1] if sp > 0 else ""
                    if (prev and "\u4e00" <= prev <= "\u9fff"
                            and prev not in _R1_OFFICE_TAIL and prev not in _R1_VERBS
                            and prev not in _R2_REL_NOUNS):
                        continue
                # (3) resolve Y by borrowing X's surname → require a canonical card
                yres = None
                tails = [yseg]
                if len(yseg) >= 2:
                    tails += [yseg[-2:], yseg[-1:]]
                for yg in tails:
                    if not _gloss_y_ok(yg):
                        continue
                    if (xsn + yg) in canon_names:
                        yres = xsn + yg
                        break
                if not yres:
                    continue
                # (4) POS-gate X's given at the gloss occurrence
                if giv_map is None:
                    giv_map = pos_giv.giv_for_juan(juan_no, paras, pos_dir)
                gset = giv_map.get(p["id"], set())
                g_start = c - len(xg)
                if not all((g_start + k) in gset for k in range(len(xg))):
                    continue
                gid = _r2_mint(x_full, xg, juan_no, meta, created, rules,
                               canon_to_pids, by_id, people_merged, carded, canon_names)
                if len(xg) == 1 and xg not in seed_mod.ANAPHORA_CHAR_EXCLUDE:
                    inv.append((gid, [juan_no], xg))
    return len(created), inv


def _r2_mint(x_full, given, juan_no, meta, created, rules, canon_to_pids, by_id,
             people_merged, carded, canon_names):
    dyn = (meta.get(juan_no, {}).get("dynasty") or "").strip()
    cs = meta.get(juan_no, {}).get("ce_start")
    card = {
        "id": seed_mod._auto_id(x_full, 8000 + len(created)),
        "canonical_name": x_full,
        "names": [given] if len(given) == 1 else [],
        "dynasty": dyn or "—",
        "era_hint": f"{dyn}人物" if dyn else "人物",
        "floruit": [cs, cs] if cs else [None, None],
        "brief": f"{dyn + '·' if dyn else ''}见于卷{juan_no:03d}（家世胡注）。",
        "identity": f"{dyn + '·' if dyn else ''}见于卷{juan_no:03d}（家世胡注）。",
        "match": [x_full],
        "juans": [juan_no],
        "confidence": "high",
    }
    created[(juan_no, x_full)] = card
    people_merged.append(card)
    by_id[card["id"]] = card
    carded.add(x_full)
    canon_names.add(x_full)
    canon_to_pids.setdefault(x_full, []).append((card["id"], {juan_no}))
    rules.setdefault(juan_no, {}).setdefault(x_full, card["id"])
    if len(given) == 1 and given not in seed_mod.ANAPHORA_CHAR_EXCLUDE:
        rules.setdefault(juan_no, {}).setdefault(given, card["id"])
    return card["id"]


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

    # R2 pre-pass — mint NEW gloss subjects (李瑑) from 「X，Y之孙也」 where Y is carded.
    # Peels 官职/爵号 off X, borrows X's surname to resolve Y, POS-gates X's given.
    # Runs after the gloss ancestor mint so canon_names includes those cards.
    r2_new, r2_inv = build_gloss_subject_cards(
        JUANS, TEXT, RULES, canon_to_pids, by_id,
        PEOPLE_MERGED, gloss_meta, OUT / "pos_giv")
    for cid, cjuans, gch in r2_inv:  # bind the bare 省称 given (瑑 → 李瑑)
        if gch and gch not in seed_mod.ANAPHORA_CHAR_EXCLUDE:
            given_of[cid] = gch
            for j in cjuans:
                minted_admit.setdefault(j, set()).add(gch)
                minted_anchor.setdefault(j, {}).setdefault(gch, cid)

    # Narrow R1 pre-pass — repair NER truncations (张禹 → 张禹谟) by minting the full
    # 姓+双名 card, registering it in RULES (longest-match beats the truncation), and
    # folding the stranded phantom tail (禹谟) into it. Runs after the gloss mint so
    # its carded set / owned-tails reflect gloss cards too. See build_truncation_cards.
    r1_new, r1_phantoms = build_truncation_cards(
        JUANS, TEXT, RULES, canon_to_pids, by_id,
        PEOPLE_MERGED, gloss_meta, OUT / "pos_giv")

    # LLM single-char 省称 bindings (卬→刘卬): admit the char + anchor it whole-卷 to the
    # LLM-named person, then let the GATED anaphora pass decide which occurrences bind.
    # setdefault so a gloss-derived anchor for the same char is never overridden; a char
    # already carrying a different anchor in this 卷 is a conflict and skipped (precision).
    for (j, ch, pid_) in _BIND_ANAPHORA:
        if pid_ not in by_id or ch in seed_mod.ANAPHORA_CHAR_EXCLUDE:
            continue
        given_of[pid_] = ch
        minted_admit.setdefault(j, set()).add(ch)
        minted_anchor.setdefault(j, {}).setdefault(ch, pid_)
    # LLM 省称 bindings are whole-卷 authoritative — unlike gloss-derived anchors they
    # must SURVIVE the per-section char_anchor reset (a 七国之乱 卷 has ①②… sections but
    # 卬→刘卬 holds across all of them). Keep them in a separate map re-applied on reset.
    llm_anchor: dict[int, dict[str, str]] = {}
    for (j, ch, pid_) in _BIND_ANAPHORA:
        if pid_ in by_id and ch not in seed_mod.ANAPHORA_CHAR_EXCLUDE:
            llm_anchor.setdefault(j, {})[ch] = pid_

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

    # ── Wave B.2: 异体/繁简 variant card merge ──
    # A person written with a variant glyph in different windows/annotations splits into
    # two AUTO cards (高騈/高骈, 张须陀/张须陁, 晁错/鼌错, 李谷/李榖). Fold them via the same
    # primitive so RULES + mentions aggregate onto one card. Curated glyph table only —
    # 封号 glue / homograph traps are excluded by construction. Runs before emission.
    var_merged, var_dropped = merge_variant_cards(
        by_id, PEOPLE_MERGED, RULES, set(JUANS), TEXT, gloss_meta)
    for did in var_dropped:
        given_of.pop(did, None)
    if var_dropped:
        for nm_, lst in list(canon_to_pids.items()):
            canon_to_pids[nm_] = [t for t in lst if t[0] not in var_dropped]

    # ── Phase 3: LLM card curation (dynasty relabel / book brief / evidence-gated merge) ──
    # Metadata-only fixes for the audit's card-quality gaps. Runs BEFORE the emission /
    # lookback passes so a merge's surviving pid aggregates all downstream mentions.
    # Never touches spans — offset precision is untouched. Keyed by canonical, resolved
    # to the nearest in-卷 card.
    card_relabeled = card_rebriefed = card_merged = 0
    _card_recs = []
    for _jn in JUANS:
        _card_recs.extend(_load_llm_card(_jn))

    def _pick_card(canon, juan):
        cands = [c for c in canon_to_pids.get(canon, []) if c[0] in by_id]
        if not cands:
            return None
        return by_id.get(min(cands, key=lambda pc: _juan_gap(pc[1], juan))[0])

    for _rec in _card_recs:                      # merges first (survivor then gets relabel)
        if not _rec["merge_into"]:
            continue
        drop = _pick_card(_rec["canonical"], _rec["juan"])
        keep = _pick_card(_rec["merge_into"], _rec["juan"])
        if not drop or not keep or drop["id"] == keep["id"]:
            continue
        _merge_xref_card(keep, drop, RULES, gloss_meta)
        _did = drop["id"]
        by_id.pop(_did, None)
        PEOPLE_MERGED[:] = [p for p in PEOPLE_MERGED if p["id"] != _did]
        given_of.pop(_did, None)
        for _nm, _lst in list(canon_to_pids.items()):
            canon_to_pids[_nm] = [t for t in _lst if t[0] != _did]
        for _jm in minted_anchor.values():       # re-point any 省称 anchor to the survivor
            for _ch, _pp in list(_jm.items()):
                if _pp == _did:
                    _jm[_ch] = keep["id"]
        for _lm in llm_anchor.values():
            for _ch, _pp in list(_lm.items()):
                if _pp == _did:
                    _lm[_ch] = keep["id"]
        card_merged += 1

    for _rec in _card_recs:                      # dynasty / brief (field edits only)
        if _rec["merge_into"]:
            continue
        card = _pick_card(_rec["canonical"], _rec["juan"])
        if not card:
            continue
        if _rec["dynasty"]:
            card["dynasty"] = _rec["dynasty"]
            card["era_hint"] = f"{_rec['dynasty']}人物"
            card_relabeled += 1
        if _rec["brief"]:
            card["brief"] = _rec["brief"]
            card["identity"] = _rec["brief"]
            card_rebriefed += 1

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

    # Wave 44 — gloss-minted 家世胡注 / 官衔连写 cards are anchored ONLY to the 卷 where
    # the office cue / gloss fired (萧复 → office-intro 卷 228); their full 姓名 recurs in
    # nearby 卷 (萧复 also in 229/231) but the lookback pass above skips them (it requires a
    # len>=3 surface and only probes 2 卷 backward), so a person card shows only part of its
    # history. Fill the gap: around each existing card 卷, probe ±GAP 卷 (era-local, no
    # century teleport) and register the full canonical wherever it literally occurs and is
    # NOT already claimed by another person (homograph-safe, mirrors the lookback guard).
    gloss_fill_added = 0
    _NONNAME_GIVEN = set("以其与为有它出入师国甲天二三日所因既将须即乃则亦焉也矣者及自等")
    for pid_, p in list(by_id.items()):
        if not re.search(r"见于卷\d{3}（(?:家世胡注|官衔连写)）。$", p.get("brief", "")):
            continue
        cn = p["canonical_name"]
        if not (2 <= len(cn) <= 3) or not seed_mod._surname_of(cn):
            continue
        # Precision guard: never EXTEND a card whose given-part is a single function
        # word / common noun — those minted cards (王以/魏以/王师/王国/马甲 = 「王，以」
        # 「royal army」「kingdom」「armor」) are 姓/封号+虚词 mis-slices, and the fill must
        # not amplify them. A real person that happens to have such a given (司马师) merely
        # forgoes the bonus fill (missing < wrong); its base mentions are untouched.
        sn = seed_mod._surname_of(cn)
        given = cn[len(sn):]
        if len(given) == 1 and given in _NONNAME_GIVEN:
            continue
        cur = sorted(j for j in p.get("juans", []) if j in juan_set)
        if not cur:
            continue
        probe = {jb for j in cur for jb in range(j - seed_mod.GAP, j + seed_mod.GAP + 1)
                 if jb in juan_set and jb not in p["juans"]}
        for jb in sorted(probe):
            if RULES.get(jb, {}).get(cn) is not None:
                continue  # claimed by another person here → homograph, skip
            if cn in _juan_text(jb):
                RULES.setdefault(jb, {})[cn] = pid_
                p["juans"].append(jb)
                gloss_fill_added += 1

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
    veto_dropped = 0                 # LLM precision-veto: mentions suppressed
    vetoed_pids: set = set()         # pids touched by veto (orphan-dropped below)
    # Wave 44 — `introduced` is the cumulative (NOT reset per 卷) set of pids whose
    # full canonical name has already appeared in corpus reading order in some EARLIER
    # 卷. Combined with the per-卷 first-occurrence position below, it drives the
    # bare-given antecedent gate: a 省称 is suppressed ONLY when it is pre-debut — the
    # full name has never been read yet AND appears later within the same 卷 (行及徐城
    # at 251#10 precedes 刘行及 at 251#19). Once introduced anywhere, every later 省称
    # (蒙逊/守光/勃勃/化及 across 卷) is kept; a person whose full name never appears in
    # the corpus at all (沮渠牧犍 → only ever 牧犍) is never gated.
    introduced: set = set()

    # Wave 44 — bare-given (surname-stripped, ≥2-char) 省称 surfaces (刘行及→行及,
    # 萧彦回→彦回 …) are antecedent-gated in the alias pass below: such a given can
    # double as an ordinary verb phrase (行及徐城 = "marched and reached 徐城"), so it
    # must not tag BEFORE the full name has been read in the same 卷. single-char 省称
    # are already antecedent-gated by the anchored anaphora pass (given_of); this
    # covers the 2+-char given the position-independent alias matcher would otherwise
    # fire ahead of its antecedent.
    bare_given_of: dict[str, str] = {}
    for pid_, p in by_id.items():
        cn = p["canonical_name"]
        sn = seed_mod._surname_of(cn)
        if sn and len(cn) - len(sn) >= 2:
            bare_given_of[pid_] = cn[len(sn):]

    for juan_no in JUANS:
        jf = TEXT / f"juan_{juan_no:03d}.json"
        if not jf.exists():
            continue
        juan = json.loads(jf.read_text(encoding="utf-8"))
        surfaces = surfaces_for(juan_no)
        pbind_list = _PARA_BINDINGS.get(juan_no, [])   # rotating 封号: (lo,hi,surf,pid)
        veto_set = _load_llm_veto(juan_no)   # LLM precision-veto surfaces for this 卷
        # Wave 44 — first reading-order position (para_idx, char_start) at which each
        # candidate person's FULL canonical name literally appears in this 卷. Used
        # below to suppress a bare-given 省称 that precedes its own full name (the
        # 行及徐城-before-刘行及 case). Position-based (not surface-registration based) so
        # it is robust when the full 4-char name is not itself a matchable surface.
        juan_blob = "\n".join(p.get("main", "") for p in juan["paragraphs"])
        intro_pos: dict[str, tuple] = {}
        cand_pids = {pid_ for _surf, pid_ in surfaces
                     if bare_given_of.get(pid_)
                     and by_id[pid_]["canonical_name"] in juan_blob}
        if cand_pids:
            for cp_idx, cpara in enumerate(juan["paragraphs"]):
                ct = cpara.get("main", "")
                for pid_ in cand_pids:
                    if pid_ in intro_pos:
                        continue
                    pos = ct.find(by_id[pid_]["canonical_name"])
                    if pos >= 0:
                        intro_pos[pid_] = (cp_idx, pos)
        admitted = ANAPHORA_RULES.get(juan_no, set()) | minted_admit.get(juan_no, set())
        char_anchor: dict[str, str] = {
            gch: pid_ for gch, pid_ in minted_anchor.get(juan_no, {}).items()
            if pid_ in by_id
        }  # given-char -> nearest antecedent person_id (卷-local)
        sec_owners: dict[str, set] = {}  # 节-local: given-char -> {owners full-named so far in 节}
        cue_idx = build_role_cue_index(juan["paragraphs"])  # P3 succession split
        mentions = []
        for p_idx, para in enumerate(juan["paragraphs"]):
            pid = para["id"]
            ce = para.get("ce_year")
            main_text = para.get("main", "")
            consumed = [False] * len(main_text)
            # Section-local anaphora gate: a numbered 节 (①②…㊿ or ASCII 51…) resets
            # the 省称 anchor table so a stale full name in an earlier section can't
            # bind a bare given char (云 in 云大破蛮 ≠ 张云). Full names recur per section.
            if _is_sec_lead(main_text):
                char_anchor.clear()
                sec_owners.clear()   # 节-local collision tally resets each numbered 节
                for _gch, _pid in llm_anchor.get(juan_no, {}).items():
                    char_anchor[_gch] = _pid   # LLM 省称 anchors are whole-卷 durable
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
            # LLM para-scoped 封号 bindings — a rotating title (赵王=刘遂 in the 七国之乱
            # paras, 刘彭祖 after the 徙…为赵王 paras) bound only inside its [lo,hi] window.
            # Runs right after 封号-glue so the specific title claims its span before the
            # whole-卷 alias pass; offsets stay deterministic (extract + consumed guard).
            if pbind_list:
                extra = [(s2, pp) for (lo, hi, s2, pp) in pbind_list if lo <= pid <= hi]
                if extra:
                    extra.sort(key=lambda t: len(t[0]), reverse=True)
                    for (s, e, pid_, surf) in extract(main_text, extra):
                        if any(consumed[s:e]):
                            continue
                        for k in range(s, e):
                            consumed[k] = True
                        mentions.append({"pid": pid, "ce_year": ce, "source": "main",
                                         "start": s, "end": e, "surface": surf,
                                         "person_id": pid_, "kind": "feng",
                                         "confidence": by_id[pid_].get("confidence", "reviewed")})
                        seen_ids.add(pid_)
                        feng_emitted += 1
            # single-char alias surfaces are never trusted (元胄→胄 etc. flow through
            # the anaphora/gloss passes which pin them to an antecedent); a bare given
            # char as a standalone alias is a precision storm.
            alias_hits = []
            for (s, e, pid_, surf) in sorted(
                    (h for h in extract(main_text, surfaces)
                     if (h[1] - h[0]) >= 2), key=lambda h: h[0]):
                if any(consumed[s:e]):
                    continue
                # #2 仆射 office-truncation guard: a discovered 姓+单名 whose tail char
                # + the following char spell 仆射 (a 官名, púyè) is office-glue, not a
                # name — 「诈称陈仆射」→ 陈仆[射]. Context-scoped: the same card's other
                # occurrence 「黟帅陈仆、祖山」 (射 does NOT follow) is untouched.
                if surf[-1:] == "仆" and main_text[e:e + 1] == "射":
                    continue
                # #4 谥号 guard: 谥曰X / 谥为X / 谥X — by classical grammar X is ALWAYS a
                # posthumous title (谥曰景庄皇帝 → 景庄 = 南诏王 世隆's 谥, not a person).
                # Left-cue only: the real-person homographs (汉使者文忠, 何文敬, 李景庄,
                # 性忠壮) never sit right after a 谥-cue, so they stay intact.
                if main_text[max(0, s - 2):s] in ("谥曰", "谥为") \
                        or main_text[max(0, s - 1):s] == "谥":
                    continue
                # only when PRE-DEBUT — its person was not introduced in an earlier 卷
                # AND its full name appears later (in reading order) within THIS 卷. A
                # given that doubles as a verb phrase (行及徐城) thus cannot tag ahead of
                # — or in place of — the real person. Cross-卷 and post-introduction 省称
                # (蒙逊/守光/化及) and full-name-never-appears persons (牧犍) are kept.
                if surf == bare_given_of.get(pid_) and pid_ not in introduced:
                    ip = intro_pos.get(pid_)
                    if ip is not None and (p_idx, s) < ip:
                        continue
                for k in range(s, e):
                    consumed[k] = True
                alias_hits.append((s, e, pid_, surf))
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
                        main_text, admitted, consumed, char_anchor, sec_owners, anchor_events, ce, by_id):
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
                    if e - s < 2:
                        continue
                    mentions.append({"pid": pid, "ce_year": ce, "source": "hu",
                                     "note_index": ni, "start": s, "end": e,
                                     "surface": surf, "person_id": pid_, "kind": "alias",
                                     "confidence": by_id[pid_].get("confidence", "reviewed")})
                    seen_ids.add(pid_)

        # ── Wave 6 — POS-gated structural 省称 resolver (recall-forward) ────
        # Runs after every per-paragraph pass so it can dedup against all main-text
        # spans emitted above (alias/feng/curated-anaphora/gloss). Binds each present
        # person's bare given to the closest antecedent in-节; single-char givens are
        # gated by the Classical-Chinese POS·Giv cache. Purely additive.
        present_pids = {m["person_id"] for m in mentions
                        if m.get("source") == "main" and m.get("person_id") in by_id}
        if present_pids:
            exist_span_main: dict = collections.defaultdict(set)
            full_anchor_main: dict = collections.defaultdict(list)
            for m in mentions:
                if m.get("source") == "main":
                    exist_span_main[m["pid"]].add((m["start"], m["end"]))
                    # title/full forms (not bare 省称) become 节-local anchors
                    if m.get("kind") != "anaphora" and m.get("person_id") in present_pids:
                        full_anchor_main[m["pid"]].append(
                            (m["start"], m["end"], m["person_id"], m["surface"]))
            giv_map = pos_giv.giv_for_juan(juan_no, juan["paragraphs"], OUT / "pos_giv")
            for (pid_para, ce_y, s, e, rpid, surf) in resolve_anaphora_pos(
                    juan["paragraphs"], present_pids, by_id, giv_map,
                    exist_span_main, full_anchor_main):
                mentions.append({"pid": pid_para, "ce_year": ce_y, "source": "main",
                                 "start": s, "end": e, "surface": surf,
                                 "person_id": rpid, "kind": "anaphora",
                                 "confidence": by_id[rpid].get("confidence", "reviewed")})
                seen_ids.add(rpid)
                anaphora_emitted += 1

        # ── LLM precision-veto (delete-only) ──────────────────────────────
        # Drop every mention whose surface the 卷's LLM audit flagged as a
        # non-person / boundary-error span. Runs after ALL passes so it uniformly
        # covers alias/anaphora/role/gloss/feng and 胡注 mentions. Delete-only: it
        # can shrink recall but never mis-bind, so precision-first is preserved.
        if veto_set:
            kept = []
            for m in mentions:
                if m["surface"] in veto_set:
                    veto_dropped += 1
                    vetoed_pids.add(m["person_id"])
                else:
                    kept.append(m)
            mentions = kept

        # Per-(person, paragraph) appearance row, 原文 preferred over 胡注.
        for m in mentions:
            key = (juan_no, m["pid"])
            slot = appearances.setdefault(m["person_id"], {})
            cur = slot.get(key)
            if cur is None or (cur["source"] == "hu" and m["source"] == "main"):
                slot[key] = {"juan": juan_no, "pid": m["pid"],
                             "ce_year": m["ce_year"], "source": m["source"]}

        # Wave 44 — every person whose full name appeared in this 卷 is now "introduced"
        # for all subsequent 卷, so their 省称 are never re-gated downstream.
        introduced.update(intro_pos)

        para_ids = {m["pid"] for m in mentions}
        per_juan_counts[juan_no] = (len(mentions), len(para_ids))
        _write_text_retry(OUT / "mentions" / f"juan_{juan_no:03d}.json",
            json.dumps({"juan_no": juan_no, "version": 1, "mentions": mentions},
                       ensure_ascii=False, indent=2))

    # ── people.json — only people actually matched somewhere ──
    # A card whose every mention was vetoed no longer appears anywhere → drop it
    # from the shipped set so it does not pollute search / people.json.
    veto_orphaned = {pid for pid in vetoed_pids if pid not in appearances}
    seen_ids -= veto_orphaned
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
    print(f"narrow R1 truncation-repair: {r1_new} cards minted, "
          f"{r1_phantoms} phantom tails folded")
    print(f"R2 gloss-subject discovery: {r2_new} cards minted")
    print(f"rc4 封号-glue (titleglue) emitted: {feng_emitted}")
    print(f"rc5 谥号/封号 fragment cards dropped: {_FRAG_DROPPED}")
    print(f"title-misread (王/公/侯/君/主) cards dropped: {_TITLE_DROPPED}")
    print(f"llm precision-veto: {veto_dropped} mentions suppressed, "
          f"{len(veto_orphaned)} orphaned cards dropped")
    print(f"llm recall-binding: {_BIND_STATS['bound']} whole-卷 + "
          f"{_BIND_STATS['para_scoped']} para-scoped + {_BIND_STATS['anaphora']} 省称 "
          f"surfaces, {_BIND_STATS['minted']} cards minted")
    print(f"llm card curation: {card_relabeled} dynasty relabel + "
          f"{card_rebriefed} book brief + {card_merged} merged")
    print(f"lookback 卷 surfaces registered: {lookback_added}")
    print(f"gloss/office card 卷 surfaces filled: {gloss_fill_added}")
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
