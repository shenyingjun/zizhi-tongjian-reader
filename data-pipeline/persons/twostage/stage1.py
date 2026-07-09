"""STAGE 1 — per-juan detection → occurrence cards.

Local-first: scans ONE juan at a time and emits one OccurrenceCard per detected
name span. It never consults people.json to decide *whether* to draw a line — it
uses only the juan text + dictionaries (name lexicon, NER surfaces) + POS·Giv +
boundary guards. Identity (which person) is deferred entirely to Stage 2.

Detectors:
  D-LIT3   literal match of any known surface, 3–4 chars (distinctive) — anywhere.
  D-GIV2   2-char surname+given names, gated by G1/G2/G3/王-gate (validated corpus
           regression guard stack). G3 (POS·Giv) carries the precision.
Boundary guards: G1 left (齐王建), G2 longest-match, G3 POS·Giv, 王 name-start gate,
  仆射 office-truncation, 谥号 posthumous-title.

Deferred (documented, NOT a regression of Stage 1): 省称 given-only anaphora
  (kind=anaphora) — that is a Stage-2 given-binding step, ported separately.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import seed as S  # noqa: E402

# ── boundary-guard character tables (from validated _corpus_regression) ──
NONNAME = set("左右上下大小前后中内外东西南北公侯王后妃氏子母兄弟长少太")
BLOCK1 = set("齐韩汉魏赵楚燕秦吴越梁陈宋鲁卫郑蔡曹许滕薛邾莒巴蜀晋隋周唐虞夏商"
             "大王侯公伯子男帝后妃太世储君贰")
BLOCK2 = {"中山", "常山", "长沙", "河间", "东平", "太子", "世子", "公子", "王子", "嗣王"}
# 王-surname name-start gate: 王X accepted only when 王 sits at a name boundary —
# preceded by punctuation or an apposition/office tail (屠者王绪 ✓); rejected when
# 王 follows a jun/place char (清河王庆 ✗ = 清河王|庆).
NAMESTART = set("、，。；：「」『』（）〔〕·！？　 \n\u0001,.;:!?\"'")
APPOS_TAIL = set("使军尉守丞卿郎牧将相者长监令傅师保都统帅侯公王后妃主民酋蛮曹夫")

WANG = "\u738b"
SHI_YUE = ("\u8c25\u66f0", "\u8c25\u4e3a")  # 谥曰 / 谥为
SHI = "\u8c25"                              # 谥
PU = "\u4ec6"                               # 仆
SHE = "\u5c04"                              # 射


def _sec_num(mt: str):
    """节 number 1..N from a paragraph lead, else None (mirrors build_persons)."""
    if not mt:
        return None
    o = ord(mt[0])
    if 0x2460 <= o <= 0x2473:
        return o - 0x2460 + 1
    if 0x3251 <= o <= 0x325F:
        return o - 0x3251 + 21
    if 0x32B1 <= o <= 0x32BF:
        return o - 0x32B1 + 36
    i = 0
    while i < len(mt) and mt[i].isdigit():
        i += 1
    if 0 < i <= 3 and i < len(mt) and "\u4e00" <= mt[i] <= "\u9fff":
        return int(mt[:i])
    return None


def detect_juan(juan_no, paras, giv, surf3, lex2, known_long):
    """Return (occurrence cards, guard counters) for one juan.

    surf3      : set of known KB surfaces length ≥3 (curated person names/aliases).
    lex2       : dict 2-char name -> [pids] (single-char surname; guarded).
    known_long : set of names length ≥3 (for the G2 longest-match guard).
    giv        : {para_id(str): set(offsets)} POS·Giv given-name char offsets.
    """
    cards = []
    sec = None
    maxL = min(max((len(x) for x in surf3), default=4), 10)
    counters = dict(g1=0, g2=0, g3=0, gw=0, lit3=0, giv2=0)
    for para in paras:
        para_id = para.get("id")
        t = para.get("main", "") or ""
        s = _sec_num(t)
        if s is not None:
            sec = s
        gset = giv.get(str(para_id), set())
        n = len(t)
        consumed = [False] * n
        for i in range(n):
            if consumed[i]:
                continue
            # ── D-LIT3: longest known surface (maxL..3 chars) ──
            hit = None
            for L in range(min(maxL, n - i), 2, -1):
                if t[i:i + L] in surf3 and not any(consumed[i:i + L]):
                    hit = t[i:i + L]
                    break
            if hit:
                L = len(hit)
                # 谥号 guard: 谥曰X / 谥为X / 谥X → posthumous title, not a person.
                if t[max(0, i - 2):i] in SHI_YUE or t[max(0, i - 1):i] == SHI:
                    continue
                for k in range(i, i + L):
                    consumed[k] = True
                cards.append(_card(juan_no, sec, para_id, i, i + L, hit, "lit3",
                                   para.get("ce_year")))
                counters["lit3"] += 1
                continue
            # ── D-GIV2: 2-char surname+given, guarded ──
            if i + 2 > n:
                continue
            cn = t[i:i + 2]
            if cn not in lex2:
                continue
            # G1 left-guard
            if (t[i - 1] if i > 0 else "") in BLOCK1 or t[i - 2:i] in BLOCK2:
                counters["g1"] += 1
                continue
            # G2 longest-match (2-char is a prefix/interior of a known ≥3 name here)
            if any(len(w) == 3 and w in known_long
                   for w in (t[i:i + 3], t[i - 1:i + 2], t[i - 2:i + 1])):
                counters["g2"] += 1
                continue
            # G3 POS·Giv
            if not ({i, i + 1} & gset):
                counters["g3"] += 1
                continue
            # 王 name-start gate (soft): 王X accepted when 王 sits at a name boundary
            # OR when the given char itself is POS·Giv-tagged (strong literal evidence
            # for a known 2-char card). Rejects 清河王|庆-style 爵+given truncation.
            if cn[0] == WANG:
                prev = t[i - 1] if i > 0 else "\u0001"
                if not (prev in NAMESTART or prev in APPOS_TAIL
                        or (i + 1) in gset):
                    counters["gw"] += 1
                    continue
            # 仆射 office-truncation (surface tail + next char spell 仆射)
            if cn[-1] == PU and t[i + 2:i + 3] == SHE:
                continue
            consumed[i] = consumed[i + 1] = True
            cards.append(_card(juan_no, sec, para_id, i, i + 2, cn, "giv2",
                               para.get("ce_year")))
            counters["giv2"] += 1
    return cards, counters


def _card(juan, sec, para_id, start, end, surface, evidence, ce):
    return {"juan": juan, "sec": sec, "para_id": para_id, "start": start,
            "end": end, "surface": surface, "evidence": evidence, "ce_year": ce}
