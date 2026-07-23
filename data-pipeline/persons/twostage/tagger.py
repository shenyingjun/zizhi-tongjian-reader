"""AGENT 1 — TAGGER (Stage 1).

Owns exactly one question: *where does an underline go?*  It scans one 卷 of text
(+ POS·Giv) and emits identity-less OccurrenceCards. It never imports people.json and
never decides *who* a span is — that is Agent 2 (Identifier).

Contract (the ONLY thing that crosses the boundary):
    OccurrenceCard = {
        juan, para_id, start, end, surface,
        chunk_type,   # fuxing | xing_ming | jue_glue | office_glue | appos | bare_given
        features: {left_char, left_is_fief, left_is_appos, left_is_boundary,
                   surname, given, pos_giv},
        ce_year, sec,
    }

Detectors (greedy longest structural chunk), all identity-independent:
  - fuxing     复姓 (COMPOUND / COMPOUND3) + 1-2 POS·Giv given chars   (司马懿, 长孙无忌)
  - xing_ming  clean 姓 + 1-2 POS·Giv given chars                       (王导, 王镇恶)
Guards (identity-independent only):
  - 复姓 left-guard  : a clean 姓 that is really a 复姓 tail (司马懿→马) is NOT a head
  - BLOCK1/BLOCK2   : 齐王建 / 中山 state+爵 boundaries
  - 爵-head gate     : 王/公/侯 accepted as a head only at a real boundary / appos / POS
  - bad_auto_surface: place / 官号 / clan / role-glue blocklist (seed)
  - stop-given / 仆射: particle-tail & office-truncation
"""
from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import seed as S  # noqa: E402

# boundary-guard tables (validated in the corpus-regression probe) ----------------
CLEAN = S.CLEAN_SURNAMES
COMPOUND = S.COMPOUND          # 2-char 复姓 (司马, 长孙, 慕容 …)
COMPOUND3 = S.COMPOUND3        # 3-char 胡/鲜卑 clan (阿史那 …)
NONNAME = set("左右上下大小前后中内外东西南北公侯王后妃氏子母兄弟长少太")
BLOCK1 = set("齐韩汉魏赵楚燕秦吴越梁陈宋鲁卫郑蔡曹许滕薛邾莒巴蜀晋隋周唐虞夏商"
             "大王侯公伯子男帝后妃太世储君贰")
BLOCK2 = {"中山", "常山", "长沙", "河间", "东平", "太子", "世子", "公子", "王子", "嗣王"}
NAMESTART = set("、，。；：「」『』（）〔〕·！？　 \n\u0001,.;:!?\"'")
APPOS_TAIL = set("使军尉守丞卿郎牧将相者长监令傅师保都统帅侯公王后妃主民酋蛮曹夫")
FANGWEI = set("东西南北")
JUE_HEAD = set("\u738b\u516c\u4faf")   # 王 公 侯 — 爵号-homograph surnames
PU, SHE = "\u4ec6", "\u5c04"           # 仆 / 射
STOP_GIVEN = set("以为之其而则也矣乃与及若因故者所焉耳于又亦皆既未无"
                 "人二三四五六七八九十众师兵家门城边内外前后中大小上下"
                 "至足肥牧望军民国主王公侯时日年月")


def _feat(t, i, end, surname, given):
    left = t[i - 1] if i > 0 else ""
    return {
        "left_char": left,
        "left_is_fief": left in FANGWEI,
        "left_is_appos": left in APPOS_TAIL,
        "left_is_boundary": (left in NAMESTART) or i == 0,
        "surname": surname,
        "given": given,
    }


def _card(juan, sec, para_id, i, end, t, ctype, surname, given, ce):
    return {"juan": juan, "sec": sec, "para_id": para_id, "start": i, "end": end,
            "surface": t[i:end], "chunk_type": ctype,
            "features": _feat(t, i, end, surname, given), "ce_year": ce}


def detect_para(juan, sec, para_id, t, gset, ce, consumed=None):
    """Emit identity-less OccurrenceCards for one paragraph. `gset` = POS·Giv offsets.
    When `consumed` (list[bool]) is given, reserved spans are skipped & marked."""
    n = len(t)
    if consumed is None:
        consumed = [False] * n
    out = []
    i = 0
    while i < n:
        if consumed[i]:
            i += 1
            continue
        c = t[i]

        # ── fuxing: 复姓 (2-char COMPOUND / 3-char COMPOUND3) + 1-2 POS·Giv given ──
        cp = None
        if t[i:i + 3] in COMPOUND3:
            cp = 3
        elif t[i:i + 2] in COMPOUND:
            cp = 2
        if cp:
            g = 0
            while g < 2 and (i + cp + g) < n and (i + cp + g) in gset:
                g += 1
            if g >= 1 and not any(consumed[i:i + cp + g]):
                surf = t[i:i + cp + g]
                if not S.bad_auto_surface(surf):
                    for k in range(i, i + cp + g):
                        consumed[k] = True
                    out.append(_card(juan, sec, para_id, i, i + cp + g, t,
                                     "fuxing", t[i:i + cp], t[i + cp:i + cp + g], ce))
                    i += cp + g
                    continue
            # a bare 复姓 with no given still blocks a wrong single-姓 read below
            i += cp
            continue

        if c not in CLEAN:
            i += 1
            continue
        # 复姓 left-guard: this clean 姓 is really the tail of a 复姓 (司马懿 → 马)
        if t[i - 1:i + 1] in COMPOUND or t[i - 2:i + 1] in COMPOUND3:
            i += 1
            continue
        # state+爵 boundary (齐王建, 中山)
        if (t[i - 1] if i > 0 else "") in BLOCK1 or t[i - 2:i] in BLOCK2:
            i += 1
            continue
        # 爵-head gate: 王/公/侯 head only at a real boundary / appos / POS-given right
        if c in JUE_HEAD:
            prev = t[i - 1] if i > 0 else "\u0001"
            if prev in FANGWEI:                       # 河西王 = 爵号 glue
                i += 1
                continue
            if not (prev in NAMESTART or prev in APPOS_TAIL or (i + 1) in gset):
                i += 1
                continue

        # ── xing_ming: clean 姓 + two POS·Giv given (3-char) ──
        if i + 2 < n and (i + 1) in gset and (i + 2) in gset \
                and not any(consumed[i:i + 3]):
            surf = t[i:i + 3]
            if not S.bad_auto_surface(surf):
                for k in range(i, i + 3):
                    consumed[k] = True
                out.append(_card(juan, sec, para_id, i, i + 3, t,
                                 "xing_ming", c, t[i + 1:i + 3], ce))
                i += 3
                continue
        # ── xing_ming: clean 姓 + one POS·Giv given (2-char) ──
        if i + 1 < n and (i + 1) in gset and t[i + 1] not in STOP_GIVEN \
                and not any(consumed[i:i + 2]):
            if not (t[i] == PU and t[i + 2:i + 3] == SHE):
                surf = t[i:i + 2]
                if not S.bad_auto_surface(surf):
                    consumed[i] = consumed[i + 1] = True
                    out.append(_card(juan, sec, para_id, i, i + 2, t,
                                     "xing_ming", c, t[i + 1], ce))
                    i += 2
                    continue
        i += 1
    return out


def _sec_num(mt):
    if not mt:
        return None
    o = ord(mt[0])
    if 0x2460 <= o <= 0x2473:
        return o - 0x2460 + 1
    if 0x3251 <= o <= 0x325F:
        return o - 0x3251 + 21
    if 0x32B1 <= o <= 0x32BF:
        return o - 0x32B1 + 36
    return None


def detect_juan(juan_no, paras, giv):
    """Return [OccurrenceCard] for one 卷. `giv` = {para_id(int): set(offsets)}."""
    out = []
    sec = None
    for para in paras:
        pid = para.get("id")
        t = para.get("main", "") or ""
        sn = _sec_num(t)
        if sn is not None:
            sec = sn
        out.extend(detect_para(juan_no, sec, pid, t, giv.get(pid, set()),
                               para.get("ce_year")))
    return out
