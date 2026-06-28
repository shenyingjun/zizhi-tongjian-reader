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
import json, datetime
from pathlib import Path

from cast import PEOPLE
import seed as seed_mod

JUANS = list(range(1, 295))  # all 294 卷 (hand 'reviewed' + auto seed layer)
REPO = Path(__file__).resolve().parents[2]
TEXT = REPO / "web" / "public" / "text"
OUT = TEXT / "persons"

# Merge the hand-curated cast with the deterministic auto-seed layer. PEOPLE is
# the combined people list (hand juans grown across batches, auto names split by
# contiguity); RULES is the collision-free {juan: {surface: person_id}} table.
# ANAPHORA_RULES is the Wave 5 single-char 省称 anchor table {juan: {char: pid}}.
PEOPLE_MERGED, RULES, ANAPHORA_RULES, SEED_STATS = seed_mod.build_seed(PEOPLE, JUANS)

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

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mentions").mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    # appearances[pid] = { (juan, pid_para): {juan, pid, ce_year, source} }
    appearances: dict[str, dict[tuple, dict]] = {}
    per_juan_counts: dict[int, tuple[int, int]] = {}
    anaphora_emitted = 0

    for juan_no in JUANS:
        jf = TEXT / f"juan_{juan_no:03d}.json"
        if not jf.exists():
            continue
        juan = json.loads(jf.read_text(encoding="utf-8"))
        surfaces = surfaces_for(juan_no)
        admitted = ANAPHORA_RULES.get(juan_no, set())
        char_anchor: dict[str, str] = {}  # given-char -> nearest antecedent person_id (卷-local)
        mentions = []
        for para in juan["paragraphs"]:
            pid = para["id"]
            ce = para.get("ce_year")
            main_text = para.get("main", "")
            alias_hits = extract(main_text, surfaces)
            for (s, e, pid_, surf) in alias_hits:
                mentions.append({"pid": pid, "ce_year": ce, "source": "main",
                                 "start": s, "end": e, "surface": surf,
                                 "person_id": pid_, "kind": "alias",
                                 "confidence": by_id[pid_].get("confidence", "reviewed")})
                seen_ids.add(pid_)
            # Wave 5 P2 — single-char 省称 anaphora on the 原文, each bound to its
            # nearest preceding full-name antecedent (disambiguates 元胄 vs 宇文胄).
            if admitted:
                consumed = [False] * len(main_text)
                anchor_events = []
                for (s, e, pid_, surf) in alias_hits:
                    for k in range(s, e):
                        consumed[k] = True
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
    print(f"title-glue aliases bound: {SEED_STATS['glue_bound']}"
          f"   missing canonical (cast to add): {SEED_STATS['glue_missing']}")
    hand_ids = {p["id"] for p in PEOPLE_MERGED if p.get("confidence") == "reviewed"}
    unmatched = sorted(hand_ids - seen_ids)
    if unmatched:
        print(f"  reviewed-but-unmatched ({len(unmatched)}):",
              ", ".join(by_id[u]["canonical_name"] for u in unmatched))


if __name__ == "__main__":
    main()
