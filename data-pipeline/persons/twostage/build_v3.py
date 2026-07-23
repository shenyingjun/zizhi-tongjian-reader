"""build_v3 — REAL two-stage local-first pipeline (independent of v1 mention JSON).

Unlike build_v2 (which unioned v1 mention files), this orchestrator PRODUCES every
underline itself from text + KB reference, by reusing the validated passes:

  Stage 1  (per-juan, local-first DETECTION → occurrence spans)
    * full names / 姓+given : stage1.detect_juan against the GLOBAL name lexicon
      (NOT the card-gated surfaces_for(juan)) — this is what recovers 王绪@254,
      a person whose card never listed juan 254.
    * feng  (封号+given, 任城王澄→元澄) : build_persons.extract_titleglue, run FIRST
      per paragraph so it reserves its span before the 姓+given detector (feng
      precedence preserved via an overlap-skip on the Stage-1 cards).
    * role  (魏主→reigning monarch)   : build_persons.extract_roles (reign tables).
    * gloss (「X，Y之Z也」戣→孔戣)      : build_persons.extract_gloss / extract_siblings.

  Stage 2  (CONSOLIDATE occurrence spans → person cards)
    * full-name spans resolve to a KB person by the era-window uniqueness merge
      (stage2.consolidate); an unresolved full name still underlines as a singleton.
    * feng/role/gloss spans already carry a KB-resolved identity; anaphora spans are
      bound in Stage 1.5 to a 节-local full-name antecedent.

  Stage 1.5 (节-local 省称 anaphora — build_persons.resolve_anaphora_pos)
    present_pids = the people actually detected in THIS juan (Stage 1), so the bare
    given 绪 borrows its surname from the full name 王绪 seen earlier in the same 节.
    Never crosses 节; single-char givens gated by the POS·Giv cache. KB = reference.

KB reference: build_persons' module-level seed KB (PEOPLE_MERGED / RULES / reign
tables). main()'s enrichment (gloss-mint / lookback / LLM binding) is NOT applied
here; any resulting recall gap surfaces as LOST in the comparison and is reported.

Usage:
  python build_v3.py                 # all juans present in the text dir
  python build_v3.py 251 252 254     # only these juans
"""
from __future__ import annotations
import json, glob, os, sys, re, io, collections
from pathlib import Path

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
import build_persons as B   # noqa: E402  (module-level: builds seed KB, no main())
import seed as S            # noqa: E402
import stage1, stage2       # noqa: E402
import pos_giv              # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "web", "public", "text"))
OUT = os.path.join(HERE, "out_v3")
POS_DIR = os.path.join(ROOT, "persons", "pos_giv")


def build_lexicon(people):
    """Global detection lexicon for Stage 1 (mirrors run.load_lexicons)."""
    surf3, lex2, known_long = set(), {}, set()
    for p in people:
        cn = p["canonical_name"]
        names = [(n.get("text", "") if isinstance(n, dict) else n)
                 for n in p.get("names", [])]
        for nm in [cn] + names:
            if not nm:
                continue
            if len(nm) >= 3:
                known_long.add(nm)
                surf3.add(nm)
        if len(cn) == 2:
            sn = S._surname_of(cn)
            if sn and len(sn) == 1 and sn in S.CLEAN_SURNAMES and cn[1] not in stage1.NONNAME:
                lex2.setdefault(cn, []).append(p["id"])
    ncf = os.path.join(HERE, "..", "ner_candidates.json")
    if os.path.exists(ncf):
        nc = json.load(open(ncf, encoding="utf-8"))
        for s, v in nc.items():
            if 3 <= len(s) <= 4 and S.ner_surface_ok(s, bool(v.get("d"))):
                known_long.add(s)
    return surf3, lex2, known_long


def load_current_mentions(juan_nums):
    by_juan = {}
    for j in juan_nums:
        f = f"{ROOT}/persons/mentions/juan_{j:03d}.json"
        if os.path.exists(f):
            by_juan[j] = json.load(open(f, encoding="utf-8")).get("mentions", [])
    return by_juan


def _overlap(consumed, s, e):
    return any(consumed[s:e])


def process_juan(juan_no, paras, by_id, canon_to_pids, surf3, lex2, known_long,
                 surface_pids, tagged):
    """Return the per-juan mention list produced entirely by the two-stage passes."""
    giv = {}
    pf = f"{POS_DIR}/juan_{juan_no:03d}.json"
    if os.path.exists(pf):
        giv = {k: set(v) for k, v in
               json.load(open(pf, encoding="utf-8")).get("giv", {}).items()}

    # ── Stage 1 full-name / 姓+given detection (global lexicon, local-first) ──
    s1cards, _ = stage1.detect_juan(juan_no, paras, giv, surf3, lex2, known_long)
    s1_by_para = collections.defaultdict(list)
    for c in s1cards:
        s1_by_para[c["para_id"]].append(c)

    rule_map = B.RULES.get(juan_no, {})
    cue_idx = B.build_role_cue_index(paras)
    mentions = []
    alias_cards = []           # Stage-1 full-name spans → consolidated in Stage 2

    for para in paras:
        pid_para = para["id"]
        ce = para.get("ce_year")
        t = para.get("main", "") or ""
        n = len(t)
        consumed = [False] * n

        # feng FIRST — reserve 封号+given spans (任城王澄) before 姓+given detection.
        feng_hits = B.extract_titleglue(t, ce, consumed, rule_map, by_id)
        for (s, e, pid_, surf) in feng_hits:
            mentions.append(_m(pid_para, ce, s, e, surf, pid_, "feng", by_id))

        # Stage-1 detected spans for this paragraph (skip anything feng already took).
        for c in sorted(s1_by_para.get(pid_para, []), key=lambda x: x["start"]):
            s, e = c["start"], c["end"]
            if _overlap(consumed, s, e):
                continue
            for k in range(s, e):
                consumed[k] = True
            alias_cards.append(c)     # identity assigned in Stage 2

        # role — reigning monarch by ce_year (reign tables), skips consumed spans.
        for (s, e, pid_, surf) in B.extract_roles(t, ce, consumed, by_id,
                                                  paras.index(para), cue_idx):
            mentions.append(_m(pid_para, ce, s, e, surf, pid_, "role", by_id))

        # gloss 「X，Y之Z也」 + sibling enumeration (bind 省姓 givens to their card).
        g_ments, _ = B.extract_gloss(t, rule_map, canon_to_pids, juan_no, by_id, consumed)
        for (s, e, pid_, surf) in g_ments:
            mentions.append(_m(pid_para, ce, s, e, surf, pid_, "gloss", by_id))
        for (s, e, pid_, surf) in B.extract_siblings(t, rule_map, canon_to_pids,
                                                     juan_no, by_id, consumed):
            mentions.append(_m(pid_para, ce, s, e, surf, pid_, "gloss", by_id))

    # ── Stage 2 — consolidate full-name occurrence spans → person ids ──
    alias_cards, _new = stage2.consolidate(alias_cards, surface_pids, tagged)
    for c in alias_cards:
        kind = "giv2" if c["evidence"] == "giv2" else "alias"
        mentions.append(_m(c["para_id"], c["ce_year"], c["start"], c["end"],
                           c["surface"], c["person_id"], kind, by_id))

    # ── Stage 1.5 — 节-local 省称 anaphora bound to Stage-1 present people ──
    present_pids = {m["person_id"] for m in mentions if m["person_id"] in by_id}
    if present_pids:
        exist_span = collections.defaultdict(set)
        full_anchor = collections.defaultdict(list)
        for m in mentions:
            exist_span[m["pid"]].add((m["start"], m["end"]))
            if m["kind"] != "anaphora" and m["person_id"] in present_pids:
                full_anchor[m["pid"]].append(
                    (m["start"], m["end"], m["person_id"], m["surface"]))
        giv_map = pos_giv.giv_for_juan(juan_no, paras, Path(POS_DIR))
        for (pid_para, ce_y, s, e, rpid, surf, kind) in B.resolve_anaphora_pos(
                paras, present_pids, by_id, giv_map, exist_span, full_anchor):
            mentions.append(_m(pid_para, ce_y, s, e, surf, rpid, kind, by_id))

    mentions.sort(key=lambda m: (m["pid"], m["start"]))
    return mentions


def _m(pid_para, ce, s, e, surface, person_id, kind, by_id):
    conf = by_id[person_id].get("confidence", "reviewed") if person_id in by_id else "low"
    return {"pid": pid_para, "ce_year": ce, "source": "main", "start": s, "end": e,
            "surface": surface, "person_id": person_id, "kind": kind, "confidence": conf}


def main(argv):
    people = B.PEOPLE_MERGED
    by_id = {p["id"]: p for p in people}
    canon_to_pids = collections.defaultdict(list)
    for pid_, p in by_id.items():
        canon_to_pids[p["canonical_name"]].append((pid_, set(p.get("juans", []))))
    surf3, lex2, known_long = build_lexicon(people)

    all_juans = sorted(int(re.search(r"juan_(\d+)", f).group(1))
                       for f in glob.glob(f"{ROOT}/juan_*.json"))
    juans = [int(a) for a in argv] if argv else all_juans
    cur_all = load_current_mentions(all_juans)
    surface_pids, tagged, _ = stage2.build_reference(people, cur_all)

    os.makedirs(f"{OUT}/mentions", exist_ok=True)
    out_by_juan = {}
    for j in juans:
        tf = f"{ROOT}/juan_{j:03d}.json"
        if not os.path.exists(tf):
            continue
        paras = json.load(open(tf, encoding="utf-8"))["paragraphs"]
        ms = process_juan(j, paras, by_id, canon_to_pids, surf3, lex2, known_long,
                           surface_pids, tagged)
        out_by_juan[j] = ms
        json.dump({"juan_no": j, "version": "twostage-v3", "mentions": ms},
                  open(f"{OUT}/mentions/juan_{j:03d}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    compare(juans, out_by_juan, cur_all)


def compare(juans, mine_by_juan, cur_all):
    agree = recover = lost2 = lost3 = 0
    rec_rows, lost_rows = [], []
    wang_254 = False
    by_kind_lost = collections.Counter()
    for j in juans:
        cur = cur_all.get(j, [])
        cur_scope = {(m["pid"], m["start"], m["end"], m["surface"]): m.get("kind")
                     for m in cur if m.get("source") == "main"}
        mine = {(m["pid"], m["start"], m["end"], m["surface"])
                for m in mine_by_juan.get(j, [])}
        curkeys = set(cur_scope)
        agree += len(curkeys & mine)
        for key in mine - curkeys:
            recover += 1
            rec_rows.append((key[3], j, key))
            if j == 254 and key[3] == "\u738b\u7eea":
                wang_254 = True
        for key in curkeys - mine:
            if len(key[3]) >= 3:
                lost3 += 1
            else:
                lost2 += 1
            by_kind_lost[cur_scope[key] or "?"] += 1
            lost_rows.append((key[3], j, key, cur_scope[key]))
    with open(f"{OUT}/compare_recover.txt", "w", encoding="utf-8") as fh:
        for surf, j, key in sorted(rec_rows):
            fh.write(f"{surf}\tj{j}\t{key}\n")
    with open(f"{OUT}/compare_lost.txt", "w", encoding="utf-8") as fh:
        for surf, j, key, kind in sorted(lost_rows):
            fh.write(f"{surf}\tj{j}\t{kind}\t{key}\n")

    wang_rec = sum(1 for r in rec_rows if r[0][0] == "\u738b")
    print("=== TWO-STAGE v3 vs CURRENT (all main-source spans) ===")
    print(f"juans compared: {len(juans)}")
    print(f"AGREE   (both draw same span): {agree}")
    print(f"RECOVER (v3 adds, current lacks): {recover}  "
          f"[\u738b={wang_rec}, non-\u738b={recover - wang_rec}]")
    print(f"LOST    (current draws, v3 misses): len\u22653={lost3}  len2={lost2}")
    print(f"LOST by kind: {dict(by_kind_lost)}")
    print(f"\u738b\u7eea@254 drawn by v3: {'YES' if wang_254 else 'NO'}")
    print(f"(details -> {OUT}/compare_recover.txt, {OUT}/compare_lost.txt)")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main(sys.argv[1:])
