"""Runner + comparison for the isolated two-stage pipeline.

Runs Stage 1 (detect) → Stage 2 (consolidate) over the requested juans, writes a
PARALLEL mention set to twostage/out/mentions/, and diffs it span-level against the
CURRENT pipeline output — WITHOUT touching the current pipeline or its files.

Comparison scope = current main-source `alias` mentions with surface length ≥2
(the class Stage 1 targets). Deferred/out-of-scope current classes (anaphora given-
binding, feng/role/gloss special passes, 胡注 notes) are reported separately so
LOST reflects only genuine full-name regressions.

Usage:
  python run.py                 # all juans present in the text dir
  python run.py 251 252 254     # only these juans
"""
from __future__ import annotations
import json, glob, os, sys, re, io, collections

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
import seed as S           # noqa: E402
import stage1, stage2      # noqa: E402

ROOT = os.path.join(HERE, "..", "..", "..", "web", "public", "text")
ROOT = os.path.normpath(ROOT)
OUT = os.path.join(HERE, "out")
ANAPHORA_DEFERRED = {"anaphora", "feng", "role", "gloss"}


def load_lexicons():
    people = json.load(open(f"{ROOT}/persons/people.json", encoding="utf-8"))["people"]
    surf3, lex2, known_long = set(), {}, set()
    for p in people:
        cn = p["canonical_name"]
        for nm in [cn] + [n.get("text", "") for n in p.get("names", [])]:
            if not nm:
                continue
            if len(nm) >= 3:
                known_long.add(nm)
                surf3.add(nm)
        if len(cn) == 2:
            sn = S._surname_of(cn)
            if sn and len(sn) == 1 and sn in S.CLEAN_SURNAMES and cn[1] not in stage1.NONNAME:
                lex2.setdefault(cn, []).append(p["id"])
    # NER surfaces (committed) are NOISY (王侍郎, 余非吾, 蒙圣恩, 黄巢北 all pass the
    # legacy surface filter). They are DISCOVERY hints, not confirmed names, so they
    # must NOT be drawn as blind literals. Keep them only in known_long (the G2
    # longest-match guard) so they can still SUPPRESS a bad 2-char split, never mint.
    ncf = os.path.join(HERE, "..", "ner_candidates.json")
    if os.path.exists(ncf):
        nc = json.load(open(ncf, encoding="utf-8"))
        for s, v in nc.items():
            if 3 <= len(s) <= 4 and S.ner_surface_ok(s, bool(v.get("d"))):
                known_long.add(s)
    return people, surf3, lex2, known_long


def load_current_mentions(juan_nums):
    by_juan = {}
    for j in juan_nums:
        f = f"{ROOT}/persons/mentions/juan_{j:03d}.json"
        if os.path.exists(f):
            by_juan[j] = json.load(open(f, encoding="utf-8")).get("mentions", [])
    return by_juan


def main(argv):
    people, surf3, lex2, known_long = load_lexicons()
    all_juans = sorted(int(re.search(r"juan_(\d+)", f).group(1))
                       for f in glob.glob(f"{ROOT}/juan_*.json"))
    juans = [int(a) for a in argv] if argv else all_juans
    # reference tagged juans come from the FULL current output (all juans), so era
    # resolution isn't starved when running a demo subset.
    cur_all = load_current_mentions(all_juans)
    surface_pids, tagged, pid_canon = stage2.build_reference(people, cur_all)

    all_cards = []
    gtot = collections.Counter()
    for j in juans:
        tf = f"{ROOT}/juan_{j:03d}.json"
        if not os.path.exists(tf):
            continue
        paras = json.load(open(tf, encoding="utf-8"))["paragraphs"]
        pf = f"{ROOT}/persons/pos_giv/juan_{j:03d}.json"
        giv = {}
        if os.path.exists(pf):
            giv = {k: set(v) for k, v in
                   json.load(open(pf, encoding="utf-8")).get("giv", {}).items()}
        cards, counters = stage1.detect_juan(j, paras, giv, surf3, lex2, known_long)
        all_cards.extend(cards)
        for k, v in counters.items():
            gtot[k] += v

    all_cards, new_clusters = stage2.consolidate(all_cards, surface_pids, tagged)
    out_by_juan = stage2.emit_mentions(all_cards)
    os.makedirs(f"{OUT}/mentions", exist_ok=True)
    for j in juans:
        ms = out_by_juan.get(j, [])
        json.dump({"juan_no": j, "version": "twostage", "mentions": ms},
                  open(f"{OUT}/mentions/juan_{j:03d}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    compare(juans, out_by_juan, cur_all, gtot, len(new_clusters))


def compare(juans, mine_by_juan, cur_all, gtot, n_new):
    agree = recover = lost2 = lost3 = 0
    rec_rows, lost_rows = [], []
    wang_254 = False
    for j in juans:
        cur = cur_all.get(j, [])
        # in-scope current = main-source alias, surface len>=2
        cur_scope = {(m["pid"], m["start"], m["end"], m["surface"])
                     for m in cur if m.get("source") == "main"
                     and m.get("kind") == "alias" and len(m.get("surface", "")) >= 2}
        mine = {(m["pid"], m["start"], m["end"], m["surface"])
                for m in mine_by_juan.get(j, [])}
        inter = cur_scope & mine
        agree += len(inter)
        for key in mine - cur_scope:
            recover += 1
            rec_rows.append((key[3], j, key))
            if j == 254 and key[3] == "\u738b\u7eea":
                wang_254 = True
        for key in cur_scope - mine:
            if len(key[3]) >= 3:
                lost3 += 1
            else:
                lost2 += 1
            lost_rows.append((key[3], j, key))
    # RECOVER precision proxy: 王 vs non-王
    wang_rec = sum(1 for r in rec_rows if r[0][0] == "\u738b")
    with open(f"{OUT}/compare_recover.txt", "w", encoding="utf-8") as fh:
        for surf, j, key in rec_rows:
            fh.write(f"{surf}\tj{j}\t{key}\n")
    with open(f"{OUT}/compare_lost.txt", "w", encoding="utf-8") as fh:
        for surf, j, key in lost_rows:
            fh.write(f"{surf}\tj{j}\t{key}\n")

    print("=== TWO-STAGE vs CURRENT (main-source, alias, surface≥2) ===")
    print(f"juans compared: {len(juans)}")
    print(f"Stage-1 detectors: lit3={gtot['lit3']}  giv2={gtot['giv2']}")
    print(f"Stage-1 guard rejects: G1={gtot['g1']} GW={gtot['gw']} "
          f"G2={gtot['g2']} G3={gtot['g3']}")
    print(f"new singleton clusters (no KB owner): {n_new}")
    print(f"AGREE   (both draw same span): {agree}")
    print(f"RECOVER (mine adds, current lacks): {recover}  "
          f"[王={wang_rec}, non-王={recover - wang_rec}]")
    print(f"LOST    (current draws, mine misses): len≥3={lost3}  len2={lost2}")
    print(f"王绪@254 drawn by two-stage: {'YES' if wang_254 else 'NO'}")
    print(f"(details → out/compare_recover.txt, out/compare_lost.txt)")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main(sys.argv[1:])
