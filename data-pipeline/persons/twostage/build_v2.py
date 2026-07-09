"""BUILD V2 — emit the production-format person dataset for the *new* (two-stage
local-first) pipeline into web/public/text/persons-v2/.

Product policy (ADD-only, precision-first): v2 is a strict SUPERSET of v1.

  v2 mentions(juan) = ALL v1 mentions(juan)                     # conformed passes
                    ∪ two-stage full-name spans not overlapping  # rebuilt layer
                      any v1 span in the same paragraph

So every underline the current pipeline draws is preserved (feng / role / 省称
anaphora / gloss are reused as-is — the "conform existing rules" half of the
rebuild), while the full-name layer is REBUILT local-first and contributes the
4,391 recovered spans (100% exact KB names, incl. 王绪@254) the card-gated v1
pipeline missed.

Outputs (mirrors text/persons/ so the frontend variant toggle just swaps dirs):
  text/persons-v2/mentions/juan_NNN.json   production format
  text/persons-v2/people.json              v1 people + synthetic singleton cards
  text/persons-v2/appearances.json         copied from v1 (cross-卷 index reuse)

Run: python build_v2.py            # all juans
     python build_v2.py 251 254    # subset (still reads full v1 for reference)
"""
from __future__ import annotations
import json, glob, os, sys, re, io, collections, shutil

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
import stage1, stage2      # noqa: E402
import run as R            # noqa: E402  (reuse load_lexicons / ROOT)

ROOT = R.ROOT
V1 = os.path.join(ROOT, "persons")
V2 = os.path.join(ROOT, "persons-v2")


def _overlaps(s, e, spans):
    return any(not (e <= a or b <= s) for (a, b) in spans)


def main(argv):
    people, surf3, lex2, known_long = R.load_lexicons()
    all_juans = sorted(int(re.search(r"juan_(\d+)", f).group(1))
                       for f in glob.glob(f"{ROOT}/juan_*.json"))
    juans = [int(a) for a in argv] if argv else all_juans

    # era reference from the FULL v1 output (so subset runs still resolve identity)
    cur_all = R.load_current_mentions(all_juans)
    surface_pids, tagged, pid_canon = stage2.build_reference(people, cur_all)

    # ── Stage 1 + Stage 2 over the requested juans → full-name occurrence spans ──
    all_cards = []
    for j in juans:
        tf = f"{ROOT}/juan_{j:03d}.json"
        if not os.path.exists(tf):
            continue
        paras = json.load(open(tf, encoding="utf-8"))["paragraphs"]
        pf = f"{V1}/pos_giv/juan_{j:03d}.json"
        giv = {}
        if os.path.exists(pf):
            giv = {k: set(v) for k, v in
                   json.load(open(pf, encoding="utf-8")).get("giv", {}).items()}
        cards, _ = stage1.detect_juan(j, paras, giv, surf3, lex2, known_long)
        all_cards.extend(cards)
    all_cards, new_clusters = stage2.consolidate(all_cards, surface_pids, tagged)
    ts_by_juan = collections.defaultdict(list)
    for c in all_cards:
        ts_by_juan[c["juan"]].append(c)

    # ── emit per-juan union (v1 complete + non-overlapping two-stage spans) ──
    os.makedirs(f"{V2}/mentions", exist_ok=True)
    used_new = set()                        # new: cluster ids that actually appear
    added_total = kept_total = 0
    for j in juans:
        v1f = f"{V1}/mentions/juan_{j:03d}.json"
        if os.path.exists(v1f):
            v1doc = json.load(open(v1f, encoding="utf-8"))
            v1ms = v1doc.get("mentions", [])
            version = v1doc.get("version", 1)
        else:
            v1ms, version = [], 1
        # existing spans per paragraph guard double-draw
        spans_by_para = collections.defaultdict(list)
        for m in v1ms:
            spans_by_para[m["pid"]].append((m["start"], m["end"]))
        out = list(v1ms)
        kept_total += len(v1ms)
        for c in ts_by_juan.get(j, []):
            pid_para, s, e = c["para_id"], c["start"], c["end"]
            if _overlaps(s, e, spans_by_para[pid_para]):
                continue                    # v1 already underlines here — no dupe
            spans_by_para[pid_para].append((s, e))
            person_id = c["person_id"]
            if person_id.startswith("new:"):
                used_new.add(person_id)
            out.append({
                "pid": pid_para, "ce_year": c.get("ce_year"), "source": "main",
                "start": s, "end": e, "surface": c["surface"],
                "person_id": person_id, "kind": "alias", "confidence": "high",
            })
            added_total += 1
        out.sort(key=lambda m: (m["pid"], m["start"]))
        json.dump({"juan_no": j, "version": version, "mentions": out},
                  open(f"{V2}/mentions/juan_{j:03d}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    # ── people.json = v1 people + synthetic minimal cards for used new: clusters ──
    ppl_doc = json.load(open(f"{V1}/people.json", encoding="utf-8"))
    synth = []
    for nid in sorted(used_new):
        surf = nid[len("new:"):]
        synth.append({
            "id": nid, "canonical_name": surf,
            "names": [{"text": surf, "type": "name"}],
            "brief": "两阶段新管线在本卷文本中识别到的姓名，暂无编者身份信息。",
            "identity": "两阶段新管线在本卷文本中识别到的姓名，暂无编者身份信息。",
            "confidence": "low",
        })
    ppl_doc["people"] = ppl_doc["people"] + synth
    json.dump(ppl_doc, open(f"{V2}/people.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # ── appearances.json — reuse v1 cross-卷 index (new: singletons simply absent) ──
    if os.path.exists(f"{V1}/appearances.json"):
        shutil.copyfile(f"{V1}/appearances.json", f"{V2}/appearances.json")

    print(f"v2 built: juans={len(juans)}  v1 kept={kept_total}  "
          f"two-stage added={added_total}  synthetic cards={len(synth)}")
    print(f"  → {V2}")


if __name__ == "__main__":
    main(sys.argv[1:])
