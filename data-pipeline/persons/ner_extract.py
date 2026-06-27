"""NER candidate harvester — proposes person surfaces from the 原文 itself.

Closes the coverage gap where a figure appears in the body text but was never
listed in a 白话导读 key_people block (so the guide-seeded layer never knew about
them). jieba's POS tagger (`nr` = person name) gives broad RECALL over the
corpus; classical Chinese makes its raw output noisy (verb-fragments, titles,
place names), so we gate every proposal through conservative guards:

  * surname-first: a 2-char name must begin with a known 姓; a 3-char name must
    begin with a known compound 姓 (司马/诸葛/…). This alone removes most places
    (秦/楚/阿房宫) and titles (大将军/二世).
  * stop chars: a name char-2 (or char-3) that is a grammatical particle, common
    verb, or admin/geographic unit signals a glued fragment (王闻/李信奔/赵地) →
    rejected.
  * seed.bad_auto_surface: shared ban on bare ranks / X王·X后·X氏 honorifics.

Survivors are written to ner_candidates.json as {surface: [juan_no, …]}. seed.py
merges them with the guide names, then its contiguity-window + per-卷 collision
guards do the final disambiguation. The product tier stays 'high' (自动识别);
the hand 'reviewed' core is untouched.

Run: python ner_extract.py   (writes ner_candidates.json next to this file)
"""
from __future__ import annotations
import json, glob, collections
from pathlib import Path

import jieba
import jieba.posseg as pseg

import seed as seed_mod  # ner_surface_ok / bad_auto_surface

jieba.setLogLevel(20)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TEXT = REPO / "web" / "public" / "text"
OUT = HERE / "ner_candidates.json"


def juans():
    for f in sorted(glob.glob(str(TEXT / "juan_*.json"))):
        p = Path(f)
        if not p.name.startswith("juan_"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if "paragraphs" not in d:
            continue
        yield d["juan_no"], d


def jieba_name_lexicon() -> set[str]:
    """Words jieba's own dictionary already tags `nr` — a trusted name lexicon
    (the `d` flag). Lets ambiguous-姓 names (高X/严X/武X) jieba knows through the
    seed guard while their bare surname-gate stays closed to glued phrases."""
    dict_path = Path(jieba.__file__).resolve().parent / "dict.txt"
    nr = set()
    for line in dict_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] == "nr":
            nr.add(parts[0])
    return nr


def name_gazetteer() -> set[str]:
    """Known person surfaces to pin into jieba's dictionary as `nr` BEFORE cutting,
    so the classical-text boundary errors that drop or truncate a known name
    (左丞李景让 → 左丞李/景/让, losing 李景让) are prevented. Sources are the layers we
    already trust: the hand cast (cast.py) and the 白话导读 key_people. This is the
    cheap, dependency-free alternative to a Classical-Chinese transformer (GuwenBERT
    / SikuBERT): it cannot discover names we don't know, but it makes jieba reliably
    keep the names we DO know intact, which is the entire truncation bug class."""
    from cast import PEOPLE
    g: set[str] = set()
    for p in PEOPLE:
        for s in [p.get("canonical_name", ""), *p.get("names", []), *p.get("match", [])]:
            if s and 2 <= len(s) <= 4:
                g.add(s)
    for nm in seed_mod.load_guide_people().keys():
        if nm and 2 <= len(nm) <= 4:
            g.add(nm)
    return {s for s in g if s not in seed_mod.WIKI_NONPERSON}


def main():
    nr_lex = jieba_name_lexicon()
    gaz = name_gazetteer()
    for nm in gaz:                       # pin known names so jieba won't split them
        jieba.add_word(nm, freq=10000, tag="nr")
    print(f"gazetteer seeded: {len(gaz)} known names")
    # surface -> {juan_no -> count}. Keep every length-2/3 nr token; the strong
    # surname / 干支 / stop-char guard runs in seed.ner_surface_ok at load time so
    # it can be tuned without re-running this slow pass.
    cand: dict[str, dict[int, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    raw_tokens = 0
    for jn, d in juans():
        text = "".join(p.get("main", "") for p in d["paragraphs"])
        for w, flag in pseg.cut(text):
            if not flag.startswith("nr"):
                continue
            raw_tokens += 1
            if 2 <= len(w) <= 3:
                cand[w][jn] += 1

    # {surface: {"j": [juan…], "n": total occurrences, "d": in jieba lexicon}}
    out = {s: {"j": sorted(jc.keys()), "n": sum(jc.values()), "d": 1 if s in nr_lex else 0}
           for s, jc in cand.items()}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    kept = {s: v for s, v in out.items() if seed_mod.ner_surface_ok(s, bool(v["d"]))}
    report = {
        "raw_nr_tokens": raw_tokens,
        "length_gated_surfaces": len(out),
        "passing_surface_guard": len(kept),
        "sample_names": sorted(kept.keys()),
    }
    (HERE / "ner_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"raw nr tokens: {raw_tokens}")
    print(f"length-gated surfaces: {len(out)}; passing guard now: {len(kept)}")
    print(f"wrote {OUT.name} + ner_report.json")


if __name__ == "__main__":
    main()
