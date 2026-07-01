"""Prep a 卷 for in-session LLM annotation.

Usage: python _prep_juan.py 254 255 256
Writes _j254.txt etc: (a) surfaces the pipeline ALREADY tags in that 卷 (so I only
annotate net-new people), then (b) the main-paragraph narrative text, numbered.
胡三省 注 is dropped from the dump (mostly phonetic glosses); names live in narrative.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEXT = REPO / "web" / "public" / "text"
MENT = TEXT / "persons" / "mentions"
PEOPLE = json.loads((TEXT / "persons" / "people.json").read_text(encoding="utf-8"))
BY_ID = {p["id"]: p for p in (PEOPLE if isinstance(PEOPLE, list)
                              else PEOPLE.get("people", []))}

for arg in sys.argv[1:]:
    n = int(arg)
    juan = json.loads((TEXT / f"juan_{n:03d}.json").read_text(encoding="utf-8"))
    mfile = MENT / f"juan_{n:03d}.json"
    tagged = set()
    canon = set()
    if mfile.exists():
        md = json.loads(mfile.read_text(encoding="utf-8"))
        for m in md.get("mentions", []):
            tagged.add(m.get("surface", ""))
            p = BY_ID.get(m.get("person_id"))
            if p:
                canon.add(p.get("canonical_name", ""))
    lines = [f"=== 卷{n:03d}  已标注人物(canonical, {len(canon)}) ==="]
    lines.append("、".join(sorted(canon)))
    lines.append(f"--- 已标注表面形({len(tagged)}) ---")
    lines.append("、".join(sorted(s for s in tagged if s)))
    lines.append("=== 正文（main only） ===")
    for i, para in enumerate(juan["paragraphs"], 1):
        mt = para.get("main", "")
        if mt:
            lines.append(f"{i}. {mt}")
    out = Path(f"_j{n:03d}.txt")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}  ({len(canon)} carded, {len(juan['paragraphs'])} paras)")
