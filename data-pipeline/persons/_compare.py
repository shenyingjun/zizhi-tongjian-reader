#!/usr/bin/env python3
"""Prep for model comparison: dump existing annotations + text for 卷 250-253."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEXT = HERE.parents[1] / "web" / "public" / "text"
OUT = TEXT / "persons"
ANN = HERE / "llm_annotations"

people = json.loads((OUT / "people.json").read_text(encoding="utf-8"))
pl = people["people"] if isinstance(people, dict) and "people" in people else people

for n in [250, 251, 252, 253]:
    # existing annotations
    af = ANN / f"juan_{n:03d}.tsv"
    existing = []
    if af.exists():
        for line in af.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                existing.append(line.split("\t")[0].strip())
    
    # text
    juan = json.loads((TEXT / f"juan_{n:03d}.json").read_text(encoding="utf-8"))
    blob = []
    for para in juan["paragraphs"]:
        blob.append(para.get("main", ""))
        for nt in para.get("notes", []):
            blob.append(nt.get("text", ""))
    fulltext = "\n".join(blob)
    
    # carded names in this juan
    carded = set()
    for p in pl:
        nm = p.get("canonical_name")
        if nm and nm in fulltext:
            carded.add(nm)
        for nm2 in (p.get("names") or []):
            t = nm2.get("text") if isinstance(nm2, dict) else nm2
            if t and t in fulltext:
                carded.add(t)
    
    print(f"\n卷{n:03d}: chars={len(fulltext)}, carded={len(carded)}, opus48_ann={len(existing)}")
    print(f"  OPUS4.8: {' '.join(existing)}")
