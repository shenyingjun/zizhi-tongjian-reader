"""Backfill llm_annotations/juan_NNN.jsonl to FULL CAST.

Full cast = every person appearing in the 卷 =
  (pipeline-carded canonicals tagged in that 卷)  +  (LLM net-new entries).

We preserve the rich LLM net-new lines (role/aliases/evidence) as carded:false,
and add carded:true lines for every other person the pipeline already tags in
that 卷, pulling aliases-in-此卷 from the mention surfaces and role from the brief.

The build loader ignores the `carded` field and skips already-carded names, so
full-cast files stay drop-in compatible.

Usage: python _fullcast.py 250 251 252 253 254 255
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEXT = REPO / "web" / "public" / "text"
MENT = TEXT / "persons" / "mentions"
LLM_ANN = Path(__file__).resolve().parent / "llm_annotations"
PEOPLE = json.loads((TEXT / "persons" / "people.json").read_text(encoding="utf-8"))
PLIST = PEOPLE if isinstance(PEOPLE, list) else PEOPLE.get("people", [])
BY_ID = {p["id"]: p for p in PLIST}


def load_netnew(n):
    """existing net-new jsonl -> (ordered list of raw recs, set of names)."""
    jl = LLM_ANN / f"juan_{n:03d}.jsonl"
    recs, header = [], []
    if not jl.exists():
        return recs, header
    for raw in jl.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s.startswith("#"):
            header.append(raw)
            continue
        if not s:
            continue
        try:
            recs.append(json.loads(s))
        except json.JSONDecodeError:
            pass
    return recs, header


for arg in sys.argv[1:]:
    n = int(arg)
    mfile = MENT / f"juan_{n:03d}.json"
    md = json.loads(mfile.read_text(encoding="utf-8"))
    # group mention surfaces by person
    surf_by_pid = {}
    canon_by_pid = {}
    for m in md.get("mentions", []):
        pid = m.get("person_id")
        p = BY_ID.get(pid)
        if not p:
            continue
        canon = p.get("canonical_name", "")
        canon_by_pid[pid] = canon
        surf_by_pid.setdefault(pid, set()).add(m.get("surface", ""))

    netnew, header = load_netnew(n)
    netnew_names = {(r.get("name") or "").strip() for r in netnew}

    lines = header[:] if header else [f"# 卷{n:03d} FULL CAST (pipeline \u222a LLM net-new)"]

    # 1) LLM net-new entries first (rich), tagged carded:false unless pipeline now has them
    carded_names = set(canon_by_pid.values())
    for r in netnew:
        nm = (r.get("name") or "").strip()
        rec = {
            "name": nm,
            "aliases": r.get("aliases", []),
            "role": r.get("role", ""),
            "confidence": r.get("confidence", "high"),
            "carded": nm in carded_names,
            "evidence": r.get("evidence", ""),
        }
        lines.append(json.dumps(rec, ensure_ascii=False))

    # 2) every other pipeline-carded person in this 卷 -> carded:true
    for pid, canon in sorted(canon_by_pid.items(), key=lambda kv: kv[1]):
        if not canon or canon in netnew_names:
            continue
        p = BY_ID.get(pid, {})
        aliases = sorted(s for s in surf_by_pid.get(pid, set())
                         if s and s != canon)
        rec = {
            "name": canon,
            "aliases": aliases,
            "role": (p.get("brief") or "").strip(),
            "confidence": "high",
            "carded": True,
            "evidence": "",
        }
        lines.append(json.dumps(rec, ensure_ascii=False))

    out = LLM_ANN / f"juan_{n:03d}.jsonl"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = len(lines) - len(header if header else [1])
    print(f"juan_{n:03d}: {len(netnew_names)} net-new + "
          f"{len(carded_names - netnew_names)} carded = {total} cast")
