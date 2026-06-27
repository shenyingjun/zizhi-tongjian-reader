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
PEOPLE_MERGED, RULES, SEED_STATS = seed_mod.build_seed(PEOPLE, JUANS)


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

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mentions").mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    # appearances[pid] = { (juan, pid_para): {juan, pid, ce_year, source} }
    appearances: dict[str, dict[tuple, dict]] = {}
    per_juan_counts: dict[int, tuple[int, int]] = {}

    for juan_no in JUANS:
        jf = TEXT / f"juan_{juan_no:03d}.json"
        if not jf.exists():
            continue
        juan = json.loads(jf.read_text(encoding="utf-8"))
        surfaces = surfaces_for(juan_no)
        mentions = []
        for para in juan["paragraphs"]:
            pid = para["id"]
            ce = para.get("ce_year")
            for (s, e, pid_, surf) in extract(para.get("main", ""), surfaces):
                mentions.append({"pid": pid, "ce_year": ce, "source": "main",
                                 "start": s, "end": e, "surface": surf,
                                 "person_id": pid_,
                                 "confidence": by_id[pid_].get("confidence", "reviewed")})
                seen_ids.add(pid_)
            for ni, note in enumerate(para.get("notes", [])):
                for (s, e, pid_, surf) in extract(note.get("text", ""), surfaces):
                    mentions.append({"pid": pid, "ce_year": ce, "source": "hu",
                                     "note_index": ni, "start": s, "end": e,
                                     "surface": surf, "person_id": pid_,
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
    hand_ids = {p["id"] for p in PEOPLE_MERGED if p.get("confidence") == "reviewed"}
    unmatched = sorted(hand_ids - seen_ids)
    if unmatched:
        print(f"  reviewed-but-unmatched ({len(unmatched)}):",
              ", ".join(by_id[u]["canonical_name"] for u in unmatched))


if __name__ == "__main__":
    main()
