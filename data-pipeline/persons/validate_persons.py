"""Validate the generated person sidecars (P0, Option 1).

Run AFTER build_persons.py. Checks the invariants the app relies on, so a bad
cast edit fails loudly here instead of leaking a spoiler or a wrong identity
into the reader:

  1. Every shipped person has a spoiler-safe `brief` and a non-empty `identity`.
  2. No `match` surface is a banned bare/ambiguous form (single char or a known
     ambiguous title used alone).
  3. Every mention offset lies within its paragraph's source text and the
     `surface` actually occurs there (the sidecar is consistent with the text).
  4. Every mention.person_id exists in people.json.
  5. appearances.json is consistent with the per-卷 mentions (same set of
     (person, juan, pid) appearance rows) and is sorted in reading order.
  6. All JSON parses; manifest counts match reality.

Exit non-zero on any failure.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEXT = REPO / "web" / "public" / "text"
OUT = TEXT / "persons"
JUANS = list(range(1, 295))

# Bare/ambiguous surfaces that must never be matched on their own. Single
# characters are rejected programmatically; these are multi-char-but-ambiguous
# titles that resolve only with context. NOTE: appellations that are unique
# within the shipped 卷 range (e.g. 二世 → only 胡亥 in 卷7–10, 汉王 → only
# 刘邦) are intentionally NOT banned — they are reviewed, single-referent forms.
BANNED = {
    "王", "公", "侯", "君", "太子", "公子", "孝公", "惠王", "武王", "文王",
    "威王", "成侯", "太后", "大王", "上", "帝",
}

errors: list[str] = []
warnings: list[str] = []


def err(msg: str): errors.append(msg)
def warn(msg: str): warnings.append(msg)


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        err(f"{p.name}: cannot parse JSON ({e})")
        return None


def main() -> int:
    people_file = load(OUT / "people.json")
    appearances_file = load(OUT / "appearances.json")
    manifest = load(OUT / "manifest.json")
    if people_file is None or appearances_file is None or manifest is None:
        _report()
        return 1

    people = {p["id"]: p for p in people_file["people"]}

    # 0: no surface may resolve to two different people within one 卷. This is
    # the core disambiguation invariant — a collision would silently mislabel a
    # mention. Derived from the emitted mentions (surface+juan -> person ids).
    surf_owners: dict[tuple, set] = {}

    # 1 + 2: per-person identity + surface safety.
    for pid, p in people.items():
        if not p.get("brief", "").strip():
            err(f"person {pid} ({p.get('canonical_name')}): empty brief (spoiler risk)")
        if not p.get("identity", "").strip():
            err(f"person {pid} ({p.get('canonical_name')}): empty identity")

    # 3/4/5: walk per-卷 mentions, cross-check text + people + appearances.
    rebuilt: dict[str, set] = {}
    total_mentions = 0
    for juan_no in JUANS:
        mf = OUT / "mentions" / f"juan_{juan_no:03d}.json"
        if not mf.exists():
            continue
        mdata = load(mf)
        if mdata is None:
            continue
        juan = load(TEXT / f"juan_{juan_no:03d}.json")
        if juan is None:
            continue
        paras = {para["id"]: para for para in juan["paragraphs"]}
        for m in mdata["mentions"]:
            total_mentions += 1
            surf = m["surface"]
            kind = m.get("kind", "alias")
            # Single-char surfaces are only permitted for position-resolved
            # anaphora (省称回指), where the offset pins one specific person; a
            # single-char *alias* would be a corpus-wide false-match storm.
            if len(surf) <= 1 and kind != "anaphora":
                err(f"juan{juan_no:03d} pid{m['pid']}: single-char surface '{surf}'")
            # Bare/ambiguous appellations are only permitted as 'role' mentions,
            # where the reigns resolver has bound them to one monarch by year.
            if surf in BANNED and kind != "role":
                err(f"juan{juan_no:03d} pid{m['pid']}: banned ambiguous surface '{surf}'")
            pid = m.get("person_id")
            if pid not in people:
                err(f"juan{juan_no:03d} pid{m['pid']}: mention person_id '{pid}' not in people.json")
            para = paras.get(m["pid"])
            if para is None:
                err(f"juan{juan_no:03d}: mention references missing paragraph {m['pid']}")
                continue
            if m["source"] == "main":
                blob = para.get("main", "")
            else:
                notes = para.get("notes", [])
                ni = m.get("note_index", 0)
                blob = notes[ni].get("text", "") if ni < len(notes) else ""
            if blob[m["start"]:m["end"]] != surf:
                err(f"juan{juan_no:03d} pid{m['pid']}: offset [{m['start']}:{m['end']}] "
                    f"!= surface '{surf}' (text mismatch)")
            if pid:
                rebuilt.setdefault(pid, set()).add((juan_no, m["pid"]))
                # Only 'alias' mentions are held to the 1:1 surface invariant.
                # 'anaphora'/'role' mentions are position-resolved, so the same
                # surface legitimately maps to different people at different
                # offsets within one 卷 (that is the whole point of Wave 5).
                if kind == "alias":
                    surf_owners.setdefault((juan_no, surf), set()).add(pid)

    # 0: surface collision report (after walking every 卷's mentions).
    for (juan_no, surf), ids in surf_owners.items():
        if len(ids) > 1:
            err(f"juan{juan_no:03d}: surface '{surf}' resolves to multiple people {sorted(ids)}")

    # 5: appearances index must equal the rebuilt (person, juan, pid) set.
    appearances = appearances_file["appearances"]
    for pid, rows in appearances.items():
        keyset = {(r["juan"], r["pid"]) for r in rows}
        if keyset != rebuilt.get(pid, set()):
            err(f"appearances[{pid}] disagrees with mentions "
                f"(index {len(keyset)} vs mentions {len(rebuilt.get(pid, set()))})")
        ordered = sorted(rows, key=lambda r: (r["juan"], r["pid"]))
        if rows != ordered:
            err(f"appearances[{pid}] not in reading order")
        if pid not in people:
            err(f"appearances[{pid}]: person not in people.json")
    for pid in rebuilt:
        if pid not in appearances:
            err(f"person {pid} has mentions but no appearances row")

    # 6: manifest sanity.
    if manifest.get("mention_count") != total_mentions:
        warn(f"manifest.mention_count {manifest.get('mention_count')} != actual {total_mentions}")
    if manifest.get("people_count") != len(people):
        warn(f"manifest.people_count {manifest.get('people_count')} != actual {len(people)}")

    _report()
    print(f"\nchecked {len(people)} people, {total_mentions} mentions across "
          f"{sum(1 for j in JUANS if (OUT/'mentions'/f'juan_{j:03d}.json').exists())} 卷")
    return 1 if errors else 0


def _report():
    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    if not errors:
        print("OK — all person-data invariants hold")


if __name__ == "__main__":
    sys.exit(main())
