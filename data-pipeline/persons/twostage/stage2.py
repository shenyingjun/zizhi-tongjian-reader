"""STAGE 2 — consolidate occurrence cards → person cards + mention spans.

Occurrence cards carry NO identity. Stage 2 assigns each to a person by the
local-first merge model (spec §4), scoped to full names here:

  * cross-juan era-window + uniqueness merge (the 节→juan→cross-juan hierarchy
    collapses to this for full-name surfaces, since identity is surface-driven):
    a surface resolves to the KB card whose tagged juans are UNIQUELY nearest
    within the era window; a tie or an out-of-window nearest leaves the card as
    its own singleton person (under-merge — the line is still drawn).

KB cards (people.json + current per-juan tags) are REFERENCE DATA for grouping
only — never a gate on detection. A surface with no KB owner still emits a span
under a fresh cluster id, so nothing detected is ever dropped for lack of identity.
"""
from __future__ import annotations

WINDOW = 40  # era-window juans for the uniqueness merge (mirrors regression harness)


def build_reference(people, mentions_by_juan):
    """surface → [pids], pid → tagged juans, pid → canonical. Pure reference data."""
    surface_pids = {}
    pid_canon = {}
    for p in people:
        cn = p["canonical_name"]
        pid_canon[p["id"]] = cn
        for nm in [cn] + [(n.get("text", "") if isinstance(n, dict) else n)
                          for n in p.get("names", [])]:
            if nm and 2 <= len(nm) <= 4:
                surface_pids.setdefault(nm, []).append(p["id"])
    for s in surface_pids:
        surface_pids[s] = sorted(set(surface_pids[s]))
    tagged = {}
    for j, ms in mentions_by_juan.items():
        for m in ms:
            pid = m.get("person_id")
            if pid:
                tagged.setdefault(pid, set()).add(j)
    return surface_pids, tagged, pid_canon


def resolve(surface, juan, surface_pids, tagged):
    """Nearest-era unique KB owner of `surface`, or None (→ singleton cluster)."""
    scored = []
    for pid in surface_pids.get(surface, []):
        tj = tagged.get(pid, set())
        if tj:
            scored.append((min(abs(x - juan) for x in tj), pid))
    if not scored:
        return None
    scored.sort()
    if len(scored) >= 2 and scored[0][0] == scored[1][0]:
        return None                     # ambiguous tie → defer identity
    if scored[0][0] > WINDOW:
        return None                     # out of era window → new person
    return scored[0][1]


def consolidate(cards, surface_pids, tagged):
    """Assign a person_id to every occurrence card. Returns cards with person_id +
    a set of newly-minted singleton cluster ids (surfaces with no KB owner)."""
    new_clusters = {}
    for c in cards:
        pid = resolve(c["surface"], c["juan"], surface_pids, tagged)
        if pid is None:
            # singleton / under-merge: cluster by surface so all its occurrences of
            # the same surface share one fresh person id (safe — never over-merges
            # two different surfaces).
            key = f"new:{c['surface']}"
            new_clusters.setdefault(key, c["surface"])
            pid = key
        c["person_id"] = pid
    return cards, new_clusters


def emit_mentions(cards):
    """Group consolidated cards into per-juan mention lists (span-level)."""
    by_juan = {}
    for c in cards:
        by_juan.setdefault(c["juan"], []).append({
            "pid": c["para_id"], "ce_year": c["ce_year"], "source": "main",
            "start": c["start"], "end": c["end"], "surface": c["surface"],
            "person_id": c["person_id"],
            "kind": "giv2" if c["evidence"] == "giv2" else "alias",
        })
    return by_juan
