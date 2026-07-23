"""STAGE 2 identity core (M3) — era-window merge that REPLACES RULES card-gated
prebinding for the literal/alias class.

RULES[juan][surface]=pid pre-decides identity at KB-build time (a surface is bound
in a juan only if some card claims that juan via NER/lookback/xref/gloss_fill). This
module instead resolves identity from the TEXT: a detected surface in juan J binds to
the KB card whose tagged juans are UNIQUELY nearest within an era window; ties or
out-of-window nearest leave it a singleton cluster (under-merge — line still drawn).

`tagged` (pid -> anchor juans) comes from the enriched KB card `juans` field, i.e.
the person's own attested appearances — NOT from the pipeline's output mentions.
"""
from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import seed as S  # noqa: E402

WINDOW = 40


def build_surface_index(people):
    """surface(len2-4) -> sorted [pid]; pid -> set(anchor juans); pid -> canonical."""
    surface_pids: dict[str, list] = {}
    tagged: dict[str, set] = {}
    pid_canon: dict[str, str] = {}
    for p in people:
        pid = p["id"]
        cn = p["canonical_name"]
        pid_canon[pid] = cn
        tagged[pid] = set(p.get("juans", []) or [])
        names = [cn] + [(n.get("text", "") if isinstance(n, dict) else n)
                        for n in p.get("names", [])]
        for nm in names:
            if nm and 2 <= len(nm) <= 4:
                surface_pids.setdefault(nm, []).append(pid)
    for s in surface_pids:
        surface_pids[s] = sorted(set(surface_pids[s]))
    return surface_pids, tagged, pid_canon


def resolve_era(surface, juan, surface_pids, tagged, window=WINDOW, pid_canon=None):
    """Nearest-era UNIQUE KB owner of surface at juan, else None (singleton).

    Tie-break: when the two nearest anchors are equidistant, prefer the single card
    whose CANONICAL name == surface (an exact full-name owner outranks a homograph /
    duplicate whose match is only via an alias). Still None if that is not unique.
    """
    scored = []
    for pid in surface_pids.get(surface, []):
        tj = tagged.get(pid)
        if tj:
            scored.append((min(abs(x - juan) for x in tj), pid))
    if not scored:
        return None
    scored.sort()
    if scored[0][0] > window:
        return None                      # nearest is out of era window → new person
    if len(scored) >= 2 and scored[0][0] == scored[1][0]:
        if pid_canon is not None:
            d0 = scored[0][0]
            exact = [pid for dist, pid in scored
                     if dist == d0 and pid_canon.get(pid) == surface]
            if len(exact) == 1:
                return exact[0]
        return None                      # ambiguous tie → defer identity
    return scored[0][1]
