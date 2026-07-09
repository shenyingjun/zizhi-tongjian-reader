# Two-stage local-first person-underlining pipeline (SHADOW / experimental)

This is an **isolated, non-production** reimplementation of person-name detection
following the two-stage *local-first* architecture (see
`session-state .../files/two_stage_spec.md`). It does **not** touch the current
production pipeline (`build_persons.py` / `seed.py` / `cast.py`) or any of its
output files. It writes a parallel mention set to `out/` and diffs it against the
current pipeline span-for-span.

## Architecture

- **Stage 1 — per-juan detection (`stage1.py`)**
  Scans one 卷 at a time and emits one *occurrence card* per detected name span.
  It never consults `people.json` to decide *whether* to underline — only the juan
  text + dictionaries + POS·Giv + boundary guards. Detectors:
  - `D-LIT3` — literal match of a curated KB name (canonical/alias), any length ≥3.
  - `D-GIV2` — 2-char surname+given, gated by the validated guard stack:
    G1 left-guard (齐王建), G2 longest-match (张守←张守一), G3 POS·Giv, soft 王
    name-start gate (accepts 王X when 王 is at a name boundary *or* the given char
    is POS·Giv-tagged), 仆射 office-truncation, 谥号 posthumous-title.
  NER candidates are treated as *discovery hints only*: they feed the G2 guard
  (to suppress bad 2-char splits) but are **never drawn as blind literals** (they
  are noisy — 王侍郎, 余非吾, 蒙圣恩 all pass the legacy surface filter).

- **Stage 2 — cross-juan merge (`stage2.py`)**
  Merges occurrence cards into person cards, safest scope outward. KB is *reference
  data*, not a detection gate, so a name attested only in the text (王绪@254) still
  underlines. Merges only, never drops; an unmerged card is a singleton that still
  underlines.

- **Runner / comparator (`run.py`)**
  `python run.py`             → all juans
  `python run.py 251 252 254` → only those juans
  Writes `out/mentions/` and diffs vs current main-source `alias` mentions (len≥2),
  reporting AGREE / RECOVER / LOST and 王绪@254.

## Corpus validation (294 卷, vs current pipeline)

| metric | value |
|---|---|
| AGREE (same span) | 56,310 |
| RECOVER (mine adds) | 4,391 — **100% exact KB names**, incl. 王绪@254, 1,200 王-names |
| LOST len≥3 (full-name) | 111; of these 51 still underlined (extent diff), 60 true misses |
| LOST len2 | 32,198 — deferred 省称 anaphora + ambiguous-surname class |

The 60 true len≥3 misses are **entirely deferred special passes** not ported in
this scope: role/appellation binding (傅太后, …可汗) and feng-title binding
without a given (梁孝王, 魏其侯). There are **zero true regressions** on Stage-1's
target class (clean full-name detection).

## Deferred (documented scope, not regressions)

- 省称 given-only anaphora (kind=anaphora) — a Stage-2 given-binding step
  (port of `resolve_anaphora_pos`).
- ambiguous / compound-surname 2-char aliases (高X, 田X, …).
- feng-title and role-appellation special passes.

`out/` is git-ignored (regenerated comparison artifacts).
