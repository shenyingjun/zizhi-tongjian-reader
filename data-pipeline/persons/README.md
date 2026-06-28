# Person recognition & identity pipeline

Builds the **person knowledge base** the reader ships with: who appears in 《资治通鉴》,
where each person is mentioned, and which surface strings (full names, 省称, 封号, 谥号,
role appellations) resolve to which person — all computed **offline, at build time**.

Current output (294 卷): **13,629 people** (179 hand-reviewed + 13,450 auto) ·
**129,796 mentions**. Run `python validate_persons.py` after every rebuild.

---

## Hard constraints (do not violate)

1. **Book-only content.** Everything shown on a person card — identity line, brief, 字,
   kinship, appearances — comes from the book itself (原文 + 胡三省音注 + 白话导读 + curated
   reign/cast tables). **Never Wikipedia bios.** Wikipedia is used *only* as an offline
   verify-only precision signal (`wiki_verify.py`), never as displayed text.
2. **Precision-first.** A *missing* underline is always better than a *wrong* one. Every
   recall rule must be gated so it cannot mis-bind. When in doubt, drop the mention.
3. **No spoilers.** A card only shows appearances at or before the reader's current
   (卷, paragraph) position. The cutoff is enforced in the web layer.

---

## Data flow

```
seed.py            → build_seed(): NER + gazetteer + title-glue → auto cards + RULES
cast.py            → 179 hand-reviewed canonical people (导读 main characters)
reigns.py          → per-dynasty reign tables, 庙号/谥号 meta, role appellations
build_persons.py   → the driver: runs every binding pass per 卷 in reading order,
                     emits mentions, aggregates appearances, writes outputs
validate_persons.py→ invariant checks (run after every build)
```

Outputs (under `web/public/text/persons/`):
- `people.json` — every person card (id, canonical_name, names[], dynasty, brief, …)
- `mentions/juan_NNN.json` — every underlined mention in a 卷 (surface, offset, person_id, kind)
- `appearances.json` — per-person appearance index, aggregated purely from mentions
- `relations.json` — kinship edges harvested from 胡注 genealogical glosses

**Key invariant:** `appearances.json` is derived *only* from emitted mentions. To make a
person show up, you must get a *mention* emitted — minting a card alone is not enough.

---

## The binding passes (the "rules")

`build_persons.py` runs these per paragraph, in reading order. Each pass marks the
character spans it consumes so later passes don't double-bind. Surfaces live in
`RULES[juan] = {surface: person_id}`; `extract()` does longest-match binding.

| Pass | What it binds | Gate summary |
|------|---------------|--------------|
| **alias** | Full 姓名 + registered 省称/别名 surfaces | longest-match; surface must be in `RULES[juan]` |
| **anaphora** (Wave 5 P2) | Bare single given char (坚→杨坚, 胄→元胄) | left/right context gates (`_ANAPHORA_LEFT/RIGHT`); char admitted per-卷 only |
| **role** (Wave 5 P3) | 语境称谓 吴主/魏主/周主/契丹主 → the monarch *reigning at that year* | `reigns.resolve_role(surf, ce)`; title-glue + same-year-succession split |
| **gloss** (RC-2b) | 家世 glosses 「X，Y之N世孙也」 → mint & bind the 省姓 relative | patrilineal kin only; surname shared; forward **and** inverse direction |
| **titleglue** (RC-4) | 封号/王号 + 名 (赵王伦→司马伦, 蜀公迥) | clan-scoped; must not global-blacklist the bare 名 |
| **lookback** | recovers 官衔-glued earlier appearances (御史中丞…夏侯孜) | scan ≤2 prior 卷; bind only if surface unclaimed there |

### Root-cause taxonomy (RC-N)

The recall/precision work is organized around named root causes, each fixed in a wave:

- **RC-1** boundary noise — 封号/官名/地名 suffix garbage, name+verb truncation.
- **RC-2 / RC-2b** 省称 anaphora + genealogical-gloss minting (孔戣; 王谱 inverse gloss).
- **RC-3** single-surname over-binning (bare 裴 → wrong 裴).
- **RC-4** 封号+名 productive binding (titleglue).
- **RC-5** 尊号/庙号/谥号 — *precision* slice (drop truncation fragments) **done**;
  *recall* slice (谥号→当朝君主 binding) **deferred** (ambiguous, see below).
- **RC-6** model 姓首 recall for 鲜卑/胡 复姓; hapax frequency-floor.
- **RC-7** dynasty metadata mislabel (周/隋 actor in a 陈纪 卷 tagged 「陈」).

---

## What's been done (wave log)

Detailed findings live in the session `findings` SQL table; high-level:

- **R12 Wave 1–2** — precision cleanup + 省姓 anaphora merge (commit 6abbe5f).
- **R13 Wave 3** — RC-6 model 姓首 recall (e53359f).
- **R14 Wave 4** — generative 省称 anaphora + RC-1 cleanup (6e88e99).
- **R15 Wave 5** — position-aware disambiguation: P2 single-char anaphora, P3 role
  appellation resolver (吴主/魏主 → per-year monarch).
- **R24** — fixed astral/surrogate-pair off-by-one underline shift (web side).
- **R25–R27** — name-boundary garbage audit, general reader-agent precision/recall
  audits (卷050/145/265), lifespan era-gate, RC-4 titleglue (赵王伦→司马伦),
  lookback pass (夏侯孜 binds 卷249+250 to one card).
- **R28 (this round):**
  - **Inverse genealogical gloss** — 「谱，珪之六世孙也」now mints 王谱 (ancestor
    surname recovered from 胡注 「王珪」) and binds the bare 谱 via anaphora.
  - **唐兴 place disambiguation** — blacklisted the 3-way-ambiguous surface
    (person 卷114 / phrase 唐兴以来 / place 唐兴县).
  - **RC-5 fragment guard** — data-driven drop of 2-char auto cards always glued to a
    封号/谥号/官名 tail and never standalone (14 dropped: 武灵/悼惠/衞思/…; homographs
    曹参/魏尚 kept).

---

## What remains (backlog)

Tracked in the `findings` table (status `open`/`partial`). By priority:

**High**
- `r26-dynasty-mislink` (RC-7) — 卷145: 北魏 people tagged [齐]/[梁]/[宋]. 通鉴 dates by
  the orthodox line, but actors are from rival states. Honest fix: show era/在世 rather
  than a wrong 朝代, or derive dynasty from the person's own reign span.
- `r26-fengtitle` (RC-4) — 封号+名 still mis-read as 王/侯-surname in some 北魏 cases
  (王宝←湘东王宝晊, 王显←阳平王元显). Extend titleglue clan scoping.

**Medium**
- `r28-dunhao-coordination` (recall) — **the 顿号 wave.** `A、B` shares POS, so a bound
  name licenses its neighbours. Currently `、` is only a splitter. A measured ~120–150
  real recoveries are available with an *interior-token + person-anchor* gate (never
  edge tokens, never place-card anchors). Designed, not yet built.
- `r26-shihao` (RC-5 recall) — bind 谥号/庙号/尊号 (天元→宇文赟, 昭宣→唐哀帝, X后→当朝皇后)
  to real monarchs. Hard: 庙号/谥号 (后主/宣帝/太祖) are shared across every dynasty, so
  binding must be 卷-local + dynasty + reign-window scoped; high re-bind risk → deferred.
- `r25-trunc` / `r26-recall` — 单姓双名 truncation (萧惠明) and 单名省称 after 封号
  (宝攸/子恪/憺/懿) recall gaps.
- `r26-pei-mislink` (RC-3) — bare 裴 over-bins to 秦裴; needs per-paragraph nearest-
  antecedent resolution for single-surname references.

**Low**
- `r26-li-anaphora` — single-char anaphora occasionally mis-fires on non-person 2-char
  contexts (勇→沈勇).

**Known recall limitation (single-occurrence persons).** A person introduced once, with
a rare given char and no title/born-place glue, can be missed by NER (the frequency
floor that suppresses hapax false-positives also costs some real one-off names). This is
a precision-first trade-off, revisited per-case.

---

## Build & validate

```powershell
cd data-pipeline\persons
python build_persons.py        # rebuild people.json + mentions + appearances
python validate_persons.py     # expect: OK ✓ all person-data invariants hold
cd ..\..\web ; npm run build   # rebuild the PWA bundle
```

If `build_persons.py` raises `OSError [Errno 22]` on `appearances.json`, the vite dev
server is holding the file — retry 2–3×.
