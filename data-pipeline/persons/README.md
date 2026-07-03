# Person recognition & identity pipeline

Builds the **person knowledge base** the reader ships with: who appears in 《资治通鉴》,
where each person is mentioned, and which surface strings (full names, 省称, 封号, 谥号,
role appellations) resolve to which person — all computed **offline, at build time**.

Current output (294 卷): **13,321 people** (179 hand-reviewed + 13,142 auto) ·
**129,337 mentions**. Run `python validate_persons.py` after every rebuild.

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
| **xref-merge** (R29) | re-unites split-window cards the book's 胡注「见N卷」 cross-references prove are one person (李洧 卷227+250) | same canonical; proximity ≤16 chars; N in earliest window ±1 |
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
- **R29 (this round):**
  - **复姓+单名 alias fix** — `_given_tail` no longer strips a 复姓 char into a false
    cross-boundary 2-char alias (夏侯孜→侯孜, 司马懿→马懿). (b907593)
  - **Lookback brief/floruit re-anchor** — when the lookback pass adds an earlier 卷,
    the auto card's `见于卷NNN` brief + floruit start are rewritten to the true first
    appearance (夏侯孜 now 卷249/850, not 卷250/860). 47 cards re-anchored. (b907593)
  - **Wave B · 胡注 见N卷 xref-merge** — auto cards that `split_windows` split across a
    large 卷 gap are re-united when the book's own 胡注「见N卷」 points from a later window
    back into an earlier one (李洧 卷227+250 → 1 card; 张建封 → 1 card). 209 merges;
    proximity-gated so cross-era homographs stay split (王郎 keeps 6 cards). floruit left
    anchored to the earliest window (retrospective allusions must not inflate lifespan).
    (96525a8)
  - **Rolling historian-audit blacklist** — 6 卷 audited across all eras (080/180/280 +
    030/120/230); **34 confirmed non-person surfaces** globally blacklisted (谥号/官署/
    藩镇/部族/城名/动宾 fragments), each verified multi-卷/multi-dynasty with no 字 bio and
    no real-person homograph. 13,420 → 13,321 cards, −448 junk mentions. (c98e451, dfe6efc)

---

## What remains (backlog)

Tracked in the `findings` table (status `open`/`partial`). By priority:

**High**
- `r29-rc7-dynasty-scoped` (RC-7) — auto cards inherit `dynasty` from their first-
  appearance 卷's orthodox narrating line, so 北朝 actors in 南朝-narrated 卷119–176 are
  mis-tagged 宋/齐/梁 (高允/源贺/崔光 → should be 北魏; 高欢/宇文泰 genuinely span two
  regimes). Shown as a card chip → user-visible. Era metadata is in
  `web/public/text/manifest.json` (dynasty/ce_start/ce_end). **~2,879 cards** in range;
  only **~77** are a "single explicit 北朝 dynasty + years fit" safe subset. The systemic
  fix (broad-era「北朝」/「存疑」 vs forced dynasty) is a **product-display decision** →
  awaiting user call before automating.
- `r29-mislink-*` / `r26-fengtitle` (RC-4 封号截断) — the **#1 remaining 错链 class**,
  present in every 卷 audit: a 封号+given is mis-bound to a 王X/公X pseudo-surname or a
  wrong-dynasty homograph (王骏→司马骏, 王亮→司马亮, 王弥→拓跋弥, 王康→刘康, 秦王炽磐→
  乞伏炽磐). titleglue (RC-4) already fixes the cases where the clan+given card exists;
  the rest need **generative card minting** (the clan-form name never appears as a clean
  token) with 萧/元 国姓-coexistence disambiguation — higher risk, needs user validation.

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

## LLM layers (v4 — two passes, then rebuild)

The LLM never touches offsets. It contributes two kinds of durable, checked-in
annotations under `llm_annotations/juan_NNN.jsonl` (schema **v4**; see that folder's
README for the record schemas). The deterministic build re-verifies every asserted span.

| Script | Role | Input | Output records |
|--------|------|-------|----------------|
| `run_llm_pass.py`  | **detection** — lists full 姓名 it sees | raw 卷 text (6k-char batches) | person lines |
| `run_llm_audit.py` | **audit** — fixes the build's own output | a compact **digest** of `people.json` + `mentions/` (no full re-read) | `veto` / `binding` / `card` |

The audit pass is the cheap one: it feeds the model a per-卷 digest (candidate cards,
distinct tagged surfaces + ±windows, unbound 封号/官职 spans) — ~5k input tokens/卷, so a
blanket **all-294** audit is well under US$1 on a mini-class model (batched). Records are
appended dedup-guarded; a 卷 that already carries v4 audit records is skipped unless
`--force`, so hand-authored pilots (016/108/195) are never clobbered.

```powershell
set LLM_API_KEY=sk-...            # OpenAI-compatible; LLM_API_BASE / LLM_MODEL optional
python run_llm_audit.py --measure          # no API: print digest sizes / cost estimate
python run_llm_audit.py --dry-run --juans 100   # no API: eyeball one 卷's prompt
python run_llm_audit.py                     # all 294 卷 (blanket), skip already-audited
python build_persons.py ; python validate_persons.py   # re-verify + re-emit
```

Precision channels the audit relies on: **veto** is delete-only; **binding** offsets are
placed + re-checked by the build; a **single-char** binding (卬→刘卬) routes through the
gated anaphora pass, never a blanket alias; **card** edits are metadata-only and a
`merge_into` is accepted only when the survivor is a real card in that 卷.

## Build & validate

```powershell
cd data-pipeline\persons
python build_persons.py        # rebuild people.json + mentions + appearances
python validate_persons.py     # expect: OK ✓ all person-data invariants hold
cd ..\..\web ; npm run build   # rebuild the PWA bundle
```

If `build_persons.py` raises `OSError [Errno 22]` on a JSON write, an external file
scanner (Windows Defender / search indexer) is momentarily locking the freshly-written
file — **not** the vite dev server. All output writes go through `_write_text_retry`
(8 tries, 0.5 s apart); a transient lock is retried automatically. If it still fails,
re-run the build.
