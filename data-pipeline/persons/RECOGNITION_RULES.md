# 资治通鉴 Reader — Person-Name Recognition Rules (verbalized pipeline)

> 本文描述当前 shipped 旧管线。两阶段目标规范见
> [`twostage/SPEC.md`](twostage/SPEC.md)；两者在迁移完成前会并存，请勿把旧管线的
> per-卷/per-paragraph 行为当作新架构约束。

This document verbalizes the **shipped** offline recognition pipeline in
`data-pipeline/persons/` (`seed.py`, `build_persons.py`, `validate_persons.py`).
It is the accumulated rule set from session waves R1–R34. It describes behavior,
not aspirations — every rule here is in the code today.

---

## 0. Governing principles

- **Precision-first.** A missing underline is acceptable; a wrong underline is not.
  Every rule below is a *gate*, not a guess: when evidence is ambiguous, emit nothing.
- **Book-only.** A person card's facts come from the text (原文) and 胡三省注 (胡注)
  plus the 白话导读 guides. No Wikipedia-derived card facts. (Wikipedia is used only
  as a *negative* filter — `WIKI_NONPERSON` / out-of-period checks — never to assert.)
- **Deterministic & reproducible.** Matching is longest-first, non-overlapping, and
  per-卷 gated. The same corpus always produces the same output.
- **Stable identity.** Each person has one stable id; recurring figures accumulate
  appearances under that id across the whole corpus.

---

## 1. Who can be a person (the cast)

Three layered sources, merged by stable id (`seed.build_seed`):

1. **Hand cast** (`cast.py`, confidence `reviewed`) — authoritative. Per person:
   `canonical_name`, `names`(aliases), `dynasty`, `era_hint`, `floruit`,
   `brief`(spoiler-safe), `identity`(may spoil), `match`(safe surfaces), `juans`.
2. **Guide auto-cast** — every 卷's 白话导读 `key_people` (confidence `high`).
3. **NER auto-cast** — dictionary + model NER over the body text (confidence `high`),
   admitted only through the surface gate (§3).

**Merge model:**
- Hand entries keep their content; their `juans` are **grown** to every 卷 the guide
  names them, but **contiguity-gated** (`GAP = 8`): a name reappearing more than 8 卷
  later is treated as a *different* instance, not teleported across centuries.
- Auto names appearing in non-contiguous 卷 windows are **split** into separate person
  instances (`split_windows`) — anti cross-century merge (e.g. two different 李崇).
- Per-卷 **collision resolution**: a surface may map to at most one person within a 卷.

---

## 2. Surname classes (the backbone of every rule)

- `SURNAMES` — the full 姓 set.
- `AMBIGUOUS_SURNAMES` = 于何方白向都任武史召国金田文成安平广万丁乐华牛高严时后那东 —
  characters that are common words *and* surnames.
- `CLEAN_SURNAMES = SURNAMES − AMBIGUOUS_SURNAMES` — safe to treat as a 姓 with no
  further evidence.
- `_surname_of(full)` returns the leading 姓 only when it is a CLEAN surname; for an
  ambiguous head it returns `None` **unless** a carded full name proves it is a real
  姓 (the R33c/R34 rule: a card for 高崇文 proves 高 is a surname here).

---

## 3. What may become a match surface (`ner_surface_ok` / `bad_auto_surface`)

A candidate surface is admitted only if:
- It is **not** a generic title, reused 庙号/谥号, clan reference, 官名, particle word,
  cross-word fragment, or a known non-person (`WIKI_NONPERSON`, blacklists).
- **3+ chars:** must begin with a CLEAN 姓 or a known 复姓 (e.g. 张延赏, 独孤信).
- **2 chars:** must begin with an **unambiguous** 姓 and char-2 must not be a particle.
- **Single char alone is never an auto surface** (handled only by anaphora/gloss, §5).
- 臣光(史评 voice), pure office-glue (司马 as 官 not 姓), 部族 names (e.g. 林蛮), and
  truncation fragments of 谥号/封号/官名 (`_drop_fragment_cards`) are excluded.

Blacklists are **surface-targeted and conservative**: a surface is only banned when no
real person legitimately uses it (e.g. 唐兴 the place is *not* blacklisted because real
唐兴 people exist; such cases are fixed 卷-locally instead).

---

## 4. Per-卷 / per-paragraph matching order (the "waves")

For each 卷, for each paragraph (`build_persons.main`), passes run **in this order**,
each consuming character spans so later passes cannot overlap them:

1. **Section-local anchor gate** (R33). If the paragraph starts a numbered section
   (①②…⑳), the single-char 省称 anchor table is cleared, so a stale full name from an
   earlier section cannot bind a bare given char (云 in 云大破蛮 ≠ 张云).
2. **封号-glue / titleglue** (`extract_titleglue`, RC-4). Reserves 「封号+given」 spans
   (任城王澄 → 元澄) *first*, so the alias matcher cannot form a false 王X glue over the
   same characters. Bound to the clan person reigning at that `ce_year` when safe.
3. **Alias match** (`extract`). Longest-first, non-overlapping surface match over the
   admitted surfaces — but **single-char alias surfaces are banned** (R34) in both 原文
   and 胡注; bare given chars flow only through anaphora/gloss.
4. **Role appellation** (`extract_roles`, RC-3/P3). Contextual titles (吴主/魏主/上…)
   resolve to the monarch reigning in that `ce_year`; same-year successions are split by
   reading position around the accession cue (`build_role_cue_index`).
5. **Single-char 省称 anaphora** (`extract_anaphora`, P2). A bare given char binds to its
   **nearest preceding full-name antecedent** (disambiguates 元胄 vs 宇文胄). Gated by an
   `admitted` set per 卷 and a common-word-bigram stop check (a char that forms a 文言
   word with its neighbor is not a person).
6. **Genealogical gloss** (`extract_gloss`, RC-2b / P5). See §6.
7. **胡注 alias pass.** Same surface match over each note's text (single-char banned, R34).

---

## 5. Anaphora rule (single-char 省称) in detail

- Only characters in the per-卷 `admitted` set can fire.
- Each match must resolve to exactly one antecedent via `char_anchor` (nearest full
  name) or an `anchor_event` produced by an alias/封号 hit earlier in the same text.
- A match adjacent to a character that forms a common 文言 bigram is rejected.
- No antecedent ⇒ no emit (precision-first).

---

## 6. Genealogical gloss rule 「X，Y之Z也」 (RC-2b, the R25→R34 line)

Pattern: `X，Y之{kin}也`, kin ∈ {子, 孙, 弟, 兄, 父, 从子/从孙/兄子/族子…, N世孙, 曾孙, 玄孙…}.
The text *states* the kinship, so the surname is recoverable by the **patrilineal
shared-surname rule** — high precision.

Two cooperating passes:

**(a) Generative pre-pass** (`build_gloss_cards`) — mint a card for a glossed person who
was never otherwise seen:
- *Forward* (X known/carded): mint the 省姓 relative `Y = 姓(X)+Y`.
- *Inverse* (X is the descendant): X shares the surname of whatever full 姓名 ending in X
  is already carded in this 卷 (高骈 in section → bare 騈 = 高骈); the ancestor merely
  confirms personhood. Guards: subject/Y must pass `_gloss_subject_ok` / `_gloss_y_ok`;
  Y morphemes that are pronouns/尊号/庙号 (上/帝/后/主/太子…) are blocked
  (`_GLOSS_Y_BLOCK`) so 姓+上 / 姓+太祖 are never minted.
- A **stale pid** in the 卷 rule map is skipped, not crashed (R34 fix — this bug had
  silently killed the *entire* gloss layer corpus-wide).

**(b) Binding pass** (`extract_gloss`) — underline X and the bare ancestor Y in body text:
- Resolve X's surname from its card; an ambiguous head (高/严) is accepted when the card
  proves it is a real 姓 (R33c/R34).
- Reconstruct `Y_full = 姓 + Y` (or use Y as-is if it is already a carded full name) and
  bind to the carded person, preferring the nearest 卷.
- **Shared-surname ancestor binding** (R34): when *neither* X nor Y is NER-mapped, bind
  **both** iff `姓+X` and `姓+Y` are carded under exactly **one** shared surname
  (瑑/仁轨之五世孙 → 刘瑑 + 刘仁轨; 譔/震之从孙 → 严震; 锴/铉之弟 → 徐锴/徐铉).
- Emits both the mention(s) and a **kinship relation** edge (subject IS object's <kin>),
  written to `relations.json`.

---

## 7. Cross-卷 consolidation

- **胡注 见N卷 xref window-merge** (`merge_xref_windows`, Wave B). A 胡注 cross-reference
  (见N卷) re-unites an auto person that `split_windows` had split across a large 卷 gap,
  when the note confirms they are the same person.
- **Lookback pass.** Recovers 官衔-glued earlier appearances of a person introduced later.
- **Truncation / elided-surname merges** (`merge_truncations`, `_elided_surname`). Within
  one 卷, a 2-char auto card that is a truncation of a 3-char person folds into it.
- **Brief enrichment** (`enrich_briefs`) — spoiler-safe one-line briefs from the guides.

---

## 8. Lifespan / era gate (Phase 3)

A full-name match is suppressed when it falls **outside the person's lifespan**
(`_lifespan_outside`) — except the two legitimate out-of-lifespan cases, which both use
full names: (a) someone discussing them (史论/citation), (b) posthumous reference. This
stops a posthumous-honorific or same-name match from binding the wrong century.

---

## 9. Output & validator invariants (`validate_persons.py`)

Outputs: `people.json` (only people matched somewhere), `mentions/juan_NNN.json`,
`appearances.json` (cross-卷, reading order, lazy-loaded), `relations.json`, `manifest.json`.

The build is shippable only when the validator passes **all** invariants:
- Every shipped person has a spoiler-safe `brief`.
- **No banned bare/ambiguous single-char surface** except `anaphora` / `gloss` kinds
  (this is the invariant the R34 single-char alias guard satisfies).
- Bare/ambiguous appellations allowed only as `role` (year-resolved) or position-pinned
  `gloss`.
- Offsets valid; `mentions ↔ people ↔ appearances` consistent; no per-卷 surface collision.

---

## 10. Precision rules added by the recent waves (quick index)

| Wave | Rule |
|---|---|
| R33 | Section-local anchor gate (clear 省称 anchors at ①②…⑳); 林蛮 部族 blacklist |
| R33b–g | Ancestor 省称 binding via gindex / shared surname; ambiguous head accepted when carded |
| R34 | (1) crash guard on stale rule-map pid; (2) shared-surname ancestor binding for X，Y之Z也; (3) single-char alias surfaces banned in 原文+胡注 |

---

## 11. Operating note (encoding)

Console output is GBK on this machine and mojibakes CJK — always verify rare characters
by codepoint from the file, not the console (e.g. 譔 = U+8B54, not U+8B34).
