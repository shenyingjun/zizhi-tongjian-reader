# ML-assisted Agent-1 person-span tagging POC

Status: reviewed and approved for a pilot-first proof of concept. This POC does not
replace `twostage/rules.py`; it first establishes trustworthy measurements and may
ultimately recommend keeping the rules.

## 1. Problem

Agent-1 person-span detection is about 7,000 lines of jie-scoped rules over a
classical-Chinese UPOS model. The current 97.2% benchmark is compatibility with a
noisy v1 reference, not true precision or recall. Approximately 28% of rule output
does not overlap v1 and is an unmeasured mixture of genuine recoveries and false
positives. Rule maintenance does not converge, patterns do not generalize, and
confidence is not tunable.

The primary bottleneck is trustworthy labels, not model architecture. A model
trained naively on v1 would learn that rule-only recoveries are negative (`O`).
The first deliverable is therefore a small human reference and reliable scoring,
not a trained model.

## 2. Scope

- Agent 1 only: where a person underline belongs. Agent 2 identity is unchanged.
- Published spans remain main-text-only.
- Notes (胡三省注) and translation may provide bounded evidence for main-text
  candidates, but are not themselves output spans.
- The POC must run on one GTX 1070 (8 GB) and support one annotator.
- No big-bang replacement, 100% automation claim, or full-corpus rule rebuild.
- The ultimate deliverable is a standalone local ML tagger. Copilot may accelerate
  label creation during development, but is not the production tagger or runtime
  dependency.

## 3. Evidence and labels

Candidate channels are v1, current rules, paragraph-scoped translation evidence,
same-jie note projection, and later model output. They are correlated candidate
generators, not independent votes and not ground truth.

Training labels use a bounded adjudicated union:

- Exact v1/rules agreements may be auto-trusted with `agreement` provenance.
- Disagreements are stratified and audited, capped at 3,000 train and 800 dev
  spans.
- `note-only` and `translation-only` candidates require span-level human
  confirmation and may never be propagated by class policy alone.
- Only `human_*` labels may enter dev or sealed evaluation data.

### 3.1 Note and translation guardrails

1. **No identity data in Agent 1.** Note mentions are sanitized before Agent-1
   ingestion to `{note_index, after, start, end, surface}`. `person_id` is never
   loaded or serialized by Agent-1 code.
2. **Same-jie only.** Note-derived evidence cannot seed another jie or form a
   juan/corpus roster. Note-derived surfaces always use strict no-cross-jie reuse.
3. **Exact translation scope.** Paragraph-approved translation evidence remains
   paragraph-scoped even when paragraphs are assembled into a jie.
4. **Earlier-anchor veto.** Notes/translation cannot alone authorize a
   single-character or given-name anaphor. A qualifying earlier main-text anchor
   in the same jie remains mandatory.
5. **Geometry only.** Evidence may suggest person-ness and boundaries, never
   identity. Note wording does not become a main-text label.

Translation is offline-only. A note-aware runtime model is permitted only as a
later, off-by-default A/B because the notes are public-domain source text.

### 3.2 Copilot teacher and assisted-label loop

P2 may use a versioned Copilot labeler as an annotation **teacher**. Its purpose is
to reduce human labeling effort and improve the labeled corpus used by the target
ML model. Copilot accuracy and target-model accuracy are separate measurements.

- The three completed pilot juans may be supplied as boundary-policy and
  span-geometry demonstrations. They are development context, not sealed data.
- The teacher may read the full current juan for narrative comprehension, but jie
  scope has higher priority for evidence authorization. Demonstration surfaces and
  other-jie context are not a person roster: an exact surface, anchor, or identity
  inferred elsewhere in the juan cannot authorize a current candidate. Each
  proposal must be justified within its target jie, with no identity KB.
- New P2 assisted rounds do not load v1, rules, or identity fields. The Copilot
  teacher may read raw Hu Sansheng note text from the current jie and modern
  translation prose fetched transiently from an approved source. Translation prose
  must match the audited source hash and align uniquely to the current paragraph or
  jie. It is never persisted in the repository or round pack.
- Notes and translation are evidence, not output. They may suggest person-ness and
  name geometry only for main-text occurrences in their approved scope. A
  single-character or given-name anaphor still requires an earlier main-text anchor
  in the same jie, and no evidence may create a cross-jie roster.
- Candidate provenance records the Copilot teacher version, prompt/example hashes,
  note/translation source hashes, target-jie scope, and the fact that full-juan
  context was visible but non-authorizing. The saved teacher output contains only
  main-text candidate geometry, not note or translation prose.
- Each expansion round is separated by whole juan: one of five juans (20%) is a
  candidate-free blind anchor and four (80%) are Copilot-assisted. The blind anchor
  must be completed and permanently locked before any assisted pack is accessible.
- Low-effort diagnostic rounds use two mutually hidden Copilot passes: A is
  recall-first and B is exact-boundary-first. Exact geometry agreed by both passes
  may be auto-accepted only when neither pass marks it explicitly low confidence.
  One-sided geometry, explicit-low output, and frozen-model-only candidates require
  focused human review. The main pipeline must independently validate both schemas,
  channels, provenance, task inventory, and geometry rather than trust teacher
  self-reports. The frozen batch remains Copilot-assisted diagnostic data: it may
  enter training, but never dev, blind-anchor, sealed evaluation, or formal metrics.
- Assisted labels cannot enter dev, blind-anchor, or sealed evaluation splits.
  Blind-anchor labels remain `human_blind_anchor` and evaluation-only.

The teacher-improvement loop is:

1. freeze Copilot teacher version `N` and generate one new assisted batch;
2. run independent A/B passes, auto-accept eligible exact consensus, and human-review
   only disagreements, explicit-low output, and model-only omissions;
3. freeze the focused corrections and measure consensus exact precision/recall/F1, overlap
   diagnostics, additions, removals, and one-to-one geometry replacements;
4. group errors by boundary policy, single-character anaphora, role/appellation,
   foreign-title structure, punctuation, and other declared challenge strata;
5. revise the teacher instructions and demonstrations only from completed
   human corrections, producing version `N+1`;
6. never regenerate or overwrite a completed batch with a later teacher.

The first double-pass batch before Round 4 covered 20 jies and 302 union candidates:
277 eligible exact-consensus spans were auto-accepted, leaving 25 (8.3%) for human
review and 284 final spans. Against those same focused-reviewed labels, consensus
precision was 99.64% and recall 97.18%; these are assisted-batch teacher diagnostics,
not independent evidence. Binding examples from that review exclude bare kinship
anaphors (`其母/母曰`) while including same-jie individualized role references
(`[使者]/[楚使者]`), with the following action `御` outside the span.

The target-model loop consumes the human-corrected assisted labels, trains a new
standalone model, and evaluates it on the locked blind anchor. Teacher promotion
requires better accuracy on the next human-reviewed assisted batch. Target-model
promotion separately requires higher blind-anchor exact F1 with no declared
challenge-stratum regression. Neither promotion substitutes for later P3 sealed
evaluation.

## 4. P0 reference

Select three juans:

1. one random juan;
2. one with high rules-v1 disagreement;
3. one rich in rare/challenge structures such as feng and foreign titles.

The annotator first marks blank main text while blind to every candidate channel.
A mandatory second recall pass shows the candidate union. About 10% of contiguous
jie is re-annotated blind after 7-10 days to measure self-agreement. The three
pilot juans are reference data, not the sealed test, and are excluded from the
future sealed-test sampling frame.

The binding boundary guide uses **name-core geometry after a prefixed title**:
`赵王[虎]`, `大将军[光]`, and `弟[亮]`. Prefixed titles, offices, and kinship
words remain outside. A conventional postposed foreign or consort title stays in
the complete established person surface: `[阿波可汗]`, `[沙钵罗叶护]`, and
`[王皇后]`; with a prefixed title, tag only the following core, as in
`柔然可汗[阿那瓌]`. An unaccompanied individualized appellation such as `始皇`
may itself be the person core. Compound surnames and the complete given name stay
inside. A role-only expression such as `帝` or `太后` is tagged only when its
current grammatical use refers to a specific person (`[帝]曰`, `[太后]怒`);
generic/class and title-conferral uses are not tagged. Agent 2 resolves the
identity. This intentionally differs from some current rule geometries.
An independently recognizable name or individualized appellation remains a person
core when embedded in a building, office, artifact, or relational compound:
`[汉高]庙` and `[启]母神`. Only that core is tagged; carrier words such as
`庙/府/宫/印` remain outside. A generic role inside an institution name is not
thereby authorized.
For a single naming predicate, tag only its result (`字[季]`, `更名曰[询]`).
When a continuous construction such as `姓 X 字 Y` jointly supplies the complete
available naming description and its pieces do not independently form that
description, tag the full expression: `即常安[姓武字仲]`.
When coordinated persons share a right-hand appellation, tag each concrete
conjunct's actual surface, including an elliptical first conjunct:
`乌孙[大]、[小昆弥]` and `[广平]、[巨鹿]、[乐成王]`. A plural expression is
taggable only when it clearly refers to a specific group whose members are anchored
in the current jie; generic or unanchored plural roles remain excluded. Juan
headings are metadata, not narrative person mentions, and contain no spans.
A concrete unnamed role may establish one continuing narrative participant within
the current jie, for example `[神策军将]诉事…擒[军将]…[军将]已死`. Pure
pronominal honorifics such as `陛下/足下` and temporary insults such as `妪`
remain excluded. A `氏` surface is taggable when it denotes one concrete person
(`向氏`, `金氏`) but not when it denotes a clan (`桓氏之甥`). Ethnonyms alone
are excluded; `[黠戛斯可汗]` is taggable only when it denotes a concrete
incumbent. A newly conferred office, reign title, or posthumous title remains
excluded at the conferral predicate and becomes taggable only in a later concrete
person reference. Person spans remain separate from a future entitlement timeline.

## 5. Geometry and evaluation

- Assemble the same numbered-jie blocks used by the rules, preserving paragraph
  boundaries with hard separators.
- For sequences beyond the model limit, use overlapping windows with one
  most-central owner per character. Context windows do not own duplicate output.
- The prediction and scoring unit remains one target jie. A model window may include
  multiple surrounding jies as soft narrative context, but non-target characters
  receive label `-100`, own no output, and cannot form a cross-jie span.
- Decode legal BIO transitions. Stray `I-PER` after `O` is promoted to `B-PER`.
  The train/dev-fixed structural decoder merges a `B-PER` directly adjacent to an
  active person span when there is no punctuation, symbol, number, separator, or
  unowned boundary. Spans cannot cross separators or begin/end on unowned
  characters.
- Primary metric: exact `[start,end)` PER precision/recall/F1.
- Diagnostic metric: overlap precision/recall/F1.
- Both use one-to-one maximum matching so one long span cannot match multiple
  references.
- Report random and challenge juans separately and stratify by structure.
- Report full-expression→name-core differences as **geometry replacements**,
  separately from additions, removals, omissions, and false positives.

### 5.1 Multi-jie model context

Context size is determined by the model token budget, not by a fixed `±1 jie`.
After reserving space for the complete target jie, fill remaining capacity with
the nearest preceding and following jies in distance order. Short targets may
therefore see several surrounding jies; long targets may see little or none.

- Preserve explicit paragraph and jie boundaries. Record each context jie's signed
  distance from the target.
- Context may help the neural model interpret narrative continuity, but it does not
  create deterministic cross-jie anchors, surface propagation, identity resolution,
  or output spans.
- Train and evaluate a target-jie-only baseline before testing context. Compare at
  least target-only, bounded-neighbor, and maximum-budget context using exact
  target-jie geometry.
- Fix context construction and checkpoint selection on training/challenge-dev only.
  Do not choose context size from the random pilot holdout or a sealed set.
- Apply a symmetric split guard band derived from the actual context graph. No
  training window may contain dev target text, even as context, and no jie visible
  in a dev window may be used as a training target. Report guard-band exclusions.
- Use the existing Transformer first. LSTM, hierarchical Transformer, Longformer,
  or other long-context architectures are later A/B options only if bounded
  Transformer context proves useful and the 512-token limit is demonstrably binding.

Tier-1 is descriptive and can only encourage or disprove:

- stop if random-juan exact F1 is more than 3 points below rules, omission
  recovery is below 25%, or an uncompensated challenge recovery regresses;
- proceed if exact F1 is within about 2 points of rules or better, at least 50%
  of confirmed rule omissions are recovered, and no systematic new false-positive
  class appears.

Only a later sealed probability sample with bootstrap confidence intervals can
support adoption. Hybrid adoption requires model exact F1 at least equal to rules
with non-overlapping 90% intervals, no challenge stratum down more than 5 points,
and a one-sided 95% precision lower bound of at least 0.98 for auto-publishing.

### 5.2 P3 sealed set

Before any P3 model inference, freeze five previously unused whole juans:

- three juans sampled uniformly without replacement from the eligible 1-294 frame;
- one role/appellation challenge juan drawn from the five highest predeclared
  raw-text term counts; and
- one foreign-title challenge juan drawn from the five highest predeclared
  raw-text term counts.

Exclude every juan used in training, development, pilot holdout, blind-anchor
evaluation, assisted annotation, or an aborted/leaked sealed set. Generate the
probability and challenge draws from a private random seed, then freeze that seed,
the selected model hash, checkpoint selection record, selection policy, task/source
hashes, and current code commit in a private manifest. Annotation task files contain
only candidate-free main text and paragraph/jie geometry. The UI must not expose
selection roles before each juan is completed and permanently locked. Do not
generate model or rule predictions until all five references are locked; P3
failures cannot select or retrain the evaluated model.

When whole-juan annotation is not feasible, a predeclared compact P3 may instead
sample numbered jies. Its probability frame is limited to previously unused jies
of 20-600 Unicode codepoints: draw 12 uniformly without replacement, plus four
private-seed draws from the top-40 role/appellation cohort and four from the
top-40 foreign-title cohort. Report probability metrics and bootstrap intervals
only on the 12 random jies; report the eight challenge jies separately. This lower-
power design cannot support claims about excluded very short or very long jies.

## 6. Model, if P0 justifies it

The plain P1 challenger uses
`KoichiYasuoka/roberta-classical-chinese-base-char` with
`AutoModelForTokenClassification` and `{O,B-PER,I-PER}`. It uses jie assembly and
constrained decoding from the start, but no self-training, agent auto-labeling,
omission recovery, overrides, calibration, or runtime notes.

The P2 Copilot teacher does not change this P1 model definition: unreviewed agent
output never trains the target model. Only human-corrected assisted spans may be
added in later training rounds.

The initial P1 timing baseline remains target-jie-only. Multi-jie soft context is a
separate P2 target-model A/B and must preserve target-only loss, output, and scoring.

Use a separate CUDA environment and measure one epoch plus full inference on the
GTX 1070 before making timing claims. Expect FP32 micro-batches and gradient
accumulation; large models are out of scope.

## 7. Phases

- **P0 (3-5 days):** prepare three juans, blind annotation, recall pass,
  self-agreement subset, exact matcher, constrained decoder, rules baseline.
- **P1 (3-5 days):** bounded adjudicated labels and plain char-BIO challenger.
- **P2:** stop, fund a real evaluation set, or expand through versioned Copilot
  teacher rounds with one blind anchor per four assisted juans. Improve teacher
  labeling efficiency and the standalone target model separately.
- **P3 (only if encouraging):** sealed probability sample and challenge set.
- **P4+ (only if Tier-2 passes):** calibration, omission channels, note-aware A/B,
  held-out-surface study, multi-seed variance, and hybrid deployment.

## 8. Risks

- Silver bias: use adjudicated labels, not raw v1.
- Long-tail blindness: maintain a non-prevalence-weighted challenge set.
- Test contamination: never feed sealed-test failures into training.
- Context leakage: construct splits after context closure; training and dev windows
  must not expose each other's target text through surrounding-jie context.
- Teacher contamination: never expose candidates before a blind anchor locks;
  never place assisted labels in dev; never use cross-jie demonstration surfaces
  as current-jie person evidence.
- Automation bias: record all Copilot accepts, rejects, boundary corrections, and
  human-added omissions; teacher output is not ground truth.
- Single annotator: report delayed blind self-agreement, not consensus.
- Scope/identity leakage: sanitize notes, preserve paragraph translation scope,
  enforce same-jie anchors, and require confirmation for note/translation-only
  candidates.
- Compute overrun: measure before committing to schedules.
- Labor relocation: compare annotation/review cost with rule-writing cost; the POC
  may validly conclude that the rules should remain.
