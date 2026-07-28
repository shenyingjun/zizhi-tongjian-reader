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

The binding boundary guide uses **name-core geometry whenever an explicit name is
present**: `赵王[虎]`, `大将军[光]`, and `弟[亮]`. Titles, offices, and kinship
words remain outside; an unaccompanied individualized appellation such as `始皇`
may itself be the person core. Compound surnames and the complete given name stay
inside. A role-only expression such as `帝` or `太后` is tagged only when its
current grammatical use refers to a specific person (`[帝]曰`, `[太后]怒`);
generic/class and title-conferral uses are not tagged. Agent 2 resolves the
identity. This intentionally differs from some current rule geometries.

## 5. Geometry and evaluation

- Assemble the same numbered-jie blocks used by the rules, preserving paragraph
  boundaries with hard separators.
- For sequences beyond the model limit, use overlapping windows with one
  most-central owner per character. Context windows do not own duplicate output.
- Decode legal BIO transitions. Stray `I-PER` after `O` is promoted to `B-PER`.
  Spans cannot cross separators or begin/end on unowned characters.
- Primary metric: exact `[start,end)` PER precision/recall/F1.
- Diagnostic metric: overlap precision/recall/F1.
- Both use one-to-one maximum matching so one long span cannot match multiple
  references.
- Report random and challenge juans separately and stratify by structure.
- Report full-expression→name-core differences as **geometry replacements**,
  separately from additions, removals, omissions, and false positives.

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

## 6. Model, if P0 justifies it

The plain P1 challenger uses
`KoichiYasuoka/roberta-classical-chinese-base-char` with
`AutoModelForTokenClassification` and `{O,B-PER,I-PER}`. It uses jie assembly and
constrained decoding from the start, but no self-training, agent auto-labeling,
omission recovery, overrides, calibration, or runtime notes.

Use a separate CUDA environment and measure one epoch plus full inference on the
GTX 1070 before making timing claims. Expect FP32 micro-batches and gradient
accumulation; large models are out of scope.

## 7. Phases

- **P0 (3-5 days):** prepare three juans, blind annotation, recall pass,
  self-agreement subset, exact matcher, constrained decoder, rules baseline.
- **P1 (3-5 days):** bounded adjudicated labels and plain char-BIO challenger.
- **P2 (1 day):** stop, expand the pilot, or fund a real evaluation set.
- **P3 (only if encouraging):** sealed probability sample and challenge set.
- **P4+ (only if Tier-2 passes):** calibration, omission channels, note-aware A/B,
  held-out-surface study, multi-seed variance, and hybrid deployment.

## 8. Risks

- Silver bias: use adjudicated labels, not raw v1.
- Long-tail blindness: maintain a non-prevalence-weighted challenge set.
- Test contamination: never feed sealed-test failures into training.
- Single annotator: report delayed blind self-agreement, not consensus.
- Scope/identity leakage: sanitize notes, preserve paragraph translation scope,
  enforce same-jie anchors, and require confirmation for note/translation-only
  candidates.
- Compute overrun: measure before committing to schedules.
- Labor relocation: compare annotation/review cost with rule-writing cost; the POC
  may validly conclude that the rules should remain.
