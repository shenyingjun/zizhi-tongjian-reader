# Agent-1 ML production program

Status: revision-9 diagnostic implementation contract. This program authorizes new
engineering and candidate-model-blind data work, but no ML candidate is authorized
for production.

## 1. Decision and objective

The POC established that the Round 7 recipe can outperform the canonical
Translation-assisted rules on exact person-span geometry. It did not pass the
predeclared production precision gate: its one-sided 95% precision lower bound was
`0.965409`, below `0.98`.

Production work therefore starts a new, prospectively declared program. It must:

1. improve precision without hiding recall or boundary regressions;
2. create fresh training and development labels without using sealed POC evaluation
   references;
3. pass a new candidate-model-blind formal evaluation with adequate power;
4. emit the existing identity-free Agent-1 occurrence contract offline; and
5. support shadowing, atomic publication, and immediate rollback.

The canonical Translation-assisted rules remain production until every adoption and
release gate in this document passes.

Round 1 and its one reserved replacement round have completed. Their best exact
3-of-3 development operating points were respectively `0.939516` precision /
`0.930140` recall and `0.965596` precision / `0.972286` recall. Neither reached
the frozen `0.99` precision gate, so no formal evaluation was created. Both
development references are consumed and may now be used only for diagnostic error
classification, never for fitting, calibration, threshold selection, checkpoint
selection, or another go/no-go decision.

## 2. Non-negotiable invariants

- Agent 1 predicts geometry only and must not read a person or identity KB.
- All person-surface evidence, anchors, recurrence, conflicts, and vetoes remain
  confined to the current numbered jie.
- Main text is the only output source. Notes and translation may provide bounded
  teacher evidence but never runtime identity or output spans.
- Translation evidence remains confined to its approved paragraph/jie mapping.
- A single-character or given-name anaphor requires an earlier qualifying main-text
  anchor in the same jie.
- Punctuation, symbols, numbers, separators, and unowned context are hard span
  boundaries.
- A real BIO continuation veto and all other hard vetoes cannot be overridden by
  score, confidence, recurrence, or ensemble votes.
- Model context is soft. Non-target jies receive no loss and own no output; they
  cannot authorize a deterministic anchor or surface propagation.
- The Round 11 promotion and Round 13 challenge references remain sealed from
  training, tuning, calibration, checkpoint selection, and error-driven policy work.
- No user is assigned candidate-free exhaustive annotation. References use
  candidate-model-blind Copilot A/B passes plus focused human review.

Any implementation that violates these invariants fails closed and produces no
publishable output.

## 3. Versioned program inputs

Every round is created in a new directory and binds the following in an immutable
`manifest.json`:

- schema and program version;
- clean Git commit;
- corpus document hashes;
- boundary-guide hash;
- base model name and pinned revision;
- teacher prompt, example, and implementation hashes;
- sampling seed commitment and sampling-frame hash;
- complete exclusion inventory with reason and source-manifest hash;
- train, development, formal-evaluation, and challenge roles;
- source and output hashes for every task and label file; and
- explicit `training_only`, `formal_evaluation`, and `eligible_for_promotion` flags.

The exclusion inventory includes every POC train, development, holdout, assisted,
promotion, challenge, aborted, leaked, and previously published formal-evaluation
item. A jie may not appear in more than one role. Target-only runs isolate at jie
level. If soft context is selected, the planner computes the symmetric exclusion
closure from the actual context graph: no training window may expose development or
evaluation target text, and no context jie visible to development or evaluation may
be a training target. The manifest records every guard-band exclusion.

Round directories are append-never: completed tasks, teacher passes, focused
decisions, datasets, predictions, and reports are never regenerated or overwritten.

## 4. Fresh improvement data

### 4.1 Sampling

Create one training round from previously unused juans:

- 80 numbered jies sampled uniformly from eligible jies of 20-600 Unicode
  codepoints;
- 20 role/appellation jies sampled from a frozen raw-text cohort;
- 5 foreign-title jies sampled from a frozen raw-text cohort; and
- 20 boundary/anaphora jies sampled from a frozen, identity-free structural cohort.

Add 15 more uniform-random jies so the 140 jies are training-only. Cohort membership
is computed from raw characters,
punctuation, and generic title/role terms before model inference. Sampling is
deduplicated by jie and then closed under the selected context graph, not by whole
juan.

The program also freezes a fresh 40-jie development set: 20 uniform random jies and
20 challenge/random jies consisting of 7 role/appellation, 5 foreign-title,
6 boundary/anaphora, and 2 additional uniform-random jies, disjoint from training
after context closure.
Development is used for checkpoint and ensemble operating-point selection and can
never become training data. A development set supports at most one training-data
revision and one declared model-selection comparison; another improvement round
requires a newly sampled development set.

Before sampling, the planner proves that the remaining eligible inventory can fund
this round, its formal evaluation, and one replacement round under the selected
context closure. The replacement round is not required to repeat an exhausted
challenge stratum; the formal reserve takes priority. In particular, at least 20
untouched foreign-title jies remain reserved for formal evaluation. If these checks
fail, the planner stops and reports the shortfall. It must not weaken exclusions,
reuse a sealed reference, or silently change the sample.

### 4.2 Candidate-model-blind labeling

Training and development tasks contain only main text, paragraph/jie geometry, and
the frozen boundary policy. Before either set is labeled, no Copilot pass or human
reviewer may see Round 7 predictions, rules, v1 spans, identities, or another pass's
output for those tasks.

Each task contains exactly one target jie. Opaque task IDs and the public task
manifest reveal neither split nor challenge stratum; the task-to-role mapping and
sampling seed live in a physically separate sealed private manifest.

For every task:

1. Copilot pass A performs recall-first tagging.
2. Independently, Copilot pass B performs exact-boundary-first tagging.
3. Exact non-low A/B consensus is accepted provisionally.
4. For training and development, a third Copilot teacher adjudicates every A/B
   disagreement and explicit-low candidate. It sees the current jie and anonymous
   union candidates, but not their source pass, confidence, prior decision, split,
   stratum, identities, model/rule/v1 output, translation, notes, or other jies.
   Its high-confidence exact accept/reject decisions pass provisionally; medium/low
   decisions and geometry conflicts require human review.
5. The human reviews all non-high third-teacher decisions and deterministic audit
   samples of 5% of A/B consensus spans plus 20% of consensus-negative jies.
6. Any incorrect audited positive or any missed person in an audited negative jie
   expands that task to 100% union-candidate review. If the stratum's audited exact
   error rate exceeds 1%, audit all union candidates in that stratum before freeze.
7. For formal evaluation, the human reviews 100% of the A/B positive union. A
   deterministic 25% of A/B-consensus-negative jies receives a third independent
   recall pass and focused review of any additions. Any miss expands the remaining
   consensus-negative jies in that stratum to a third pass.

The pipeline validates task inventory, provenance, source hashes, bounds, surfaces,
non-overlap, and legal geometry independently of teacher self-reports. Persist only
main-text geometry and provenance; note or translation prose is transient.

Copilot passes are one correlated model family, not independent ground truth. The
third teacher reduces routine training/development adjudication; it does not reduce
formal-positive review. Full formal-positive review prevents shared false-positive
biases from inflating measured precision, and the adaptive negative audit measures
shared omissions without assigning blank-text exhaustive annotation to the user.

### 4.3 Error-directed improvement

The Round 7 recipe is the baseline, not training input. After the fresh development
reference is locked, run the baseline and classify every development error into:

- boundary replacement;
- punctuation or numeric-boundary violation;
- role/appellation;
- foreign title;
- single-character or given-name anaphora;
- embedded person core;
- false person lexicalization;
- omission; or
- other, with an explicit explanation.

Only training-set labels and development aggregate/error classes may guide changes.
No surface blacklist or identity-specific feature is allowed. Structural fixes must
be represented in the boundary policy, constrained decoder, sampling strata, or
general training data.

## 5. Candidate model and selection

The first production candidate retains the pinned classical-Chinese character
Transformer and `{O,B-PER,I-PER}` labels. Train deterministic seeds
`20260727`, `20260728`, and `20260729` with the Round 7 learning rate `3e-5`,
target-only loss, pinned tokenizer/model revision, and the existing constrained
decoder.

Evaluate, on fresh development only:

- each seed;
- exact-geometry 2-of-3 voting; and
- exact-geometry 3-of-3 voting.

Individual seeds are stability diagnostics, not deployable operating points. The
predeclared precision mechanism is stricter exact-geometry voting plus the existing
hard decoder. Select 2-of-3 or 3-of-3 by this frozen order:

1. precision at least `0.99`;
2. exact recall at least `0.95`;
3. every challenge stratum reports its complete reference-span count; strata with
   fewer than 50 spans are descriptive safety audits rather than statistical gates;
4. the lower bound of the paired 90% jie-bootstrap exact-F1 difference from rules
   is greater than zero overall;
5. no challenge stratum's paired 90% difference lower bound is below `-0.03`;
6. highest exact precision; then
7. the stricter vote threshold.

If no operating point reaches `0.99` development precision, do not create a formal
evaluation. Perform one new training-data round under section 4; do not tune on a
formal or sealed set. Architecture changes, probability calibration, note-aware
runtime inference, self-training, and identity features require a new spec revision.

### 5.1 Revision-2 precision recovery after the replacement round

The replacement round is exhausted. No further training/development sample may be
drawn from the 160-jie formal reserve. A complete audit of the replacement 3-of-3
errors found 15 false positives and 12 false negatives. Ten false positives came
from the boundary/anaphora stratum; recurrent structural families were
single-character partials, standalone court-role references, generic offices or
relations, and one exact boundary replacement. These are policy families, not a
surface blacklist. Development surfaces and identities must not become runtime
features.

Revision 2 authorizes one precision controller consisting only of:

- exact-geometry vote threshold `2` or `3`;
- a minimum span-confidence threshold from the fixed grid
  `{0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94,
  0.96, 0.97, 0.98, 0.99}`.

For one seed, a predicted span's confidence is the geometric mean of the
**pre-constraint, per-position softmax** probabilities of the final emitted labels:
the
single character of a one-character span uses `B-PER`; longer spans use `B-PER`
then `I-PER`. A decoder-forced label uses that label's pre-constraint probability,
not the original argmax probability. Compute the geometric mean in log space. Equal
logits use the frozen label order.
For vote threshold `k`, ensemble confidence is the `k`th-highest seed confidence,
with a non-supporting seed assigned zero. This is the weakest required supporter
and has the same definition for both vote thresholds.

The confidence controller is an explicitly authorized decoder-adjacent global
abstention mechanism. It cannot contain a learned calibrator, minimum-length filter,
surface feature, title lexicon, identity, translation, rule output, or development
prediction. In particular, valid anchored single-character anaphors remain eligible.

For revision 2, this section supersedes section 5's fresh-development selection.
Partition the cumulative 280 former training jies once into `fit`, `calibration`,
and `confirmation` groups with target proportions `5:1:1`:

1. Group by juan; a juan may appear in only one partition. Use the sealed private
   role manifests only to balance split membership, never as model input.
2. For every juan, compute the vector `(examples, reference spans,
   uniform, role/appellation, foreign-title, boundary/anaphora)`. Set each
   partition's target vector to its `5:1:1` share of the global vector.
3. Order juans by descending examples, descending spans, descending maximum
   stratum count, then ascending SHA-256 of `20260807:<juan>`.
4. Place each juan in the partition that minimizes the sum, over every partition
   and vector component, of
   `((current_after_placement - target) / max(1, target)) ** 2`.
   Ties use `fit`, then `calibration`, then `confirmation`.
5. Freeze the partition before model training. Calibration and confirmation must
   each contain every stratum, and their example and span totals must be within
   20% of target. Otherwise stop without manually changing the partition.

Before model training, upgrade calibration and confirmation to formal-grade
references. A human reviews 100% of their A/B positive union; 25% of
consensus-negative jies receives an independent candidate-blind recall pass and
focused review of additions, sampled with seed `20260809`, with the expansion rules
from section 4.2. The review
tasks cannot show model predictions, controller scores, split role, identities,
translation, notes, rules, or v1. Freeze and hash-bind the upgraded references.

An explicitly labeled **AI-assisted diagnostic route** may instead carry forward the
already frozen high-confidence teacher/reference decisions and send only new recall
findings or genuinely unresolved candidates to a human. This route reduces duplicate
label review, but it is not a formal-grade reference and cannot support the production
precision claim, promotion gate, or formal-evaluation authorization in this spec.
Training-grade labels are insufficient for a `0.99` selection claim.

Train the three production seeds on `fit` only. Use exactly epoch 5; neither
calibration nor confirmation may select a checkpoint. The exact three resulting
model artifacts are the only deployable revision-2 models and must not be retrained
after controller selection.

Evaluate all 30 predeclared vote/confidence operating points on calibration. An
operating point is calibration-eligible only if it has at least 300 predictions,
precision at least `0.99`, recall at least `0.95`, and a one-sided 95% span-level
Wilson precision lower bound at least `0.98`. Choose highest recall, then highest
precision, then higher confidence threshold, then higher vote threshold. Record
the complete 30-point table. If none is eligible, stop without reading
confirmation.

Evaluate the one selected controller exactly once on confirmation. It passes only
if confirmation contains at least 40 jies and 300 predictions, precision is at
least `0.99`, recall is at least `0.95`, and the boundary-safe precision lower
bound below is at least `0.98`. Confirmation cannot change the controller.

The boundary-safe one-sided 95% precision lower bound is the minimum of the
span-level Wilson bound, every leave-one-jie-out span-level Wilson bound, and the
jie-level BCa bootstrap lower bound using 50,000 replicates and seed `20260808`.
If BCa is undefined or non-finite at the zero-error boundary, omit BCa and use the
minimum Wilson bounds. An evidence-size failure is reported as an underpowered
partition, not a model-quality failure. Any powered quality failure terminates
revision 2 with canonical rules retained; section 5's additional-data fallback no
longer applies.

The partition, three model artifacts, pre-constraint character probabilities,
complete controller table, confirmation result, and selected operating point are
append-never and hash-bound. Round 1 and Round 2 development remain diagnostic
history only and cannot alter this selection. Formal tasks and candidate-model-blind
references must be frozen before the exact confirmed models run on them. The formal
evaluation remains the first and only promotion decision for revision 2.

The selected bundle contains all three model artifacts, tokenizer files, decoder
configuration, vote threshold, hashes, environment versions, and selection report.
Inference must reject a missing or mismatched artifact.

### 5.2 Revision-3 frozen candidate generator plus span verifier

Revision 2 is terminally blocked. On its AI-assisted calibration reference, the best
powered high-recall point was exact 3-of-3 at confidence `0.50`: 459 predictions,
447 true positives, precision `0.973856`, recall `0.953092`, and one-sided 95%
Wilson precision lower bound `0.958553`. The only points reaching precision `0.99`
had fewer than 300 predictions and recall at most `0.603412`. Revision-2 confirmation
has not been read. This result rejects global confidence abstention as the sole
precision mechanism; it does not reject ML candidate generation.

Revision 3 is a new, explicitly diagnostic two-stage ML program:

1. The three immutable revision-2 epoch-5 models are the **frozen candidate
   generator**. They are not retrained or checkpoint-selected.
2. A new binary **span verifier** receives a candidate's exact geometry and only the
   current numbered jie's main text. It decides whether that exact occurrence is a
   person span.
3. A structural geometry layer runs before the verifier and may hard-veto only
   candidates that are intrinsically invalid.
4. The verified ML output is evaluated independently, then may enter production only
   as non-overlapping additions to the canonical Translation-assisted rules. It does
   not delete, resize, or relabel a canonical rule span.

Revision 3 supersedes revision 2 only after its artifacts are frozen. Until every
revision-3 gate passes, canonical Translation-assisted rules remain production.

#### 5.2.1 Candidate lattice and generator gate

Build one exact candidate lattice from the frozen three models. Include the union of
all exact spans supported by at least one seed with per-seed span confidence at least
`0.30`. Preserve each seed's pre-constraint character probabilities, exact support
bit, confidence, and emitted BIO geometry. The threshold and union rule are fixed
before verifier training and cannot be selected from verifier results.

On every labeled verifier-fit or verifier-calibration jie, report candidate-lattice
recall before any veto or verifier. Candidate-lattice recall must be at least `0.98`
overall. Report recall separately for one-character spans, multi-character spans,
role/appellation, foreign-title, and boundary/anaphora groups. A missing reference
span can never be recovered by the verifier; failure of the overall candidate-recall
gate blocks the revision before threshold selection. Compute this gate immediately
from the already frozen calibration predictions before creating folds, embeddings, or
verifier artifacts; it is the revision's first executable stop condition.

The structural layer may hard-veto only:

- out-of-bounds, empty, cross-paragraph, cross-jie, or hard-separator geometry;
- geometry containing punctuation, symbols, a newline, or a character functioning as
  quantity/enumeration rather than as part of a supported name core; numeral-shaped
  name characters such as `万` are not rejected by shape alone;
- geometry not wholly owned by target-text tokenizer positions;
- an impossible BIO decode that cannot map to one source-verified paragraph span; or
- a duplicate exact geometry after deterministic provenance merging;
- a single-character or given-name anaphor without a qualifying earlier main-text
  anchor in the same numbered jie; or
- a current surface made explicitly plural by a numeral or `诸/群/众`.

Single-character spans with the required earlier anchor, concrete singular roles,
offices, honorifics, royal references, foreign-name shapes, and a span contained in
or containing another plausible candidate are not hard-vetoed on shape alone. They
require verifier or conflict-resolution evidence. A hard veto must describe the
current occurrence and cannot use another jie, a person surface list, an identity, or
cumulative score. Validation reports zero unanchored anaphors and zero explicit
plural-marker spans at every downstream gate.

#### 5.2.2 Verifier examples and allowed evidence

Revision-2 calibration is now consumed development evidence. It may supply verifier
fit examples and error classes, but it can never again report an unbiased metric.
Revision-2 confirmation remains sealed and is not verifier fit, threshold, calibration,
checkpoint-selection, or hyperparameter-selection input.

Both the consumed calibration labels and their out-of-fold verifier metrics are
AI-assisted and may share correlated teacher/model errors. Their apparent precision
is potentially optimistic and is used only to choose whether the diagnostic verifier
is worth the one-shot confirmation; formal-grade evaluation remains the generalization
and precision check. The frozen encoder can also memorize surface patterns implicitly,
so a ban on explicit lookup features does not upgrade this evidence grade.

For every candidate-lattice geometry on consumed calibration:

- label `1` only when it exactly equals one frozen reference geometry;
- label every non-exact, partial, overlong, merged, role-only, lexicalized, or otherwise
  incorrect geometry `0`;
- retain overlapping alternatives as separate examples; and
- group all examples from one jie together in every split or fold.

The 12 false positives and 22 false negatives at revision 2's 3-of-3 `0.50` point are
error-analysis inventory, not a surface blacklist. The observed families include
single-character lexicalization, polity/people lexicalization, person-reference
titles, partial spans, merged adjacent spans, and missed long or foreign names.
Every family drives a general feature, example generator, or audit stratum; no
literal surface or identity may become a feature. Exact reference equality always
dominates the negative-family examples above: a concrete role occurrence present in
the frozen reference is positive, never a `role-only` negative.

The verifier is a binary span classifier over
`KoichiYasuoka/roberta-classical-chinese-base-char` revision
`51e91a5270ce5e68eb31b1c828598c09c3a5e4c6`. Its input is:

- target-only text from the current numbered jie;
- the candidate token/character mask and immediate left/right boundary positions;
- pooled encoder states for the candidate and its jie-local context;
- span length and boundary categories; and
- the frozen generator family's support count and per-seed confidences.

The verifier cannot read translation, notes, canonical rule output, person or identity
KBs, juan rosters, other jies, development identities, or literal surface lookup
features. Generator POS, BIO, NER, vote, and confidence remain one correlated model
family; the verifier must not describe them as independent votes.

Use a separately initialized verifier head and a frozen encoder for the first
revision-3 experiment. The fixed head is a two-layer MLP with hidden width `256`,
GELU, dropout `0.10`, and one sigmoid output. Train for exactly 20 epochs with AdamW,
learning rate `1e-4`, weight decay `0.01`, batch size `32`, seed `20260810`, and
positive class weight `1.0`. Neither fold metrics nor confirmation may select an
epoch. Any encoder fine-tuning, alternative architecture, loss weighting, synthetic
negative generation, or additional hyperparameter grid requires another written
revision before execution.

#### 5.2.3 Cross-fitted threshold and conflict resolution

Because the consumed calibration jies train the verifier, select its threshold only
from grouped out-of-fold predictions:

1. The frozen calibration contains exactly four juans (`105`, `151`, `278`, and
   `282`). Assign them to four leave-one-juan-out folds by ascending SHA-256 of
   `20260811:<juan>`; each fold holds one complete juan.
2. For each fold, train the exact fixed verifier on the other three juans and emit
   scores once for the held-out juan.
3. Concatenate the four held-out outputs. No example may be scored by a verifier that
   trained on its jie or juan.
4. Evaluate thresholds
   `{0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94,
   0.96, 0.97, 0.98, 0.99}`.

For a threshold `t`, retain candidates at or above `t`, then resolve overlaps within
each paragraph by deterministic weighted interval scheduling. Quantize each weight
to the integer
`round(1_000_000 * (logit(clamp(score,1e-6,1-1e-6)) - logit(t)))`.
This threshold-relative per-span penalty prevents raw score summation from
systematically preferring multiple weak fragments over one strong complete span.
Maximize total integer weight; ties prefer the greater additive sum of supporting-seed
counts, then the greater additive sum of each candidate's minimum supporting-seed
confidence quantized to integer millionths, then greater total covered length, then
fewer spans, then the lexicographically ascending ordered tuple of all
`(start,end)` geometries. Every optimization key before the final geometry tuple is
additive, so the interval DP carries and compares the complete key exactly. Candidate
and DP iteration order is this same total order; floating-point sums are never
compared. Conflict resolution cannot invent, extend, or split geometry.

An out-of-fold point is diagnostic-eligible only with at least 300 predictions,
precision at least `0.99`, recall at least `0.95`, and one-sided 95% Wilson precision
lower bound at least `0.98`. Select highest recall, then precision, then higher
threshold. Freeze the complete fold assignment, four model artifacts, all scores,
the 15-point table, and the selected threshold. If no point is eligible, stop without
reading confirmation. Precision and recall are end-to-end exact metrics against every
frozen reference span in consumed calibration; lattice misses, hard-veto removals, and
conflict-resolution losses remain in the recall denominator.

After selection, train one final verifier with the same fixed recipe on every consumed
calibration jie. The out-of-fold threshold is provisional because four three-juan
heads and the final four-juan head need not have identical score distributions. No
post-hoc score mapping, Platt scaling, isotonic calibration, or threshold change is
allowed; one-shot confirmation measures the transfer directly. This final verifier
is the only verifier allowed to read revision-2 confirmation. No full-data retraining
occurs after confirmation.

#### 5.2.4 One-shot diagnostic confirmation

Run the frozen generator, hard-veto layer, final verifier, selected threshold, and
conflict resolver exactly once on the still-unread 46-jie revision-2 confirmation
reference. Confirmation cannot alter a threshold, model, feature, veto, tie break, or
candidate-lattice rule.

The current confirmation reference is explicitly AI-assisted. Its precision estimate
may be upward-biased by correlated teacher/model errors and is diagnostic only; it
cannot seed formal-evaluation power or support a production claim.

Diagnostic confirmation passes only with:

- all 46 jies and complete reference counts;
- at least 300 predictions;
- candidate-lattice recall at least `0.98`;
- final exact precision at least `0.99`;
- final exact recall at least `0.95`;
- boundary-safe one-sided 95% precision lower bound, as defined in section 5.1, at
  least `0.98`; and
- zero cross-jie, source, hard-boundary, overlap, identity-leakage, unanchored-anaphor,
  or explicit-plural-marker violations.

The AI-assisted reference cannot authorize production even if these gates pass. A
pass authorizes only a candidate-model-blind upgrade of the already fixed 46-jie
confirmation reference: a human reviews 100% of the A/B positive union and the
predeclared negative-recall audit additions without seeing model predictions or
scores. Predictions remain frozen; after the upgraded reference is locked, rescore
the already frozen geometry without rerunning inference. The upgraded result must
pass every diagnostic confirmation gate and is the only revision-3 confirmation
bound allowed to seed the section 6 power plan. A quality failure permanently
consumes confirmation for revision 3, blocks the verifier, and cannot tune a
successor. A failure caused only by fewer than 300 predictions is reported as
underpowered rather than model-quality failure, but the read confirmation remains
consumed and still cannot be enlarged post hoc.

#### 5.2.5 Add-only rules integration

Formal and shadow comparisons evaluate three outputs separately:

1. canonical Translation-assisted rules;
2. standalone verified ML geometry; and
3. canonical rules plus ML additions.

The combined output preserves every canonical rule span byte-for-byte. An exact ML
match is attribution-only and is not a new addition. Any ML span overlapping a
canonical span is suppressed from the combined output; it cannot replace a boundary.
Only source-verified, non-overlapping ML-only geometry is added. Report raw ML
additions, suppressed overlaps, exact agreements, reference recoveries, false
additions, and net combined change separately.

The formal adoption gate applies both to standalone verified ML and to ML-only
additions. ML-only additions require exact precision at least `0.99` and a one-sided
95% lower bound at least `0.98`; combined output must also pass section 6.1 and improve
paired exact F1 over canonical rules. High combined precision cannot hide imprecise ML
additions.

### 5.3 Revision-4 fit-only structural hard-negative verifier

Revision 3 is terminally blocked before confirmation. Its four leave-one-juan-out
verifier heads did not improve the frontier: the highest-recall powered point predicted
474 spans at threshold `0.50`, with precision `0.953586` and recall `0.963753`; the
highest-precision point with at least 300 predictions used threshold `0.90`, with
precision `0.971429` and recall `0.652452`. Held-out folds contained only 6–26
negative candidates, and several unseen negative types received scores above `0.92`.
Revision-3 confirmation remains unread.

Revision 4 keeps the exact frozen candidate generator, `0.30` candidate-lattice rule,
structural hard vetoes, threshold table, conflict resolver, confirmation firewall, and
add-only rules integration from section 5.2. It replaces only verifier training and
feature inputs. Revision 3's consumed calibration labels, OOF scores, and errors are
architecture diagnostics; they are not revision-4 training examples.

#### 5.3.1 Clean data roles

- **Verifier fit:** all 189 jies in the frozen `fit` partition, grouped by their
  28 original juans. They may train verifier parameters but cannot select a threshold.
- **Verifier calibration:** the 45 consumed calibration jies. They select one threshold
  from the unchanged 15-point grid but do not train verifier parameters, fit a scaler,
  choose an epoch, choose a negative policy, or alter features.
- **Diagnostic confirmation:** the still-unread 46 confirmation jies. They retain the
  one-shot rules and gates from section 5.2.4.

The frozen deployment generator was trained on verifier-fit, so its fit predictions
are in-sample and cannot identify realistic whole-span mistakes. Do not use those
predictions for verifier examples. Instead create a mining-only out-of-fold generator:

1. Assign all 28 fit juans to five folds with the normalized vector objective from
   section 5.1, targeting one fifth of every vector component per fold, replacing the
   ordering seed with `20260813`, and resolving placement ties by fold number `1..5`.
2. For each fold, train the exact three pinned candidate seeds for fixed epoch 5 on
   the other four folds. Neither the held-out fold nor any development data selects a
   checkpoint.
3. Infer the unchanged one-seed/`0.30` lattice exactly once on the held-out fold.
4. Concatenate all held-out geometries. No mining prediction may come from a model
   trained on its jie or juan.

These 15 mining models and their predictions are verifier-training artifacts only and
are never deployment candidates. Freeze every fold assignment, dataset, model,
prediction, environment, and hash. The concatenated OOF lattice must cover at least
`0.98` of all fit reference spans; otherwise stop before verifier training. All
reported selection metrics still begin on verifier calibration.

#### 5.3.2 Deterministic fit example construction

For every fit jie, positive verifier examples are every exact frozen reference span.
Build negative examples from the following closed policies, in this order:

1. **Generator mistakes:** every mining OOF candidate at the unchanged
   one-seed/`0.30` lattice rule that is not an exact reference geometry.
2. **Strict partials:** for every reference span of length at least two, remove exactly
   one character from the left to create `[start+1,end)`, and separately remove
   exactly one character from the right to create `[start,end-1)`. A length-two
   reference therefore produces two one-character partials. Retain each source-valid
   non-empty geometry that is not another exact reference.
3. **One-character overreach:** extend each reference exactly one character left,
   right, and both directions when the extension remains in the same paragraph,
   contains no hard separator, punctuation, symbol, or functional numeral, and is not
   another exact reference.
4. **Adjacent merge:** when two reference spans satisfy
   `left.end == right.start` in one paragraph, add their exact union if that union is
   not itself a reference.

Deduplicate exact geometry before training. A geometry that exactly equals any
reference is always positive and cannot be relabeled by a negative policy. When
multiple negative policies produce one geometry, retain the earliest policy in the
numbered list above as its primary provenance and retain the complete ordered policy
membership list for audit/counting. Do not
generate arbitrary O-text substrings, literal error surfaces, identity-derived
variants, cross-jie examples, or random negatives.

Sort negatives by `(juan,jie_index,para_id,start,end,policy)` and retain at most four
negatives per positive within each jie, taking policies round-robin in the order above.
Within one policy, take examples in ascending
`(juan,jie_index,para_id,start,end)` order.
Every jie with positives retains at least one example from each available negative
policy before a policy receives its second example. Freeze the complete pre-cap and
post-cap inventories, policy counts, exact geometries, source hashes, and discarded
rows. Every floor applies to the post-cap training inventory. Require at least 2,000
total negatives, at least 150 OOF generator mistakes, at least 100 strict-partial
examples, at least 100 one-character-overreach examples, and at least 20 of the
frozen 28 fit juans with negatives; otherwise stop before training. Adjacent-merge
counts are reported but have no minimum because the corpus determines availability.
Policy floors count complete policy membership after the cap, not only primary
provenance, so a generator mistake that is also a strict partial contributes to both
audited families without becoming two training rows.

The functional-numeral check is the exact hash-bound implementation used by section
5.2.1; negative generation cannot reinterpret it. AI-assisted fit references may
contain teacher boundary errors, so record every synthetic negative that overlaps a
positive geometry as a label-noise audit stratum. Geometry remains negative unless it
exactly equals another frozen reference; this training-grade uncertainty is one reason
revision 4 remains diagnostic.

#### 5.3.3 Verifier features and fixed training

Use the same pinned frozen encoder and candidate/left-boundary/right-boundary/jie
context pooling from revision 3. Remove generator support count and confidence from
the verifier input so synthetic negatives cannot be distinguished by missing
generator metadata. The only non-encoder inputs are:

- `log1p(span length)`;
- one-hot Unicode boundary categories for the immediate left and right characters; and
- two edge bits stating whether the candidate starts or ends its paragraph.

Fit feature mean and scale on all verifier-fit examples only and freeze them before
reading calibration. Use the unchanged two-layer MLP: hidden width `256`, GELU,
dropout `0.10`, AdamW `1e-4`, weight decay `0.01`, batch size `32`, positive class
weight `1.0`, seed `20260812`, and exactly 20 epochs. Calibration cannot select an
epoch. Encoder fine-tuning, generator features, class reweighting, focal loss,
additional negative policies, or hyperparameter search requires another revision.

Train one verifier once on the complete frozen fit example inventory. Emit calibration
scores once, apply all section 5.2 hard vetoes, and evaluate the unchanged 15
thresholds with the unchanged deterministic conflict resolver. End-to-end recall uses
all 469 calibration reference spans, including lattice misses and vetoed geometries.
The section 5.2.3 prediction-count, precision, recall, Wilson, selection order, and
stop rules apply. Calibration is acknowledged as repeatedly consumed diagnostic
evidence; its metrics cannot support a production claim.

If and only if one threshold is eligible, freeze the exact already-trained verifier,
scaler, threshold, calibration scores, and table. Do not retrain on calibration.
Proceed to the same one-shot diagnostic confirmation and candidate-model-blind
formal-grade upgrade in section 5.2.4. A blocked calibration point leaves confirmation
unread and terminates revision 4.

Even a confirmation pass is only an AI-assisted diagnostic result. Repeated
architecture work on fit/calibration and the mining-model lineage do not gain
production weight; only the blinded formal-grade upgrade and section 6 can do so.

### 5.4 Revision-5 mining-purpose gate correction

Revision 4 is terminally blocked before hard-negative freeze or verifier training.
Its 15 mining models and OOF predictions were successfully frozen, but their
concatenated lattice covered `2393/2483 = 0.963754` fit reference spans, below the
revision-4 `0.98` gate. The same artifacts produced 297 source-verified OOF generator
mistakes and a valid 7,363-row post-cap negative inventory across all 28 fit juans;
every other section 5.3.2 floor passed. Confirmation remains unread.

The failed gate measured a property not used by the verifier:

- every exact fit reference is independently included as a positive verifier example;
- a mining-model miss is absence, not a negative label;
- mining recall does not bound deployment candidate recall or any calibration metric;
  the full-fit deployment lattice already passed its separate `460/469 = 0.980810`
  calibration recall gate; and
- four-of-five-fold mining models are expected to have lower recall than the full-fit
  deployment generator, while their purpose is to expose realistic wrong geometries.

Revision 5 prospectively corrects only this mining-purpose gate. It reuses the exact
immutable revision-4 fold plan, 15 model artifacts, and 15 held-out prediction
artifacts; retraining or replacing any of them is forbidden. Mining OOF recall becomes
a pipeline-sanity tripwire of `0.90`, derived before verifier training as deployment
calibration recall minus a fixed `0.08` out-of-fold tolerance. A lower value indicates
broken folds or mining training and still stops the program.

The actual authorization to freeze hard negatives is the complete post-cap evidence
gate from section 5.3.2: at least 2,000 total negatives, 150 OOF generator mistakes,
100 strict-partial memberships, 100 one-character-overreach memberships, and 20 fit
juans with negatives. Record mining recall and every miss for audit, but do not use it
as a model-quality or deployment-recall claim.

All other revision-4 inputs, feature bans, fixed verifier controls, calibration gates,
confirmation firewall, formal-grade upgrade, add-only integration, and stop rules
remain unchanged. This correction has no production weight and is valid only because
neither revision-4 verifier training nor confirmation inference occurred before it
was written.

### 5.5 Revision-6 separated existence and boundary objectives

Revision 5 is terminally blocked before confirmation. Its fit-only binary verifier
mixed 2,483 positives with 7,363 negatives and improved calibration precision at
threshold `0.50` to `0.979499`, but recall fell to `0.916844`. Of 39 end-to-end misses,
30 were lattice-covered exact positives scored below `0.50`; 9 were deployment-lattice
misses. Nine false positives remained. Raising the threshold monotonically reduced
recall; no point reached `0.95`. The failure shows that one binary objective cannot
simultaneously learn occurrence existence from realistic OOF mistakes and exact
boundary preference from synthetic overlapping alternatives.

Revision 6 reuses the exact immutable revision-5 mining plan, 15 mining models,
15 OOF predictions, hard-negative inventories, pinned encoder, deployment lattice,
calibration reference, structural vetoes, and confirmation firewall. It replaces only
the verifier head, training objective, and overlap resolver with two separately
supervised components.

#### 5.5.1 Existence head

Train the existence head only on real OOF generator candidates from section 5.3.1.
For half-open intervals in the same paragraph, define occurrence overlap as
`candidate_start < reference_end and reference_start < candidate_end`:

- positive: every OOF candidate overlapping at least one fit reference;
- negative: every OOF generator candidate overlapping no fit reference; and
- excluded: all synthetic partial, overreach, and adjacent-merge geometries not emitted
  by an OOF generator.

This occurrence-level label intentionally allows an imperfect boundary to reach the
boundary ranker; it does not authorize that geometry as output. The frozen inventory
must contain exactly the source-verified OOF lattice before synthetic augmentation.
Require at least 2,000 positive candidates, 150 negative candidates, and 20 fit juans
with negatives. A candidate may appear once after exact geometry dedup. Freeze its
label, overlapping reference geometries, source fold, and source prediction bindings.

Use the pinned frozen encoder poolings plus:

- `log1p(span length)`;
- left/right Unicode boundary categories; and
- paragraph-edge bits.

The head is the fixed two-layer MLP from revision 3: width `256`, GELU, dropout `0.10`,
one sigmoid output, AdamW `1e-4`, weight decay `0.01`, batch size `32`, seed `20260814`,
and exactly 20 epochs. Use class-balanced binary cross entropy with immutable row
weight `N / (2 * N_class)` computed from the complete frozen existence inventory.
Normalize the mean loss by the sum of row weights in each batch. Fit scaler and head
on fit OOF candidates only. No synthetic row enters this loss, and no generator
support or confidence feature is permitted because mining-fold and full-fit confidence
scales differ.

#### 5.5.2 Boundary ranker

Train a separate score head on each frozen fit reference and its source-valid
overlapping negative alternatives from the complete strict-partial,
one-character-overreach, adjacent-merge, and OOF-generator-mistake memberships.
Only candidates whose intervals overlap a positive in the same paragraph enter a
ranking group. Whole-span negatives with no positive overlap remain existence-only.

For every `(positive, negative)` pair in a group, the head emits an unbounded scalar
logit and optimizes `max(0, 1 - logit_positive + logit_negative)`. A geometry equal to
any frozen reference is always on the positive side. If an alternative overlaps two
references, create one pair against each overlapping exact reference. Cap at eight
negatives per positive by membership priority `generator_mistake`, `strict_partial`,
`one_character_overreach`, `adjacent_merge`, then ascending
`(juan,jie_index,para_id,start,end)`. Freeze all pairs, membership sets, and discarded
alternatives. A geometry may occur in both the existence and ranking inventories but
enters each loss only once per frozen row or pair. Require at least 2,000 pairs
spanning at least 20 fit juans and at least 100 pairs from strict-partial and
one-character-overreach membership.

The rank head uses the same frozen encoder poolings plus only `log1p(span length)`,
boundary categories, and paragraph-edge bits; it cannot read generator support or
confidence. It has the same two-layer architecture without an output sigmoid, AdamW
`1e-4`, weight decay `0.01`, batch size `32` pairs, seed `20260815`, and exactly 20
epochs. Batch loss is the arithmetic mean of pair hinge losses. The existence and
rank heads share no learned parameters and cannot select each other's epoch.

#### 5.5.3 Fixed inference and calibration

For each deployment-lattice candidate:

1. apply the unchanged structural hard vetoes;
2. compute an existence probability;
3. retain it when probability is at least threshold `t`; and
4. within each paragraph, sort retained candidates by descending unbounded rank logit,
   then descending generator support, descending summed generator confidence,
   descending covered length, and ascending `(start,end)`;
5. greedily accept a candidate if it overlaps no already accepted candidate.

This ordinal resolver never adds rank logits and therefore does not assume that
pairwise scores are cardinally comparable. The rank logit has no global admission
threshold and cannot delete a non-overlapping candidate that passed existence.
Generator metadata is used only as a deterministic inference tie-break and is not a
learned feature. Freeze exact float-to-order behavior: reject non-finite values and
compare the stored IEEE-754 float32 rank logits directly before integer/string
tie-break keys. Quantize summed generator confidence with round-to-nearest,
ties-to-even at `1e-6` before comparing it.

Train both heads once on their frozen fit inventories. Score calibration once and
evaluate only the unchanged 15 existence thresholds. End-to-end recall uses all 469
calibration reference spans. The section 5.2.3 minimum predictions, precision,
recall, Wilson lower bound, threshold selection, artifact freeze, and stop rules
remain unchanged. Neither calibration nor revision-5 scores may select loss weights,
pair policy, head width, epoch, or rank combination.

If no threshold is eligible, terminate revision 6 without reading confirmation. If a
threshold is eligible, freeze both already-trained heads and proceed exactly once
through section 5.2.4. This remains AI-assisted diagnostic evidence with no production
weight.

### 5.6 Revision-7 fit-only lexical hard-negative mining

Revision 6 is terminally blocked before confirmation. At existence threshold `0.50`
it produced 456 exact true positives, 21 false positives, precision `0.955975`, and
recall `0.972281`. Exact error decomposition found:

- 20 selected false positives overlapping no calibration reference;
- 1 selected boundary overreach replacing a covered exact reference;
- 3 covered exact references rejected by the existence threshold;
- 1 covered exact reference replaced by the ranker; and
- the unchanged 9 deployment-lattice misses.

The boundary ranker therefore selected the exact candidate in 459 of 460
lattice-covered reference cases when the exact candidate was admitted. The dominant
failure is whole-span existence discrimination. Revision 7 must not alter or retrain
the revision-6 ranker, resolver, generator, lattice, reference, encoder, thresholds,
or gates.

#### 5.6.1 Frozen mining model and candidate universe

Use only the frozen revision-6 existence head, scaler, and pinned encoder as a mining
model. Validate their hashes against the terminal revision-6 manifest. The mining
model and every source label are fit-only; revision-6 calibration scores and
calibration examples are forbidden inputs.

For every target paragraph in each of the 189 frozen fit jies, enumerate every
contiguous source span of 1 through 8 Unicode code points. A mining candidate must:

- stay inside one target paragraph;
- overlap no frozen fit reference under section 5.5.1's half-open rule;
- pass the unchanged intrinsic hard vetoes; and
- not duplicate a geometry in the frozen revision-6 existence inventory.

Score every surviving candidate with the frozen revision-6 existence head. Within
each jie, sort by descending stored float32 score, then ascending
`(para_id,start,end)`, and retain exactly
`min(candidate_count, max(8, 2 * reference_count))`. No score threshold, surface
list, calibration observation, or morphology heuristic may change this quota or
ordering.

Freeze every pool geometry and float32 score in deterministic per-jie NPZ arrays,
every retained row in reviewer-readable JSONL, per-jie pool/retained counts, source
text and reference bindings, mining-model hashes, and exact ordering. Require at
least 4,000 retained candidates spanning all 28 fit juans; otherwise stop before
review.

#### 5.6.2 Source-hidden negative verification

The frozen fit references are training references, not candidate-free exhaustive
formal annotation. A retained non-reference geometry therefore cannot become a
negative label from absence alone.

Run two independent AI teachers over every retained candidate. Each sees the full
current numbered jie and the highlighted geometry, but not its mining score, rank,
model identity, other teacher response, translation, notes, person KB, or identity
KB. Each returns exactly:

- `definitely_not_person`; or
- `possible_person_or_boundary`;

plus a confidence and short source-grounded rationale. Only unanimous
`definitely_not_person` decisions with both confidences at least `0.95` pass
provisionally. Send every other item, plus a deterministic 10% audit sample of the
provisional agreements selected by SHA-256 order over the canonical geometry key, to
a source-hidden third AI teacher with the same two labels and visibility restrictions.
A third-teacher `definitely_not_person` at confidence at least `0.95` passes; every
other item requires human review. The audit size is `ceil(0.10 * provisional_count)`;
sort the lowercase hexadecimal SHA-256 digests of the ASCII
`juan:jie_index:para_id:start:end` keys and take the first audit-size rows.

The human reviewer sees only the jie, highlighted geometry, and three rationales,
without scores or model identities, and chooses `not_person` or
`exclude_from_negative_training`. A possible person or boundary error is only
excluded; it does not add or change a positive reference. Freeze all prompts,
responses, confidence values, rationales, task bindings, human state, and completion
receipt. Each AI response must validate against a pinned JSON schema containing the
closed label enum, a finite confidence in `[0,1]`, and a non-empty rationale. Require
at least 2,000 verified mined negatives spanning all 28 fit juans; otherwise stop
before training.

#### 5.6.3 Existence-only retraining

Create the revision-7 existence inventory as the exact union of:

- all 2,696 frozen revision-6 real OOF existence rows with their unchanged labels; and
- all section 5.6.2 verified mined rows, each labeled negative.

Deduplicate by the canonical
`(juan,jie_index,para_id,start,end)` key and stop on any collision rather than
selecting a provenance. Any serialized `id` is only the canonical jie identifier for
the same `(juan,jie_index)` and is not an independent key. Use the exact section 5.5.1
features, architecture, optimizer, batch size, and 20 fixed epochs, changing only the
seed to `20260816`. Assign aggregate loss mass prospectively by stratum:

- `0.50` to real OOF occurrence-positive rows;
- `0.25` to real OOF non-overlap negative rows; and
- `0.25` to verified mined negative rows.

For a dataset of `N` rows, each row in stratum `s` receives immutable weight
`mass_s * N / N_s`; optimize the arithmetic mean of weighted per-row BCE losses in
each shuffled batch. Fit a new scaler on this complete fit-only inventory. Do not
initialize from the mining head.

Copy the revision-6 rank head and rank scaler byte-for-byte into the revision-7
artifact and verify their hashes before and after copying. Do not extract rank
training features or execute a rank optimizer.

#### 5.6.4 Fixed calibration and stop rule

Score the unchanged 517-candidate calibration lattice once with the new existence
head and copied ranker. Apply section 5.5.3's veto, unchanged 15 thresholds, ordinal
resolver, 469-reference denominator, Wilson gate, selection order, and artifact
freeze exactly.

If no threshold is eligible, terminate revision 7 without reading confirmation. If a
threshold is eligible, freeze the already-trained existence head and copied ranker
and proceed exactly once through section 5.2.4. This remains AI-assisted diagnostic
evidence with no production weight. Repeated mining, a second retraining pass, or
changing the candidate quota after seeing calibration is forbidden.

### 5.7 Revision-8 categorical fourth-teacher correction

Revision 7's source-hidden A/B/C labeling completed before existence retraining and
without reading calibration or confirmation. The scalar-confidence gate would pass
2,828 candidates and route 2,366 to humans. Of those human-routed candidates, 2,270
had the third teacher's categorical `definitely_not_person` decision but a
self-reported confidence below `0.95`. This confirms the pre-implementation challenge
that raw self-reported confidence is not calibrated across agents.

Revision 8 changes only training-label adjudication. It reuses the immutable mining
inventory and frozen A/B/C outputs and does not alter candidates, model features,
training controls, calibration, confirmation, or production gates.

Freeze the already-used teacher provenance as A/B/C model ID
`claude-sonnet-5` and D model ID `gpt-5.6-sol`, including every batch agent/run
identifier. A candidate can pass only with categorical support from both the Claude
and GPT model families.

1. Run a source-hidden fourth teacher over all 5,194 candidates, including the 1,553
   not read by C. It sees only the
   current numbered jie and highlighted geometry, with the same KB, translation,
   score, source, and cross-jie restrictions. It cannot see A/B/C decisions or
   rationales.
2. For a candidate read by C, provisionally pass a fourth-teacher
   `definitely_not_person` decision only when at least two of A/B/C also returned
   `definitely_not_person`.
3. For a candidate not read by C, provisionally pass only when A and B were the
   original high-confidence `definitely_not_person` agreement and D also returned
   `definitely_not_person`.
4. Every other item requires human review.
5. From all provisional passes, select
   `ceil(0.10 * provisional_pass_count)` by the exact section 5.6.2 canonical
   geometry SHA-256 order for human audit. The auditor sees the current jie,
   highlighted geometry, all available rationales (A/B/C/D when C read the item,
   otherwise A/B/D), and approved same-jie translation evidence when available. The
   auditor cannot see scores, KBs, identities, or other jies. Any possible-person
   decision must be confirmed against same-jie source and, when available,
   translation evidence; a confirmed `exclude_from_negative_training` decision stops
   Revision 8 before training and cannot be handled by deleting only that row.

The fourth teacher uses the same closed categorical labels and rationale schema.
D confidence remains recorded audit metadata but has no gating role; the frozen
A/B `0.95` gate only defines the already-existing Revision-7 routing partition.
Before D routing, assert that every candidate not read by C is exactly an A/B
high-confidence agreement excluded only because it was outside the deterministic C
audit; stop on any missing or differently routed item. Freeze the exact model IDs,
JSON schema hash, all routing, prompts, responses, and bindings. The majority rule is
fixed because it requires D plus two same-family source-hidden votes, not to clear the
2,000 floor. After human resolution, require at least 2,000 verified negatives
spanning all 28 fit juans or stop before training. This correction is valid only
because no mined negative has entered training and no human decision, calibration
score, confirmation example, or downstream metric was read before it was written.

### 5.8 Revision-9 existing-label safety veto

Revision 8 is stopped by user decision before any human review or training. Its
cross-family routing produced 4,701 provisional passes and 493 disagreements, but the
mandatory audit plus disagreements would require 964 new human decisions.

The mined spans carry `O` in the fit BIO sequences only because they do not overlap
the non-exhaustive fit references. That absence does not authorize a negative label.
Revision 9 instead narrows teacher-authorized candidates to the strongest
cross-family agreement stratum and measures its residual false-negative rate.

Freeze a safe-negative candidate inventory using only these rules:

1. retain only the 3,127 cross-family-unanimous candidates for which Claude runs A,
   B, and C plus GPT run D all chose `definitely_not_person`; these are two model
   families, not four independent families; discard the 1,453 A/B/D passes not read
   by C, the 121 candidates with only two-of-three original-family support, and all
   493 cross-family or majority disagreements without adjudicating them;
2. discard any candidate overlapping any same-paragraph, same-numbered-jie candidate
   geometry in the hash-validated
   `data-pipeline/persons/twostage/translation/evidence/` whose mapping status binds
   it to that canonical paragraph and numbered jie.

Translation is a positive-side veto only. Its absence cannot authorize a negative,
and it cannot add or alter a positive fit reference. The evidence must predate and be
independent of sealed references, calibration, and confirmation. Validate exact NFC
paragraph-text SHA-256 equality among the fit task, canonical text, and translation
evidence; jie indexes; every per-juan artifact hash; source surface; and half-open
code-point geometry before applying the veto. Do not use the current canonical
Agent-1 output as a second veto: its recorded translation-evidence manifest SHA-256
does not equal the present evidence manifest and therefore fails the required binding.

Require at least 2,000 candidates spanning all 28 fit juans after vetoes or stop.
Select a deterministic digest-ordered simple random sample of
`ceil(log(0.05) / log(0.99)) = 299` candidates, or the full inventory if smaller, for
source-hidden human audit. With zero observed errors this gives a one-sided 95%
binomial upper bound below 1% on the residual false-negative rate. Show the numbered
jie, exact highlighted geometry, and approved same-jie translation when available,
but no teacher decision, rationale, model score, intended negative label, agreement
count, or audit purpose before the human's initial judgment is immutably recorded.
Only after that judgment may the reviewer reveal the four teacher rationales. The
post-reveal decision may remain unchanged or become the more conservative
`exclude_from_negative_training`; it cannot change an exclusion to `not_person`.

The reviewer chooses only `not_person` or `exclude_from_negative_training`. Review may
stop at the first exclusion; any exclusion stops Revision 9 before training. Only a
complete zero-error audit authorizes the unaudited candidate inventory as the frozen
safe-negative stratum. The statistical claim bounds only the marginal candidate-level
false-negative proportion, not the absence of a recurring surface-level error family.
Freeze every kept, vetoed, excluded, and audited geometry;
veto memberships; source manifests and hashes; complete A/B/C/D provenance; audit
selection digest; exact confidence calculation; and the abandoned Revision-8
human-pack manifest hash.

Train the revision-9 existence head exactly as section 5.6.3, using the frozen safe
negatives as the `0.25` mined-negative loss stratum. Before calibration, score the
2,525 real-OOF occurrence-positive fit rows and require at least `0.95` recall at
existence threshold `0.50`; this is only a fit-pipeline sanity tripwire, not a
negative-label safety or generalization claim. Assert the exact 2,525/171 grouped-data
inventory and reject any canonical-geometry collision before training. Copy the
revision-6 ranker byte-for-byte.

Then execute section 5.6.4's single unchanged calibration pass and stop rules. No
additional mining, relabeling, human review, threshold change, or second training pass
is permitted. Confirmation remains unread unless the unchanged calibration gate
passes.

Revision 9 completed its 299-candidate blind audit with zero exclusions and froze
3,126 safe negatives across all 28 fit juans. The single training run passed the
fit-positive sanity tripwire at `1.0`, but no calibration threshold passed. At
threshold `0.94`, the model retained recall `0.9530916844` but reached only precision
`0.9738562092` and one-sided 95% Wilson lower precision `0.9585532989`
(`447` true positives from `459` predictions). Revision 9 is therefore blocked;
confirmation remains unread and the artifact has no production weight.

### 5.9 Revision-10 candidate-conditioned same-jie context

Revision 9 demonstrated that a frozen candidate/left/right/global-mean representation
cannot separate the remaining context-dependent negatives. Revision 10 changes only
same-jie context aggregation. It must not load Revision-9 calibration candidates,
scores, references, errors, or thresholds, and it must not read confirmation before
the fit-only gate below passes.

Reuse the exact Revision-9 frozen encoder and 5,822-row existence inventory: 2,525 real
OOF occurrence-positive rows, 171 real OOF negatives, and 3,126 zero-error-audited
mined negatives. Preserve the `.50/.25/.25` stratum masses. No new label, teacher
decision, translation evidence, person KB, identity KB, neighboring jie, or
cross-jie recurrence feature is permitted.

Encode each numbered jie with the unchanged overlapping 512-token windows and
single-owner character geometry. For every candidate, compute its candidate mean and
immediate left/right character vectors. Replace the global context mean with one
candidate-conditioned attention layer:

- query: a learned 128-wide projection of the candidate mean;
- keys: a learned 128-wide projection of every owned non-newline character in the
  current numbered jie;
- values: the unchanged frozen character vectors;
- additive learned scalar biases for relative-distance buckets
  `inside`, `1`, `2-4`, `5-16`, `17-64`, `65+`, and `other_paragraph`; distance is
  Unicode code points from the nearest candidate boundary, `inside` takes precedence,
  and `other_paragraph` takes precedence over numeric distance; and
- softmax only over owned characters in the current jie.

Concatenate candidate, left, right, attended-context, current-paragraph mean, and
whole-jie mean vectors with section 5.3.3's existing boundary features plus normalized
candidate start/end within its paragraph, normalized paragraph ordinal within the jie,
and paragraph-count `log1p`. Feed this to one 256-wide GELU/dropout binary existence
head. The encoder remains frozen. Use attention width `128`, dropout `0.1`, AdamW
learning rate `1e-4`, weight decay `0.01`, 20 epochs, and seed `20260817`. Candidate
scores, generator support, teacher confidence, surface identity, and cross-jie
features remain forbidden.

Fit two scalers on each fold's 24-juan training subset only: one per-dimension
mean/std over every owned non-newline frozen character vector, applied to all
candidate, boundary, attention-key/value, paragraph, and jie vectors; and one
mean/std over the candidate-row numeric features. Clamp either scale to `1.0` below
`1e-6`. Missing left/right boundaries remain exact zero vectors after scaling. Apply
the frozen fold scalers unchanged to its four held-out juans.

Cache each fold's frozen character vectors once. Training examples are candidate rows,
not variable-size jie batches: shuffle rows uniformly with batch size `32`. Give every
row in stratum `s` immutable weight `mass_s * N / N_s` and optimize the arithmetic
mean of weighted per-row BCE in each batch, exactly as section 5.6.3. A jie may share
one cached character tensor across its rows, but this must not change row-level
shuffling, batch membership, or loss normalization. Recompute `N` and every `N_s`
from each fold's 24-juan training subset; only the final full fit uses the complete
28-juan counts.

Before any training, freeze seven folds by sorting the 28 fit juans numerically and
assigning them round-robin to folds `0..6`; each fold therefore holds four complete
juans. For each fold, fit a fresh attention/head/scaler on the other 24 juans and emit
scores for every row in the four held-out juans. No row, candidate from the same jie,
or fitted scaler may cross the fold boundary.

Evaluate the concatenated OOF scores at the unchanged 15 thresholds. A threshold is
fit-eligible only when:

- OOF recall over the 2,525 real positives is at least `0.98`;
- OOF precision over all labeled OOF rows is at least `0.99`;
- the one-sided 95% Wilson precision lower bound is at least `0.985`;
- mined-negative rejection is at least `0.99`;
- real-OOF-negative rejection is at least `0.98`; and
- every fold has positive recall at least `0.95`.

The real-negative rejection gate is the primary fit-only evidence for the stated
context-separation goal. Mined-negative rejection is only a safety sanity check:
those candidates were selected with a full-fit mining model and are not unbiased OOF
generalization evidence. The row-level Wilson bound is uncorrected for jie/juan
clustering and is also only a fit diagnostic. Both remain necessary-but-not-sufficient
conservative eligibility floors; confirmation remains the sole fresh transfer gate.

Select the eligible threshold by descending global recall, precision, then threshold.
If none is eligible, stop Revision 10 without fitting a full model or reading
confirmation. If one is eligible, freeze the OOF table and selected threshold, fit
exactly one final model on all 28 fit juans with the same controls, copy the
Revision-6 ranker byte-for-byte, and evaluate confirmation exactly once at that fixed
existence threshold through section 5.2.4's complete generator, hard-veto, copied
ranker, ordinal resolver, lattice-recall, precision, recall, Wilson, and minimum-count
gates. No old calibration pass, threshold adjustment, second model, or
post-confirmation retry is permitted.

Revision 10 completed the seven-fold fit-only OOF selector and stopped without a
final fit or confirmation read. At threshold `0.50`, precision was `0.9321579149`,
recall was `0.9631683168`, mined-negative rejection was `0.9772872681`, and
real-negative rejection was only `0.3801169591`. Raising the threshold to `0.99`
improved real-negative rejection only to `0.5087719298` while recall fell to
`0.9259405941`. No threshold was eligible. Candidate-conditioned same-jie attention
therefore did not solve cross-juan separation of the 171 real hard negatives.

### 5.10 Revision-11 candidate-marked encoder fine-tuning

Revision 10 shows that a learned head over frozen character vectors is insufficient.
Revision 11 permits the encoder itself to adapt, but remains a fit-only experiment:
it must not load any calibration or confirmation candidate, reference, prediction,
score, error inventory, or threshold. The seven Revision-10 juan folds and all
eligibility gates remain unchanged.

Reuse the exact 2,696 real OOF rows and 3,126 audited mined negatives. Relabel the real
OOF rows from their frozen geometry and `overlapping_references` only:

- `exact_person`: the candidate exactly equals an overlapping reference (`2,393`);
- `boundary_alternative`: it overlaps a reference but is not exact (`132`); and
- `not_person`: it overlaps no reference (`171`).

Every audited mined negative is also `not_person`, but remains a distinct
`mined_not_person` loss stratum. Do not invent semantic subtype labels from surfaces,
teacher rationales, lexicons, translations, or model scores. In particular,
collective, role, and metalinguistic errors are learned only through the observed
`not_person` decision unless a future candidate-blind human protocol supplies those
subtype labels.

For each row, construct one candidate-marked input containing only its numbered jie.
Segment A is the exact candidate surface. Segment B is the assembled current-jie text
with reserved in-vocabulary marker tokens immediately before and after the occurrence
identified by the row's frozen paragraph-local `[start,end)` geometry, never by a
surface search, and a reserved paragraph marker replacing each newline. Before
training, bind the sentinels to `㈠` (left), `㈡` (right), and `㈢` (paragraph).
Require their distinct tokenizer IDs `8680`, `11019`, and `12821`, their absence from
every fit-jie source string, and exactly one token for each whitespace-delimited
literal. Sentinel-delimiting ASCII spaces are model-input syntax, not source text.
Adding tokens or resizing embeddings is forbidden.

The `384`-wordpiece limit includes Segment A, Segment B, all three marker types, and
every tokenizer special token. Tokenize complete code points; never split a code point
or candidate. First try the complete marked jie. If it does not fit, define one fixed
expansion sequence from the complete candidate geometry: prepend one code point, then
append one, alternating from the left; when one jie edge is exhausted, continue only
on the other side. Select the longest prefix of that expansion sequence whose complete
paired tokenization is at most `384` wordpieces. Find the prefix by binary search over
its integer length and assert that the selected prefix fits while the next prefix does
not. This tokenizer's character-level token count must be nondecreasing over that
sequence; a violated assertion stops the run. Freeze the resulting assembled-text
codepoint bounds and exact token IDs. Paragraph markers inside the slice count against
the same budget.

Preflight every row before fold training. Segment A must contain at least one
non-special token; both occurrence markers must appear exactly once in Segment B in
the correct order; and at least one Segment-B token must lie between them. Any failure
stops the complete Revision-11 run rather than dropping a row or changing a metric
denominator. Candidate text, marker IDs, and current-jie text are the only inputs;
generator support, confidence, identity, person KBs, neighboring jies, and cross-jie
recurrence remain forbidden.

Initialize from the exact Revision-9 encoder. Fine-tune all encoder parameters and a
new three-class head. Concatenate the final-layer `[CLS]` vector, the mean Segment-A
candidate vector, and the mean marked-occurrence vector; apply dropout `0.10` and one
linear three-logit classifier. Use cross entropy with immutable fold-local stratum
masses `0.45` exact person, `0.15` boundary alternative, `0.20` real not-person, and
`0.20` mined not-person. For each fold, compute `N` and `N_s` from only that fold's
24-juan training subset and give each row weight `mass_s * N / N_s`; only a permitted
final full fit uses all 28-juan counts. For each deterministic accumulation group,
sum weighted per-row losses and divide by the actual row count in that group. Split
that already-known divisor across its physical micro-batches before backpropagation;
the last partial group therefore uses its own actual row count. Use seed `20260818`,
exactly 3 epochs, Adafactor encoder learning rate `1e-5`, classifier learning rate `1e-4`,
weight decay `0.01`, physical batch size `1`, gradient accumulation `32`, gradient norm
cap `1.0`, and no scheduler, warmup, early stopping, augmentation, or hyperparameter
search. Shuffle rows deterministically each epoch. Fit each fold from the same
untouched Revision-9 initialization; no model or optimizer state crosses folds.
Use Transformers Adafactor with `scale_parameter=False`, `relative_step=False`,
`warmup_init=False`, and no first-moment term. Keep model parameters, gradients, and
factored optimizer state on the GPU.

The admission score is softmax probability of `exact_person`. Evaluate the concatenated
OOF scores at the unchanged 15 threshold values, with Revision-10 gates restated for
the new exact-geometry labels. A threshold is eligible only when:

- recall over the fixed 2,393 `exact_person` rows is at least `0.98`;
- precision counts only admitted `exact_person` rows as true positives and is at least
  `0.99`;
- the corresponding one-sided 95% Wilson precision lower bound is at least `0.985`;
- rejection of the 3,126 mined not-person rows is at least `0.99`;
- rejection of the 171 real not-person rows is at least `0.98`;
- rejection of the 132 boundary alternatives is at least `0.90`; and
- every fold's recall over its exact-person rows is at least `0.95`.

Report the complete three-by-three argmax confusion matrix and score distributions by
all four loss strata, but neither may change the fixed threshold table or controls.
At inference, sort threshold-admitted overlapping candidates first by descending
stored float32 exact-person probability, then by the copied Revision-6 rank logit and
its existing deterministic tie-breaks. The copied ranker always runs against its
original pinned Revision-9 encoder and scaler, never the fine-tuned encoder; inference
therefore loads the fine-tuned encoder for admission and the pinned encoder for
ranking.

If no threshold is eligible, stop without a full fit or confirmation read. If one is
eligible, freeze the OOF artifact and selected threshold, fit one final model on all 28
fit juans, and run the unchanged one-shot confirmation workflow with the copied
Revision-6 ranker. No calibration read, threshold adjustment, retry, semantic-label
expansion, or post-confirmation retraining is permitted.

Revision 11 completed all seven OOF folds and stopped without a full fit or
confirmation read. At threshold `0.50`, exact recall was `0.9870455495`, precision was
`0.9391650099`, real-negative rejection was `0.6140350877`, and boundary-alternative
rejection was `0.4621212121`. Higher thresholds improved rejection only by sacrificing
recall; no threshold was eligible. Encoder adaptation therefore preserved exact
positives but did not establish the required transfer separation.

### 5.11 Revision-12 blind hard-label audit

Before another model revision, audit all 303 difficult fit geometries: the 171 real
OOF rows labeled not-person and the 132 occurrence-positive rows whose geometry does
not exactly equal an overlapping reference. This is label diagnosis, not model
selection. Revision-11 scores, predictions, argmax classes, fold metrics, and error
inventories must not enter a review task or reviewer prompt. Calibration and
confirmation remain unread.

Create one immutable task per numbered jie containing its complete current-jie text,
paragraph geometry, and every selected candidate's source-verified paragraph-local
`[start,end)` geometry and surface. Hide the frozen class, overlapping references,
policy provenance, generator score/support, model score, and all other candidates
outside the 303-row inventory. Candidate IDs are SHA-256 digests of exact geometry and
the bound grouped-data manifest. Randomize candidate display order by digest while
retaining complete numbered-jie context. No reviewer may use neighboring jies, a
person or identity KB, translation evidence, or cross-jie recurrence.
Publish one atomic audit root containing sibling `reviewer-tasks` and
`sealed-original-labels` directories. Distribute only `reviewer-tasks`; its manifest
contains no answer key or path to the sealed sibling.

Two independent model families judge every candidate using exactly four labels:

- `exact_person`: the candidate geometry is exactly one individual-person mention;
- `wrong_boundary`: a person is present but the candidate includes too much, omits
  part of the name/appellation, or merges multiple mentions;
- `not_person`: the candidate is not an individual-person mention; and
- `uncertain`: the text does not support a reliable decision.

Each judgment includes confidence in `[0,1]` and a non-empty same-jie rationale.
Reviewers are source-hidden from one another. Validate complete candidate coverage,
task hashes, labels, confidence, and provenance before comparison.
Each validated output declares a model-family identifier, and routing must reject
equal family identifiers.

Accept an AI audit decision only when both families give the same non-`uncertain`
label with confidence at least `0.90`. Route every disagreement, uncertainty, or
lower-confidence agreement to one blind human review page. The page initially hides
AI labels and rationales; the human chooses the same four labels from current-jie text.
AI rationales may be revealed only after the immutable first judgment. No AI agreement
or human decision enters training until the complete 303-row audit is frozen.
Routing likewise publishes one atomic root with a sealed adjudication sibling and a
human-review sibling containing only current-jie tasks and candidate geometry.
For this Revision-12 audit, the user explicitly authorizes a fresh source-hidden
Copilot teacher to act as the human adjudicator. Its first judgments use the identical
label-free human bundle and immutable state API; they carry human-gate authority for
this diagnostic training audit, but do not convert the underlying AI-assisted
references into formal-grade evidence.

After freeze, report the original-to-audited confusion matrix separately for the 171
real negatives and 132 boundary alternatives. A row audited `exact_person` is removed
from negative training and added as an exact positive only if its geometry is
source-valid and does not conflict with another audited exact positive. A
`wrong_boundary` row may enter only a pairwise boundary objective against the audited
exact geometry identified by a subsequent candidate-blind geometry review; the audit
label alone may not invent that target. A `not_person` row remains a semantic negative.
Any final `uncertain` row is excluded from Revision-12 training. Specify and challenge
the pairwise training revision only after these counts and geometries are frozen.

## 6. Fresh formal evaluation

Formal evaluation is sampled and completely labeled before candidate inference.
It uses previously unused jies, applies the selected context closure, and excludes
all program training/development data plus every POC-consumed or leaked item.

Freeze:

- at least 100 uniform-random eligible jies from the full 20-600-codepoint frame,
  closed under the selected context graph; and
- 20 role/appellation, 20 foreign-title, and 20 boundary/anaphora challenge jies from
  predeclared raw-text cohorts.

The planner may increase the random sample before task generation to satisfy the
power rule below, but may never decrease it afterward. Challenge jies do not enter
the probability estimate.

The candidate-model-blind A/B and focused-review protocol from section 4.2 applies.
The formal reference, audit decisions, candidate hash, and analysis code hash are
locked before inference. Evaluation is single-use: failure cannot select, tune, or
retrain the evaluated candidate.

### 6.1 Power and adoption gates

Before sampling, the planner computes and records the minimum random-jie count needed
for 80% power, clustered at the jie level. Anticipated precision is the lower of
`0.985` and the active revision's confirmation boundary-safe precision lower bound;
consumed development is not an input. For revision 3 this means the one-shot
geometry rescored against the formal-grade upgraded confirmation from section 5.2.4,
never the AI-assisted diagnostic score or its out-of-fold verifier-fit result.
The power simulation resamples complete confirmation jies, uses 50,000 replicates,
and records its code hash and assumptions.
At least 1,000 predicted random-set spans and 100 random jies are required regardless
of the estimate. If the fixed 160-jie reserve cannot fund the required random count
plus all 60 challenge jies, formal evaluation is not created; no consumed inventory
may supply extra tasks. If inference produces fewer than 1,000 predictions, the
evaluation is underpowered and fails without adding post-hoc tasks.

If that conservative anticipated precision is at or below `0.98`, the candidate
cannot be powered for the adoption alternative and formal evaluation is not created.

Adoption requires all of:

- random-set exact precision boundary-safe one-sided 95% lower bound, using the
  confirmation definition with 50,000 BCa replicates and a span-level/leave-one-jie
  Wilson fallback, at least `0.98`;
- lower bound of the paired 90% jie-bootstrap exact-F1 difference from canonical
  Translation-assisted rules greater than zero;
- exact recall at least `0.95`;
- challenge strata with at least 50 reference spans have no paired 90% exact-F1
  difference lower bound below `-0.05`; smaller strata are exhaustively error-audited
  and must have no systematic invariant or boundary regression;
- zero hard-boundary or cross-jie invariant violations;
- focused-review audit acceptance; and
- reproducible byte-identical decoded geometry on a second run.

Reproducibility sets deterministic PyTorch algorithms, deterministic cuDNN behavior,
disabled TF32, pinned CUDA/cuDNN/PyTorch/Transformers versions, and a deterministic
equal-logit label-order tie break. The gate hashes decoded geometry and manifests,
not floating-point logits. CPU inference on the targeted set is the tie-break
reference if the pinned GPU stack cannot supply a deterministic kernel.

Report exact and overlap metrics, per-stratum metrics, raw additions, removals, net
growth, reference recoveries, reference regressions, and geometry replacements.
A net count is never described as new spans.

## 7. Offline production output

The production tagger is an offline build step, not a browser or service dependency.
It reads corpus text plus the immutable selected model bundle and writes one file per
juan with the existing Agent-1 contract:

```json
{
  "schema_version": 2,
  "juan": 1,
  "source": "ml-ensemble",
  "model_bundle_sha256": "...",
  "decoder_sha256": "...",
  "occurrences": [
    {
      "juan": 1,
      "para_id": 0,
      "start": 0,
      "end": 2,
      "surface": "人物",
      "chunk_type": "ml_person",
      "rule": "ml_ensemble_3_of_3",
      "scope": "numbered-jie",
      "ce_year": null,
      "field": "main"
    }
  ]
}
```

Offsets are paragraph-local Unicode codepoint `[start,end)` geometry. Output must be
sorted, non-overlapping, main-text-only, source-verified, and byte-identical across
repeated builds. The manifest records corpus, model, decoder, code, and per-juan
hashes plus counts.

For revision 2, `source` is `ml-ensemble`; `rule` is exactly
`ml_ensemble_2_of_3` or `ml_ensemble_3_of_3` as bound by the selected bundle. For
revision 3 standalone output, `source` is `ml-span-verifier` and `rule` is exactly
`ml_span_verifier_v1`. Combined add-only output preserves each canonical rule's
original provenance and gives only accepted ML additions the
`ml_span_verifier_v1` rule. Any other value is invalid.

Agent 2 may reuse identity only for exact matching geometry. Missing identity leaves
an unresolved underline; identity data never changes Agent-1 output.

## 8. Rollout and rollback

1. **Targeted shadow:** build positive, known-false-positive, regression, and
   unchanged-control juans into a new directory. Inspect every addition, removal,
   regression, and geometry replacement.
2. **Full shadow:** after targeted geometry is explained, build all 294 juans into a
   new immutable directory. Do not overwrite rules or production output.
3. **Release audit:** inspect every formal-evaluation error, every targeted delta,
   all invariant violations, and a deterministic stratified sample of full-shadow
   ML-only and rules-only geometries.
4. **Canary data release:** materialize the existing v2 Agent-1 directory from an
   explicit juan allowlist: selected juans copy immutable ML sidecars and all other
   juans copy the frozen Translation-assisted rules baseline. The release manifest
   records the origin and hash of every sidecar; it is invalid if a file has no
   declared origin. v1 remains the default reader variant.
5. **Expand:** increase the allowlist only after data-integrity checks and manual
   reader smoke tests. Expansion changes data/configuration only, never model bytes.
6. **Default:** making ML the default requires a separate product approval after
   canary completion. This program does not silently flip the reader default.

Publication copies only a previously validated immutable shadow artifact. It never
runs inference in place. The release manifest names the exact model bundle and
allowlist. Rollback restores the previous manifest/allowlist and requires no rebuild;
old artifacts remain available until the release is closed.

Mixed canary geometry is an accepted temporary condition because ML follows the
name-core policy while some rules differ. Release checks report exact-geometry Agent
2 identity reuse separately for ML and rule-origin juans; a drop does not authorize
fuzzy identity binding. The canary UI and audit identify the active origin.

Any source-hash mismatch, missing model, nondeterministic output, invalid geometry,
identity leakage, metric-gate failure, or canary data error blocks expansion and
restores the previous release manifest.

## 9. Human gates

Human input is required only for:

1. focused review of fresh training/development A/B disagreements and audit samples;
2. focused review of the formal reference before it is locked;
3. release-audit disposition of unexplained systematic errors; and
4. approval to expand canary coverage or make ML the default.

Engineering may proceed without further product decisions through immutable round
planning, task generation, schema validation, teacher-pass ingestion, model training,
shadow generation, and automated evaluation. It must stop at each listed gate rather
than infer approval.

## 10. Required implementation order

```text
program spec and exclusion review
  -> immutable training/development plan
  -> candidate-blind A/B task generation
  -> focused human review
  -> dataset freeze
  -> deterministic three-seed training
  -> development-only operating-point selection
  -> prospective formal plan and power check
  -> formal A/B tasks and focused human review
  -> reference lock
  -> one-time candidate and rules inference
  -> adoption decision
  -> targeted shadow and complete delta inspection
  -> full immutable shadow
  -> release audit
  -> allowlisted canary
  -> explicit default-product decision
```

If review or audit finds a structural error, return to a new training round. Never
retain statistics from a model, policy, label set, or decoder that changed afterward.

Revision 3 uses this more specific order before the prospective formal-plan step:

```text
freeze revision-2 generator and consumed calibration
  -> build and validate the 0.30 one-seed candidate lattice
  -> freeze four leave-one-juan-out verifier folds
  -> train fixed frozen-encoder verifier heads
  -> emit exactly-once out-of-fold scores
  -> select or block from the fixed 15-point threshold table
  -> train one final verifier on all consumed calibration
  -> one-shot diagnostic confirmation
  -> stop if diagnostic gates fail
  -> formal-grade plan and human gate if diagnostic gates pass
```
