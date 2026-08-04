# Agent-1 ML production program

Status: draft implementation contract. This program authorizes new engineering and
data collection, but it does not authorize publishing the existing Round 7 model.

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
- 20 foreign-title jies sampled from a frozen raw-text cohort; and
- 20 boundary/anaphora jies sampled from a frozen, identity-free structural cohort.

The 140 jies are training-only. Cohort membership is computed from raw characters,
punctuation, and generic title/role terms before model inference. Sampling is
deduplicated by jie and then closed under the selected context graph, not by whole
juan.

The program also freezes a fresh 40-jie development set: 20 uniform random jies and
20 balanced challenge jies, disjoint from training after context closure.
Development is used for checkpoint and ensemble operating-point selection and can
never become training data. A development set supports at most one training-data
revision and one declared model-selection comparison; another improvement round
requires a newly sampled development set.

Before sampling, the planner proves that the remaining eligible inventory can fund
this round, its formal evaluation, and one complete replacement round under the
selected context closure. If it cannot, it stops and reports the shortfall. It must
not weaken exclusions, reuse a sealed reference, or silently change the sample.

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
4. For training and development, the human reviews all disagreements, explicit-low
   candidates, and deterministic audit samples of 20% of consensus spans plus 20%
   of consensus-negative jies.
5. Any incorrect audited positive or any missed person in an audited negative jie
   expands that task to 100% union-candidate review. If the stratum's audited exact
   error rate exceeds 1%, audit all union candidates in that stratum before freeze.
6. For formal evaluation, the human reviews 100% of the A/B positive union. A
   deterministic 25% of A/B-consensus-negative jies receives a third independent
   recall pass and focused review of any additions. Any miss expands the remaining
   consensus-negative jies in that stratum to a third pass.

The pipeline validates task inventory, provenance, source hashes, bounds, surfaces,
non-overlap, and legal geometry independently of teacher self-reports. Persist only
main-text geometry and provenance; note or translation prose is transient.

Copilot agreement is never treated as independent ground truth. Full formal-positive
review prevents shared false-positive biases from inflating measured precision, and
the adaptive negative audit measures shared omissions without assigning blank-text
exhaustive annotation to the user.

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
3. every challenge stratum contains at least 200 reference spans, expanding its
   pre-inference development sample if necessary;
4. the lower bound of the paired 90% jie-bootstrap exact-F1 difference from rules
   is greater than zero overall;
5. no challenge stratum's paired 90% difference lower bound is below `-0.03`;
6. highest exact precision; then
7. the stricter vote threshold.

If no operating point reaches `0.99` development precision, do not create a formal
evaluation. Perform one new training-data round under section 4; do not tune on a
formal or sealed set. Architecture changes, probability calibration, note-aware
runtime inference, self-training, and identity features require a new spec revision.

The selected bundle contains all three model artifacts, tokenizer files, decoder
configuration, vote threshold, hashes, environment versions, and selection report.
Inference must reject a missing or mismatched artifact.

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
for 80% power, clustered at the jie level. Its anticipated precision is the lower of
`0.985` and the development one-sided 95% cluster-bootstrap precision lower bound;
it must not use the selected operating point's optimistic point estimate. The power
simulation resamples complete development jies, uses 50,000 replicates, and records
its code hash and assumptions. At least 1,000 predicted random-set spans and 100
random jies are required regardless of the estimate. If inference produces fewer
than 1,000 predictions, the evaluation is underpowered and fails without adding
post-hoc tasks.

If that conservative anticipated precision is at or below `0.98`, the candidate
cannot be powered for the adoption alternative and formal evaluation is not created.

Adoption requires all of:

- random-set exact precision one-sided 95% BCa jie-bootstrap lower bound, using
  50,000 replicates, at least `0.98`; also report a span-level Wilson sensitivity
  interval without using it for the decision;
- lower bound of the paired 90% jie-bootstrap exact-F1 difference from canonical
  Translation-assisted rules greater than zero;
- exact recall at least `0.95`;
- every formal challenge stratum contains at least 200 reference spans before
  inference, and no stratum's paired 90% exact-F1 difference lower bound is below
  `-0.05`;
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

`source` is `ml-ensemble`; `rule` is exactly `ml_ensemble_2_of_3` or
`ml_ensemble_3_of_3` as bound by the selected bundle. Any other value is invalid.

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
