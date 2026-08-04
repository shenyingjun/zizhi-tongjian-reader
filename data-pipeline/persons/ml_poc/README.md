# Agent-1 ML POC

This folder contains the pilot-first proof of concept described in `SPEC.md`
and its Chinese counterpart `SPEC.zh.md`.
It is isolated from the production two-stage pipeline until the P0/P1 gates are
met.

The completed POC did not authorize production adoption. The prospective training,
evaluation, rollout, and rollback contract is in `PRODUCTION_SPEC.md`.

After assembling a complete exact-jie exclusion inventory from the POC artifacts,
freeze the first candidate-blind production training/development round:

```powershell
python production_exclusions.py `
  --artifact <POC-artifact-directory> `
  --output <exact-jie-exclusions.json>

python production_program.py `
  --exclusions <exact-jie-exclusions.json> `
  --seed <prospectively-recorded-seed> `
  --output <new-round-directory>
```

The command requires a clean commit, never overwrites a round, and emits tasks that
contain no model, rules, v1, identity, translation, note, or challenge-role data.

After the two mutually hidden Copilot passes are complete, validate them and freeze
the focused-review pack:

```powershell
python production_review.py `
  --round <frozen-round-directory> `
  --teachers <pass-a-and-pass-b-directory> `
  --audit-seed <prospectively-recorded-seed> `
  --output <new-review-directory>
```

Merge the independent audit of sampled A/B-negative jies before human review:

```powershell
python production_negative_audit.py `
  --review <review-directory> `
  --audit <negative-audit-pass-directory> `
  --output <new-complete-review-directory>
```

Prepare source-hidden third-teacher adjudication tasks. After an independent teacher
produces one matching output per task, merge its high-confidence accept/reject
decisions into a new immutable review version:

```powershell
python production_third_teacher.py prepare `
  --review <complete-review-directory> `
  --output <new-third-teacher-task-directory>

python production_third_teacher.py merge `
  --review <complete-review-directory> `
  --tasks <third-teacher-task-directory> `
  --adjudications <third-teacher-output-directory> `
  --output <new-third-teacher-review-directory>
```

Run the ported local web UI against the immutable third-teacher review pack. Human
decisions are written only to the separate state directory:

```powershell
python production_review_server.py `
  --review-dir <complete-review-directory> `
  --state-dir <new-human-review-state-directory>
```

Open `http://127.0.0.1:18766`. Rejecting an audited consensus candidate, accepting
a negative-jie recall addition, or overriding any automatic third-teacher decision
expands that task to full union-candidate review before it can be locked.

Run focused tests:

```powershell
python -m unittest discover -s . -p "test_*.py" -v
```

Current scope:

- numbered-jie assembly with explicit paragraph geometry;
- identity-safe sanitization of 胡三省注 person evidence;
- constrained BIO decoding;
- one-to-one exact and overlap span metrics.

No model is trained and no production output is modified at this stage.

Prepare deterministic, candidate-blind pilot tasks:

```powershell
python pilot.py --output <artifact-directory>
```

The generated directory contains `manifest.json` and one `blind_juan_NNN.json`
per selected juan. Generated tasks intentionally omit v1, rules, translation,
notes, and identity fields.

Build the physically separate recall packs:

```powershell
python recall.py --blind-dir <blind-directory> --output <recall-directory>
```

After all recall passes are complete, build the separate specific-role audit packs:

```powershell
python role_audit.py `
  --blind-dir <blind-directory> `
  --recall-dir <recall-directory> `
  --state-dir <annotation-state-directory> `
  --output <role-audit-directory>
```

The audit proposes uncovered title/office surfaces for review. It does not alter the
locked blind or recall phases and contains no identity fields.

Start the local-only annotation UI:

```powershell
python server.py `
  --blind-dir <blind-directory> `
  --recall-dir <recall-directory> `
  --role-audit-dir <role-audit-directory> `
  --state-dir <annotation-state-directory>
```

Open `http://127.0.0.1:18765`. The server never returns recall evidence before
that juan's blind phase is completed and locked, or role-audit evidence before
recall is completed and locked. Each phase is persisted and locked independently.
Annotation state is written atomically outside the repository.

Freeze the completed audit as an identity-free char-BIO P1 dataset:

```powershell
python p1_dataset.py `
  --blind-dir <blind-directory> `
  --state-dir <annotation-state-directory> `
  --report-dir <report-directory> `
  --boundary-guide BOUNDARY_GUIDE.md `
  --output <p1-dataset-directory>
```

Run the plain CUDA challenger from a separate environment:

```powershell
python p1_train.py `
  --dataset <p1-dataset-directory> `
  --output <p1-run-directory> `
  --epochs 1
```

The initial run is FP32 and reports the contiguous Juan 27 challenge dev block
separately from the random Juan 52 pilot holdout. The holdout is not a sealed test.
Training selects the saved checkpoint only by challenge-dev exact F1. Decoding
forces punctuation, symbols, and numeric characters to `O`, and merges an adjacent
`B-PER` into the active person span when there is no hard boundary. These structural
constraints are derived from the frozen training/dev labels and do not use person
surfaces or the random pilot holdout.

Audit a trained challenger against the frozen references and current rule output:

```powershell
python p1_audit.py `
  --dataset <p1-dataset-directory> `
  --predictions <p1-run-directory> `
  --rules ..\..\..\web\public\text\persons-v2\agent1 `
  --blind-dir <blind-directory> `
  --output <audit-report.json>
```

The audit uses one-to-one exact geometry accounting and reports rule-omission
recoveries, rule true-positive regressions, boundary replacements, pure additions,
and pure misses separately for challenge dev and random pilot holdout.

Build a v1-free P2 expansion with one blind anchor and four assisted juans. This
command creates ML seed packs for the versioned Copilot teacher; the seed packs are
not shown directly to the annotator:

```powershell
python p2_assisted.py `
  --model <selected-model-directory> `
  --output <round-directory>

python server.py `
  --blind-dir <round-directory>\tasks `
  --assisted-dir <round-directory>\copilot-v1 `
  --recall-dir <round-directory>\unused-recall `
  --role-audit-dir <round-directory>\unused-role-audit `
  --state-dir <round-directory>\state
```

The Copilot teacher reviews the ML seeds using the bounded evidence stream below and
writes immutable packs under `copilot-v1`. Assisted tasks remain inaccessible until
the blind anchor is completed and locked. Neither stage reads v1, rules, identity
data, or blind-anchor labels; note and translation prose used by the teacher remains
transient.

After every task is locked, freeze the human corrections and measure candidate
accuracy:

```powershell
python p2_round.py `
  --tasks <round-directory>\tasks `
  --assisted <round-directory>\copilot-v1 `
  --state <round-directory>\state `
  --output <frozen-round-directory>
```

Only assisted labels enter the next training round. The blind anchor remains
evaluation-only. Promote a new candidate model only when exact F1 improves on the
blind anchor and no declared challenge stratum regresses; never regenerate or
rescore a completed assisted batch in place.

When generating Copilot-teacher candidates, stream bounded evidence for one target
jie without persisting translation prose:

```powershell
python teacher_evidence.py --juan 12 --jie-index 5
```

The teacher sees full-juan main-text context, but other jies are explicitly
non-authorizing. Hu Sansheng notes are limited to the target jie. Translation prose
must match the audited source hash and an approved unique pair-to-jie mapping.

Train a corrected assisted round while keeping the blind anchor evaluation-only:

```powershell
python p1_train.py `
  --train-file <frozen-round>\train_assisted.jsonl `
  --dev-file <p1-dataset>\dev.jsonl `
  --evaluation-file <frozen-round>\dev_blind_anchor.jsonl `
  --evaluation-name locked_blind_anchor `
  --context-mode target_only `
  --epochs 5 `
  --output <run-directory>
```

Run the same command with `bounded_neighbor` and `max_budget` for the soft-context
A/B. Select the mode and checkpoint only by challenge-dev exact F1; blind-anchor
results are reported afterward and must not change that selection. Context characters
receive `-100`, own no output, and whole-juan-disjoint splits require no additional
guard-band exclusions.

Before streaming translation evidence to the teacher, regenerate the identity-free
scope sidecar outside Agent 1:

```powershell
python export_translation_scope.py `
  --mapping ..\twostage\translation\mapping.json `
  --output ..\twostage\translation\agent1_translation_scope.json
```

`teacher_evidence.py` reads only this sidecar, never the identity-bearing mapping.

Freeze the P3 candidate-blind tasks only after selecting the Round 1 checkpoint:

```powershell
python p3_sealed.py `
  --output <new-p3-task-directory> `
  --model <selected-model-directory> `
  --selected-model <round1-selected.json>
```

The command verifies the model hash, excludes every consumed or aborted sealed
juan, writes three uniform-random tasks and two private-seed draws from the
predeclared top-five challenge cohorts, and records selection metadata only in the
private manifest. Do not run model or rule inference until all five blind tasks are
complete and locked.

If the user explicitly abandons formal P3 in favor of lower-effort Copilot review,
package validated teacher outputs as a **diagnostic-only** workflow:

```powershell
python p3_diagnostic.py `
  --sealed-tasks <sealed-p3>\tasks `
  --copilot-packs <copilot-output-directory> `
  --output <new-diagnostic-directory>
```

The diagnostic UI pre-accepts high/medium-confidence auto-tags and leaves only
low-confidence candidates unresolved. These labels and any resulting model
comparison are assisted diagnostics, not an independent sealed-test metric.
