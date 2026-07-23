# Repository instructions

## Two-stage person-underlining pipeline

These instructions apply when changing `data-pipeline/persons/twostage/`.
Rule semantics are defined in `data-pipeline/persons/twostage/SPEC.md`.

### Scope invariants

- Agent 1 must not read a person or identity KB.
- Every person-surface-derived anchor, recurrence, morphology aggregation, conflict,
  and veto must be confined to the current numbered jie.
- Juan is a file, chronology, and batch boundary, not an Agent 1 person-resolution
  scope.
- Translation evidence may only act within its approved paragraph/jie scope.
- Hard vetoes cannot be overridden by cumulative scores.
- If implementation and `SPEC.md` disagree, fix the implementation before evaluating
  coverage.

### Targeted-first workflow

Do not start a 294-juan rebuild while rules are still changing.

1. Add focused tests for the behavior, especially scope boundaries, BIO continuation,
   and hard vetoes.
2. Rebuild only affected juans, including positive examples, known false positives,
   regressions, and unchanged controls.
3. Compare raw exact geometry, not only total span counts.
4. Inspect every targeted addition and removal.
5. Run the full assisted build only after targeted output is stable. A full default
   build is optional diagnostic output, never the reported final result.

If any targeted geometry remains unexplained, do not proceed to a full build.

### Full-build gate

Before a full rebuild:

- focused tests pass;
- evidence names and rule behavior are stable;
- targeted default and assisted deltas have been reviewed;
- no `document_*` or other cross-jie person-surface evidence remains;
- the comparison baseline and expected delta are explicit;
- output uses a new directory and does not overwrite a baseline.

The canonical full build and benchmark must load approved translation evidence.
The formal benchmark and every reported final coverage/gap metric must additionally
use the audited v1 reference through `benchmark_reference.py` and
`benchmark-reference-exclusions.jsonl`. Therefore **audited Translation-assisted** is
the only canonical result. Raw production-v1 counts and default output may be retained
for compatibility accounting, attribution, or ablation, but must be labeled diagnostic
and must never be reported as the final metric. Reference exclusions are benchmark
corrections, not rule recoveries or closed gaps. If rules change during a run, stop or
discard that run. Do not repeatedly launch full builds as a substitute for targeted
diagnosis.

### Delta accounting

Never describe output-count difference as “new spans.” Report separately:

- raw additions;
- removals;
- net growth (`additions - removals`);
- v1 recoveries;
- v1 regressions;
- geometry replacements.

A net number can hide many additions/removals or misrepresent a boundary replacement
as recall gain.

### Precision audit

After a full rule change:

1. Review every new non-v1 geometry.
2. Review every v1 regression and classify it as a true miss, removal of a bad partial
   span, fuller-span replacement, or deliberate loss caused by scope tightening.
3. Group candidates by policy, score, family signature, conflict, and veto.
4. Fix systematic errors structurally; do not add surface blacklists.
5. Return to targeted validation after any audit-driven rule change.

Save detailed manual audit output as a session artifact. Write only final reproducible
results to `BENCHMARK.md`.

### Evidence-design checks

- Correlated POS, BIO, and NER outputs are one model family, not independent votes.
- Same-jie recurrence is one family and must not be duplicated through another view.
- Punctuation or a hard boundary is not person syntax.
- Surname shape requires current-occurrence morphology support.
- A BIO `I` veto must represent a real name continuation; punctuation mis-tagged as
  `B` must not propagate a veto.
- Same-jie trusted exact-surface propagation differs from identity resolution.
- Given-name anaphora requires an earlier anchor in the same jie.
- Never authorize Agent 1 from another jie, a juan-level roster, or a later identity.

### Required run order

```text
spec/scope review
  -> focused tests
  -> targeted default rebuild
  -> targeted assisted rebuild
  -> inspect every targeted delta
  -> full assisted rebuild
  -> raw geometry comparison
  -> full candidate/non-v1/regression audit
  -> canonical audited assisted benchmark
  -> documentation update
```

If the full audit finds a structural error, return to targeted testing. Do not retain
statistics from a build whose rules have since changed.

### Incident lesson

The jie-only audit started full default and assisted rebuilds before the partial-name
continuation guard was stable. Later targeted review of `沙漠汗/漠汗`,
`斛拔弥俄突/拔`, quotation-boundary BIO, and standalone-handle controls forced full
runs to be stopped or repeated. Stabilize such guards on targeted juans first, then
perform one final pair of full rebuilds.
