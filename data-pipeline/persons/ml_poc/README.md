# Agent-1 ML POC

This folder contains the pilot-first proof of concept described in `SPEC.md`
and its Chinese counterpart `SPEC.zh.md`.
It is isolated from the production two-stage pipeline until the P0/P1 gates are
met.

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
