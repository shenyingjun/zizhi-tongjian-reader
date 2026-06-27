"""Tier-2 NER candidate generator — a Classical-Chinese transformer pass.

The jieba `nr` harvester (ner_extract.py) plus the Han-surname gate (seed.py)
structurally cannot reach person names that (a) jieba never tokenizes as one unit
(e.g. the 5-char Türkic compound 阿史那思摩 → n=0) or (b) begin with a non-Han
surname the whitelist will never contain (突利, 柴绍-class foreign / rare names).

This script runs KoichiYasuoka/roberta-classical-chinese-base-upos — a char-level
UPOS tagger whose PROPN labels carry a NameType feature that separates PERSON
(NameType=Prs/Sur/Giv) from PLACE / STATE (Case=Loc / NameType=Geo|Nat). We take
maximal runs of person-tagged characters as person surfaces. Because the model is
character-level it keeps long foreign compounds intact, and because it is
context-driven it does not depend on a surname list at all.

Output: ner_model_candidates.json = {surface: {"j": [juan…], "n": total}}.
seed.py reads this as a SEPARATE, surname-gate-exempt feed (the whole point is to
admit names the surname gate rejects); the usual frequency / per-卷 collision /
WIKI_NONPERSON guards still apply downstream, and the tier stays 自动识别 (high).

Run (needs the .venv-ner interpreter with torch+transformers):
  ..\.venv-ner\Scripts\python.exe ner_model.py
This is a slow one-time offline pass (~3.1M chars on CPU). Re-run only when the
corpus text changes — NOT for seed/surname tuning.
"""
from __future__ import annotations
import json, glob, collections, re, time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

import seed as seed_mod  # WIKI_NONPERSON

MODEL = "KoichiYasuoka/roberta-classical-chinese-base-upos"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TEXT = REPO / "web" / "public" / "text"
OUT = HERE / "ner_model_candidates.json"

WINDOW = 460          # chars per inference window (model max len 512, leave margin)
BATCH = 16
MIN_LEN, MAX_LEN = 2, 6   # person surface length bounds
# Split points: never cut a window mid-name. Punctuation is a safe boundary.
SPLIT_RE = re.compile(r"[。！？；：，、（）「」『』《》〈〉\s]")


def juans():
    for f in sorted(glob.glob(str(TEXT / "juan_*.json"))):
        p = Path(f)
        if not p.name.startswith("juan_"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if "paragraphs" not in d:
            continue
        yield d["juan_no"], d


def windows_for(text: str):
    """Yield (offset, chunk) windows <= WINDOW chars, cutting only on punctuation
    so a name is never split across two windows."""
    i, n = 0, len(text)
    while i < n:
        end = min(i + WINDOW, n)
        if end < n:
            # back up to the last safe split inside the window
            m = None
            for mt in SPLIT_RE.finditer(text, i, end):
                m = mt
            if m and m.end() > i + 40:
                end = m.end()
        yield i, text[i:end]
        i = end


def is_person(label: str) -> bool:
    return ("NameType=Prs" in label or "NameType=Sur" in label
            or "NameType=Giv" in label)


def main():
    print(f"loading {MODEL} …")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForTokenClassification.from_pretrained(MODEL)
    model.eval()
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    id2label = model.config.id2label

    cand: dict[str, dict[int, int]] = collections.defaultdict(lambda: collections.defaultdict(int))

    # Gather all windows first so we can batch across juan boundaries.
    jobs: list[tuple[int, str]] = []   # (juan_no, chunk)
    for jn, d in juans():
        text = "".join(p.get("main", "") for p in d["paragraphs"])
        for _, chunk in windows_for(text):
            if chunk.strip():
                jobs.append((jn, chunk))
    print(f"{len(jobs)} windows over 294 juan")

    t0 = time.time()
    for b in range(0, len(jobs), BATCH):
        batch = jobs[b:b + BATCH]
        texts = [c for _, c in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=512, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")
        with torch.no_grad():
            logits = model(**enc).logits
        preds = logits.argmax(-1)
        for r, (jn, chunk) in enumerate(batch):
            run = []  # list of (char) for the current person span
            for (a, bb), pid in zip(offs[r].tolist(), preds[r].tolist()):
                if a == bb:
                    continue  # special / pad token
                lab = id2label[pid]
                if is_person(lab):
                    run.append(chunk[a:bb])
                else:
                    if run:
                        surf = "".join(run)
                        if MIN_LEN <= len(surf) <= MAX_LEN:
                            cand[surf][jn] += 1
                        run = []
            if run:
                surf = "".join(run)
                if MIN_LEN <= len(surf) <= MAX_LEN:
                    cand[surf][jn] += 1
        if (b // BATCH) % 25 == 0:
            done = b + len(batch)
            rate = done / max(1e-6, time.time() - t0)
            eta = (len(jobs) - done) / max(1e-6, rate)
            print(f"  {done}/{len(jobs)} windows  {rate:.1f}/s  eta {eta/60:.1f}m  surfaces={len(cand)}")

    # Drop known non-persons; keep raw (no surname gate — that's the point).
    out = {s: {"j": sorted(jc.keys()), "n": sum(jc.values())}
           for s, jc in cand.items() if s not in seed_mod.WIKI_NONPERSON}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT.name}: {len(out)} person surfaces in {(time.time()-t0)/60:.1f}m")
    # quick sanity on the targets
    for nm in ("突利", "阿史那思摩", "柴绍", "李景让", "阿史那思"):
        v = out.get(nm)
        print(f"  {nm}: {v}")


if __name__ == "__main__":
    main()
