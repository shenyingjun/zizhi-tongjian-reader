"""Classical-Chinese POS·Giv oracle for the anaphora resolver, with an on-disk cache.

The single-char 省称 resolver (resolve_anaphora_pos in build_persons.py) admits a
bare given char only when the Classical-Chinese UPOS tagger labels it
``PROPN | NameType=Giv`` at that offset. Running the transformer model on every
build over 294 卷 is slow, so results are cached per 卷 in
``web/public/text/persons/pos_giv/juan_NNN.json``.

The cache is keyed by a SHA-256 of the 卷's paragraph main texts, so it is
transparently regenerated when the source text changes. ``torch`` /
``transformers`` are imported lazily — only on a cache miss — so a normal build
against an up-to-date cache never touches the heavy dependency and stays fast.

Cache format:
    {"sha": "<hex>", "giv": {"<para_id>": [offset, ...], ...}}
where each offset is a code-point index into that paragraph's ``main`` string
whose POS label contains both ``PROPN`` and ``Giv``.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path

MODEL = "KoichiYasuoka/roberta-classical-chinese-base-upos"

_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is None:
        # Force HF offline: the model is cached locally, and letting transformers
        # phone home mid-build risks a ReadTimeout that kills a long run. setdefault
        # so an operator can still opt back online by exporting the vars themselves.
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        # Lazy: torch + transformers load only when a 卷 must be (re)tagged.
        from transformers import pipeline
        _pipe = pipeline("token-classification", model=MODEL,
                         aggregation_strategy="none")
    return _pipe


def _split_sents(mt: str):
    """Yield (offset, sentence) pairs split on 。！？ (matches the validated demo)."""
    sents = []
    st = 0
    for i, ch in enumerate(mt):
        if ch in "。！？":
            sents.append((st, mt[st:i + 1]))
            st = i + 1
    if st < len(mt):
        sents.append((st, mt[st:]))
    return sents


def _giv_from_result(result) -> set[int]:
    """Extract local PROPN|Giv offsets from one pipeline result list."""
    out = set()
    for r in result:
        lab = r.get("entity", "") or ""
        if "PROPN" in lab and "Giv" in lab:
            for i in range(r["start"], r["end"]):
                out.add(i)
    return out


def _sha_of(paras) -> str:
    h = hashlib.sha256()
    for p in paras:
        h.update(str(p.get("id", "")).encode("utf-8"))
        h.update(b"\x00")
        h.update((p.get("main", "") or "").encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


def giv_for_juan(juan_no: int, paras, cache_dir: Path) -> dict[int, set[int]]:
    """Return {para_id: set(giv_offsets)} for one 卷, using / refreshing the cache.

    On a cache hit (matching text SHA) no model is loaded. On a miss the model
    tags every paragraph and the cache file is (re)written. All sentences in the
    卷 are tagged in one batched pipeline call for throughput.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cf = cache_dir / f"juan_{juan_no:03d}.json"
    sha = _sha_of(paras)
    if cf.exists():
        try:
            blob = json.loads(cf.read_text(encoding="utf-8"))
            if blob.get("sha") == sha:
                return {int(k): set(v) for k, v in blob.get("giv", {}).items()}
        except (ValueError, OSError):
            pass  # corrupt / unreadable cache → regenerate

    # Flatten every non-empty sentence across the 卷 into one batch. Each item
    # carries (para_id, sentence_offset) so results map back to paragraph offsets.
    keys, texts = [], []
    for p in paras:
        mt = p.get("main", "") or ""
        if not mt.strip():
            continue
        for so, stext in _split_sents(mt):
            if stext.strip():
                keys.append((p["id"], so))
                texts.append(stext)

    giv: dict[int, set[int]] = {}
    if texts:
        pipe = _get_pipe()
        results = pipe(texts, batch_size=16)
        if isinstance(results, dict) or (results and isinstance(results[0], dict)):
            results = [results]  # single-input safety (never expected here)
        for (pid, so), res in zip(keys, results):
            for local in _giv_from_result(res):
                giv.setdefault(pid, set()).add(so + local)

    cf.write_text(
        json.dumps({"sha": sha,
                    "giv": {str(k): sorted(v) for k, v in giv.items()}},
                   ensure_ascii=False),
        encoding="utf-8")
    return giv
