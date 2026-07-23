"""Classical-Chinese POS oracle for the person pipeline, with an on-disk cache.

The single-char 省称 resolver consumes the derived ``PROPN|NameType=Giv``
offsets. Cache v3 also retains the useful structured output for every model
token so later rules can distinguish names from verbs and other parts of speech
without rerunning the model.

Cache v3 stores the source SHA-256, model identity/revision, sentence bounds,
and token records containing source text, absolute paragraph-local offsets,
UPOS, the model's complete entity tag, and its confidence score. The current
model emits a mix of unprefixed tags such as ``PROPN|NameType=Giv`` and BIO
tags for grouped tokens such as ``B-NUM``/``I-NUM``. The optional ``bio`` field
is stored only where the model's tag actually contains that prefix.
The compact derived ``giv`` and ``giv_spans`` maps remain in the file for
inspection, while readers derive their Giv evidence from v3 token records.

Legacy v1 (no ``version``) and v2 caches remain readable. A normal read never
runs the model: missing, stale, or invalid caches raise ``CacheMissError`` with
the explicit refresh command. This prevents a benchmark loop from silently
refreshing all 294 volumes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
import uuid


MODEL = "KoichiYasuoka/roberta-classical-chinese-base-upos"
CACHE_VERSION = 3

_pipe = None


class CacheMissError(RuntimeError):
    """Raised when an explicit POS cache refresh is required."""


@dataclass(frozen=True)
class PosToken:
    """One model token with paragraph-local source offsets."""

    text: str
    start: int
    end: int
    pos: str
    tag: str
    bio: str | None = None
    score: float | None = None

    @property
    def is_giv(self) -> bool:
        return _is_giv(self.tag)

    def shifted(self, delta: int) -> "PosToken":
        return replace(self, start=self.start + delta, end=self.end + delta)


class GivOffsets(set):
    """Giv offsets plus optional complete v3 POS-token evidence."""

    def __init__(self, offsets=(), spans=(), tokens=()):
        super().__init__(int(offset) for offset in offsets)
        self.spans = tuple((int(start), int(end)) for start, end in spans)
        self.tokens = tuple(tokens)

    def token_at(self, offset: int) -> PosToken | None:
        return next(
            (token for token in self.tokens if token.start <= offset < token.end),
            None,
        )


def _contiguous_spans(offsets) -> list[tuple[int, int]]:
    """Best-effort spans for v1 caches that stored offsets only."""
    points = sorted(set(offsets))
    if not points:
        return []
    spans = []
    start = previous = points[0]
    for point in points[1:]:
        if point != previous + 1:
            spans.append((start, previous + 1))
            start = point
        previous = point
    spans.append((start, previous + 1))
    return spans


def _get_pipe():
    global _pipe
    if _pipe is None:
        # The model is cached locally. Avoid a network timeout in a long refresh,
        # while still allowing an operator to opt online by setting either value.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import pipeline

        _pipe = pipeline(
            "token-classification",
            model=MODEL,
            aggregation_strategy="none",
        )
    return _pipe


def _split_sents(mt: str):
    """Yield paragraph-local ``(start, text)`` pairs split on 。！？."""
    sents = []
    start = 0
    for index, char in enumerate(mt):
        if char in "。！？":
            sents.append((start, mt[start:index + 1]))
            start = index + 1
    if start < len(mt):
        sents.append((start, mt[start:]))
    return sents


def _tag_parts(tag: str) -> tuple[str | None, str]:
    """Return an actually emitted BIO prefix, if any, and the base UPOS."""
    bio = None
    complete = tag
    if len(tag) > 2 and tag[0] in "BIEOS" and tag[1] == "-":
        bio, complete = tag[0], tag[2:]
    return bio, complete.split("|", 1)[0]


def _is_giv(tag: str) -> bool:
    _, complete = _tag_parts(tag)
    if complete != "PROPN":
        return False
    parts = tag.split("-", 1)[-1].split("|")[1:]
    return "NameType=Giv" in parts


def _giv_from_result(result) -> tuple[set[int], list[tuple[int, int]]]:
    """Extract offsets and model-bounded Giv spans from token-like records."""
    offsets: set[int] = set()
    spans: list[tuple[int, int]] = []
    current = None
    for record in sorted(result, key=lambda item: (item["start"], item["end"])):
        tag = str(record.get("tag", record.get("entity", "")) or "")
        if not _is_giv(tag):
            if current is not None:
                spans.append(current)
                current = None
            continue
        start, end = int(record["start"]), int(record["end"])
        offsets.update(range(start, end))
        bio = record.get("bio")
        if bio is None:
            bio, _ = _tag_parts(tag)
        if bio == "I" and current is not None and current[1] == start:
            current = (current[0], end)
        else:
            if current is not None:
                spans.append(current)
            current = (start, end)
    if current is not None:
        spans.append(current)
    return offsets, spans


def _sha_of(paras) -> str:
    """Hash paragraph IDs and main texts in source order."""
    digest = hashlib.sha256()
    for para in paras:
        digest.update(str(para.get("id", "")).encode("utf-8"))
        digest.update(b"\x00")
        digest.update((para.get("main", "") or "").encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()


def _evidence_from_legacy(blob) -> dict[int, GivOffsets]:
    span_map = blob.get("giv_spans")
    evidence = {}
    for key, offsets in blob.get("giv", {}).items():
        spans = (
            span_map.get(str(key), ())
            if span_map is not None
            else _contiguous_spans(offsets)
        )
        evidence[int(key)] = GivOffsets(offsets, spans)
    return evidence


def _evidence_from_v3(blob) -> dict[int, GivOffsets]:
    """Derive compatibility evidence from v3 token records, not summary maps."""
    giv: dict[int, set[int]] = {}
    spans: dict[int, list[tuple[int, int]]] = {}
    paragraphs = blob.get("paragraphs")
    if not isinstance(paragraphs, dict):
        raise ValueError("v3 cache has no paragraphs mapping")
    tokens_by_pid: dict[int, list[PosToken]] = {}
    for para_key, paragraph in paragraphs.items():
        pid = int(para_key)
        sentences = paragraph.get("sentences", [])
        if not isinstance(sentences, list):
            raise ValueError(f"v3 paragraph {para_key} has invalid sentences")
        for sentence in sentences:
            tokens = sentence.get("tokens", [])
            if not isinstance(tokens, list):
                raise ValueError(f"v3 paragraph {para_key} has invalid tokens")
            local_offsets, local_spans = _giv_from_result(tokens)
            for token in tokens:
                tokens_by_pid.setdefault(pid, []).append(PosToken(
                    text=str(token["text"]),
                    start=int(token["start"]),
                    end=int(token["end"]),
                    pos=str(token["pos"]),
                    tag=str(token["tag"]),
                    bio=token.get("bio"),
                    score=(
                        float(token["score"])
                        if token.get("score") is not None
                        else None
                    ),
                ))
            if local_offsets:
                giv.setdefault(pid, set()).update(local_offsets)
            if local_spans:
                spans.setdefault(pid, []).extend(local_spans)
    return {
        pid: GivOffsets(
            giv.get(pid, ()),
            spans.get(pid, ()),
            tokens,
        )
        for pid, tokens in tokens_by_pid.items()
    }


def _evidence_from_blob(blob) -> dict[int, GivOffsets]:
    version = int(blob.get("version", 1))
    if version == CACHE_VERSION:
        return _evidence_from_v3(blob)
    if version in (1, 2):
        return _evidence_from_legacy(blob)
    raise ValueError(f"unsupported POS cache version {version}")


def _model_metadata(pipe) -> dict:
    config = pipe.model.config
    metadata = {
        "id": getattr(pipe.model, "name_or_path", None) or MODEL,
        "revision": getattr(config, "_commit_hash", None),
        "task": "token-classification",
        "aggregation_strategy": "none",
        "tokenizer": getattr(pipe.tokenizer, "name_or_path", None),
        "model_type": getattr(config, "model_type", None),
        "config_transformers_version": getattr(
            config, "transformers_version", None
        ),
    }
    try:
        import transformers

        metadata["transformers_version"] = transformers.__version__
    except (ImportError, AttributeError):
        pass
    return {key: value for key, value in metadata.items() if value is not None}


def _token_record(raw, sentence: str, sentence_start: int) -> dict:
    try:
        local_start = int(raw["start"])
        local_end = int(raw["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"model token has invalid offsets: {raw!r}") from exc
    if not 0 <= local_start < local_end <= len(sentence):
        raise ValueError(
            f"model token offsets [{local_start}, {local_end}) outside "
            f"sentence length {len(sentence)}"
        )

    tag = str(raw.get("entity", "") or "")
    bio, pos = _tag_parts(tag)
    text = sentence[local_start:local_end]
    record = {
        "text": text,
        "start": sentence_start + local_start,
        "end": sentence_start + local_end,
        "pos": pos,
        "tag": tag,
    }
    if bio is not None:
        record["bio"] = bio
    if "score" in raw and raw["score"] is not None:
        score = float(raw["score"])
        if not math.isfinite(score):
            raise ValueError(f"model token has non-finite score: {raw!r}")
        record["score"] = score
    model_text = raw.get("word")
    if model_text is not None and str(model_text) != text:
        record["model_text"] = str(model_text)
    return record


def _build_v3_payload(paras, sha: str) -> tuple[dict, dict[int, GivOffsets]]:
    keys: list[tuple[int, int, int, str]] = []
    texts: list[str] = []
    for para in paras:
        main = para.get("main", "") or ""
        if not main.strip():
            continue
        for sentence_start, sentence in _split_sents(main):
            if sentence.strip():
                keys.append((int(para["id"]), sentence_start,
                             sentence_start + len(sentence), sentence))
                texts.append(sentence)

    paragraph_records: dict[str, dict] = {}
    giv: dict[int, set[int]] = {}
    giv_spans: dict[int, list[tuple[int, int]]] = {}
    metadata = {
        "id": MODEL,
        "task": "token-classification",
        "aggregation_strategy": "none",
    }
    if texts:
        pipe = _get_pipe()
        results = pipe(texts, batch_size=16)
        if isinstance(results, dict) or (
            results and isinstance(results[0], dict)
        ):
            results = [results]
        if not isinstance(results, list) or len(results) != len(texts):
            actual = len(results) if isinstance(results, list) else type(results)
            raise ValueError(
                f"model returned {actual!r} results for {len(texts)} sentences"
            )
        metadata = _model_metadata(pipe)

        for (pid, sentence_start, sentence_end, sentence), raw_tokens in zip(
            keys, results
        ):
            if not isinstance(raw_tokens, list):
                raise ValueError(
                    f"model result for paragraph {pid} is not a token list"
                )
            tokens = [
                _token_record(raw, sentence, sentence_start)
                for raw in raw_tokens
            ]
            sentence_record = {
                "start": sentence_start,
                "end": sentence_end,
                "tokens": tokens,
            }
            paragraph_records.setdefault(
                str(pid), {"sentences": []}
            )["sentences"].append(sentence_record)
            local_offsets, local_spans = _giv_from_result(tokens)
            if local_offsets:
                giv.setdefault(pid, set()).update(local_offsets)
            if local_spans:
                giv_spans.setdefault(pid, []).extend(local_spans)

    payload = {
        "version": CACHE_VERSION,
        "sha": sha,
        "model": metadata,
        "paragraphs": paragraph_records,
        "giv": {str(key): sorted(value) for key, value in giv.items()},
        "giv_spans": {
            str(key): spans for key, spans in giv_spans.items()
        },
    }
    evidence = _evidence_from_v3(payload)
    return payload, evidence


def _write_atomic(path: Path, payload: dict) -> None:
    """Write complete JSON beside the target, then atomically replace it."""
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def cache_version(
    juan_no: int, paras, cache_dir: Path
) -> int | None:
    """Return the matching cache version, or ``None`` if absent/stale/invalid."""
    path = cache_dir / f"juan_{juan_no:03d}.json"
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
        if blob.get("sha") != _sha_of(paras):
            return None
        version = int(blob.get("version", 1))
        _evidence_from_blob(blob)
        return version
    except (OSError, ValueError, TypeError, KeyError):
        return None


def giv_for_juan(
    juan_no: int, paras, cache_dir: Path, *, refresh: bool = False
) -> dict[int, GivOffsets]:
    """Return backward-compatible Giv evidence for one volume.

    Matching v1/v2/v3 caches are read without loading the model. Set
    ``refresh=True`` only from an explicit refresh command to generate v3.
    Generation is validated before an atomic per-volume replacement.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"juan_{juan_no:03d}.json"
    sha = _sha_of(paras)
    if not refresh:
        try:
            blob = json.loads(cache_file.read_text(encoding="utf-8"))
            if blob.get("sha") == sha:
                return _evidence_from_blob(blob)
        except (OSError, ValueError, TypeError, KeyError):
            pass
        raise CacheMissError(
            f"POS cache for juan {juan_no:03d} is missing, stale, or invalid; "
            "run refresh_pos_giv.py explicitly"
        )

    payload, evidence = _build_v3_payload(paras, sha)
    # Verify the serialized source of truth reproduces the compatibility view.
    derived = _evidence_from_v3(payload)
    if derived != evidence or any(
        derived[pid].spans != item.spans for pid, item in evidence.items()
    ):
        raise ValueError("v3 Giv compatibility derivation did not round-trip")
    _write_atomic(cache_file, payload)
    return evidence
