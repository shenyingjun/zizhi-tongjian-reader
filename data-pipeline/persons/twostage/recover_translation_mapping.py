"""Recover prose-free Translation evidence mappings one juan at a time.

The source page is read in memory and is never persisted. Person entities detected
in each translated segment must either occur in its paired source-original segment
or align to an abbreviated mention in the corresponding source sentence. Each
resulting canonical occurrence is then restricted to a uniquely aligned numbered
jie before it can become eligible evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.parse
import time

from bs4 import BeautifulSoup
from opencc import OpenCC
import requests


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TEXT = REPO / "web" / "public" / "text"
SOURCE_URL = (
    "http://www.ziyexing.com/files-5/zizhitongjian/"
    "zizhitongjian_{juan:03d}.htm"
)
MODEL = "uer/roberta-base-finetuned-cluener2020-chinese"
MIN_NER_SCORE = 0.75
MIN_SENTENCE_ALIGNMENT_SCORE = 0.17
MIN_PARAGRAPH_ALIGNMENT_MARGIN = 0.05
ORIGINAL_MARKER = "\u3010\u539f\u6587\u3011"
TRANSLATION_MARKER = "\u3010\u8bd1\u6587\u3011"
MISSING_TRANSLATION_NOTICE = "\u672c\u5377\u8bd1\u6587\u7f3a\u5931"
_CC = OpenCC("t2s")
_CJK = re.compile(r"[\u3400-\u9fff]+")
_SENTENCE = re.compile(r"[^\u3002\uff01\uff1f\uff1b]+[\u3002\uff01\uff1f\uff1b]?")
_HU_SANSHENG_NOTE = re.compile(
    r"\u3014\u3016\u80e1\u4e09\u7701\u6ce8\u3017.*?\u3015",
    re.DOTALL,
)
_SESSION = requests.Session()
_COREFERENCE_HANDLE_VETO = set(
    "\u4e4b\u5176\u4e43\u800c\u4ee5\u4e8e\u4e0e\u4e3a\u6240"
    "\u8005\u4e5f\u77e3\u7109\u4e4e\u516e"
)
_PERSON_TITLE_CONTINUATIONS = (
    "\u5b50", "\u516c", "\u4faf", "\u738b", "\u541b", "\u540e", "\u5983",
    "\u59ec", "\u592b\u4eba", "\u5148\u751f",
)


@dataclass(frozen=True)
class SourcePair:
    index: int
    original: str
    translation: str


@dataclass(frozen=True)
class PersonEntity:
    surface: str
    score: float


@dataclass(frozen=True)
class Jie:
    index: int
    number: int | None
    paragraphs: tuple[dict, ...]

    @property
    def text(self) -> str:
        return "".join(paragraph.get("main", "") or "" for paragraph in self.paragraphs)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize(text: str) -> str:
    return "".join(_CJK.findall(_CC.convert(text)))


def _jie_number(text: str) -> int | None:
    if not text:
        return None
    codepoint = ord(text[0])
    if 0x2460 <= codepoint <= 0x2473:
        return codepoint - 0x2460 + 1
    if 0x3251 <= codepoint <= 0x325F:
        return codepoint - 0x3251 + 21
    if 0x32B1 <= codepoint <= 0x32BF:
        return codepoint - 0x32B1 + 36
    return None


def _jies(paragraphs: list[dict]) -> tuple[Jie, ...]:
    grouped: list[tuple[int, int | None, list[dict]]] = []
    jie_index = 0
    number = None
    for paragraph in paragraphs:
        next_number = _jie_number(paragraph.get("main", "") or "")
        if next_number is not None:
            jie_index += 1
            number = next_number
            grouped.append((jie_index, number, [paragraph]))
        elif grouped:
            grouped[-1][2].append(paragraph)
        else:
            grouped.append((jie_index, number, [paragraph]))
    return tuple(
        Jie(index, jie_number, tuple(rows))
        for index, jie_number, rows in grouped
    )


def _text_between(start, end) -> str:
    chunks = []
    for element in start.next_elements:
        if element is end:
            break
        if (
            isinstance(element, str)
            and element.parent is not None
            and element.parent.name not in {"script", "style"}
        ):
            chunks.append(element)
    return re.sub(r"\s+", " ", "".join(chunks)).strip()


def parse_source(source_bytes: bytes) -> tuple[SourcePair, ...]:
    soup = BeautifulSoup(source_bytes.decode("gb18030"), "html.parser")
    markers = [
        node
        for node in soup.find_all(
            string=lambda value: value
            and value.strip() in {ORIGINAL_MARKER, TRANSLATION_MARKER}
        )
    ]
    pairs = []
    for marker_index, marker in enumerate(markers):
        if (
            marker.strip() != ORIGINAL_MARKER
            or marker_index + 1 >= len(markers)
            or markers[marker_index + 1].strip() != TRANSLATION_MARKER
        ):
            continue
        translation_marker = markers[marker_index + 1]
        next_marker = (
            markers[marker_index + 2]
            if marker_index + 2 < len(markers)
            else None
        )
        pairs.append(
            SourcePair(
                index=len(pairs),
                original=_text_between(marker, translation_marker),
                translation=_text_between(translation_marker, next_marker),
            )
        )
    if not pairs:
        raise ValueError("source page has no paired original/translation segments")
    return tuple(pairs)


def _translation_is_explicitly_missing(source_bytes: bytes) -> bool:
    text = BeautifulSoup(
        source_bytes.decode("gb18030"), "html.parser"
    ).get_text(" ", strip=True)
    return MISSING_TRANSLATION_NOTICE in text


def fetch_source(juan: int) -> tuple[str, bytes]:
    url = SOURCE_URL.format(juan=juan)
    return url, _fetch(url)


def _fetch(url: str) -> bytes:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/126 Safari/537.36 "
            "zizhi-tongjian-reader-translation-recovery/1"
        )
    }
    last_error = None
    for attempt in range(3):
        try:
            response = _SESSION.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            return response.content
        except requests.RequestException as error:
            last_error = error
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"failed to fetch translation source: {url}") from last_error


def _alternate_translation_url(source_url: str, source_bytes: bytes) -> str | None:
    soup = BeautifulSoup(source_bytes.decode("gb18030"), "html.parser")
    link = next(
        (
            anchor.get("href")
            for anchor in soup.find_all("a", href=True)
            if TRANSLATION_MARKER[-3:-1] in anchor.get_text()
        ),
        None,
    )
    return urllib.parse.urljoin(source_url, link) if link else None


def _whole_page_pair(
    source_bytes: bytes,
    translation_bytes: bytes,
) -> tuple[SourcePair, ...]:
    source = BeautifulSoup(
        source_bytes.decode("gb18030"), "html.parser"
    ).get_text(" ", strip=True)
    translation = BeautifulSoup(
        translation_bytes.decode("gb18030"), "html.parser"
    ).get_text(" ", strip=True)
    if not source or not translation:
        raise ValueError("separate source/translation page is empty")
    return (SourcePair(0, source, translation),)


def _ngrams(text: str, size: int) -> set[str]:
    if len(text) < size:
        return {text} if text else set()
    return {text[index:index + size] for index in range(len(text) - size + 1)}


def aligned_jies(original: str, jies: tuple[Jie, ...]) -> tuple[Jie, ...]:
    normalized_original = _normalize(original)
    aligned = []
    for jie in jies:
        if jie.index == 0:
            continue
        normalized_jie = _normalize(jie.text)
        size = 4 if min(len(normalized_original), len(normalized_jie)) >= 4 else 2
        source_grams = _ngrams(normalized_original, size)
        jie_grams = _ngrams(normalized_jie, size)
        shared = len(source_grams & jie_grams)
        if not shared:
            continue
        source_ratio = shared / max(1, len(source_grams))
        jie_ratio = shared / max(1, len(jie_grams))
        minimum_shared = min(3, len(jie_grams))
        if shared >= minimum_shared and max(source_ratio, jie_ratio) >= 0.05:
            aligned.append(jie)
    return tuple(aligned)


def _chunks(text: str, limit: int = 450):
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            split = max(text.rfind(char, start, end) for char in "\u3002\uff01\uff1f\uff1b")
            if split >= start + 50:
                end = split + 1
        yield text[start:end]
        start = end


class TranslationNer:
    def __init__(self):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import pipeline

        self._pipeline = pipeline(
            "token-classification",
            model=MODEL,
            aggregation_strategy="simple",
            device=-1,
        )

    def people(self, text: str) -> tuple[PersonEntity, ...]:
        best = {}
        for chunk in _chunks(text):
            records = []
            for entity in self._pipeline(chunk):
                if str(entity.get("entity_group", "")).lower() not in {
                    "name",
                    "person",
                    "per",
                }:
                    continue
                score = float(entity["score"])
                surface = "".join(_CJK.findall(str(entity.get("word", ""))))
                if not 2 <= len(surface) <= 8 or score < MIN_NER_SCORE:
                    continue
                records.append(
                    (
                        int(entity["start"]),
                        int(entity["end"]),
                        surface,
                        score,
                    )
                )
            merged = []
            for start, end, surface, score in records:
                if merged and merged[-1][1] == start:
                    previous = merged.pop()
                    combined = "".join(_CJK.findall(chunk[previous[0]:end]))
                    if len(combined) <= 8:
                        merged.append(
                            (previous[0], end, combined, min(previous[3], score))
                        )
                        continue
                    merged.append(previous)
                merged.append((start, end, surface, score))
            for _, _, surface, score in merged:
                best[surface] = max(score, best.get(surface, 0.0))
        return tuple(
            PersonEntity(surface, score)
            for surface, score in sorted(best.items())
        )


def _occurrences(text: str, normalized_surface: str):
    normalized_text = _CC.convert(text)
    if len(normalized_text) != len(text):
        return
    start = normalized_text.find(normalized_surface)
    while start >= 0:
        yield start, start + len(normalized_surface)
        start = normalized_text.find(normalized_surface, start + 1)


def _coreference_handles(normalized_identity: str) -> tuple[str, ...]:
    if len(normalized_identity) < 2 or normalized_identity.endswith("\u66f0"):
        return ()
    lengths = (2, 1) if len(normalized_identity) >= 3 else (1,)
    return tuple(
        normalized_identity[-length:]
        for length in lengths
        if (
            normalized_identity[-length:] != normalized_identity
            and not set(normalized_identity[-length:]) <= _COREFERENCE_HANDLE_VETO
        )
    )


def _occurrence_matches_source_pair(
    text: str,
    start: int,
    end: int,
    normalized_source: str,
) -> bool:
    normalized_handle = _normalize(text[start:end])
    sentence_start = max(
        (text.rfind(terminator, 0, start) for terminator in "\u3002\uff01\uff1f\uff1b"),
        default=-1,
    ) + 1
    sentence_ends = [
        position
        for terminator in "\u3002\uff01\uff1f\uff1b"
        if (position := text.find(terminator, end)) >= 0
    ]
    sentence_end = min(sentence_ends, default=len(text))
    left = _normalize(text[max(sentence_start, start - 12):start])
    right = _normalize(text[end:min(sentence_end, end + 12)])
    for width in range(6, 1, -1):
        left_context = left[-width:]
        right_context = right[:width]
        if (
            len(left_context) + len(right_context) >= 4
            and left_context + normalized_handle + right_context
            in normalized_source
        ):
            return True
    return False


def _sentences(text: str, *, strip_hu_notes: bool = False) -> tuple[str, ...]:
    if strip_hu_notes:
        text = _HU_SANSHENG_NOTE.sub("", text)
    return tuple(
        normalized
        for match in _SENTENCE.finditer(text)
        if (normalized := _normalize(match.group()))
    )


def _aligned_source_sentences(
    pair: SourcePair,
    normalized_identity: str,
    normalized_handle: str,
) -> tuple[str, ...]:
    source_sentences = tuple(
        sentence
        for sentence in _sentences(pair.original, strip_hu_notes=True)
        if normalized_handle in sentence
    )
    translation_sentences = tuple(
        sentence
        for sentence in _sentences(pair.translation)
        if normalized_identity in sentence
    )
    if not source_sentences or not translation_sentences:
        return ()

    scores = [
        [
            difflib.SequenceMatcher(
                None,
                source_sentence,
                translation_sentence,
                autojunk=False,
            ).ratio()
            for translation_sentence in translation_sentences
        ]
        for source_sentence in source_sentences
    ]
    best = [
        [0.0] * (len(translation_sentences) + 1)
        for _ in range(len(source_sentences) + 1)
    ]
    decision = [
        [""] * (len(translation_sentences) + 1)
        for _ in range(len(source_sentences) + 1)
    ]
    for source_index in range(1, len(source_sentences) + 1):
        for translation_index in range(1, len(translation_sentences) + 1):
            options = [
                (best[source_index - 1][translation_index], "source"),
                (best[source_index][translation_index - 1], "translation"),
            ]
            score = scores[source_index - 1][translation_index - 1]
            if score >= MIN_SENTENCE_ALIGNMENT_SCORE:
                options.append(
                    (
                        best[source_index - 1][translation_index - 1] + score,
                        "match",
                    )
                )
            best[source_index][translation_index], decision[source_index][
                translation_index
            ] = max(options, key=lambda option: (option[0], option[1] == "match"))

    aligned_indexes = []
    source_index = len(source_sentences)
    translation_index = len(translation_sentences)
    while source_index and translation_index:
        step = decision[source_index][translation_index]
        if step == "match":
            aligned_indexes.append(source_index - 1)
            source_index -= 1
            translation_index -= 1
        elif step == "source":
            source_index -= 1
        else:
            translation_index -= 1
    return tuple(source_sentences[index] for index in reversed(aligned_indexes))


def _paragraph_alignment_score(source_sentence: str, paragraph_text: str) -> float:
    paragraph_sentences = _sentences(paragraph_text)
    if not paragraph_sentences:
        return 0.0
    return max(
        difflib.SequenceMatcher(
            None,
            source_sentence,
            paragraph_sentence,
            autojunk=False,
        ).ratio()
        for paragraph_sentence in paragraph_sentences
    )


def _paragraph_mapped_occurrences(
    pair: SourcePair,
    jies: tuple[Jie, ...],
    normalized_identity: str,
    normalized_surface: str,
) -> tuple[tuple[Jie, int, dict, int, int], ...]:
    source_sentences = _aligned_source_sentences(
        pair,
        normalized_identity,
        normalized_surface,
    )
    if not source_sentences:
        return ()

    occurrences = []
    for jie in jies:
        if jie.index == 0:
            continue
        for paragraph_index, paragraph in enumerate(jie.paragraphs):
            text = paragraph.get("main", "") or ""
            for start, end in _occurrences(text, normalized_surface):
                occurrences.append((jie, paragraph_index, paragraph, start, end))

    mapped = {}
    for source_sentence in source_sentences:
        by_paragraph = {}
        for occurrence in occurrences:
            jie, paragraph_index, paragraph, start, end = occurrence
            text = paragraph.get("main", "") or ""
            normalized_paragraph = _normalize(text)
            paragraph_contained = (
                normalized_paragraph
                and (
                    normalized_paragraph in source_sentence
                    or source_sentence in normalized_paragraph
                )
            )
            if not paragraph_contained and not _occurrence_matches_source_pair(
                text, start, end, source_sentence
            ):
                continue
            by_paragraph.setdefault(int(paragraph["id"]), []).append(occurrence)
        if not by_paragraph:
            continue

        ranked = sorted(
            (
                _paragraph_alignment_score(
                    source_sentence,
                    rows[0][2].get("main", "") or "",
                ),
                pid,
                rows,
            )
            for pid, rows in by_paragraph.items()
        )
        best_score, _, best_rows = ranked[-1]
        if (
            len(ranked) > 1
            and best_score - ranked[-2][0] < MIN_PARAGRAPH_ALIGNMENT_MARGIN
        ):
            continue
        for occurrence in best_rows:
            paragraph = occurrence[2]
            key = (
                int(paragraph["id"]),
                occurrence[3],
                occurrence[4],
            )
            mapped[key] = occurrence
    return tuple(mapped[key] for key in sorted(mapped))


def _map_translation_coreference(
    juan: int,
    pair: SourcePair,
    aligned: tuple[Jie, ...],
    entity: PersonEntity,
    source_url: str,
    handle_owners: dict[str, set[str]],
) -> list[dict]:
    normalized_entity = _normalize(entity.surface)
    normalized_pair_source = _normalize(pair.original)
    if normalized_entity in normalized_pair_source:
        return []

    handles = [
        handle
        for handle in _coreference_handles(normalized_entity)
        if (
            handle in normalized_pair_source
            and handle_owners.get(handle) == {normalized_entity}
        )
    ]
    if not handles:
        return []
    normalized_handle = handles[0]
    mapped_occurrences = _paragraph_mapped_occurrences(
        pair,
        aligned,
        normalized_entity,
        normalized_handle,
    )
    if not mapped_occurrences:
        return []

    anchors_by_jie = {}
    for jie in aligned:
        anchors = []
        for paragraph_index, paragraph in enumerate(jie.paragraphs):
            text = paragraph.get("main", "") or ""
            for start, end in _occurrences(text, normalized_entity):
                if any(
                    text.startswith(suffix, end)
                    for suffix in _PERSON_TITLE_CONTINUATIONS
                ):
                    continue
                anchors.append(
                    (
                        paragraph_index,
                        start,
                        text[start:end],
                    )
                )
        if anchors:
            anchors_by_jie[jie.index] = min(anchors)

    rows = []
    for jie, paragraph_index, paragraph, start, end in mapped_occurrences:
        text = paragraph.get("main", "") or ""
        anchor = anchors_by_jie.get(jie.index)
        if anchor is not None:
            anchor_paragraph, anchor_start, identity_surface = anchor
            if (paragraph_index, start) <= (anchor_paragraph, anchor_start):
                continue
            status = "mapped_translation_coreference_paragraph"
        elif entity.score >= 0.90:
            anchor_start = -1
            identity_surface = entity.surface
            status = "mapped_translation_expansion_paragraph"
        else:
            continue
        rows.append(
            {
                "juan": juan,
                "repo_para_id": int(paragraph["id"]),
                "repo_jie_index": jie.index,
                "repo_jie_number": jie.number,
                "identity_surface": identity_surface,
                "translation_ner_name": entity.surface,
                "translation_ner_score": entity.score,
                "original_start": start,
                "original_end": end,
                "original_surface": text[start:end],
                "normalized_original_surface": normalized_handle,
                "transfer_mode": "anchor_given",
                "mapping_status": status,
                "source_kind": (
                    "ziyexing_modern_chinese_translation_coreference"
                ),
                "source_page": f"{source_url}#pair-{pair.index + 1}",
            }
        )
    return rows


def _drop_ambiguous_coreferences(rows: list[dict]) -> list[dict]:
    identities_by_geometry = {}
    identities_by_jie_handle = {}
    for row in rows:
        if str(row["mapping_status"]).startswith("flagged_"):
            continue
        geometry = (
            int(row["repo_para_id"]),
            int(row["original_start"]),
            int(row["original_end"]),
        )
        identities_by_geometry.setdefault(geometry, set()).add(
            str(row["identity_surface"])
        )
        jie = int(row["repo_jie_index"])
        identity = _normalize(str(row["identity_surface"]))
        for handle in _coreference_handles(identity):
            identities_by_jie_handle.setdefault((jie, handle), set()).add(identity)
    return [
        row
        for row in rows
        if (
            row.get("transfer_mode") != "anchor_given"
            or (
                len(
                    identities_by_geometry[
                        (
                            int(row["repo_para_id"]),
                            int(row["original_start"]),
                            int(row["original_end"]),
                        )
                    ]
                ) == 1
                and len(
                    identities_by_jie_handle.get(
                        (
                            int(row["repo_jie_index"]),
                            _normalize(str(row["original_surface"])),
                        ),
                        (),
                    )
                ) == 1
            )
        )
    ]


def map_pair(
    juan: int,
    pair: SourcePair,
    jies: tuple[Jie, ...],
    people: tuple[PersonEntity, ...],
    source_url: str,
) -> list[dict]:
    source_original = _normalize(pair.original)
    aligned = aligned_jies(pair.original, jies)
    handle_owners: dict[str, set[str]] = {}
    for entity in people:
        normalized_entity = _normalize(entity.surface)
        if normalized_entity.endswith("\u66f0"):
            continue
        if normalized_entity in source_original:
            continue
        for handle in _coreference_handles(normalized_entity):
            if handle in source_original:
                handle_owners.setdefault(handle, set()).add(normalized_entity)
                break
    rows = []
    for entity in people:
        normalized_entity = _normalize(entity.surface)
        if normalized_entity.endswith("\u66f0"):
            continue
        if len(normalized_entity) < 2 or normalized_entity not in source_original:
            if len(normalized_entity) >= 2:
                rows.extend(
                    _map_translation_coreference(
                        juan,
                        pair,
                        aligned,
                        entity,
                        source_url,
                        handle_owners,
                    )
                )
            continue
        matching_jies = [
            jie
            for jie in aligned
            if normalized_entity in _CC.convert(jie.text)
        ]
        if not matching_jies:
            continue
        unique_jie = len(matching_jies) == 1
        mapped_geometry = {
            (int(paragraph["id"]), start, end)
            for _, _, paragraph, start, end in (
                _paragraph_mapped_occurrences(
                    pair,
                    tuple(matching_jies),
                    normalized_entity,
                    normalized_entity,
                )
                if not unique_jie
                else ()
            )
        }
        for jie in matching_jies:
            for paragraph in jie.paragraphs:
                text = paragraph.get("main", "") or ""
                for start, end in _occurrences(text, normalized_entity):
                    surface = text[start:end]
                    geometry = (int(paragraph["id"]), start, end)
                    occurrence_status = (
                        "mapped_exact_unique_jie"
                        if unique_jie
                        else (
                            "mapped_exact_paragraph"
                            if geometry in mapped_geometry
                            else "flagged_multi_jie_identity"
                        )
                    )
                    if (
                        unique_jie
                        and any(
                            text.startswith(suffix, end)
                            for suffix in _PERSON_TITLE_CONTINUATIONS
                        )
                    ):
                        occurrence_status = "flagged_person_title_continuation"
                    rows.append(
                        {
                            "juan": juan,
                            "repo_para_id": int(paragraph["id"]),
                            "repo_jie_index": jie.index,
                            "repo_jie_number": jie.number,
                            "identity_surface": surface,
                            "translation_ner_name": entity.surface,
                            "translation_ner_score": entity.score,
                            "original_start": start,
                            "original_end": end,
                            "original_surface": surface,
                            "normalized_original_surface": normalized_entity,
                            "transfer_mode": "exact",
                            "mapping_status": occurrence_status,
                            "source_kind": "ziyexing_modern_chinese_translation",
                            "source_page": f"{source_url}#pair-{pair.index + 1}",
                        }
                    )
    return rows


def recover_juan(juan: int, ner: TranslationNer) -> tuple[dict, list[dict]]:
    text_path = TEXT / f"juan_{juan:03d}.json"
    paragraphs = json.loads(text_path.read_text(encoding="utf-8"))["paragraphs"]
    jies = _jies(paragraphs)
    source_url, source_bytes = fetch_source(juan)
    alternate_url = None
    alternate_bytes = None
    try:
        pairs = parse_source(source_bytes)
    except ValueError:
        alternate_url = _alternate_translation_url(source_url, source_bytes)
        if alternate_url is None:
            if not _translation_is_explicitly_missing(source_bytes):
                raise
            return (
                {
                    "juan": juan,
                    "source_kind": "ziyexing_translation_explicitly_missing",
                    "source_page": source_url,
                    "source_sha256": _sha256(source_bytes),
                    "source_pairs": 0,
                },
                [],
            )
        alternate_bytes = _fetch(alternate_url)
        pairs = _whole_page_pair(source_bytes, alternate_bytes)
    rows = []
    for pair in pairs:
        rows.extend(
            map_pair(
                juan,
                pair,
                jies,
                ner.people(pair.translation),
                source_url,
            )
        )
    rows = _drop_ambiguous_coreferences(rows)
    deduplicated = {}
    for row in rows:
        key = (
            row["repo_para_id"],
            row["original_start"],
            row["original_end"],
            row["identity_surface"],
            row["mapping_status"],
        )
        previous = deduplicated.get(key)
        if (
            previous is None
            or row["translation_ner_score"] > previous["translation_ner_score"]
        ):
            deduplicated[key] = row
    source = {
        "juan": juan,
        "source_kind": "ziyexing_modern_chinese_translation",
        "source_page": source_url,
        "source_sha256": _sha256(source_bytes),
        "source_pairs": len(pairs),
    }
    if alternate_url is not None and alternate_bytes is not None:
        source["translation_page"] = alternate_url
        source["translation_page_sha256"] = _sha256(alternate_bytes)
    return source, sorted(
        deduplicated.values(),
        key=lambda row: (
            row["repo_para_id"],
            row["original_start"],
            row["original_end"],
            row["identity_surface"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--juans", type=int, nargs="+", required=True)
    parser.add_argument("--mapping-json", type=Path, required=True)
    args = parser.parse_args()

    mapping = (
        json.loads(args.mapping_json.read_text(encoding="utf-8"))
        if args.mapping_json.is_file()
        else {}
    )
    requested = set(args.juans)
    sources = [
        source
        for source in mapping.get("sources", [])
        if int(source["juan"]) not in requested
    ]
    candidates = [
        row
        for row in mapping.get("all_candidates", [])
        if int(row["juan"]) not in requested
    ]
    method = {
        "alignment_scope": "canonical_numbered_jie",
        "translation_ner_model": MODEL,
        "minimum_ner_score": MIN_NER_SCORE,
        "policy": (
            "person NER plus paired source-original exact identity or unique "
            "sentence-aligned abbreviated mention; canonical occurrence must "
            "align to one unique numbered jie"
        ),
    }

    def write() -> None:
        payload = {
            "method": method,
            "sources": sorted(sources, key=lambda source: int(source["juan"])),
            "all_candidates": sorted(
                candidates,
                key=lambda row: (
                    int(row["juan"]),
                    int(row["repo_para_id"]),
                    int(row["original_start"]),
                    int(row["original_end"]),
                ),
            ),
        }
        args.mapping_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.mapping_json.with_suffix(args.mapping_json.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.mapping_json)

    ner = TranslationNer()
    for juan in sorted(requested):
        source, rows = recover_juan(juan, ner)
        sources.append(source)
        candidates.extend(rows)
        eligible = sum(
            not row["mapping_status"].startswith("flagged_") for row in rows
        )
        print(f"juan {juan}: {eligible}/{len(rows)} eligible mapped candidates")
        write()

    print(f"wrote {len(candidates)} candidates to {args.mapping_json}")


if __name__ == "__main__":
    main()
