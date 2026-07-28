from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


HARD_SEPARATOR = "\n"


@dataclass(frozen=True, order=True)
class Span:
    para_id: int
    start: int
    end: int
    surface: str = ""

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid span [{self.start},{self.end})")


@dataclass(frozen=True)
class Segment:
    para_id: int
    assembled_start: int
    assembled_end: int


@dataclass(frozen=True)
class Jie:
    index: int
    number: int | None
    text: str
    segments: tuple[Segment, ...]


@dataclass(frozen=True)
class Metrics:
    true_positive: int
    predicted: int
    reference: int

    @property
    def precision(self) -> float:
        return self.true_positive / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.true_positive / self.reference if self.reference else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


def jie_number(text: str) -> int | None:
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


def assemble_jies(paragraphs: Sequence[dict]) -> list[Jie]:
    grouped: list[tuple[int | None, list[dict]]] = []
    current_number: int | None = None
    for paragraph in paragraphs:
        number = jie_number(str(paragraph.get("main", "") or ""))
        if number is not None:
            current_number = number
        if number is not None or not grouped:
            grouped.append((current_number, [paragraph]))
        else:
            grouped[-1][1].append(paragraph)

    result = []
    for index, (number, block) in enumerate(grouped):
        parts: list[str] = []
        segments: list[Segment] = []
        cursor = 0
        for paragraph in block:
            if parts:
                parts.append(HARD_SEPARATOR)
                cursor += len(HARD_SEPARATOR)
            text = str(paragraph.get("main", "") or "")
            start = cursor
            parts.append(text)
            cursor += len(text)
            segments.append(Segment(int(paragraph["id"]), start, cursor))
        result.append(Jie(index, number, "".join(parts), tuple(segments)))
    return result


def sanitize_note_mentions(
    paragraphs: Sequence[dict], mentions: Iterable[dict]
) -> list[dict]:
    paragraphs_by_id = {int(row["id"]): row for row in paragraphs}
    sanitized = []
    for mention in mentions:
        if mention.get("source") != "hu":
            continue
        para_id = int(mention["pid"])
        note_index = int(mention["note_index"])
        paragraph = paragraphs_by_id.get(para_id)
        if paragraph is None:
            raise ValueError(f"note mention references missing paragraph {para_id}")
        notes = paragraph.get("notes", [])
        if not 0 <= note_index < len(notes):
            raise ValueError(
                f"paragraph {para_id} has no note index {note_index}"
            )
        note = notes[note_index]
        start, end = int(mention["start"]), int(mention["end"])
        surface = str(mention["surface"])
        note_text = str(note.get("text", ""))
        if not 0 <= start < end <= len(note_text):
            raise ValueError("note mention geometry is out of bounds")
        if note_text[start:end] != surface:
            raise ValueError("note mention surface does not match note text")
        sanitized.append(
            {
                "para_id": para_id,
                "note_index": note_index,
                "after": int(note["after"]),
                "start": start,
                "end": end,
                "surface": surface,
            }
        )
    return sanitized


def decode_bio(
    labels: Sequence[str],
    *,
    owned: Sequence[bool] | None = None,
    separators: Sequence[bool] | None = None,
) -> list[tuple[int, int]]:
    size = len(labels)
    owned = tuple(owned) if owned is not None else (True,) * size
    separators = (
        tuple(separators) if separators is not None else (False,) * size
    )
    if len(owned) != size or len(separators) != size:
        raise ValueError("labels, owned, and separators must have equal length")

    spans: list[tuple[int, int]] = []
    active: int | None = None

    def close(end: int) -> None:
        nonlocal active
        if active is not None:
            spans.append((active, end))
            active = None

    for offset, raw_label in enumerate(labels):
        if not owned[offset] or separators[offset]:
            close(offset)
            continue
        label = raw_label.upper()
        if label in {"B", "B-PER"}:
            close(offset)
            active = offset
        elif label in {"I", "I-PER"}:
            if active is None:
                active = offset
        elif label in {"O", ""}:
            close(offset)
        else:
            raise ValueError(f"unsupported BIO label: {raw_label!r}")
    close(size)
    return spans


def _exact(left: Span, right: Span) -> bool:
    return (
        left.para_id == right.para_id
        and left.start == right.start
        and left.end == right.end
    )


def _overlap(left: Span, right: Span) -> bool:
    return (
        left.para_id == right.para_id
        and left.start < right.end
        and right.start < left.end
    )


def _maximum_matches(
    references: Sequence[Span],
    predictions: Sequence[Span],
    matches: Callable[[Span, Span], bool],
) -> int:
    edges = [
        [index for index, prediction in enumerate(predictions)
         if matches(reference, prediction)]
        for reference in references
    ]
    assigned: dict[int, int] = {}

    def augment(reference_index: int, visited: set[int]) -> bool:
        for prediction_index in edges[reference_index]:
            if prediction_index in visited:
                continue
            visited.add(prediction_index)
            previous = assigned.get(prediction_index)
            if previous is None or augment(previous, visited):
                assigned[prediction_index] = reference_index
                return True
        return False

    return sum(
        augment(reference_index, set())
        for reference_index in range(len(references))
    )


def score_spans(
    references: Iterable[Span],
    predictions: Iterable[Span],
    *,
    overlap: bool = False,
) -> Metrics:
    reference_rows = tuple(references)
    prediction_rows = tuple(predictions)
    matcher = _overlap if overlap else _exact
    matched = _maximum_matches(reference_rows, prediction_rows, matcher)
    return Metrics(matched, len(prediction_rows), len(reference_rows))
