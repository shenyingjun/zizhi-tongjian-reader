from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from core import HARD_SEPARATOR, Span, decode_bio
from p1_dataset import LABELS


LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


@dataclass
class TokenWindow:
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int] | None
    labels: list[int]
    offsets: list[tuple[int, int]]
    owned_tokens: list[bool]


def _token_label(char_labels: Sequence[str]) -> str:
    if not char_labels:
        raise ValueError("token does not own any characters")
    if all(label == "O" for label in char_labels):
        return "O"
    if char_labels[0] == "B-PER" and all(
        label == "I-PER" for label in char_labels[1:]
    ):
        return "B-PER"
    if all(label == "I-PER" for label in char_labels):
        return "I-PER"
    raise ValueError(f"token crosses incompatible BIO labels: {char_labels}")


def build_windows(
    tokenizer,
    example: dict,
    *,
    max_length: int = 512,
    stride: int = 128,
) -> list[TokenWindow]:
    text = str(example["text"])
    char_labels = list(example["labels"])
    if len(char_labels) != len(text):
        raise ValueError("text and labels must have equal character length")
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    offsets_by_window = [
        [tuple(pair) for pair in offsets]
        for offsets in encoded["offset_mapping"]
    ]
    candidates: list[list[tuple[int, int, int]]] = [
        [] for _ in text
    ]
    for window_index, offsets in enumerate(offsets_by_window):
        real = [(start, end) for start, end in offsets if end > start]
        if not real:
            continue
        content_start = min(start for start, _ in real)
        content_end = max(end for _, end in real)
        for token_index, (start, end) in enumerate(offsets):
            for char_index in range(start, end):
                margin = min(
                    char_index - content_start,
                    content_end - 1 - char_index,
                )
                candidates[char_index].append(
                    (margin, -window_index, token_index)
                )
    owner: list[tuple[int, int] | None] = []
    for rows in candidates:
        if not rows:
            owner.append(None)
            continue
        _, negative_window, token_index = max(rows)
        owner.append((-negative_window, token_index))

    windows = []
    for window_index, offsets in enumerate(offsets_by_window):
        token_labels = []
        owned_tokens = []
        for token_index, (start, end) in enumerate(offsets):
            owned = end > start and all(
                owner[char_index] == (window_index, token_index)
                for char_index in range(start, end)
            )
            owned_tokens.append(owned)
            token_labels.append(
                LABEL_TO_ID[_token_label(char_labels[start:end])]
                if owned
                else -100
            )
        token_type_ids = encoded.get("token_type_ids")
        windows.append(TokenWindow(
            input_ids=list(encoded["input_ids"][window_index]),
            attention_mask=list(encoded["attention_mask"][window_index]),
            token_type_ids=(
                list(token_type_ids[window_index])
                if token_type_ids is not None
                else None
            ),
            labels=token_labels,
            offsets=offsets,
            owned_tokens=owned_tokens,
        ))
    return windows


def merge_predictions(
    text: str,
    windows: Sequence[TokenWindow],
    prediction_ids: Sequence[Sequence[int]],
) -> tuple[list[str], list[bool]]:
    if len(windows) != len(prediction_ids):
        raise ValueError("window and prediction counts differ")
    labels = ["O"] * len(text)
    owned = [False] * len(text)
    for window, predictions in zip(windows, prediction_ids):
        if len(window.offsets) != len(predictions):
            raise ValueError("token and prediction counts differ")
        for token_index, ((start, end), prediction) in enumerate(
            zip(window.offsets, predictions)
        ):
            if not window.owned_tokens[token_index]:
                continue
            label = LABELS[int(prediction)]
            if label == "B-PER":
                expanded = ["B-PER"] + ["I-PER"] * (end - start - 1)
            else:
                expanded = [label] * (end - start)
            for char_index, char_label in zip(range(start, end), expanded):
                if owned[char_index]:
                    raise ValueError("character has multiple owning predictions")
                labels[char_index] = char_label
                owned[char_index] = True
    return labels, owned


def labels_to_spans(
    example: dict,
    labels: Sequence[str],
    owned: Sequence[bool],
) -> list[Span]:
    text = str(example["text"])
    geometry = decode_bio(
        labels,
        owned=owned,
        separators=[char == HARD_SEPARATOR for char in text],
    )
    result = []
    for start, end in geometry:
        segment = next(
            (
                row for row in example["segments"]
                if (
                    int(row["assembled_start"]) <= start
                    and end <= int(row["assembled_end"])
                )
            ),
            None,
        )
        if segment is None:
            raise ValueError(f"decoded span crosses paragraph geometry: {start}:{end}")
        assembled_start = int(segment["assembled_start"])
        result.append(Span(
            int(segment["para_id"]),
            start - assembled_start,
            end - assembled_start,
            text[start:end],
        ))
    return result
