from __future__ import annotations

from collections import defaultdict

from core import HARD_SEPARATOR


CONTEXT_MODES = ("target_only", "bounded_neighbor", "max_budget")


def validate_whole_juan_splits(**splits: list[dict]) -> dict:
    juans = {
        name: {int(row["juan"]) for row in rows}
        for name, rows in splits.items()
    }
    names = list(juans)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = juans[left] & juans[right]
            if overlap:
                values = ", ".join(str(value) for value in sorted(overlap))
                raise ValueError(
                    f"{left} and {right} share juan(s): {values}; "
                    "soft-context splits must be whole-juan disjoint"
                )
    return {
        "guard_band_exclusions": 0,
        "guard_reason": (
            "validated whole-juan-disjoint splits have disjoint context graphs"
        ),
        "split_juans": {
            name: sorted(values) for name, values in juans.items()
        },
    }


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        verbose=False,
    )["input_ids"])


def add_soft_context(
    examples: list[dict],
    tokenizer,
    *,
    mode: str,
    max_length: int,
) -> tuple[list[dict], dict]:
    if mode not in CONTEXT_MODES:
        raise ValueError(f"unsupported context mode: {mode}")
    by_juan: dict[int, list[dict]] = defaultdict(list)
    for example in examples:
        by_juan[int(example["juan"])].append(example)
    for rows in by_juan.values():
        rows.sort(key=lambda row: int(row["jie_index"]))

    result = []
    context_counts = []
    for example in examples:
        target = str(example["text"])
        if mode == "target_only" or _token_count(
            tokenizer, target
        ) > max_length:
            row = dict(example)
            row["target_mask"] = [True] * len(target)
            row["context_jies"] = []
            result.append(row)
            context_counts.append(0)
            continue

        rows = by_juan[int(example["juan"])]
        position = next(
            index for index, row in enumerate(rows)
            if int(row["jie_index"]) == int(example["jie_index"])
        )
        limit = 1 if mode == "bounded_neighbor" else len(rows)
        selected: dict[int, dict] = {}
        blocked_sides: set[int] = set()
        for distance in range(1, limit + 1):
            for signed_distance in (-distance, distance):
                side = -1 if signed_distance < 0 else 1
                if side in blocked_sides:
                    continue
                candidate_position = position + signed_distance
                if not 0 <= candidate_position < len(rows):
                    blocked_sides.add(side)
                    continue
                candidate = rows[candidate_position]
                trial = dict(selected)
                trial[signed_distance] = candidate
                parts = [
                    str(trial[key]["text"])
                    for key in sorted(key for key in trial if key < 0)
                ]
                parts.append(target)
                parts.extend(
                    str(trial[key]["text"])
                    for key in sorted(key for key in trial if key > 0)
                )
                if _token_count(
                    tokenizer, HARD_SEPARATOR.join(parts)
                ) <= max_length:
                    selected = trial
                else:
                    blocked_sides.add(side)

        preceding = [
            (distance, selected[distance])
            for distance in sorted(key for key in selected if key < 0)
        ]
        following = [
            (distance, selected[distance])
            for distance in sorted(key for key in selected if key > 0)
        ]
        prefix_parts = [str(row["text"]) for _, row in preceding]
        suffix_parts = [str(row["text"]) for _, row in following]
        prefix = (
            HARD_SEPARATOR.join(prefix_parts) + HARD_SEPARATOR
            if prefix_parts else ""
        )
        suffix = (
            HARD_SEPARATOR + HARD_SEPARATOR.join(suffix_parts)
            if suffix_parts else ""
        )
        text = prefix + target + suffix
        target_start = len(prefix)
        row = dict(example)
        row["text"] = text
        row["labels"] = (
            ["O"] * target_start
            + list(example["labels"])
            + ["O"] * len(suffix)
        )
        row["target_mask"] = (
            [False] * target_start
            + [True] * len(target)
            + [False] * len(suffix)
        )
        row["segments"] = [
            {
                **segment,
                "assembled_start": (
                    int(segment["assembled_start"]) + target_start
                ),
                "assembled_end": int(segment["assembled_end"]) + target_start,
            }
            for segment in example["segments"]
        ]
        row["context_jies"] = [
            {
                "distance": distance,
                "jie_index": int(context["jie_index"]),
            }
            for distance, context in preceding + following
        ]
        result.append(row)
        context_counts.append(len(selected))

    return result, {
        "mode": mode,
        "examples": len(result),
        "examples_with_context": sum(count > 0 for count in context_counts),
        "context_jies_total": sum(context_counts),
        "max_context_jies": max(context_counts, default=0),
        "guard_band_exclusions": 0,
        "guard_reason": "reported after whole-juan split validation",
    }
