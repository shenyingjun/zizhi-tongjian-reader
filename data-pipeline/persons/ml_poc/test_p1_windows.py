import unittest

from p1_windows import (
    build_windows,
    constrain_predictions,
    labels_to_spans,
    merge_predictions,
)


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        max_length = kwargs["max_length"]
        stride = kwargs["stride"]
        content = max_length - 2
        step = content - stride
        starts = list(range(0, len(text), step))
        windows = []
        masks = []
        offsets = []
        for start in starts:
            end = min(len(text), start + content)
            pairs = [(0, 0)] + [(i, i + 1) for i in range(start, end)]
            pairs += [(0, 0)] * (max_length - len(pairs))
            windows.append([101] + list(range(start, end)) + [0] * (
                max_length - (end - start) - 1
            ))
            masks.append([1] * (end - start + 1) + [0] * (
                max_length - (end - start) - 1
            ))
            offsets.append(pairs)
            if end == len(text):
                break
        return {
            "input_ids": windows,
            "attention_mask": masks,
            "offset_mapping": offsets,
        }


class P1WindowTest(unittest.TestCase):
    def test_overlapping_windows_assign_each_character_once(self):
        example = {
            "text": "甲乙丙丁戊己庚辛",
            "labels": [
                "O", "B-PER", "I-PER", "O",
                "O", "B-PER", "I-PER", "O",
            ],
            "segments": [{
                "para_id": 2, "assembled_start": 0, "assembled_end": 8,
            }],
        }
        windows = build_windows(
            FakeTokenizer(), example, max_length=6, stride=2
        )
        predictions = [
            [0 if label == -100 else label for label in window.labels]
            for window in windows
        ]

        labels, owned = merge_predictions(
            example["text"], windows, predictions
        )

        self.assertEqual(example["labels"], labels)
        self.assertEqual([True] * 8, owned)

    def test_separator_is_unowned_and_closes_span(self):
        example = {
            "text": "曹操\n刘备",
            "labels": ["B-PER", "I-PER", "O", "B-PER", "I-PER"],
            "segments": [
                {"para_id": 2, "assembled_start": 0, "assembled_end": 2},
                {"para_id": 3, "assembled_start": 3, "assembled_end": 5},
            ],
        }
        labels = list(example["labels"])
        owned = [True, True, False, True, True]

        spans = labels_to_spans(example, labels, owned)

        self.assertEqual(
            [(2, 0, 2, "曹操"), (3, 0, 2, "刘备")],
            [(row.para_id, row.start, row.end, row.surface) for row in spans],
        )

    def test_context_characters_do_not_own_loss_or_output(self):
        example = {
            "text": "甲\n曹操\n乙",
            "labels": [
                "O", "O", "B-PER", "I-PER", "O", "O",
            ],
            "target_mask": [False, False, True, True, False, False],
            "segments": [{
                "para_id": 2, "assembled_start": 2, "assembled_end": 4,
            }],
        }

        windows = build_windows(
            FakeTokenizer(), example, max_length=8, stride=2
        )
        predictions = [
            [0 if label == -100 else label for label in window.labels]
            for window in windows
        ]
        labels, owned = merge_predictions(
            example["text"], windows, predictions
        )

        self.assertEqual(
            [False, False, True, True, False, False], owned
        )
        self.assertEqual(
            [(2, 0, 2, "曹操")],
            [
                (row.para_id, row.start, row.end, row.surface)
                for row in labels_to_spans(example, labels, owned)
            ],
        )

    def test_constraints_remove_punctuation_and_merge_adjacent_b(self):
        text = "严延年，「"
        labels = [
            "B-PER", "B-PER", "I-PER", "O", "B-PER",
        ]

        constrained = constrain_predictions(
            text, labels, [True] * len(text)
        )

        self.assertEqual(
            ["B-PER", "I-PER", "I-PER", "O", "O"],
            constrained,
        )

    def test_constraints_do_not_join_across_punctuation(self):
        text = "尧、舜"
        labels = ["B-PER", "B-PER", "B-PER"]

        constrained = constrain_predictions(
            text, labels, [True] * len(text)
        )

        self.assertEqual(["B-PER", "O", "B-PER"], constrained)


if __name__ == "__main__":
    unittest.main()
