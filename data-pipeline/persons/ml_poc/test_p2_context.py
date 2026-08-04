import unittest

from p2_context import add_soft_context, validate_whole_juan_splits


class CharTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [101] + list(range(len(text))) + [102]}


def example(index, text):
    return {
        "id": f"juan-001-jie-{index:04d}",
        "juan": 1,
        "jie_index": index,
        "jie_number": index,
        "text": text,
        "labels": ["O"] * len(text),
        "segments": [{
            "para_id": index,
            "assembled_start": 0,
            "assembled_end": len(text),
        }],
    }


class P2ContextTest(unittest.TestCase):
    def test_split_guard_rejects_shared_juan(self):
        with self.assertRaisesRegex(ValueError, "whole-juan disjoint"):
            validate_whole_juan_splits(
                train=[example(0, "甲")],
                dev=[example(1, "乙")],
                evaluation=[],
            )

    def test_split_guard_reports_disjoint_juan_sets(self):
        train = example(0, "甲")
        dev = {**example(0, "乙"), "juan": 2}

        report = validate_whole_juan_splits(
            train=[train], dev=[dev], evaluation=[]
        )

        self.assertEqual(0, report["guard_band_exclusions"])
        self.assertEqual({"train": [1], "dev": [2], "evaluation": []},
                         report["split_juans"])

    def test_max_budget_adds_nearest_jies_and_masks_context(self):
        rows = [example(0, "甲"), example(1, "曹操"), example(2, "乙")]

        contextualized, report = add_soft_context(
            rows, CharTokenizer(), mode="max_budget", max_length=8
        )
        target = contextualized[1]

        self.assertEqual("甲\n曹操\n乙", target["text"])
        self.assertEqual(
            [False, False, True, True, False, False],
            target["target_mask"],
        )
        self.assertEqual(2, target["segments"][0]["assembled_start"])
        self.assertEqual([-1, 1], [
            row["distance"] for row in target["context_jies"]
        ])
        self.assertEqual(3, report["examples_with_context"])

    def test_target_only_preserves_geometry(self):
        rows = [example(0, "曹操")]

        contextualized, report = add_soft_context(
            rows, CharTokenizer(), mode="target_only", max_length=8
        )

        self.assertEqual(rows[0]["text"], contextualized[0]["text"])
        self.assertEqual([True, True], contextualized[0]["target_mask"])
        self.assertEqual(0, report["context_jies_total"])

    def test_max_budget_does_not_skip_oversized_nearest_jie(self):
        rows = [
            example(0, "甲"),
            example(1, "乙" * 10),
            example(2, "曹操"),
        ]

        contextualized, _ = add_soft_context(
            rows, CharTokenizer(), mode="max_budget", max_length=8
        )

        self.assertEqual("曹操", contextualized[2]["text"])
        self.assertEqual([], contextualized[2]["context_jies"])


if __name__ == "__main__":
    unittest.main()
