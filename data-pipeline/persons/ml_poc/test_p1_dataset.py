import unittest

from p1_dataset import build_examples, choose_contiguous_dev


class P1DatasetTest(unittest.TestCase):
    def test_maps_paragraph_geometry_to_assembled_char_bio(self):
        task = {"jies": [{
            "jie_index": 0,
            "jie_number": 1,
            "text": "①曹操至。\n刘备来。",
            "segments": [
                {"para_id": 2, "assembled_start": 0, "assembled_end": 5},
                {"para_id": 3, "assembled_start": 6, "assembled_end": 10},
            ],
        }]}
        state = {"role_audit": {
            "complete": True,
            "annotations": [
                {"para_id": 2, "start": 1, "end": 3, "surface": "曹操"},
                {"para_id": 3, "start": 0, "end": 2, "surface": "刘备"},
            ],
        }}

        examples = build_examples(1, task, state)

        self.assertEqual(1, len(examples))
        self.assertEqual(
            ["O", "B-PER", "I-PER", "O", "O", "O",
             "B-PER", "I-PER", "O", "O"],
            examples[0]["labels"],
        )
        self.assertEqual("O", examples[0]["labels"][5])
        self.assertEqual("human_audited", examples[0]["label_provenance"])

    def test_dev_is_one_contiguous_block_and_train_is_disjoint(self):
        examples = [
            {
                "id": str(index),
                "jie_index": index,
                "text": "单于" if index == 2 else "文",
                "span_count": count,
            }
            for index, count in enumerate((2, 2, 2, 2, 2))
        ]

        train, dev = choose_contiguous_dev(examples, target_fraction=0.4)

        dev_indexes = [row["jie_index"] for row in dev]
        self.assertEqual(
            list(range(dev_indexes[0], dev_indexes[-1] + 1)),
            dev_indexes,
        )
        self.assertTrue({row["id"] for row in train}.isdisjoint(
            {row["id"] for row in dev}
        ))
        self.assertIn(2, dev_indexes)


if __name__ == "__main__":
    unittest.main()
