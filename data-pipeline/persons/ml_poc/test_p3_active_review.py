import unittest
from unittest.mock import patch

from p3_active_review import build_review_pack


class P3ActiveReviewTest(unittest.TestCase):
    def test_union_keeps_teacher_initial_and_adds_ml_only_review(self):
        task = {
            "jies": [{
                "text": "①曹操与刘备。",
                "segments": [{
                    "para_id": 2,
                    "assembled_start": 0,
                    "assembled_end": 7,
                }],
            }],
        }
        teacher = {
            "juan": 1,
            "phase": "assisted",
            "diagnostic_only": True,
            "candidates": [{
                "id": "copilot:2:1:3",
                "para_id": 2,
                "start": 1,
                "end": 3,
                "surface": "曹操",
                "channels": ["copilot_round3_teacher"],
                "confidence": "high",
                "review_reason": "",
            }],
        }
        ml_seed = {
            "juan": 1,
            "phase": "assisted",
            "candidate_model": {"sha256": "model-hash"},
            "candidates": [{
                "id": "2:1:3",
                "para_id": 2,
                "start": 1,
                "end": 3,
                "surface": "曹操",
            }, {
                "id": "2:4:6",
                "para_id": 2,
                "start": 4,
                "end": 6,
                "surface": "刘备",
            }],
        }

        with patch("p3_active_review.EXPECTED_MODEL_SHA256", "model-hash"):
            pack, counts = build_review_pack(task, teacher, ml_seed, 1)

        self.assertEqual(2, len(pack["candidates"]))
        self.assertEqual(1, counts["ml_only"])
        self.assertEqual(1, counts["review_candidates"])
        self.assertEqual(
            [{"para_id": 2, "start": 1, "end": 3, "surface": "曹操"}],
            pack["initial_annotations"],
        )
        self.assertEqual(
            {"copilot:2:1:3": "accept"},
            pack["initial_decisions"],
        )


if __name__ == "__main__":
    unittest.main()
