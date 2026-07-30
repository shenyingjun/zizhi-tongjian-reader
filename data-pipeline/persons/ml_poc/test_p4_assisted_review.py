import unittest

from p4_assisted_review import build_review_pack


class P4AssistedReviewTest(unittest.TestCase):
    def test_accepts_consensus_and_reviews_disagreement_and_model_only(self):
        task = {
            "juan": 1,
            "jies": [{
                "text": "曹操见刘备",
                "segments": [{
                    "para_id": 2,
                    "assembled_start": 0,
                    "assembled_end": 6,
                }],
            }],
        }

        def pack(channel, teacher_pass, rows):
            return {
                "phase": "assisted",
                "juan": 1,
                "diagnostic_only": True,
                "teacher_pass": teacher_pass,
                "candidates": [{
                    "id": f"copilot:2:{start}:{end}",
                    "para_id": 2,
                    "start": start,
                    "end": end,
                    "surface": "曹操见刘备"[start:end],
                    "channels": [channel],
                    "confidence": confidence,
                    "review_reason": "" if confidence == "high" else "check",
                } for start, end, confidence in rows],
            }

        result, counts = build_review_pack(
            task,
            pack(
                "copilot_independent_a",
                "A-recall-first",
                [(0, 2, "high"), (3, 5, "high")],
            ),
            pack(
                "copilot_independent_b",
                "B-boundary-first",
                [(0, 2, "high")],
            ),
            [{"para_id": 2, "start": 0, "end": 2, "surface": "曹操"},
             {"para_id": 2, "start": 2, "end": 3, "surface": "见"}],
            1,
        )

        self.assertEqual(3, counts["candidate_union"])
        self.assertEqual(1, counts["auto_accepted_consensus"])
        self.assertEqual(1, counts["teacher_disagreement_review"])
        self.assertEqual(1, counts["model_only_review"])
        self.assertEqual([{
            "para_id": 2, "start": 0, "end": 2, "surface": "曹操",
        }], result["initial_annotations"])
        self.assertEqual(
            {"copilot:2:0:2": "accept"}, result["initial_decisions"]
        )


if __name__ == "__main__":
    unittest.main()
