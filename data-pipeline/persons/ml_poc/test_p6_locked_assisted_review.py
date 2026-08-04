import unittest

from p6_locked_assisted_review import build_review_pack


def _task():
    return {
        "jies": [{
            "text": "甲乙丙",
            "segments": [{
                "para_id": 1,
                "assembled_start": 0,
                "assembled_end": 3,
            }],
        }],
    }


def _pack(pass_name, channel, candidates):
    return {
        "phase": "assisted",
        "diagnostic_only": True,
        "juan": 1,
        "teacher_pass": pass_name,
        "candidates": [{
            "id": f"copilot:1:{start}:{end}",
            "para_id": 1,
            "start": start,
            "end": end,
            "surface": "甲乙丙"[start:end],
            "channels": [channel],
            "confidence": confidence,
            "review_reason": "" if confidence == "high" else "uncertain",
        } for start, end, confidence in candidates],
    }


class LockedAssistedReviewTest(unittest.TestCase):
    def test_auto_accepts_consensus_and_reviews_disagreement(self):
        pass_a = _pack(
            "A-recall-first", "copilot_independent_a",
            [(0, 1, "high"), (2, 3, "high")],
        )
        pass_b = _pack(
            "B-boundary-first", "copilot_independent_b",
            [(0, 1, "high")],
        )
        review, counts = build_review_pack(
            _task(), pass_a, pass_b, 1, set()
        )
        self.assertEqual(1, counts["auto_accepted_consensus"])
        self.assertEqual(1, counts["teacher_disagreement_review"])
        self.assertEqual(
            {"copilot:1:0:1": "accept"}, review["initial_decisions"]
        )

    def test_routes_predeclared_consensus_audit_to_review(self):
        pass_a = _pack(
            "A-recall-first", "copilot_independent_a", [(0, 1, "high")]
        )
        pass_b = _pack(
            "B-boundary-first", "copilot_independent_b", [(0, 1, "high")]
        )
        review, counts = build_review_pack(
            _task(), pass_a, pass_b, 1, {(1, 0, 1)}
        )
        self.assertEqual(1, counts["consensus_audit_review"])
        self.assertEqual({}, review["initial_decisions"])
        self.assertEqual("low", review["candidates"][0]["confidence"])


if __name__ == "__main__":
    unittest.main()
