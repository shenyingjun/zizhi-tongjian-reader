import tempfile
import unittest
from pathlib import Path

from p10_independent_dev_review import build_review_pack


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


class IndependentDevReviewTest(unittest.TestCase):
    def test_refuses_existing_output(self):
        from p10_independent_dev_review import prepare_independent_dev_review

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_independent_dev_review(
                    root / "tasks", root / "teachers", output
                )

    def test_reviews_only_teacher_disagreement_and_explicit_low(self):
        pass_a = _pack(
            "A-recall-first",
            "copilot_independent_a",
            [(0, 1, "high"), (1, 2, "low"), (2, 3, "high")],
        )
        pass_b = _pack(
            "B-boundary-first",
            "copilot_independent_b",
            [(0, 1, "high"), (1, 2, "high")],
        )
        review, counts = build_review_pack(_task(), pass_a, pass_b, [], 1)
        self.assertEqual(1, counts["auto_accepted_consensus"])
        self.assertEqual(1, counts["explicit_low_review"])
        self.assertEqual(1, counts["teacher_disagreement_review"])
        self.assertEqual(0, counts["model_only_review"])
        self.assertEqual(
            {"copilot:1:0:1": "accept"}, review["initial_decisions"]
        )
        self.assertFalse(any(
            "round3_ml_omission_check" in row["channels"]
            for row in review["candidates"]
        ))

    def test_rejects_wrong_teacher_provenance(self):
        pass_a = _pack(
            "A-recall-first", "copilot_independent_b", [(0, 1, "high")]
        )
        pass_b = _pack(
            "B-boundary-first", "copilot_independent_b", [(0, 1, "high")]
        )
        with self.assertRaises(ValueError):
            build_review_pack(_task(), pass_a, pass_b, [], 1)


if __name__ == "__main__":
    unittest.main()
