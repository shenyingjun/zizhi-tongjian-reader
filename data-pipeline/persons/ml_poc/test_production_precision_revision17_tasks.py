import unittest

from production_precision_revision17_tasks import (
    MAX_PER_JIE,
    MAX_PER_JUAN,
    select_candidates,
)


def _row(index, *, score, support, confidence, juan=None, jie=None):
    return {
        "id": f"j-{index}",
        "juan": index // 50 if juan is None else juan,
        "jie_index": index // 4 if jie is None else jie,
        "para_id": index,
        "start": 0,
        "end": 1,
        "surface": "甲",
        "stage1_probability": score,
        "generator_support": support,
        "maximum_generator_confidence": confidence,
    }


class Revision17TasksTest(unittest.TestCase):
    def test_selects_strata_in_frozen_priority_order(self):
        rows = [
            _row(1, score=0.9, support=2, confidence=0.8),
            _row(2, score=0.8, support=1, confidence=0.7),
            _row(3, score=0.4, support=3, confidence=0.9),
        ]
        selected = select_candidates(rows)

        self.assertEqual(
            ["hard_disagreement", "hard_disagreement", "low_stage1"],
            [row["selection_stratum"] for row in selected],
        )
        self.assertEqual(["j-2", "j-1", "j-3"], [row["id"] for row in selected])

    def test_applies_shared_jie_and_juan_caps(self):
        rows = [
            _row(
                index,
                score=0.8,
                support=1,
                confidence=0.5 + index / 1000,
                juan=1,
                jie=1 if index < 10 else index,
            )
            for index in range(80)
        ]
        selected = select_candidates(rows)

        self.assertLessEqual(
            sum(row["jie_index"] == 1 for row in selected), MAX_PER_JIE
        )
        self.assertLessEqual(len(selected), MAX_PER_JUAN)
        self.assertEqual(len(selected), len({
            (row["id"], row["para_id"], row["start"], row["end"])
            for row in selected
        }))

    def test_rejects_duplicate_geometry(self):
        row = _row(1, score=0.8, support=1, confidence=0.7)
        with self.assertRaisesRegex(ValueError, "duplicate geometry"):
            select_candidates([row, dict(row)])


if __name__ == "__main__":
    unittest.main()
