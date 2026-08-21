import unittest

import numpy as np

from production_precision_revision17_mining_stage1 import attach_scores


class Revision17MiningStage1Test(unittest.TestCase):
    def test_attaches_float32_scores_without_reordering(self):
        rows = [{"id": "a"}, {"id": "b"}]
        result = attach_scores(rows, np.asarray([0.1, 0.9], dtype=np.float32))

        self.assertEqual(["a", "b"], [row["id"] for row in result])
        self.assertEqual(
            [float(np.float32(0.1)), float(np.float32(0.9))],
            [row["stage1_probability"] for row in result],
        )

    def test_rejects_nonfinite_or_incomplete_scores(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            attach_scores([{"id": "a"}], np.asarray([], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "coverage"):
            attach_scores(
                [{"id": "a"}], np.asarray([np.nan], dtype=np.float32)
            )


if __name__ == "__main__":
    unittest.main()
