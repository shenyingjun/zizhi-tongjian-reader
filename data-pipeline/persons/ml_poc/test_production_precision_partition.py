import unittest

from production_precision_partition import assign_juans


class PrecisionPartitionTest(unittest.TestCase):
    def test_assignment_is_deterministic_and_juan_grouped(self):
        groups = {
            juan: {
                "examples": 7 + juan % 3,
                "spans": 70 + juan,
                "uniform_random": 4 + juan % 2,
                "role_appellation": 1,
                "foreign_title": int(juan % 6 == 0),
                "boundary_anaphora": 1,
            }
            for juan in range(1, 37)
        }

        first = assign_juans(groups)
        second = assign_juans(groups)

        self.assertEqual(first, second)
        self.assertEqual(set(groups), set(first))
        self.assertEqual(
            {"fit", "calibration", "confirmation"}, set(first.values())
        )


if __name__ == "__main__":
    unittest.main()
