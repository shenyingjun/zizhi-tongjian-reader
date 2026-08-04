import unittest

from p4_round4_compare import _rename_hit_buckets


class Round4CompareTest(unittest.TestCase):
    def test_renames_round_hit_buckets(self):
        value = {
            "reference_length_recall": {
                "2": {"round2_hits": 3, "round3_hits": 4},
            },
            "reference_term_recall": {
                "太后": {"round2_hits": 1, "round3_hits": 2},
            },
        }
        _rename_hit_buckets(value)
        self.assertEqual({
            "round3_hits": 3, "round4_hits": 4,
        }, value["reference_length_recall"]["2"])


if __name__ == "__main__":
    unittest.main()
