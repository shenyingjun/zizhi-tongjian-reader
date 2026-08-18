from __future__ import annotations

import unittest

import numpy as np

from production_precision_lexical_verifier import (
    STRATUM_MASSES,
    _stratum_weights,
)


class LexicalVerifierContractTest(unittest.TestCase):
    def test_stratum_weights_assign_exact_prospective_mass(self):
        strata = [
            *(["real_positive"] * 5),
            *(["real_negative"] * 2),
            *(["mined_negative"] * 7),
        ]

        weights, inventory = _stratum_weights(strata)

        total = len(strata)
        for name, mass in STRATUM_MASSES.items():
            indices = [index for index, value in enumerate(strata)
                       if value == name]
            self.assertAlmostEqual(
                mass,
                float(weights[indices].sum()) / total,
                places=7,
            )
            self.assertEqual(len(indices), inventory[name]["rows"])

    def test_missing_stratum_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "all be non-empty"):
            _stratum_weights(["real_positive", "real_negative"])


if __name__ == "__main__":
    unittest.main()
