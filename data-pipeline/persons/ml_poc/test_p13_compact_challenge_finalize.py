import tempfile
import unittest
from pathlib import Path

from p13_compact_challenge_finalize import (
    finalize_compact_challenge_reference,
)


class CompactChallengeFinalizeTest(unittest.TestCase):
    def test_refuses_existing_output_before_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                finalize_compact_challenge_reference(
                    root / "review", root / "state", output
                )


if __name__ == "__main__":
    unittest.main()
