import unittest

from p3_active_finalize import _validate_decisions
from p4_assisted_finalize import finalize_assisted_review


class P4AssistedFinalizeTest(unittest.TestCase):
    def test_refuses_existing_output_before_reading_inputs(self):
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                finalize_assisted_review(Path(temporary) / "missing", output)

    def test_rejects_decision_annotation_mismatch(self):
        task = {"jies": [{
            "text": "曹操",
            "segments": [{
                "para_id": 1,
                "assembled_start": 0,
                "assembled_end": 2,
            }],
        }]}
        pack = {"candidates": [{
            "id": "copilot:1:0:2",
            "para_id": 1,
            "start": 0,
            "end": 2,
            "surface": "曹操",
        }]}
        assisted = {
            "decisions": {"copilot:1:0:2": "accept"},
            "annotations": [],
        }
        with self.assertRaisesRegex(ValueError, "decision differs"):
            _validate_decisions(1, task, pack, assisted)


if __name__ == "__main__":
    unittest.main()
