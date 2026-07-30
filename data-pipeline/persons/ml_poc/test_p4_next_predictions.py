import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from p4_next_predictions import predict_tasks


class NextPredictionsTest(unittest.TestCase):
    def test_refuses_existing_output_before_loading_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.json"
            output.write_text("existing", encoding="utf-8")
            with (
                patch(
                    "p4_next_predictions._git_commit_clean",
                    side_effect=AssertionError("must not inspect git"),
                ),
                self.assertRaises(FileExistsError),
            ):
                predict_tasks(Path(temporary), Path(temporary), output)


if __name__ == "__main__":
    unittest.main()
