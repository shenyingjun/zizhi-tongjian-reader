import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from p6_seed_replication_train import (
    CUBLAS_WORKSPACE_CONFIG,
    DATASETS,
    _configure_determinism,
    run_seed_replication,
)


class SeedReplicationTrainTest(unittest.TestCase):
    def test_round7_dataset_is_registered(self):
        self.assertEqual(
            DATASETS["round7"]["status"],
            "round7_controlled_training_dataset",
        )

    def test_rejects_unregistered_experiment_before_git_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch(
                    "p6_seed_replication_train._git_commit_clean",
                    side_effect=AssertionError("must not run"),
                ),
                self.assertRaises(ValueError),
            ):
                run_seed_replication(
                    root / "dataset",
                    "round7",
                    20260727,
                    root / "output",
                    "unknown",
                )

    def test_lower_lr_requires_round7_before_git_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch(
                    "p6_seed_replication_train._git_commit_clean",
                    side_effect=AssertionError("must not run"),
                ),
                self.assertRaises(ValueError),
            ):
                run_seed_replication(
                    root / "dataset",
                    "round6",
                    20260727,
                    root / "output",
                    "round8-lr2e-5",
                )

    def test_refuses_existing_output_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            with (
                patch(
                    "p6_seed_replication_train._git_commit_clean",
                    side_effect=AssertionError("must not run"),
                ),
                self.assertRaises(FileExistsError),
            ):
                run_seed_replication(
                    root / "dataset", "round4", 20260728, output
                )

    def test_refuses_unregistered_seed_before_git_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch(
                    "p6_seed_replication_train._git_commit_clean",
                    side_effect=AssertionError("must not run"),
                ),
                self.assertRaises(ValueError),
            ):
                run_seed_replication(
                    root / "dataset", "round4", 20260726, root / "output"
                )

    @patch("p6_seed_replication_train.torch")
    def test_enables_strict_cuda_determinism(self, torch):
        _configure_determinism()
        torch.use_deterministic_algorithms.assert_called_once_with(
            True, warn_only=False
        )
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)
        import os
        self.assertEqual(
            CUBLAS_WORKSPACE_CONFIG,
            os.environ["CUBLAS_WORKSPACE_CONFIG"],
        )


if __name__ == "__main__":
    unittest.main()
