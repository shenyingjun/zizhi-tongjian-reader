import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from p3_round3_train import (
    CONTROL,
    MODEL_NAME,
    MODEL_REVISION,
    run_controlled_training,
)


class Round3ControlledTrainTest(unittest.TestCase):
    def test_enforces_control_and_binds_model_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            outputs = {}
            for name in ("train.jsonl", "dev.jsonl", "evaluation.jsonl"):
                path = dataset / name
                path.write_text("{}\n", encoding="utf-8")
                outputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            (dataset / "manifest.json").write_text(json.dumps({
                "status": "round3_controlled_training_dataset",
                "formal_evaluation": False,
                "outputs": outputs,
            }), encoding="utf-8")

            def fake_train(args):
                self.assertEqual(MODEL_NAME, args.model)
                self.assertEqual(MODEL_REVISION, args.model_revision)
                self.assertEqual(CONTROL["epochs"], args.epochs)
                model = args.output / "model"
                model.mkdir()
                (model / "model.safetensors").write_bytes(b"model")
                for name in (
                    "history.json",
                    "dev_predictions.json",
                    "evaluation_predictions.json",
                ):
                    (args.output / name).write_text("[]\n", encoding="utf-8")
                inputs = {
                    name: str(path) for name, path in {
                        "train": args.train_file,
                        "dev": args.dev_file,
                        "evaluation": args.evaluation_file,
                    }.items()
                }
                inputs.update({
                    f"{name}_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in {
                        "train": args.train_file,
                        "dev": args.dev_file,
                        "evaluation": args.evaluation_file,
                    }.items()
                })
                config = {
                    key: value for key, value in CONTROL.items()
                    if key not in {"eval_batch_size", "max_grad_norm"}
                }
                config["selected_epoch"] = 5
                return {
                    "model": MODEL_NAME,
                    "model_revision": MODEL_REVISION,
                    "config": config,
                    "inputs": inputs,
                    "dev_challenge": {"exact": {}},
                    "evaluation": {
                        "name": "locked_blind_anchor_diagnostic",
                        "exact": {},
                    },
                }

            output = root / "output"
            manifest_sha256 = hashlib.sha256(
                (dataset / "manifest.json").read_bytes()
            ).hexdigest()
            with patch(
                "p3_round3_train._git_commit_clean",
                return_value="commit",
            ), patch(
                "p3_round3_train.EXPECTED_DATASET_MANIFEST_SHA256",
                manifest_sha256,
            ), patch("p3_round3_train.train", side_effect=fake_train):
                report = run_controlled_training(dataset, output)

            self.assertEqual(
                hashlib.sha256(b"model").hexdigest(),
                report["round3_control"]["model_artifact"]["files"][
                    "model.safetensors"
                ],
            )
            self.assertFalse(report["round3_control"]["formal_evaluation"])
            self.assertTrue((output / "report.json").is_file())


if __name__ == "__main__":
    unittest.main()
