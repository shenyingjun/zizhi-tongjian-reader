from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

from p1_windows import (
    build_windows,
    constrain_predictions,
    labels_to_spans,
    merge_predictions,
)
from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from p4_fresh_sealed import (
    EXPECTED_MODEL_ARTIFACT_SHA256,
    EXPECTED_MODEL_REPORT_SHA256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict_tasks(
    tasks_root: Path,
    model_root: Path,
    output_path: Path,
) -> dict:
    if output_path.exists():
        raise FileExistsError(f"prediction bundle exists: {output_path}")
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    git_commit = _git_commit_clean()
    manifest_path = tasks_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_jies = {
        (int(row["juan"]), int(row["jie_index"]))
        for row in manifest.get("selected_jies", [])
    }
    selections = manifest.get("selected", [])
    if (
        manifest.get("status")
        != "copilot_double_pass_tasks_before_labeling"
        or len(selected_jies) != 60
        or len(selections) != 60
        or len({int(row["juan"]) for row in selections}) != 60
        or any(int(row.get("sampled_jies", 0)) != 1 for row in selections)
        or manifest.get("model_predictions_used_for_selection") is not False
        or _sha256(model_root / "report.json")
        != EXPECTED_MODEL_REPORT_SHA256
        or _model_artifact(model_root / "model")["combined_sha256"]
        != EXPECTED_MODEL_ARTIFACT_SHA256
    ):
        raise ValueError("next-round task or model provenance differs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_root / "model", use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_root / "model")
    model.to(device)
    model.eval()
    predictions = []
    seen = set()
    with torch.inference_mode():
        for selection in selections:
            juan = int(selection["juan"])
            task_path = tasks_root / str(selection["task"])
            if _sha256(task_path) != selection["task_sha256"]:
                raise ValueError(f"task hash differs: {juan}")
            task = json.loads(task_path.read_text(encoding="utf-8"))
            for jie in task["jies"]:
                jie_index = int(jie["jie_index"])
                key = (juan, jie_index)
                if key not in selected_jies or key in seen:
                    raise ValueError(f"prediction task inventory differs: {key}")
                seen.add(key)
                example = {
                    "id": f"juan-{juan:03d}-jie-{jie_index:04d}",
                    "text": jie["text"],
                    "labels": ["O"] * len(jie["text"]),
                    "segments": jie["segments"],
                }
                windows = build_windows(
                    tokenizer, example, max_length=512, stride=128
                )
                prediction_ids = []
                for window in windows:
                    inputs = {
                        "input_ids": torch.tensor(
                            [window.input_ids], dtype=torch.long, device=device
                        ),
                        "attention_mask": torch.tensor(
                            [window.attention_mask],
                            dtype=torch.long,
                            device=device,
                        ),
                    }
                    if window.token_type_ids is not None:
                        inputs["token_type_ids"] = torch.tensor(
                            [window.token_type_ids],
                            dtype=torch.long,
                            device=device,
                        )
                    prediction_ids.append(
                        model(**inputs).logits[0].argmax(dim=-1).cpu().tolist()
                    )
                labels, owned = merge_predictions(
                    example["text"], windows, prediction_ids
                )
                labels = constrain_predictions(example["text"], labels, owned)
                spans = labels_to_spans(example, labels, owned)
                predictions.append({
                    "id": example["id"],
                    "prediction_spans": [span.__dict__ for span in spans],
                })
    if seen != selected_jies:
        raise ValueError("not every selected jie was predicted")
    bundle = {
        "schema_version": 1,
        "status": "round3_model_omission_predictions",
        "diagnostic_only": True,
        "model_artifact_sha256": EXPECTED_MODEL_ARTIFACT_SHA256,
        "model_report_sha256": EXPECTED_MODEL_REPORT_SHA256,
        "source_manifest_sha256": _sha256(manifest_path),
        "git_commit": git_commit,
        "predictions": predictions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}-",
        dir=output_path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        temporary_path.write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.chmod(stat.S_IREAD)
        try:
            os.link(temporary_path, output_path)
        except FileExistsError:
            raise FileExistsError(
                f"prediction bundle was created concurrently: {output_path}"
            )
    finally:
        if temporary_path.exists():
            temporary_path.chmod(stat.S_IWRITE)
            temporary_path.unlink()
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen Round 3 omission checks on next assisted tasks."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = predict_tasks(args.tasks, args.model_root, args.output)
    print(json.dumps({
        "jies": len(bundle["predictions"]),
        "spans": sum(
            len(row["prediction_spans"]) for row in bundle["predictions"]
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
