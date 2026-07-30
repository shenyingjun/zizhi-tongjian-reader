from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import tempfile
from dataclasses import asdict
from pathlib import Path

from core import Span
from p1_windows import (
    build_windows,
    constrain_predictions,
    labels_to_spans,
    merge_predictions,
)
from p3_compact import (
    EXPECTED_MODEL_SHA256,
    FOREIGN_TERMS,
    MAX_CHARS,
    MIN_CHARS,
    ROLE_TERMS,
    _git_commit_clean,
)
from p3_compact_evaluate import _model_artifact
from pilot import RULES, TEXT, _load


PREVIOUSLY_USED_JUANS = {
    12, 13, 21, 24, 27, 37, 44, 46, 52, 76, 177, 201, 204, 205,
    207, 225, 248, 251,
}
STRATA = (
    ("model_rule_disagreement", 20),
    ("model_uncertainty", 20),
    ("role_appellation_density", 10),
    ("foreign_title_density", 10),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _score_key(row: dict, stratum: str) -> tuple:
    if stratum == "model_rule_disagreement":
        return (
            row["model_rule_symmetric_difference"],
            row["uncertainty"],
            row["role_count"] + row["foreign_count"],
            -row["characters"],
            -row["juan"],
            -row["jie_index"],
        )
    if stratum == "model_uncertainty":
        return (
            row["uncertainty"],
            row["model_rule_symmetric_difference"],
            row["role_count"] + row["foreign_count"],
            -row["characters"],
            -row["juan"],
            -row["jie_index"],
        )
    if stratum == "role_appellation_density":
        return (
            row["role_count"] / row["characters"],
            row["role_count"],
            row["model_rule_symmetric_difference"],
            row["uncertainty"],
            -row["characters"],
            -row["juan"],
            -row["jie_index"],
        )
    if stratum == "foreign_title_density":
        return (
            row["foreign_count"] / row["characters"],
            row["foreign_count"],
            row["model_rule_symmetric_difference"],
            row["uncertainty"],
            -row["characters"],
            -row["juan"],
            -row["jie_index"],
        )
    raise ValueError(f"unknown active-learning stratum: {stratum}")


def select_active_rows(rows: list[dict]) -> list[dict]:
    selected = []
    used_juans = set()
    used_jies = set()
    for stratum, count in STRATA:
        ranked = sorted(
            rows,
            key=lambda row: _score_key(row, stratum),
            reverse=True,
        )
        chosen = []
        for row in ranked:
            if (
                row["juan"] in used_juans
                or (row["juan"], row["jie_index"]) in used_jies
                or (
                    stratum in {
                        "model_rule_disagreement",
                        "role_appellation_density",
                        "foreign_title_density",
                    }
                    and _score_key(row, stratum)[0] <= 0
                )
            ):
                continue
            chosen.append(row)
            used_juans.add(int(row["juan"]))
            used_jies.add((int(row["juan"]), int(row["jie_index"])))
            if len(chosen) == count:
                break
        if len(chosen) < count:
            raise ValueError(f"not enough rows for {stratum}")
        for row in chosen:
            selected.append({"selection_stratum": stratum, **row})
    if len(selected) != sum(count for _, count in STRATA):
        raise AssertionError("active selection count differs")
    return selected


def _eligible_rows(excluded_juans: set[int]) -> list[dict]:
    from core import assemble_jies

    rows = []
    for juan in range(1, 295):
        if juan in excluded_juans:
            continue
        source_path = TEXT / f"juan_{juan:03d}.json"
        if not source_path.is_file():
            continue
        source = _load(source_path)
        for jie in assemble_jies(source["paragraphs"]):
            if (
                jie.number is None
                or not MIN_CHARS <= len(jie.text) <= MAX_CHARS
            ):
                continue
            rows.append({
                "juan": juan,
                "jie_index": int(jie.index),
                "jie_number": jie.number,
                "text": jie.text,
                "segments": [asdict(segment) for segment in jie.segments],
                "characters": len(jie.text),
                "role_count": sum(
                    jie.text.count(term) for term in ROLE_TERMS
                ),
                "foreign_count": sum(
                    jie.text.count(term) for term in FOREIGN_TERMS
                ),
            })
    return rows


def _infer_rows(rows: list[dict], model_dir: Path) -> None:
    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
    )

    if _sha256(model_dir / "model.safetensors") != EXPECTED_MODEL_SHA256:
        raise ValueError("active selection requires frozen Round 2 model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    rule_documents = {}
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            example = {
                "id": (
                    f"juan-{row['juan']:03d}-"
                    f"jie-{row['jie_index']:04d}"
                ),
                "text": row["text"],
                "labels": ["O"] * len(row["text"]),
                "segments": row["segments"],
            }
            windows = build_windows(
                tokenizer, example, max_length=512, stride=128
            )
            prediction_ids = []
            uncertainty_values = []
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
                logits = model(**inputs).logits[0]
                probabilities = logits.softmax(dim=-1)
                prediction_ids.append(
                    probabilities.argmax(dim=-1).cpu().tolist()
                )
                for token_index, owned in enumerate(window.owned_tokens):
                    if not owned:
                        continue
                    token_probs = probabilities[token_index]
                    entropy = -float(
                        (token_probs * token_probs.clamp_min(1e-12).log()).sum()
                    ) / math.log(probabilities.shape[-1])
                    start, end = window.offsets[token_index]
                    uncertainty_values.append(entropy)
            labels, owned = merge_predictions(
                row["text"], windows, prediction_ids
            )
            labels = constrain_predictions(row["text"], labels, owned)
            model_spans = labels_to_spans(example, labels, owned)
            row["model_predictions"] = [
                span.__dict__ for span in model_spans
            ]
            uncertainty_values.sort(reverse=True)
            top_count = max(1, min(20, len(uncertainty_values)))
            row["uncertainty"] = (
                sum(uncertainty_values[:top_count]) / top_count
                if uncertainty_values else 0.0
            )
            juan = int(row["juan"])
            if juan not in rule_documents:
                rule_documents[juan] = _load(
                    RULES / f"juan_{juan:03d}.json"
                )
            paragraphs = {}
            for segment in row["segments"]:
                start = int(segment["assembled_start"])
                end = int(segment["assembled_end"])
                paragraphs[int(segment["para_id"])] = row["text"][start:end]
            rule_spans = set()
            for occurrence in rule_documents[juan].get("occurrences", []):
                para_id = int(occurrence["para_id"])
                if (
                    occurrence.get("field", "main") != "main"
                    or para_id not in paragraphs
                ):
                    continue
                start = int(occurrence["start"])
                end = int(occurrence["end"])
                paragraph = paragraphs[para_id]
                if not 0 <= start < end <= len(paragraph):
                    raise ValueError("rule geometry is outside paragraph")
                rule_spans.add(Span(
                    para_id, start, end, paragraph[start:end]
                ))
            row["model_rule_symmetric_difference"] = len(
                set(model_spans) ^ rule_spans
            )
            if index % 500 == 0:
                print(json.dumps({
                    "inferred": index,
                    "total": len(rows),
                }))


def prepare_active_round(
    model_dir: Path,
    compact_manifest_path: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"active round exists: {output_dir}")
    git_commit = _git_commit_clean()
    compact_manifest_bytes = compact_manifest_path.read_bytes()
    compact_manifest_sha256 = hashlib.sha256(
        compact_manifest_bytes
    ).hexdigest()
    compact_manifest = json.loads(compact_manifest_bytes)
    model_artifact = _model_artifact(model_dir)
    if (
        model_artifact["files"].get("model.safetensors")
        != EXPECTED_MODEL_SHA256
    ):
        raise ValueError("active selection model artifact differs")
    private_compact_juans = {
        int(row["juan"])
        for row in compact_manifest.get("private_selected_jies", [])
    }
    public_compact_juans = {
        int(row["juan"])
        for row in compact_manifest.get("selected", [])
    }
    compact_juans = private_compact_juans | public_compact_juans
    if (
        compact_manifest.get("formal_p3") is not True
        or not compact_juans
        or private_compact_juans - public_compact_juans
    ):
        raise ValueError("compact manifest has invalid selection exclusions")
    excluded = PREVIOUSLY_USED_JUANS | compact_juans
    rows = _eligible_rows(excluded)
    _infer_rows(rows, model_dir)
    if (
        _model_artifact(model_dir) != model_artifact
        or _sha256(compact_manifest_path) != compact_manifest_sha256
    ):
        raise RuntimeError("active selection inputs changed during inference")
    selected = select_active_rows(rows)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        tasks_dir = staging / "tasks"
        seeds_dir = staging / "ml-seeds"
        tasks_dir.mkdir()
        seeds_dir.mkdir()
        public = []
        private = []
        for row in selected:
            juan = int(row["juan"])
            task = {
                "schema_version": 1,
                "phase": "assisted",
                "juan": juan,
                "instructions": (
                    "Copilot teacher must independently mark every main-text "
                    "person span in this jie, then compare with the ML seed. "
                    "Use jie-local evidence only."
                ),
                "jies": [{
                    "jie_index": row["jie_index"],
                    "jie_number": row["jie_number"],
                    "text": row["text"],
                    "segments": row["segments"],
                    "annotations": [],
                }],
            }
            task_path = tasks_dir / f"blind_juan_{juan:03d}.json"
            task_path.write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            seed = {
                "schema_version": 1,
                "phase": "assisted",
                "juan": juan,
                "candidates": [
                    {
                        "id": (
                            f"{span['para_id']}:{span['start']}:{span['end']}"
                        ),
                        **span,
                        "channels": ["round2_ml_constrained"],
                    }
                    for span in row["model_predictions"]
                ],
                "candidate_model": {
                    "sha256": EXPECTED_MODEL_SHA256,
                },
            }
            seed_path = seeds_dir / f"assisted_juan_{juan:03d}.json"
            seed_path.write_text(
                json.dumps(seed, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            public.append({
                "juan": juan,
                "mode": "teacher_task",
                "task": task_path.name,
                "task_sha256": _sha256(task_path),
                "ml_seed": seed_path.name,
                "ml_seed_sha256": _sha256(seed_path),
            })
            private.append({
                key: value for key, value in row.items()
                if key not in {
                    "text", "segments", "model_predictions",
                }
            })
        manifest = {
            "schema_version": 1,
            "status": "round3_active_learning_teacher_tasks",
            "formal_evaluation": False,
            "eligible_for_training_after_human_review": True,
            "selection_policy": dict(STRATA),
            "unique_juan_required": True,
            "excluded_juans": sorted(excluded),
            "model_sha256": EXPECTED_MODEL_SHA256,
            "model_artifact": model_artifact,
            "compact_manifest_sha256": compact_manifest_sha256,
            "git_commit": git_commit,
            "selected": public,
            "private_selection": private,
        }
        manifest_path = tasks_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for directory in (tasks_dir, seeds_dir):
            for path in directory.iterdir():
                path.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select 60 high-information jies for Round 3 teachers."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--compact-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_active_round(
        args.model, args.compact_manifest, args.output
    )
    print(json.dumps({
        "selected_jies": len(manifest["selected"]),
        "strata": manifest["selection_policy"],
        "unique_juans": len({
            row["juan"] for row in manifest["selected"]
        }),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
