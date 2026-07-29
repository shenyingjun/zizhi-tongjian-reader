from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from pilot import TEXT, _load, build_blind_task
from p1_train import evaluate


SEED = 20260728
EXCLUDED_JUANS = {13, 27, 52}
ANAPHORA_TERMS = ("帝", "上", "后")
ROLE_TERMS = ("太子", "太后", "将军", "皇帝", "皇后", "单于")
FOREIGN_TERMS = ("单于", "可汗", "谷蠡王", "左贤王", "右贤王")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_text(source: dict) -> str:
    return "".join(
        str(row.get("main", "") or "") for row in source["paragraphs"]
    )


def _term_score(text: str, terms: tuple[str, ...]) -> int:
    return sum(text.count(term) for term in terms)


def select_expansion(
    sources: dict[int, dict],
    *,
    seed: int = SEED,
) -> list[tuple[str, str, int]]:
    available = sorted(set(sources) - EXCLUDED_JUANS)
    if len(available) < 5:
        raise ValueError("at least five unconsumed juans are required")
    rng = random.Random(seed)
    blind_anchor = rng.choice(available)
    available.remove(blind_anchor)
    assisted_random = rng.choice(available)
    available.remove(assisted_random)
    texts = {juan: _source_text(sources[juan]) for juan in available}

    def take_max(terms: tuple[str, ...]) -> int:
        juan = max(
            available,
            key=lambda value: (
                _term_score(texts[value], terms),
                len(texts[value]),
                -value,
            ),
        )
        available.remove(juan)
        return juan

    assisted_anaphora = take_max(ANAPHORA_TERMS)
    assisted_role = take_max(ROLE_TERMS)
    assisted_foreign = take_max(FOREIGN_TERMS)
    return [
        ("blind_anchor", "random_blind_anchor", blind_anchor),
        ("assisted", "random_assisted", assisted_random),
        ("assisted", "single_anaphora_challenge", assisted_anaphora),
        ("assisted", "role_appellation_challenge", assisted_role),
        ("assisted", "foreign_title_challenge", assisted_foreign),
    ]


def prepare_tasks(output_dir: Path) -> dict:
    sources = {}
    source_paths = {}
    for juan in range(1, 295):
        path = TEXT / f"juan_{juan:03d}.json"
        if path.is_file():
            sources[juan] = _load(path)
            source_paths[juan] = path
    selected = select_expansion(sources)
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode, role, juan in selected:
        task = build_blind_task(juan, sources[juan], role)
        task["instructions"] = (
            "Mark main-text person spans without candidate evidence."
            if mode == "blind_anchor"
            else (
                "Review Copilot-teacher candidates and add, remove, or correct "
                "person spans."
            )
        )
        path = tasks_dir / f"blind_juan_{juan:03d}.json"
        path.write_text(
            json.dumps(task, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        rows.append({
            "mode": mode,
            "role": role,
            "juan": juan,
            "source_sha256": _sha256(source_paths[juan]),
            "task": path.name,
        })
    manifest = {
        "schema_version": 2,
        "selection_seed": SEED,
        "candidate_sources": ["ml"],
        "v1_used": False,
        "rules_used": False,
        "identity_fields_present": False,
        "selected": rows,
    }
    (tasks_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_assisted_packs(
    manifest: dict,
    tasks_dir: Path,
    output_dir: Path,
    model_dir: Path,
    *,
    max_length: int = 512,
    stride: int = 128,
    batch_size: int = 2,
) -> list[Path]:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to(device)
    model_hash = _sha256(model_dir / "model.safetensors")
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for selection in manifest["selected"]:
        if selection["mode"] != "assisted":
            continue
        juan = int(selection["juan"])
        task = _load(tasks_dir / f"blind_juan_{juan:03d}.json")
        examples = [{
            "id": f"juan-{juan:03d}-jie-{int(jie['jie_index']):04d}",
            "text": jie["text"],
            "labels": ["O"] * len(jie["text"]),
            "segments": jie["segments"],
        } for jie in task["jies"]]
        _, predictions = evaluate(
            model,
            tokenizer,
            examples,
            device,
            max_length=max_length,
            stride=stride,
            batch_size=batch_size,
        )
        candidates = []
        for prediction in predictions:
            for row in prediction["prediction_spans"]:
                candidates.append({
                    "id": (
                        f"{row['para_id']}:{row['start']}:{row['end']}"
                    ),
                    **row,
                    "channels": ["ml_constrained"],
                })
        pack = {
            "schema_version": 1,
            "phase": "assisted",
            "juan": juan,
            "candidates": sorted(
                candidates,
                key=lambda row: (
                    row["para_id"], row["start"], row["end"]
                ),
            ),
            "note_evidence": [],
            "candidate_model": {
                "sha256": model_hash,
                "checkpoint_selection": "challenge-dev exact F1",
            },
            "provenance_contract": {
                "v1_used": False,
                "rules_used": False,
                "identity_fields_present": False,
                "candidate_scope": "current jie only",
                "human_review_required": True,
            },
        }
        path = output_dir / f"assisted_juan_{juan:03d}.json"
        path.write_text(
            json.dumps(pack, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build P2 round tasks and ML seed packs for a Copilot teacher."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_tasks(args.output)
    paths = build_assisted_packs(
        manifest,
        args.output / "tasks",
        args.output / "assisted",
        args.model,
    )
    print(json.dumps({
        "selected": manifest["selected"],
        "ml_seed_packs": [str(path) for path in paths],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
