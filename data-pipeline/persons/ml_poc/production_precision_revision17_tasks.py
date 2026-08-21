from __future__ import annotations

import argparse
import hashlib
import json
import math
import secrets
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from p3_compact import _git_commit_clean
from production_precision_revision17_candidates import CANDIDATE_STATUS
from production_precision_revision17_mining_stage1 import MINING_STAGE1_STATUS
from production_precision_revision17_plan import PLAN_STATUS, SEED, _read, _sha256
from production_train import _make_read_only


REVISION = 17
TASK_STATUS = "ml_production_precision_revision17_review_task"
TASKS_STATUS = "ml_production_precision_revision17_review_tasks"
LABELS = ("exact_person", "wrong_boundary", "not_person", "uncertain")
SALT_BYTES = 32
STRATUM_LIMITS = (
    ("hard_disagreement", 600),
    ("low_stage1", 300),
    ("supported_control", 300),
)
MAX_PER_JIE = 4
MAX_PER_JUAN = 40
MIN_TASKS = 800
MIN_JUANS = 20
MIN_STRATUM_JUANS = 10


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _geometry(row: dict) -> tuple[str, int, int, int]:
    return (
        str(row["id"]),
        int(row["para_id"]),
        int(row["start"]),
        int(row["end"]),
    )


def _digest(row: dict, domain: str) -> str:
    payload = "\x1f".join(
        (
            str(SEED),
            domain,
            str(row["id"]),
            str(int(row["para_id"])),
            str(int(row["start"])),
            str(int(row["end"])),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _score(row: dict) -> np.float32:
    value = np.float32(float(row["stage1_probability"]))
    if not np.isfinite(value):
        raise ValueError("Revision-17 candidate score is not finite")
    return value


def _confidence(row: dict) -> float:
    value = float(row["maximum_generator_confidence"])
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("Revision-17 candidate confidence differs")
    return value


def select_candidates(rows: list[dict]) -> list[dict]:
    by_geometry = {_geometry(row): row for row in rows}
    if len(by_geometry) != len(rows):
        raise ValueError("Revision-17 scored candidates contain duplicate geometry")
    for row in rows:
        support = int(row["generator_support"])
        if support not in (1, 2, 3):
            raise ValueError("Revision-17 candidate support differs")
        _score(row)
        _confidence(row)

    def rank(stratum: str, row: dict):
        score = float(_score(row))
        support = int(row["generator_support"])
        confidence = _confidence(row)
        digest = _digest(row, "selection")
        if stratum == "hard_disagreement":
            return support, confidence, -score, digest
        return -support, -confidence, -score, digest

    selected: list[dict] = []
    selected_keys = set()
    jie_counts: Counter[tuple[int, int]] = Counter()
    juan_counts: Counter[int] = Counter()
    for stratum, limit in STRATUM_LIMITS:
        if stratum == "hard_disagreement":
            eligible = [
                row for row in rows if _score(row) >= np.float32(0.50)
            ]
        elif stratum == "low_stage1":
            eligible = [
                row for row in rows if _score(row) < np.float32(0.50)
            ]
        else:
            eligible = list(rows)
        accepted = 0
        for row in sorted(eligible, key=lambda item: rank(stratum, item)):
            key = _geometry(row)
            jie = int(row["juan"]), int(row["jie_index"])
            juan = int(row["juan"])
            if (
                key in selected_keys
                or jie_counts[jie] >= MAX_PER_JIE
                or juan_counts[juan] >= MAX_PER_JUAN
            ):
                continue
            selected.append({**row, "selection_stratum": stratum})
            selected_keys.add(key)
            jie_counts[jie] += 1
            juan_counts[juan] += 1
            accepted += 1
            if accepted == limit:
                break
    return selected


def _opaque_id(salt: bytes, domain: str, row: dict) -> str:
    return hashlib.sha256(
        salt + b"\0" + _digest(row, domain).encode("ascii")
    ).hexdigest()[:24]


def _validate_task(task: dict) -> None:
    if set(task) != {
        "schema_version",
        "status",
        "phase",
        "task_id",
        "review_scope",
        "protocol",
        "jie",
        "candidate",
        "allowed_labels",
    }:
        raise ValueError("Revision-17 task fields differ")
    if set(task["jie"]) != {"text", "segments"}:
        raise ValueError("Revision-17 task jie fields differ")
    if set(task["candidate"]) != {
        "candidate_id",
        "para_id",
        "start",
        "end",
        "surface",
    }:
        raise ValueError("Revision-17 task candidate fields differ")
    if tuple(task["allowed_labels"]) != LABELS:
        raise ValueError("Revision-17 task labels differ")


def freeze_tasks(
    plan_root: Path,
    candidate_root: Path,
    stage1_root: Path,
    output_dir: Path,
    *,
    salt: bytes | None = None,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Revision-17 tasks exist: {output_dir}")
    salt = secrets.token_bytes(SALT_BYTES) if salt is None else salt
    if len(salt) != SALT_BYTES:
        raise ValueError("Revision-17 task salt must contain 32 bytes")

    plan_manifest_path = plan_root / "manifest.json"
    mining_path = plan_root / "mining.jsonl"
    candidate_manifest_path = candidate_root / "manifest.json"
    candidates_path = candidate_root / "candidates.jsonl"
    stage1_manifest_path = stage1_root / "manifest.json"
    scores_path = stage1_root / "scored-candidates.jsonl"
    plan = _read(plan_manifest_path)
    candidate_manifest = _read(candidate_manifest_path)
    stage1_manifest = _read(stage1_manifest_path)
    if (
        plan.get("status") != PLAN_STATUS
        or plan.get("confirmation_read") is not False
        or plan.get("formal_reserve_text_read") is not False
        or plan.get("outputs", {}).get("mining_sha256") != _sha256(mining_path)
        or candidate_manifest.get("status") != CANDIDATE_STATUS
        or candidate_manifest.get("bindings", {}).get("plan_manifest_sha256")
        != _sha256(plan_manifest_path)
        or candidate_manifest.get("outputs", {}).get("candidates_sha256")
        != _sha256(candidates_path)
        or stage1_manifest.get("status") != MINING_STAGE1_STATUS
        or stage1_manifest.get("confirmation_read") is not False
        or stage1_manifest.get("formal_reserve_text_read") is not False
        or stage1_manifest.get("new_judgments_read") is not False
        or stage1_manifest.get("bindings", {}).get("candidate_manifest_sha256")
        != _sha256(candidate_manifest_path)
        or stage1_manifest.get("bindings", {}).get("candidates_sha256")
        != _sha256(candidates_path)
        or stage1_manifest.get("outputs", {}).get("scores_sha256")
        != _sha256(scores_path)
    ):
        raise ValueError("Revision-17 task source binding differs")

    example_rows = _read_jsonl(mining_path)
    examples = {str(row["id"]): row for row in example_rows}
    scored = _read_jsonl(scores_path)
    if len(examples) != len(example_rows) or not scored:
        raise ValueError("Revision-17 task source inventory differs")
    selected = select_candidates(scored)
    stratum_juans = {
        stratum: {
            int(row["juan"])
            for row in selected
            if row["selection_stratum"] == stratum
        }
        for stratum, _ in STRATUM_LIMITS
    }
    if (
        len(selected) < MIN_TASKS
        or len({int(row["juan"]) for row in selected}) < MIN_JUANS
        or any(
            0 < sum(row["selection_stratum"] == stratum for row in selected)
            and len(stratum_juans[stratum]) < MIN_STRATUM_JUANS
            for stratum, _ in STRATUM_LIMITS
        )
    ):
        raise ValueError("Revision-17 selected review inventory is insufficient")

    prepared = []
    for row in selected:
        example = examples.get(str(row["id"]))
        if example is None:
            raise ValueError("Revision-17 selected example is absent")
        candidate_id = _opaque_id(salt, "candidate", row)
        task_id = _opaque_id(salt, "task", row)
        prepared.append((row, example, candidate_id, task_id))
    if (
        len({item[2] for item in prepared}) != len(prepared)
        or len({item[3] for item in prepared}) != len(prepared)
    ):
        raise ValueError("Revision-17 opaque ID collision")
    prepared.sort(key=lambda item: _opaque_id(salt, "order", item[0]))

    git_commit = _git_commit_clean()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        tasks_dir = staging / "reviewer-tasks"
        sealed_dir = staging / "sealed-selection"
        tasks_dir.mkdir()
        sealed_dir.mkdir()
        task_hashes = []
        selection_rows = []
        for order, (row, example, candidate_id, task_id) in enumerate(prepared):
            task = {
                "schema_version": 1,
                "status": TASK_STATUS,
                "phase": "revision-17-blind-semantic-negative-mining",
                "task_id": task_id,
                "review_scope": "current-numbered-jie-only",
                "protocol": {
                    "decision": (
                        "Judge only whether the marked source occurrence is an "
                        "exact individual-person span, a wrong-boundary person "
                        "span, or not a person."
                    ),
                    "evidence": "Use only the complete numbered jie in this task.",
                    "independence": (
                        "Return one first judgment without seeking sibling tasks."
                    ),
                },
                "jie": {
                    "text": str(example["text"]),
                    "segments": [
                        {
                            "para_id": int(segment["para_id"]),
                            "assembled_start": int(segment["assembled_start"]),
                            "assembled_end": int(segment["assembled_end"]),
                        }
                        for segment in example["segments"]
                    ],
                },
                "candidate": {
                    "candidate_id": candidate_id,
                    "para_id": int(row["para_id"]),
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "surface": str(row["surface"]),
                },
                "allowed_labels": list(LABELS),
            }
            _validate_task(task)
            task_path = tasks_dir / f"task_{task_id}.json"
            task_path.write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            task_hashes.append({
                "task_id": task_id,
                "task_sha256": _sha256(task_path),
            })
            selection_rows.append({
                "random_order": order,
                "task_id": task_id,
                "candidate_id": candidate_id,
                **row,
            })

        hashes_path = staging / "task-hashes.jsonl"
        _write_jsonl(
            hashes_path,
            sorted(task_hashes, key=lambda row: row["task_id"]),
        )
        selection_path = sealed_dir / "selection.jsonl"
        _write_jsonl(selection_path, selection_rows)
        salt_path = sealed_dir / "salt.txt"
        salt_path.write_text(salt.hex() + "\n", encoding="ascii")
        manifest = {
            "schema_version": 1,
            "status": TASKS_STATUS,
            "revision": REVISION,
            "fit_only": True,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "formal_reserve_text_read": False,
            "one_candidate_per_task": True,
            "reviewer_progress_disclosure": False,
            "git_commit": git_commit,
            "bindings": {
                "plan_manifest_sha256": _sha256(plan_manifest_path),
                "mining_sha256": _sha256(mining_path),
                "candidate_manifest_sha256": _sha256(
                    candidate_manifest_path
                ),
                "candidates_sha256": _sha256(candidates_path),
                "stage1_manifest_sha256": _sha256(stage1_manifest_path),
                "scores_sha256": _sha256(scores_path),
            },
            "selector": {
                "seed": SEED,
                "threshold_type": "numpy.float32",
                "threshold": 0.5,
                "stratum_limits": dict(STRATUM_LIMITS),
                "max_per_jie": MAX_PER_JIE,
                "max_per_juan": MAX_PER_JUAN,
            },
            "counts": {
                "tasks": len(selection_rows),
                "juans": len({int(row["juan"]) for row in selected}),
                "strata": {
                    stratum: sum(
                        row["selection_stratum"] == stratum for row in selected
                    )
                    for stratum, _ in STRATUM_LIMITS
                },
                "stratum_juans": {
                    stratum: len(stratum_juans[stratum])
                    for stratum, _ in STRATUM_LIMITS
                },
            },
            "outputs": {
                "task_hashes_sha256": _sha256(hashes_path),
                "selection_sha256": _sha256(selection_path),
                "salt_sha256": _sha256(salt_path),
            },
            "claim_limit": (
                "Candidate-blind fit-only training audit; not fresh "
                "generalization evidence."
            ),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze Revision-17 blind semantic-negative review tasks."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_tasks(
        args.plan, args.candidates, args.stage1, args.output
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
