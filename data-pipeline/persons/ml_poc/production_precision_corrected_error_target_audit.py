from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_corrected_error_audit import (
    ARTIFACT_STATUS as ERROR_ARTIFACT_STATUS,
    TASK_STATUS as ERROR_TASK_STATUS,
)
from production_precision_corrected_error_freeze import FROZEN_STATUS
from production_precision_grouped_verifier import _read, _read_jsonl, _sha256
from production_train import _make_read_only


REVISION = 14
TASK_STATUS = "ml_production_precision_corrected_error_target_task"
ARTIFACT_STATUS = "ml_production_precision_corrected_error_target_tasks"
EXPECTED_TASKS = 6
SALT_BYTES = 32
DECISIONS = ("targets", "uncertain")


def _opaque_id(salt: bytes, domain: str, task_id: str, candidate_id: str) -> str:
    payload = f"{domain}\x1f{task_id}\x1f{candidate_id}".encode("ascii")
    return hashlib.sha256(salt + b"\0" + payload).hexdigest()[:24]


def _validate_public_task(task: dict) -> None:
    if set(task) != {
        "schema_version",
        "status",
        "phase",
        "task_id",
        "review_scope",
        "protocol",
        "jie",
        "source_candidate",
        "allowed_decisions",
    }:
        raise ValueError("corrected error target task fields differ")
    if set(task["protocol"]) != {"decision", "evidence", "independence"}:
        raise ValueError("corrected error target protocol fields differ")
    if set(task["jie"]) != {"text", "segments"}:
        raise ValueError("corrected error target jie fields differ")
    if set(task["source_candidate"]) != {
        "candidate_id",
        "para_id",
        "start",
        "end",
        "surface",
    }:
        raise ValueError("corrected error target candidate fields differ")
    if tuple(task["allowed_decisions"]) != DECISIONS:
        raise ValueError("corrected error target decisions differ")


def build_tasks(
    error_task_root: Path,
    frozen_root: Path,
    output_dir: Path,
    *,
    salt: bytes | None = None,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"corrected error target output exists: {output_dir}")
    salt = secrets.token_bytes(SALT_BYTES) if salt is None else salt
    if len(salt) != SALT_BYTES:
        raise ValueError("corrected error target salt must contain 32 bytes")

    error_manifest_path = error_task_root / "manifest.json"
    error_hashes_path = error_task_root / "task-hashes.jsonl"
    frozen_manifest_path = frozen_root / "manifest.json"
    decisions_path = frozen_root / "decisions.jsonl"
    error_manifest = _read(error_manifest_path)
    frozen_manifest = _read(frozen_manifest_path)
    if error_manifest.get("status") != ERROR_ARTIFACT_STATUS:
        raise ValueError("corrected error target source task status differs")
    if (
        frozen_manifest.get("status") != FROZEN_STATUS
        or frozen_manifest.get("confirmation_read") is not False
        or int(frozen_manifest.get("counts", {}).get("uncertain", -1)) != 0
        or int(frozen_manifest.get("counts", {}).get("wrong_boundary", -1))
        != EXPECTED_TASKS
        or frozen_manifest.get("bindings", {}).get("task_manifest_sha256")
        != _sha256(error_manifest_path)
        or frozen_manifest.get("outputs", {}).get("decisions_sha256")
        != _sha256(decisions_path)
        or error_manifest.get("outputs", {}).get("task_hashes_sha256")
        != _sha256(error_hashes_path)
    ):
        raise ValueError("corrected error target frozen binding differs")
    task_hashes = {
        str(row["task_id"]): str(row["task_sha256"])
        for row in _read_jsonl(error_hashes_path)
    }
    wrong = [
        row for row in _read_jsonl(decisions_path)
        if row["label"] == "wrong_boundary"
    ]
    if len(wrong) != EXPECTED_TASKS:
        raise ValueError("corrected error target decision inventory differs")

    prepared = []
    for decision in wrong:
        source_task_id = str(decision["task_id"])
        source_path = (
            error_task_root / "reviewer-tasks" / f"task_{source_task_id}.json"
        )
        source_task = _read(source_path)
        if (
            source_task.get("status") != ERROR_TASK_STATUS
            or source_task.get("task_id") != source_task_id
            or _sha256(source_path) != task_hashes.get(source_task_id)
            or source_task.get("candidate", {}).get("candidate_id")
            != decision["candidate_id"]
            or decision["task_sha256"] != task_hashes.get(source_task_id)
        ):
            raise ValueError("corrected error target source task differs")
        target_task_id = _opaque_id(
            salt, "revision-14-target-task", source_task_id, decision["candidate_id"]
        )
        target_candidate_id = _opaque_id(
            salt,
            "revision-14-target-candidate",
            source_task_id,
            decision["candidate_id"],
        )
        prepared.append((
            hashlib.sha256(
                salt + b"\0order\0" + source_task_id.encode("ascii")
            ).hexdigest(),
            source_task_id,
            source_task,
            target_task_id,
            target_candidate_id,
        ))
    prepared.sort(key=lambda item: item[0])

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        tasks_dir = staging / "reviewer-tasks"
        tasks_dir.mkdir()
        sealed_dir = staging / "sealed-mapping"
        sealed_dir.mkdir()
        selected = []
        mapping = []
        for order, (
            _,
            source_task_id,
            source_task,
            task_id,
            candidate_id,
        ) in enumerate(prepared):
            source_candidate = source_task["candidate"]
            task = {
                "schema_version": 1,
                "status": TASK_STATUS,
                "phase": "revision-14-blind-exact-target-audit",
                "task_id": task_id,
                "review_scope": "current-numbered-jie-only",
                "protocol": {
                    "decision": (
                        "Return every source-exact complete person span that corrects "
                        "the marked non-exact occurrence, or uncertain."
                    ),
                    "evidence": "Use only the complete numbered jie in this task.",
                    "independence": (
                        "Return one first judgment without seeking sibling tasks."
                    ),
                },
                "jie": source_task["jie"],
                "source_candidate": {
                    "candidate_id": candidate_id,
                    "para_id": int(source_candidate["para_id"]),
                    "start": int(source_candidate["start"]),
                    "end": int(source_candidate["end"]),
                    "surface": str(source_candidate["surface"]),
                },
                "allowed_decisions": list(DECISIONS),
            }
            _validate_public_task(task)
            path = tasks_dir / f"task_{task_id}.json"
            path.write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            selected.append({
                "task_id": task_id,
                "task_sha256": _sha256(path),
            })
            mapping.append({
                "random_order": order,
                "task_id": task_id,
                "candidate_id": candidate_id,
                "source_task_id": source_task_id,
                "source_candidate_id": str(source_candidate["candidate_id"]),
            })
        selected_path = staging / "task-hashes.jsonl"
        selected_path.write_text(
            "".join(
                json.dumps(row, separators=(",", ":")) + "\n"
                for row in sorted(selected, key=lambda item: item["task_id"])
            ),
            encoding="utf-8",
        )
        mapping_path = sealed_dir / "mapping.jsonl"
        mapping_path.write_text(
            "".join(
                json.dumps(row, separators=(",", ":")) + "\n" for row in mapping
            ),
            encoding="utf-8",
        )
        salt_path = sealed_dir / "salt.txt"
        salt_path.write_text(salt.hex() + "\n", encoding="ascii")
        manifest = {
            "schema_version": 1,
            "status": ARTIFACT_STATUS,
            "revision": REVISION,
            "formal_grade": False,
            "fit_only": True,
            "confirmation_read": False,
            "one_candidate_per_task": True,
            "prior_judgments_hidden": True,
            "prior_rationales_hidden": True,
            "git_commit": _git_commit_clean(),
            "bindings": {
                "error_task_manifest_sha256": _sha256(error_manifest_path),
                "frozen_manifest_sha256": _sha256(frozen_manifest_path),
                "decisions_sha256": _sha256(decisions_path),
            },
            "counts": {"tasks": EXPECTED_TASKS},
            "outputs": {
                "task_hashes_sha256": _sha256(selected_path),
                "mapping_sha256": _sha256(mapping_path),
                "salt_sha256": _sha256(salt_path),
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build blind Revision-14 exact-target tasks."
    )
    parser.add_argument("--error-task-root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_tasks(
        args.error_task_root, args.frozen_root, args.output_dir
    ), indent=2))


if __name__ == "__main__":
    main()
