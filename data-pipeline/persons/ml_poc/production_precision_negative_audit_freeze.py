from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_lexical_mining import (
    MINING_STATUS,
    _canonical_key,
    _sha256,
)
from production_precision_lexical_safe_review import SAFE_REVIEW_STATUS
from production_precision_negative_audit_server import (
    SafeNegativeAuditStore,
)
from production_train import _make_read_only


FROZEN_STATUS = "ml_production_precision_safe_negatives_frozen"
EXPECTED_AUDIT = 299
EXPECTED_SAFE = 3126
EXPECTED_JUANS = 28


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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


def freeze_safe_negatives(
    review_root: Path,
    state_root: Path,
    mining_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"safe-negative freeze exists: {output_dir}")
    store = SafeNegativeAuditStore(review_root, state_root)
    review_manifest_path = review_root / "manifest.json"
    review_manifest = _read(review_manifest_path)
    routing_path = review_root / "safe-routing.jsonl"
    routing = _read_jsonl(routing_path)
    if (
        review_manifest.get("status") != SAFE_REVIEW_STATUS
        or review_manifest.get("revision") != 9
        or review_manifest.get("confirmation_read") is not False
        or review_manifest.get("safe_routing_sha256") != _sha256(routing_path)
        or int(review_manifest.get("counts", {}).get("safe_candidates", -1))
        != EXPECTED_SAFE
        or int(review_manifest.get("counts", {}).get("audit_candidates", -1))
        != EXPECTED_AUDIT
    ):
        raise ValueError("safe-negative review binding differs")

    route_by_key = {
        (
            int(row["juan"]),
            int(row["jie_index"]),
            int(row["para_id"]),
            int(row["start"]),
            int(row["end"]),
        ): row
        for row in routing
    }
    if len(route_by_key) != len(routing):
        raise ValueError("duplicate safe-negative routing geometry")
    safe_routes = {
        key: row for key, row in route_by_key.items() if not row["vetoes"]
    }
    audit_routes = {
        key: row for key, row in safe_routes.items()
        if row.get("audit_selected") is True
    }
    if (
        len(safe_routes) != EXPECTED_SAFE
        or len(audit_routes) != EXPECTED_AUDIT
        or len({key[0] for key in safe_routes}) != EXPECTED_JUANS
    ):
        raise ValueError("safe-negative routing counts differ")

    audited_keys = set()
    state_inventory = {}
    receipt_rows = []
    for task_id in store.order:
        task, _ = store._sources(task_id)
        state = store._load_state(task_id, task)
        state_path = store._state_path(task_id)
        if (
            state.get("complete") is not True
            or not isinstance(state.get("completion_receipt"), str)
            or len(state["completion_receipt"]) != 64
            or set(state["decisions"]) != {
                str(row["candidate_id"]) for row in task["candidates"]
            }
            or any(
                decision.get("initial") != "not_person"
                or decision.get("rationales_revealed") is not True
                or decision.get("final") != "not_person"
                for decision in state["decisions"].values()
            )
        ):
            raise ValueError(f"incomplete or non-zero audit state: {task_id}")
        for candidate in task["candidates"]:
            key = (
                int(task["juan"]),
                int(task["jie_index"]),
                int(candidate["para_id"]),
                int(candidate["start"]),
                int(candidate["end"]),
            )
            if (
                key not in audit_routes
                or audit_routes[key]["candidate_id"]
                != candidate["candidate_id"]
            ):
                raise ValueError(f"audit candidate binding differs: {key}")
            audited_keys.add(key)
        state_inventory[task_id] = {
            "path": str(state_path.resolve()),
            "sha256": _sha256(state_path),
            "completion_receipt": state["completion_receipt"],
            "candidates": len(task["candidates"]),
        }
        receipt_rows.append({
            "task_id": task_id,
            "state_sha256": _sha256(state_path),
            "completion_receipt": state["completion_receipt"],
            "candidates": len(task["candidates"]),
        })
    if audited_keys != set(audit_routes):
        raise ValueError("completed audit inventory differs")

    mining_manifest_path = mining_root / "manifest.json"
    mining_manifest = _read(mining_manifest_path)
    retained_path = mining_root / "retained.jsonl"
    retained = _read_jsonl(retained_path)
    if (
        mining_manifest.get("status") != MINING_STATUS
        or mining_manifest.get("confirmation_read") is not False
        or mining_manifest.get("outputs", {}).get("retained_sha256")
        != _sha256(retained_path)
    ):
        raise ValueError("safe-negative mining binding differs")
    retained_by_key = {_canonical_key(row): row for row in retained}
    if len(retained_by_key) != len(retained):
        raise ValueError("duplicate lexical mining geometry")
    safe_rows = []
    for key in sorted(safe_routes):
        mined = retained_by_key.get(key)
        route = safe_routes[key]
        if (
            mined is None
            or mined["candidate_id"] != route["candidate_id"]
            or mined["surface"] != route["surface"]
        ):
            raise ValueError(f"safe-negative mining geometry differs: {key}")
        safe_rows.append({
            **mined,
            "label": 0,
            "provenance": "revision-9-cross-family-unanimous-zero-error-audit",
            "audit_selected": key in audit_routes,
        })

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        safe_path = staging / "safe-negatives.jsonl"
        receipts_path = staging / "audit-receipts.jsonl"
        _write_jsonl(safe_path, safe_rows)
        _write_jsonl(receipts_path, sorted(
            receipt_rows, key=lambda row: row["task_id"]
        ))
        manifest = {
            "schema_version": 1,
            "status": FROZEN_STATUS,
            "revision": 9,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "git_commit": _git_commit_clean(),
            "bindings": {
                "review_manifest_sha256": _sha256(review_manifest_path),
                "safe_routing_sha256": _sha256(routing_path),
                "mining_manifest_sha256": _sha256(mining_manifest_path),
                "mining_retained_sha256": _sha256(retained_path),
                "grouped_manifest_sha256": mining_manifest["bindings"][
                    "grouped_manifest_sha256"
                ],
                "revision6_manifest_sha256": mining_manifest["bindings"][
                    "revision6_manifest_sha256"
                ],
            },
            "counts": {
                "safe_negatives": len(safe_rows),
                "safe_juans": len({int(row["juan"]) for row in safe_rows}),
                "audit_candidates": len(audited_keys),
                "audit_tasks": len(receipt_rows),
                "audit_exclusions": 0,
            },
            "audit": {
                "one_sided_confidence": 0.95,
                "candidate_false_negative_upper_bound": 0.01,
                "observed_errors": 0,
            },
            "state_inventory": state_inventory,
            "outputs": {
                "safe_negatives_sha256": _sha256(safe_path),
                "audit_receipts_sha256": _sha256(receipts_path),
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the completed Revision-9 zero-error audit."
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--mining", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_safe_negatives(
        args.review, args.state, args.mining, args.output
    )
    print(json.dumps({
        "status": manifest["status"],
        "counts": manifest["counts"],
        "audit": manifest["audit"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
