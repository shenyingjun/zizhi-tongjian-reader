from __future__ import annotations

import argparse
import json
import shutil
import stat
import tempfile
from pathlib import Path

from production_review import _read, _sha256, _validate_teacher


def apply_negative_audit(
    review_root: Path,
    audit_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"negative-audit review output exists: {output_dir}")
    source_manifest_path = review_root / "manifest.json"
    source_manifest = _read(source_manifest_path)
    private_path = review_root / "private" / "selection.json"
    private = _read(private_path)
    if (
        source_manifest.get("schema_version") != 1
        or source_manifest.get("status") != "ml_production_focused_review"
        or source_manifest.get("candidate_model_blind") is not True
        or source_manifest.get("model_predictions_used") is not False
        or source_manifest.get("private_selection_sha256") != _sha256(private_path)
        or len(source_manifest.get("selected", [])) != 180
        or private.get("schema_version") != 1
        or private.get("status") != "ml_production_private_review_selection"
    ):
        raise ValueError("focused review source binding differs")
    task_ids = set(private.get("negative_audit_task_ids", []))
    selections = {
        str(row["task_id"]): row for row in source_manifest.get("selected", [])
    }
    if not task_ids or not task_ids.issubset(selections):
        raise ValueError("negative-audit task inventory differs")
    expected_names = {f"task_{task_id}.json" for task_id in task_ids}
    audit_paths = list(audit_root.glob("*.json"))
    audit_by_name = {path.name: path for path in audit_paths}
    if len(audit_paths) != len(audit_by_name) or set(audit_by_name) != expected_names:
        raise ValueError("negative-audit output inventory differs")

    validated = {}
    additions = 0
    for task_id in sorted(task_ids):
        selection = selections[task_id]
        task_path = review_root / str(selection["task"])
        task = _read(task_path)
        audit_path = audit_by_name[f"task_{task_id}.json"]
        payload = _read(audit_path)
        candidates = _validate_teacher(
            task,
            task_id,
            _sha256(task_path),
            payload,
            expected_pass="C-negative-recall-audit",
            expected_channel="copilot_independent_c",
            expected_phase="negative-audit",
        )
        validated[task_id] = (audit_path, candidates)
        additions += len(candidates)

    manifest = {
        **source_manifest,
        "status": "ml_production_focused_review_with_negative_audit",
        "source_review_manifest_sha256": _sha256(source_manifest_path),
        "negative_audit_inventory": {},
        "counts": {
            **source_manifest["counts"],
            "candidate_union": (
                int(source_manifest["counts"]["candidate_union"]) + additions
            ),
            "negative_audit_review": additions,
        },
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        for name in ("tasks", "review", "private"):
            shutil.copytree(review_root / name, staging / name)
        for task_id, (audit_path, candidates) in validated.items():
            selection = selections[task_id]
            pack_path = staging / str(selection["review"])
            pack = _read(pack_path)
            if pack.get("candidates") or pack.get("initial_annotations"):
                raise ValueError(
                    f"negative-audit source was not A/B-negative: {task_id}"
                )
            rows = []
            for geometry, source in sorted(candidates.items()):
                para_id, start, end = geometry
                teacher_reason = str(source.get("review_reason") or "").strip()
                reason = "Independent negative-jie recall audit found this candidate."
                if teacher_reason:
                    reason += f" {teacher_reason}"
                rows.append({
                    "id": f"copilot:{task_id}:{para_id}:{start}:{end}",
                    "para_id": para_id,
                    "start": start,
                    "end": end,
                    "surface": source["surface"],
                    "channels": ["copilot_independent_c"],
                    "confidence": "low",
                    "review_reason": reason,
                    "pass_confidence": {
                        "a": None,
                        "b": None,
                        "c": source["confidence"],
                    },
                })
            pack["human_review_scope"] = (
                "teacher_disagreement_explicit_low_consensus_audit_and_"
                "negative_jie_recall_audit"
            )
            pack["candidates"] = rows
            pack["negative_audit_sha256"] = _sha256(audit_path)
            pack_path.chmod(stat.S_IWRITE)
            pack_path.write_text(
                json.dumps(pack, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            selection["review_sha256"] = _sha256(pack_path)
            selection["review_candidates"] = len(rows)
            manifest["negative_audit_inventory"][task_id] = {
                "sha256": _sha256(audit_path),
                "candidates": len(rows),
            }
        manifest["private_selection_sha256"] = _sha256(
            staging / "private" / "selection.json"
        )
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in staging.rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge independent negative-jie audit into focused review."
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = apply_negative_audit(args.review, args.audit, args.output)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
