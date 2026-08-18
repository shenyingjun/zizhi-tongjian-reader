from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import unicodedata
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_lexical_adjudication import _read, _teacher_decisions
from production_precision_lexical_final_review import (
    FINAL_REVIEW_STATUS,
    TEACHER_MODELS,
)
from production_precision_lexical_mining import _sha256
from production_precision_lexical_review import TASK_STATUS
from production_train import _make_read_only


SAFE_REVIEW_STATUS = "ml_production_precision_lexical_safe_negative_audit"
AUDIT_SALT = "revision-9-safe-negative-audit-v1"
AUDIT_SIZE = math.ceil(math.log(0.05) / math.log(0.99))
EXPECTED_UNANIMOUS = 3127
MIN_SAFE_NEGATIVES = 2000
EXPECTED_JUANS = 28
MAPPED_STATUSES = {
    "mapped_exact_paragraph",
    "mapped_exact_unique_jie",
    "mapped_translation_coreference_paragraph",
    "mapped_translation_expansion_paragraph",
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _audit_digest(task: dict, candidate: dict) -> str:
    key = ":".join(str(value) for value in (
        AUDIT_SALT,
        int(task["juan"]),
        int(task["jie_index"]),
        int(candidate["para_id"]),
        int(candidate["start"]),
        int(candidate["end"]),
    ))
    return hashlib.sha256(key.encode("ascii")).hexdigest()


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _paragraphs(task: dict, canonical: dict) -> dict[int, str]:
    canonical_paragraphs = {
        int(row["id"]): unicodedata.normalize("NFC", str(row["main"]))
        for row in canonical["paragraphs"]
    }
    jie = task["jie"]
    jie_text = unicodedata.normalize("NFC", str(jie["text"]))
    paragraphs = {}
    for segment in jie["segments"]:
        para_id = int(segment["para_id"])
        start = int(segment["assembled_start"])
        end = int(segment["assembled_end"])
        if para_id in paragraphs or not 0 <= start < end <= len(jie_text):
            raise ValueError("invalid lexical task paragraph geometry")
        text = jie_text[start:end]
        if canonical_paragraphs.get(para_id) != text:
            raise ValueError(
                f"lexical task differs from canonical paragraph: {para_id}"
            )
        paragraphs[para_id] = text
    return paragraphs


def _candidate_vetoes(
    task: dict,
    candidate: dict,
    paragraphs: dict[int, str],
    evidence: dict,
) -> list[str]:
    para_id = int(candidate["para_id"])
    start = int(candidate["start"])
    end = int(candidate["end"])
    paragraph = paragraphs.get(para_id)
    if (
        paragraph is None
        or not 0 <= start < end <= len(paragraph)
        or paragraph[start:end] != candidate["surface"]
    ):
        raise ValueError(f"invalid lexical candidate geometry: {candidate}")
    target = (start, end)
    vetoes = []
    evidence_paragraph = evidence.get("paragraphs", {}).get(str(para_id))
    if evidence_paragraph is not None:
        if (
            evidence_paragraph.get("text_sha256") != _text_sha256(paragraph)
            or int(evidence_paragraph.get("jie_index", -1))
            != int(task["jie_index"])
        ):
            raise ValueError(
                f"translation evidence paragraph binding differs: {para_id}"
            )
        for identity in evidence_paragraph.get("identities", []):
            for row in identity.get("candidates", []):
                mapped = str(row.get("mapping_status", ""))
                geometry = (int(row["start"]), int(row["end"]))
                if (
                    mapped in MAPPED_STATUSES
                    and paragraph[geometry[0]:geometry[1]] == row.get("surface")
                    and _overlaps(target, geometry)
                ):
                    vetoes.append("approved_translation_evidence_overlap")
                    break
            if vetoes:
                break
    return vetoes


def prepare_safe_review(
    task_root: Path,
    teacher_a_root: Path,
    teacher_b_root: Path,
    teacher_c_root: Path,
    teacher_d_root: Path,
    final_review_root: Path,
    translation_evidence_root: Path,
    canonical_text_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"safe review output exists: {output_dir}")
    task_manifest_path = task_root / "manifest.json"
    task_manifest = _read(task_manifest_path)
    final_manifest_path = final_review_root / "manifest.json"
    final_manifest = _read(final_manifest_path)
    final_routing_path = final_review_root / "final-routing.jsonl"
    if (
        task_manifest.get("status") != TASK_STATUS
        or final_manifest.get("status") != FINAL_REVIEW_STATUS
        or final_manifest.get("revision") != 8
        or final_manifest.get("confirmation_read") is not False
        or final_manifest.get("final_routing_sha256")
        != _sha256(final_routing_path)
        or final_manifest.get("bindings", {}).get(
            "source_task_manifest_sha256"
        ) != _sha256(task_manifest_path)
    ):
        raise ValueError("Revision-8 routing binding differs")

    roots = {
        "A": teacher_a_root,
        "B": teacher_b_root,
        "C": teacher_c_root,
        "D": teacher_d_root,
    }
    decisions = {}
    for teacher, root in roots.items():
        _, decisions[teacher] = _teacher_decisions(root, teacher)
        if final_manifest["bindings"][
            f"teacher_{teacher.lower()}_manifest_sha256"
        ] != _sha256(root / "manifest.json"):
            raise ValueError(f"Revision-8 teacher {teacher} binding differs")

    tasks = {}
    candidates = {}
    for selected in task_manifest["selected"]:
        task_id = str(selected["task_id"])
        task_path = task_root / selected["task"]
        task = _read(task_path)
        if _sha256(task_path) != selected["task_sha256"]:
            raise ValueError(f"source task hash differs: {task_id}")
        tasks[task_id] = (selected, task)
        for candidate in task["candidates"]:
            key = (task_id, str(candidate["candidate_id"]))
            if key in candidates:
                raise ValueError(f"duplicate source candidate: {key}")
            candidates[key] = candidate

    routes = [
        json.loads(line)
        for line in final_routing_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    route_by_key = {
        (str(row["task_id"]), str(row["candidate_id"])): row for row in routes
    }
    if len(route_by_key) != len(routes) or set(route_by_key) != set(candidates):
        raise ValueError("Revision-8 routing inventory differs")

    unanimous = []
    for key, route in route_by_key.items():
        if not (
            route.get("provisional_pass") is True
            and route.get("read_by_c") is True
            and int(route.get("original_definite_support", 0)) == 3
            and route.get("d_label") == "definitely_not_person"
        ):
            continue
        task_id, candidate_id = key
        if any(
            decisions[teacher][task_id][candidate_id]["label"]
            != "definitely_not_person"
            for teacher in ("A", "B", "C", "D")
        ):
            raise ValueError(f"unanimous teacher decision differs: {key}")
        unanimous.append(key)
    if len(unanimous) != EXPECTED_UNANIMOUS:
        raise ValueError(
            f"expected {EXPECTED_UNANIMOUS} unanimous candidates, "
            f"found {len(unanimous)}"
        )

    evidence_manifest_path = translation_evidence_root / "manifest.json"
    evidence_manifest = _read(evidence_manifest_path)
    documents = {}
    source_hashes = {}
    routing = []
    safe = []
    for task_id, candidate_id in sorted(unanimous):
        _, task = tasks[task_id]
        juan = int(task["juan"])
        if juan not in documents:
            canonical_path = canonical_text_root / f"juan_{juan:03d}.json"
            evidence_path = translation_evidence_root / f"juan_{juan:03d}.json"
            canonical = _read(canonical_path)
            evidence = _read(evidence_path)
            if (
                int(canonical.get("juan_no", -1)) != juan
                or int(evidence.get("juan", -1)) != juan
                or evidence.get("mapping_sha256")
                != evidence_manifest.get("mapping_sha256")
            ):
                raise ValueError(f"juan source binding differs: {juan}")
            documents[juan] = (canonical, evidence)
            source_hashes[str(juan)] = {
                "canonical_text_sha256": _sha256(canonical_path),
                "translation_evidence_sha256": _sha256(evidence_path),
            }
        canonical, evidence = documents[juan]
        paragraphs = _paragraphs(task, canonical)
        candidate = candidates[(task_id, candidate_id)]
        vetoes = _candidate_vetoes(
            task, candidate, paragraphs, evidence
        )
        row = {
            "task_id": task_id,
            "candidate_id": candidate_id,
            "juan": juan,
            "jie_index": int(task["jie_index"]),
            "para_id": int(candidate["para_id"]),
            "start": int(candidate["start"]),
            "end": int(candidate["end"]),
            "surface": candidate["surface"],
            "vetoes": vetoes,
        }
        routing.append(row)
        if not vetoes:
            safe.append(row)

    safe_juans = {row["juan"] for row in safe}
    if len(safe) < MIN_SAFE_NEGATIVES or len(safe_juans) != EXPECTED_JUANS:
        raise RuntimeError("Revision-9 safe-negative floors not met")
    audit_count = min(AUDIT_SIZE, len(safe))
    audit = sorted(
        safe,
        key=lambda row: (
            _audit_digest(
                tasks[row["task_id"]][1],
                candidates[(row["task_id"], row["candidate_id"])],
            ),
            row["task_id"],
            row["candidate_id"],
        ),
    )[:audit_count]
    audit_keys = {
        (row["task_id"], row["candidate_id"]) for row in audit
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        task_dir = staging / "tasks"
        rationale_dir = staging / "rationales"
        task_dir.mkdir()
        rationale_dir.mkdir()
        selected_rows = []
        for task_id in sorted({row["task_id"] for row in audit}):
            source_selected, source_task = tasks[task_id]
            selected_candidates = [
                candidates[key] for key in sorted(audit_keys)
                if key[0] == task_id
            ]
            review_task = {
                "schema_version": 1,
                "status": SAFE_REVIEW_STATUS,
                "phase": "revision-9-blind-negative-audit",
                "task_id": task_id,
                "source_task_sha256": source_selected["task_sha256"],
                "juan": int(source_task["juan"]),
                "jie_index": int(source_task["jie_index"]),
                "instructions": (
                    "Using only this numbered jie and the exact highlighted span, "
                    "decide whether the span denotes a person occurrence."
                ),
                "jie": source_task["jie"],
                "approved_translation": {
                    "available": False,
                    "reason": "No approved translation prose is bound to this task.",
                },
                "candidates": selected_candidates,
            }
            rationale_rows = []
            for candidate in selected_candidates:
                candidate_id = str(candidate["candidate_id"])
                rationale_rows.append({
                    "candidate_id": candidate_id,
                    "judgments": [{
                        "teacher": teacher,
                        "model": TEACHER_MODELS[teacher],
                        "rationale": decisions[teacher][task_id][
                            candidate_id
                        ]["rationale"],
                    } for teacher in ("A", "B", "C", "D")],
                })
            target = task_dir / f"task_{task_id}.json"
            rationale_target = rationale_dir / f"task_{task_id}.json"
            target.write_text(
                json.dumps(review_task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            rationale_target.write_text(
                json.dumps({
                    "schema_version": 1,
                    "phase": "revision-9-post-judgment-rationales",
                    "task_id": task_id,
                    "task_sha256": _sha256(target),
                    "candidates": rationale_rows,
                }, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            selected_rows.append({
                "task_id": task_id,
                "juan": int(source_task["juan"]),
                "jie_index": int(source_task["jie_index"]),
                "task": str(Path("tasks") / target.name),
                "task_sha256": _sha256(target),
                "rationales": str(Path("rationales") / rationale_target.name),
                "rationales_sha256": _sha256(rationale_target),
                "candidates": len(selected_candidates),
            })
        routing_path = staging / "safe-routing.jsonl"
        _write_jsonl(routing_path, [{
            **row,
            "audit_selected": (
                row["task_id"], row["candidate_id"]
            ) in audit_keys,
        } for row in routing])
        manifest = {
            "schema_version": 1,
            "status": SAFE_REVIEW_STATUS,
            "revision": 9,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "candidate_model_blind": True,
            "model_predictions_used": False,
            "git_commit": _git_commit_clean(),
            "audit": {
                "salt": AUDIT_SALT,
                "sample_size": audit_count,
                "one_sided_confidence": 0.95,
                "candidate_false_negative_upper_bound": 0.01,
                "stop_on_first_exclusion": True,
            },
            "bindings": {
                "source_task_manifest_sha256": _sha256(task_manifest_path),
                "revision_8_manifest_sha256": _sha256(final_manifest_path),
                "translation_evidence_manifest_sha256": _sha256(
                    evidence_manifest_path
                ),
                **{
                    f"teacher_{teacher.lower()}_manifest_sha256": _sha256(
                        root / "manifest.json"
                    )
                    for teacher, root in roots.items()
                },
            },
            "counts": {
                "source_candidates": len(candidates),
                "cross_family_unanimous": len(unanimous),
                "vetoed": len(routing) - len(safe),
                "safe_candidates": len(safe),
                "safe_juans": len(safe_juans),
                "audit_candidates": audit_count,
                "audit_tasks": len(selected_rows),
            },
            "source_juan_hashes": source_hashes,
            "safe_routing_sha256": _sha256(routing_path),
            "selected": selected_rows,
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
        description="Prepare the Revision-9 blind safe-negative audit."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--teacher-a", type=Path, required=True)
    parser.add_argument("--teacher-b", type=Path, required=True)
    parser.add_argument("--teacher-c", type=Path, required=True)
    parser.add_argument("--teacher-d", type=Path, required=True)
    parser.add_argument("--revision-8", type=Path, required=True)
    parser.add_argument("--translation-evidence", type=Path, required=True)
    parser.add_argument("--canonical-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_safe_review(
        args.tasks,
        args.teacher_a,
        args.teacher_b,
        args.teacher_c,
        args.teacher_d,
        args.revision_8,
        args.translation_evidence,
        args.canonical_text,
        args.output,
    )
    print(json.dumps({
        "status": manifest["status"],
        "counts": manifest["counts"],
        "audit": manifest["audit"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
