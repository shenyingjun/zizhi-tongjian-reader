from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
UI_GEOMETRY_VERSION = 1
EXPECTED_TASKS = 180
FINAL_STATUS = "ml_production_focused_review_with_third_teacher"
REDUCED_STATUS = "ml_production_focused_review_with_reduced_audit"


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_file(root: Path, value: object, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise ValueError(f"{label} path must be relative")
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes review directory") from error
    if not path.is_file():
        raise ValueError(f"{label} file is missing")
    return path


class ProductionReviewStore:
    def __init__(self, review_dir: Path, state_dir: Path):
        self.review_dir = review_dir.resolve()
        self.state_dir = state_dir.resolve()
        if self.review_dir == self.state_dir:
            raise ValueError("review and state directories must differ")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.manifest_path = self.review_dir / "manifest.json"
        self.manifest_sha256 = _sha256(self.manifest_path)
        manifest = _read(self.manifest_path)
        if (
            manifest.get("schema_version") != 1
            or manifest.get("status") not in {FINAL_STATUS, REDUCED_STATUS}
            or manifest.get("candidate_model_blind") is not True
            or manifest.get("model_predictions_used") is not False
        ):
            raise ValueError("unsupported production review manifest")
        if manifest.get("status") == REDUCED_STATUS and (
            float(manifest.get("source_consensus_audit_rate", -1)) != 0.20
            or float(manifest.get("consensus_audit_rate", -1))
            not in {0.0, 0.05}
            or not isinstance(
                manifest.get("source_review_manifest_sha256"), str
            )
            or _sha256(_artifact_file(
                self.review_dir,
                manifest.get("source_review_manifest"),
                "source review manifest",
            ))
            != manifest.get("source_review_manifest_sha256")
            or not isinstance(
                manifest.get("prior_human_state_inventory"), dict
            )
            or int(manifest.get("expected_carried_human_decisions", -1))
            != int(
                manifest.get("counts", {}).get(
                    "carried_human_decisions", -2
                )
            )
        ):
            raise ValueError("reduced consensus-audit binding differs")
        selected = manifest.get("selected")
        if not isinstance(selected, list) or len(selected) != EXPECTED_TASKS:
            raise ValueError(
                f"production review must contain {EXPECTED_TASKS} tasks"
            )
        self.selected = {}
        self.order = []
        for row in selected:
            task_id = str(row.get("task_id", ""))
            if (
                len(task_id) != 20
                or any(char not in "0123456789abcdef" for char in task_id)
                or task_id in self.selected
            ):
                raise ValueError(f"invalid production review task ID: {task_id}")
            self._bound_path(row, "task", "task_sha256")
            self._bound_path(row, "review", "review_sha256")
            self.selected[task_id] = row
            self.order.append(task_id)
        if manifest.get("status") == REDUCED_STATUS:
            state_inventory = manifest["prior_human_state_inventory"]
            if not set(state_inventory).issubset(self.selected) or any(
                not isinstance(value, dict)
                or _sha256(_artifact_file(
                    self.review_dir,
                    value.get("path"),
                    "prior human state",
                ))
                != value.get("sha256")
                for value in state_inventory.values()
            ):
                raise ValueError("prior human state binding differs")
        audit_inventory = manifest.get("negative_audit_inventory")
        third_inventory = manifest.get("third_teacher_inventory")
        counts = manifest.get("counts")
        bound_third_reviews = {}
        for task_id, selection in self.selected.items():
            review = _read(self._bound_path(
                selection, "review", "review_sha256"
            ))
            has_hash = isinstance(review.get("third_teacher_sha256"), str)
            has_candidates = any(
                isinstance(candidate.get("third_teacher"), dict)
                for candidate in review.get("candidates", [])
            )
            if has_hash != has_candidates:
                raise ValueError(
                    f"incomplete third-teacher review binding: {task_id}"
                )
            if has_hash:
                bound_third_reviews[task_id] = review
        if (
            not isinstance(audit_inventory, dict)
            or not audit_inventory
            or not isinstance(counts, dict)
            or len(audit_inventory)
            != int(counts.get("negative_jie_third_pass", -1))
            or sum(
                int(value.get("candidates", -1))
                for value in audit_inventory.values()
                if isinstance(value, dict)
            )
            != int(counts.get("negative_audit_review", -1))
            or not set(audit_inventory).issubset(self.selected)
            or any(
                not isinstance(value, dict)
                or not isinstance(value.get("sha256"), str)
                or int(value.get("candidates", -1)) < 0
                for value in audit_inventory.values()
            )
        ):
            raise ValueError("negative-jie audit binding differs")
        if (
            not isinstance(third_inventory, dict)
            or not third_inventory
            or not isinstance(
                manifest.get("third_teacher_task_manifest_sha256"), str
            )
            or set(third_inventory) != set(bound_third_reviews)
            or sum(
                int(value.get("decisions", -1))
                for value in third_inventory.values()
                if isinstance(value, dict)
            )
            != int(counts.get("third_teacher_decisions", -1))
            or any(
                not isinstance(value, dict)
                or not isinstance(value.get("sha256"), str)
                or int(value.get("decisions", -1)) < 0
                or int(value.get("additions", -1)) < 0
                or int(value.get("novel_additions", -1)) < 0
                or int(value.get("duplicate_existing", -1)) < 0
                or int(value.get("additions", -1))
                != (
                    int(value.get("novel_additions", -1))
                    + int(value.get("duplicate_existing", -1))
                )
                for value in third_inventory.values()
            )
        ):
            raise ValueError("third-teacher binding differs")
        for task_id, inventory in third_inventory.items():
            review = bound_third_reviews[task_id]
            third_candidates = [
                candidate
                for candidate in review.get("candidates", [])
                if isinstance(candidate.get("third_teacher"), dict)
            ]
            novel_additions = [
                candidate for candidate in third_candidates
                if candidate.get("channels")
                == ["copilot_independent_c_adjudicator"]
            ]
            decisions = [
                candidate for candidate in third_candidates
                if candidate not in novel_additions
            ]
            if (
                review.get("third_teacher_sha256")
                != inventory["sha256"]
                or len(decisions) != int(inventory["decisions"])
                or len(novel_additions)
                != int(inventory["novel_additions"])
            ):
                raise ValueError(
                    f"third-teacher review binding differs: {task_id}"
                )

    def _bound_path(self, row: dict, key: str, hash_key: str) -> Path:
        path = (self.review_dir / str(row[key])).resolve()
        try:
            path.relative_to(self.review_dir)
        except ValueError as error:
            raise ValueError(f"{key} path escapes review directory") from error
        expected = row.get(hash_key)
        if not path.is_file() or not isinstance(expected, str):
            raise ValueError(f"missing bound {key}: {path}")
        if _sha256(path) != expected:
            raise PermissionError(f"{key} hash differs: {path.name}")
        return path

    def _sources(self, task_id: str) -> tuple[dict, dict, dict]:
        row = self._selection(task_id)
        if _sha256(self.manifest_path) != self.manifest_sha256:
            raise PermissionError("production review manifest changed")
        task = _read(self._bound_path(row, "task", "task_sha256"))
        review = _read(self._bound_path(row, "review", "review_sha256"))
        if (
            task.get("schema_version") != 1
            or review.get("schema_version") != 1
            or review.get("phase") != "assisted"
            or review.get("candidate_model_blind") is not True
            or review.get("task_id") != task_id
            or int(review.get("juan")) != int(task.get("juan"))
            or len(task.get("jies", [])) != 1
            or int(review.get("jie_index"))
            != int(task["jies"][0].get("jie_index"))
        ):
            raise ValueError(f"task/review provenance differs: {task_id}")
        self._candidate_index(task_id, task, review)
        return row, task, review

    def _selection(self, task_id: str) -> dict:
        if task_id not in self.selected:
            raise ValueError(f"unknown production review task: {task_id}")
        return self.selected[task_id]

    @staticmethod
    def _paragraphs(task: dict) -> dict[int, str]:
        jie = task["jies"][0]
        text = str(jie["text"])
        paragraphs = {}
        for segment in jie.get("segments", []):
            para_id = int(segment["para_id"])
            start = int(segment["assembled_start"])
            end = int(segment["assembled_end"])
            if (
                para_id in paragraphs
                or not 0 <= start <= end <= len(text)
            ):
                raise ValueError("invalid task paragraph geometry")
            paragraphs[para_id] = text[start:end]
        if not paragraphs:
            raise ValueError("production review task has no paragraphs")
        return paragraphs

    def _candidate_index(
        self, task_id: str, task: dict, review: dict
    ) -> dict[str, dict]:
        paragraphs = self._paragraphs(task)
        result = {}
        geometries = set()
        for candidate in review.get("candidates", []):
            candidate_id = str(candidate.get("id", ""))
            para_id = int(candidate["para_id"])
            start = int(candidate["start"])
            end = int(candidate["end"])
            geometry = (para_id, start, end)
            paragraph = paragraphs.get(para_id)
            if (
                candidate_id
                != f"copilot:{task_id}:{para_id}:{start}:{end}"
                or candidate_id in result
                or geometry in geometries
                or paragraph is None
                or not 0 <= start < end <= len(paragraph)
                or paragraph[start:end] != candidate.get("surface")
            ):
                raise ValueError(f"invalid review candidate: {candidate_id}")
            result[candidate_id] = candidate
            geometries.add(geometry)
        initial = review.get("initial_decisions")
        if (
            not isinstance(initial, dict)
            or not set(initial).issubset(result)
            or any(value not in {"accept", "reject"} for value in initial.values())
        ):
            raise ValueError(f"invalid initial decisions: {task_id}")
        initial_annotations = self._validate_annotations(
            task, review.get("initial_annotations")
        )
        annotation_geometry = {
            (row["para_id"], row["start"], row["end"])
            for row in initial_annotations
        }
        accepted_geometry = {
            (
                int(result[candidate_id]["para_id"]),
                int(result[candidate_id]["start"]),
                int(result[candidate_id]["end"]),
            )
            for candidate_id, decision in initial.items()
            if decision == "accept"
        }
        if annotation_geometry != accepted_geometry:
            raise ValueError(
                f"initial decisions differ from annotations: {task_id}"
            )
        return result

    @staticmethod
    def _validate_annotations(task: dict, rows: object) -> list[dict]:
        if not isinstance(rows, list):
            raise ValueError("annotations must be a list")
        paragraphs = ProductionReviewStore._paragraphs(task)
        result = []
        geometries = set()
        by_para: dict[int, list[tuple[int, int]]] = {}
        for raw in rows:
            para_id = int(raw["para_id"])
            start = int(raw["start"])
            end = int(raw["end"])
            paragraph = paragraphs.get(para_id)
            geometry = (para_id, start, end)
            if (
                geometry in geometries
                or paragraph is None
                or not 0 <= start < end <= len(paragraph)
                or paragraph[start:end] != raw.get("surface")
            ):
                raise ValueError("invalid annotation geometry or surface")
            geometries.add(geometry)
            by_para.setdefault(para_id, []).append((start, end))
            result.append({
                "para_id": para_id,
                "start": start,
                "end": end,
                "surface": paragraph[start:end],
            })
        for spans in by_para.values():
            spans.sort()
            if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
                raise ValueError("annotations may not overlap")
        return sorted(
            result,
            key=lambda row: (row["para_id"], row["start"], row["end"]),
        )

    def _state_path(self, task_id: str) -> Path:
        return self.state_dir / f"task_{task_id}.json"

    def _receipt_path(self, task_id: str) -> Path:
        return self.state_dir / "completed" / f"task_{task_id}.json"

    def _new_state(self, task_id: str, row: dict, review: dict) -> dict:
        return {
            "schema_version": 1,
            "task_id": task_id,
            "source_manifest_sha256": self.manifest_sha256,
            "task_sha256": row["task_sha256"],
            "review_sha256": row["review_sha256"],
            "complete": False,
            "expanded_full_union": False,
            "annotations": review["initial_annotations"],
            "human_decisions": {},
        }

    def _state(
        self, task_id: str, row: dict, task: dict, review: dict
    ) -> dict:
        path = self._state_path(task_id)
        state = _read(path) if path.is_file() else self._new_state(
            task_id, row, review
        )
        if (
            set(state) != {
                "schema_version",
                "task_id",
                "source_manifest_sha256",
                "task_sha256",
                "review_sha256",
                "complete",
                "expanded_full_union",
                "annotations",
                "human_decisions",
            }
            or state.get("schema_version") != 1
            or state.get("task_id") != task_id
            or state.get("source_manifest_sha256") != self.manifest_sha256
            or state.get("task_sha256") != row["task_sha256"]
            or state.get("review_sha256") != row["review_sha256"]
            or not isinstance(state.get("complete"), bool)
            or not isinstance(state.get("expanded_full_union"), bool)
        ):
            raise PermissionError(f"review state source binding differs: {task_id}")
        annotations = self._validate_annotations(task, state["annotations"])
        candidates = self._candidate_index(task_id, task, review)
        human_decisions = state.get("human_decisions")
        if (
            not isinstance(human_decisions, dict)
            or not set(human_decisions).issubset(candidates)
            or any(
                decision not in {"accept", "reject"}
                for decision in human_decisions.values()
            )
        ):
            raise ValueError(f"invalid persisted decisions: {task_id}")
        expansion_required = any(
            self._triggers_expansion(
                candidates[candidate_id],
                decision,
                review["initial_decisions"],
            )
            for candidate_id, decision in human_decisions.items()
        )
        if expansion_required and not state["expanded_full_union"]:
            raise ValueError(f"persisted expansion state differs: {task_id}")
        effective = self._effective_decisions(review, state)
        self._validate_decision_geometry(candidates, effective, annotations)
        if state["complete"]:
            unresolved = self._required_ids(review, state) - set(
                human_decisions
            )
            if unresolved:
                raise ValueError(
                    f"completed review has {len(unresolved)} unresolved candidates"
                )
        receipt_path = self._receipt_path(task_id)
        if state["complete"]:
            if not receipt_path.is_file():
                raise PermissionError(
                    f"completed review receipt is missing: {task_id}"
                )
            receipt = _read(receipt_path)
            state_path = self._state_path(task_id)
            if (
                set(receipt) != {
                    "schema_version",
                    "task_id",
                    "source_manifest_sha256",
                    "state_sha256",
                }
                or receipt.get("schema_version") != 1
                or receipt.get("task_id") != task_id
                or receipt.get("source_manifest_sha256")
                != self.manifest_sha256
                or receipt.get("state_sha256") != _sha256(state_path)
            ):
                raise PermissionError(
                    f"completed review receipt differs: {task_id}"
                )
        elif receipt_path.exists() or receipt_path.is_symlink():
            raise PermissionError(
                f"completed review cannot be reopened: {task_id}"
            )
        state["annotations"] = annotations
        return state

    @staticmethod
    def _required_ids(review: dict, state: dict) -> set[str]:
        candidate_ids = {
            str(candidate["id"]) for candidate in review["candidates"]
        }
        if state["expanded_full_union"]:
            return candidate_ids
        return candidate_ids - set(review["initial_decisions"])

    @staticmethod
    def _effective_decisions(review: dict, state: dict) -> dict[str, str]:
        decisions = (
            {} if state["expanded_full_union"]
            else dict(review["initial_decisions"])
        )
        decisions.update(state["human_decisions"])
        return decisions

    @staticmethod
    def _triggers_expansion(
        candidate: dict, decision: str, initial_decisions: dict
    ) -> bool:
        reason = str(candidate.get("review_reason", ""))
        candidate_id = str(candidate["id"])
        return (
            candidate_id in initial_decisions
            and decision != initial_decisions[candidate_id]
        ) or (
            reason.startswith("Predeclared ")
            and " audit of exact non-low A/B consensus." in reason
            and decision == "reject"
        ) or (
            reason.startswith("Independent negative-jie recall audit")
            and decision == "accept"
        )

    @staticmethod
    def _validate_decision_geometry(
        candidates: dict[str, dict],
        decisions: dict[str, str],
        annotations: list[dict],
    ) -> None:
        geometries = {
            (row["para_id"], row["start"], row["end"]) for row in annotations
        }
        for candidate_id, decision in decisions.items():
            candidate = candidates[candidate_id]
            geometry = (
                int(candidate["para_id"]),
                int(candidate["start"]),
                int(candidate["end"]),
            )
            if (decision == "accept") != (geometry in geometries):
                raise ValueError(
                    f"candidate decision differs from annotation: {candidate_id}"
                )

    def index(self) -> dict:
        with self._lock:
            rows = []
            complete_count = 0
            for position, task_id in enumerate(self.order, start=1):
                row, task, review = self._sources(task_id)
                state = self._state(task_id, row, task, review)
                required = self._required_ids(review, state)
                unresolved = required - set(state["human_decisions"])
                complete_count += int(state["complete"])
                jie = task["jies"][0]
                rows.append({
                    "task_id": task_id,
                    "position": position,
                    "total": len(self.order),
                    "juan": int(task["juan"]),
                    "jie_index": int(jie["jie_index"]),
                    "jie_number": jie.get("jie_number"),
                    "complete": bool(state["complete"]),
                    "expanded_full_union": bool(
                        state["expanded_full_union"]
                    ),
                    "required": len(required),
                    "unresolved": len(unresolved),
                })
            return {
                "tasks": rows,
                "complete": complete_count,
                "total": len(rows),
            }

    def payload(self, task_id: str) -> dict:
        with self._lock:
            row, task, review = self._sources(task_id)
            state = self._state(task_id, row, task, review)
            return {
                "task": task,
                "review": review,
                "state": {
                    **state,
                    "effective_decisions": self._effective_decisions(
                        review, state
                    ),
                    "required_ids": sorted(
                        self._required_ids(review, state)
                    ),
                },
                "locked": bool(state["complete"]),
            }

    def save(self, task_id: str, payload: dict) -> dict:
        with self._lock:
            row, task, review = self._sources(task_id)
            state = self._state(task_id, row, task, review)
            if state["complete"]:
                raise PermissionError("production review task is locked")
            annotations = self._validate_annotations(
                task, payload.get("annotations")
            )
            human_decisions = payload.get("human_decisions")
            candidates = self._candidate_index(task_id, task, review)
            if (
                not isinstance(human_decisions, dict)
                or not set(human_decisions).issubset(candidates)
                or any(
                    decision not in {"accept", "reject"}
                    for decision in human_decisions.values()
                )
            ):
                raise ValueError("invalid human review decisions")
            expanded = bool(state["expanded_full_union"]) or any(
                self._triggers_expansion(
                    candidates[candidate_id],
                    decision,
                    review["initial_decisions"],
                )
                for candidate_id, decision in human_decisions.items()
            )
            candidate_state = {
                **state,
                "expanded_full_union": expanded,
                "human_decisions": dict(human_decisions),
            }
            effective = self._effective_decisions(review, candidate_state)
            self._validate_decision_geometry(
                candidates, effective, annotations
            )
            state["expanded_full_union"] = expanded
            state["annotations"] = annotations
            state["human_decisions"] = dict(human_decisions)
            self._write_state(task_id, state)
            return self.payload(task_id)["state"]

    def complete(self, task_id: str) -> dict:
        with self._lock:
            row, task, review = self._sources(task_id)
            state = self._state(task_id, row, task, review)
            if state["complete"]:
                return state
            candidates = self._candidate_index(task_id, task, review)
            required = self._required_ids(review, state)
            unresolved = required - set(state["human_decisions"])
            if unresolved:
                raise ValueError(
                    f"production review has {len(unresolved)} unresolved candidates"
                )
            effective = self._effective_decisions(review, state)
            self._validate_decision_geometry(
                candidates, effective, state["annotations"]
            )
            state["complete"] = True
            self._write_state(task_id, state)
            self._write_receipt(task_id)
            return state

    def _write_state(self, task_id: str, state: dict) -> None:
        path = self._state_path(task_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _write_receipt(self, task_id: str) -> None:
        path = self._receipt_path(task_id)
        if path.exists() or path.is_symlink():
            raise PermissionError(
                f"completed review receipt already exists: {task_id}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({
                "schema_version": 1,
                "task_id": task_id,
                "source_manifest_sha256": self.manifest_sha256,
                "state_sha256": _sha256(self._state_path(task_id)),
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        path.chmod(stat.S_IREAD)


class Handler(SimpleHTTPRequestHandler):
    store: ProductionReviewStore

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, status: HTTPStatus, payload: dict) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _error(self, error: Exception) -> None:
        status = (
            HTTPStatus.FORBIDDEN
            if isinstance(error, PermissionError)
            else HTTPStatus.BAD_REQUEST
        )
        self._json(status, {"error": str(error)})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/production-review/index":
                self._json(HTTPStatus.OK, self.store.index())
                return
            if parsed.path == "/api/production-review/task":
                query = parse_qs(parsed.query)
                self._json(
                    HTTPStatus.OK,
                    self.store.payload(str(query["task_id"][0])),
                )
                return
            if parsed.path == "/":
                self.path = "/production_review.html"
        except (KeyError, ValueError, PermissionError, FileNotFoundError) as error:
            self._error(error)
            return
        super().do_GET()

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if payload.get("geometry_version") != UI_GEOMETRY_VERSION:
                raise ValueError(
                    "production review UI is outdated; refresh the browser page"
                )
            task_id = str(payload["task_id"])
            if self.path == "/api/production-review/save":
                result = self.store.save(task_id, payload)
            elif self.path == "/api/production-review/complete":
                result = self.store.complete(task_id)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._json(HTTPStatus.OK, result)
        except (
            KeyError,
            ValueError,
            PermissionError,
            FileNotFoundError,
            json.JSONDecodeError,
        ) as error:
            self._error(error)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the local production focused-review UI."
    )
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18766)
    args = parser.parse_args()
    Handler.store = ProductionReviewStore(args.review_dir, args.state_dir)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"production review UI: http://127.0.0.1:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
