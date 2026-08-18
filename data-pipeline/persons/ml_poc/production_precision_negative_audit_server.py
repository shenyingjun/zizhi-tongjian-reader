from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from production_precision_lexical_safe_review import SAFE_REVIEW_STATUS


HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
LABELS = {"not_person", "exclude_from_negative_training"}


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: dict) -> str:
    raw = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class SafeNegativeAuditStore:
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
        selected = manifest.get("selected")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("status") != SAFE_REVIEW_STATUS
            or manifest.get("revision") != 9
            or manifest.get("confirmation_read") is not False
            or manifest.get("candidate_model_blind") is not True
            or manifest.get("model_predictions_used") is not False
            or not isinstance(selected, list)
            or not selected
        ):
            raise ValueError("unsupported safe-negative audit manifest")
        self.selected = {}
        self.order = []
        total = 0
        for row in selected:
            task_id = str(row.get("task_id", ""))
            if (
                len(task_id) != 20
                or any(char not in "0123456789abcdef" for char in task_id)
                or task_id in self.selected
            ):
                raise ValueError(f"invalid audit task ID: {task_id}")
            task = self._bound(row, "task", "task_sha256")
            rationales = self._bound(
                row, "rationales", "rationales_sha256"
            )
            self._validate_sources(
                task_id, task, rationales, str(row["task_sha256"])
            )
            count = len(task["candidates"])
            if count != int(row.get("candidates", -1)):
                raise ValueError(f"audit candidate count differs: {task_id}")
            total += count
            self.selected[task_id] = row
            self.order.append(task_id)
        if (
            total != int(manifest.get("audit", {}).get("sample_size", -1))
            or total != int(
                manifest.get("counts", {}).get("audit_candidates", -2)
            )
        ):
            raise ValueError("audit sample count differs")

    def _bound(self, row: dict, key: str, hash_key: str) -> dict:
        relative = Path(str(row.get(key, "")))
        if relative.is_absolute():
            raise ValueError(f"{key} path must be relative")
        path = (self.review_dir / relative).resolve()
        try:
            path.relative_to(self.review_dir)
        except ValueError as error:
            raise ValueError(f"{key} path escapes review directory") from error
        if (
            not path.is_file()
            or _sha256(path) != row.get(hash_key)
        ):
            raise PermissionError(f"{key} hash differs")
        return _read(path)

    @staticmethod
    def _candidate_index(task: dict) -> dict[str, dict]:
        jie = task["jie"]
        text = str(jie["text"])
        paragraphs = {}
        for segment in jie["segments"]:
            para_id = int(segment["para_id"])
            start = int(segment["assembled_start"])
            end = int(segment["assembled_end"])
            if para_id in paragraphs or not 0 <= start < end <= len(text):
                raise ValueError("invalid audit paragraph geometry")
            paragraphs[para_id] = text[start:end]
        candidates = {}
        geometries = set()
        for row in task["candidates"]:
            candidate_id = str(row.get("candidate_id", ""))
            para_id = int(row["para_id"])
            start = int(row["start"])
            end = int(row["end"])
            geometry = (para_id, start, end)
            paragraph = paragraphs.get(para_id)
            if (
                len(candidate_id) != 20
                or candidate_id in candidates
                or geometry in geometries
                or paragraph is None
                or not 0 <= start < end <= len(paragraph)
                or paragraph[start:end] != row.get("surface")
            ):
                raise ValueError(f"invalid audit candidate: {candidate_id}")
            candidates[candidate_id] = row
            geometries.add(geometry)
        if not candidates:
            raise ValueError("audit task has no candidates")
        return candidates

    def _validate_sources(
        self,
        task_id: str,
        task: dict,
        rationales: dict,
        expected_task_sha256: str,
    ) -> None:
        if (
            task.get("schema_version") != 1
            or task.get("status") != SAFE_REVIEW_STATUS
            or task.get("phase") != "revision-9-blind-negative-audit"
            or task.get("task_id") != task_id
            or rationales.get("schema_version") != 1
            or rationales.get("phase")
            != "revision-9-post-judgment-rationales"
            or rationales.get("task_id") != task_id
            or rationales.get("task_sha256") != expected_task_sha256
        ):
            raise ValueError(f"audit task provenance differs: {task_id}")
        candidates = self._candidate_index(task)
        rationale_rows = rationales.get("candidates")
        rationale_index = {
            str(row.get("candidate_id", "")): row
            for row in rationale_rows
        } if isinstance(rationale_rows, list) else {}
        if (
            set(rationale_index) != set(candidates)
            or len(rationale_index) != len(rationale_rows)
            or any(
                not isinstance(row.get("judgments"), list)
                or len(row["judgments"]) != 4
                or any(
                    set(judgment) != {"teacher", "model", "rationale"}
                    or not str(judgment["rationale"]).strip()
                    for judgment in row["judgments"]
                )
                for row in rationale_index.values()
            )
        ):
            raise ValueError(f"audit rationale inventory differs: {task_id}")

    def _sources(self, task_id: str) -> tuple[dict, dict]:
        if _sha256(self.manifest_path) != self.manifest_sha256:
            raise PermissionError("audit manifest changed")
        row = self.selected.get(task_id)
        if row is None:
            raise ValueError(f"unknown audit task: {task_id}")
        task = self._bound(row, "task", "task_sha256")
        rationales = self._bound(
            row, "rationales", "rationales_sha256"
        )
        self._validate_sources(
            task_id, task, rationales, str(row["task_sha256"])
        )
        return task, rationales

    def _state_path(self, task_id: str) -> Path:
        return self.state_dir / f"task_{task_id}.json"

    def _new_state(self, task_id: str) -> dict:
        row = self.selected[task_id]
        return {
            "schema_version": 1,
            "status": SAFE_REVIEW_STATUS,
            "task_id": task_id,
            "task_sha256": row["task_sha256"],
            "rationales_sha256": row["rationales_sha256"],
            "decisions": {},
            "complete": False,
            "completion_receipt": None,
        }

    def _validate_state(
        self, task_id: str, state: dict, task: dict
    ) -> dict:
        candidates = self._candidate_index(task)
        row = self.selected[task_id]
        if (
            state.get("schema_version") != 1
            or state.get("status") != SAFE_REVIEW_STATUS
            or state.get("task_id") != task_id
            or state.get("task_sha256") != row["task_sha256"]
            or state.get("rationales_sha256") != row["rationales_sha256"]
            or not isinstance(state.get("decisions"), dict)
            or not set(state["decisions"]).issubset(candidates)
        ):
            raise ValueError(f"invalid audit state: {task_id}")
        for decision in state["decisions"].values():
            initial = decision.get("initial")
            final = decision.get("final")
            revealed = decision.get("rationales_revealed")
            if (
                initial not in LABELS
                or not isinstance(revealed, bool)
                or final not in {*LABELS, None}
                or (initial == "exclude_from_negative_training"
                    and final != initial)
                or (initial == "not_person" and final is not None
                    and not revealed)
            ):
                raise ValueError(f"invalid audit decision: {task_id}")
        complete = state.get("complete")
        receipt = state.get("completion_receipt")
        all_final = (
            set(state["decisions"]) == set(candidates)
            and all(
                row["final"] in LABELS
                for row in state["decisions"].values()
            )
        )
        if not isinstance(complete, bool) or (complete and not all_final):
            raise ValueError(f"invalid audit completion: {task_id}")
        if complete:
            expected = _canonical_sha256({
                key: value for key, value in state.items()
                if key != "completion_receipt"
            })
            if receipt != expected:
                raise PermissionError(f"audit completion receipt differs: {task_id}")
        elif receipt is not None:
            raise ValueError(f"unfinished audit has receipt: {task_id}")
        return state

    def _load_state(self, task_id: str, task: dict) -> dict:
        path = self._state_path(task_id)
        state = _read(path) if path.is_file() else self._new_state(task_id)
        return self._validate_state(task_id, state, task)

    def _write_state(self, task_id: str, state: dict) -> None:
        path = self._state_path(task_id)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.state_dir,
            prefix=f".{path.name}-", delete=False,
        ) as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(path)

    def _stopped(self) -> bool:
        for task_id in self.order:
            task, _ = self._sources(task_id)
            state = self._load_state(task_id, task)
            if any(
                row.get("final") == "exclude_from_negative_training"
                for row in state["decisions"].values()
            ):
                return True
        return False

    def index(self) -> dict:
        with self._lock:
            stopped = self._stopped()
            rows = []
            complete = 0
            decided = 0
            total_candidates = 0
            for position, task_id in enumerate(self.order, 1):
                task, _ = self._sources(task_id)
                state = self._load_state(task_id, task)
                count = len(task["candidates"])
                final_count = sum(
                    row.get("final") in LABELS
                    for row in state["decisions"].values()
                )
                complete += int(state["complete"])
                decided += final_count
                total_candidates += count
                rows.append({
                    "task_id": task_id,
                    "position": position,
                    "total": len(self.order),
                    "juan": int(task["juan"]),
                    "jie_index": int(task["jie_index"]),
                    "candidates": count,
                    "decided": final_count,
                    "complete": state["complete"],
                })
            return {
                "tasks": rows,
                "complete": complete,
                "total": len(rows),
                "decided": decided,
                "total_candidates": total_candidates,
                "stopped": stopped,
            }

    def payload(self, task_id: str) -> dict:
        with self._lock:
            task, rationales = self._sources(task_id)
            state = self._load_state(task_id, task)
            rationale_index = {
                row["candidate_id"]: row["judgments"]
                for row in rationales["candidates"]
            }
            revealed = {
                candidate_id: rationale_index[candidate_id]
                for candidate_id, decision in state["decisions"].items()
                if decision["rationales_revealed"]
            }
            return {
                "task": task,
                "state": state,
                "revealed_rationales": revealed,
                "stopped": self._stopped(),
            }

    def all_payloads(self) -> dict:
        with self._lock:
            stopped = self._stopped()
            payloads = []
            decided = 0
            complete = 0
            total_candidates = 0
            for task_id in self.order:
                task, rationales = self._sources(task_id)
                state = self._load_state(task_id, task)
                rationale_index = {
                    row["candidate_id"]: row["judgments"]
                    for row in rationales["candidates"]
                }
                revealed = {
                    candidate_id: rationale_index[candidate_id]
                    for candidate_id, decision in state["decisions"].items()
                    if decision["rationales_revealed"]
                }
                final_count = sum(
                    row.get("final") in LABELS
                    for row in state["decisions"].values()
                )
                decided += final_count
                complete += int(state["complete"])
                total_candidates += len(task["candidates"])
                payloads.append({
                    "task": task,
                    "state": state,
                    "revealed_rationales": revealed,
                })
            return {
                "payloads": payloads,
                "complete": complete,
                "total": len(payloads),
                "decided": decided,
                "total_candidates": total_candidates,
                "stopped": stopped,
            }

    def initial(self, task_id: str, candidate_id: str, label: str) -> dict:
        with self._lock:
            if label not in LABELS:
                raise ValueError("invalid initial audit label")
            if self._stopped():
                raise PermissionError("audit stopped after exclusion")
            task, _ = self._sources(task_id)
            state = self._load_state(task_id, task)
            if state["complete"] or candidate_id in state["decisions"]:
                raise PermissionError("initial audit judgment is immutable")
            if candidate_id not in self._candidate_index(task):
                raise ValueError("unknown audit candidate")
            excluded = label == "exclude_from_negative_training"
            state["decisions"][candidate_id] = {
                "initial": label,
                "rationales_revealed": False,
                "final": label if excluded else None,
            }
            self._write_state(task_id, state)
            return state["decisions"][candidate_id]

    def reveal(self, task_id: str, candidate_id: str) -> list[dict]:
        with self._lock:
            task, rationales = self._sources(task_id)
            state = self._load_state(task_id, task)
            decision = state["decisions"].get(candidate_id)
            if decision is None:
                raise PermissionError("initial judgment required before reveal")
            if (
                decision["initial"] == "exclude_from_negative_training"
                or state["complete"]
            ):
                raise PermissionError("rationale reveal is unavailable")
            decision["rationales_revealed"] = True
            self._write_state(task_id, state)
            return next(
                row["judgments"] for row in rationales["candidates"]
                if row["candidate_id"] == candidate_id
            )

    def reveal_task(self, task_id: str) -> dict[str, list[dict]]:
        with self._lock:
            if self._stopped():
                raise PermissionError("audit stopped after exclusion")
            task, rationales = self._sources(task_id)
            state = self._load_state(task_id, task)
            candidates = self._candidate_index(task)
            if (
                set(state["decisions"]) != set(candidates)
                or any(
                    row["initial"] != "not_person"
                    or row["final"] not in {None, "not_person"}
                    for row in state["decisions"].values()
                )
                or (
                    state["complete"]
                    and any(
                        row["final"] != "not_person"
                        or not row["rationales_revealed"]
                        for row in state["decisions"].values()
                    )
                )
            ):
                raise PermissionError(
                    "all independent initial judgments are required"
                )
            for decision in state["decisions"].values():
                decision["rationales_revealed"] = True
            self._write_state(task_id, state)
            return {
                row["candidate_id"]: row["judgments"]
                for row in rationales["candidates"]
            }

    def reveal_all(self) -> dict[str, dict[str, list[dict]]]:
        with self._lock:
            if self._stopped():
                raise PermissionError("audit stopped after exclusion")
            prepared = []
            revealed = {}
            for task_id in self.order:
                task, rationales = self._sources(task_id)
                state = self._load_state(task_id, task)
                candidates = self._candidate_index(task)
                if (
                    set(state["decisions"]) != set(candidates)
                    or any(
                        row["initial"] != "not_person"
                        or row["final"] not in {None, "not_person"}
                        for row in state["decisions"].values()
                    )
                    or (
                        state["complete"]
                        and any(
                            row["final"] != "not_person"
                            or not row["rationales_revealed"]
                            for row in state["decisions"].values()
                        )
                    )
                ):
                    raise PermissionError(
                        "all independent initial judgments are required"
                    )
                prepared.append((task_id, state))
                revealed[task_id] = {
                    row["candidate_id"]: row["judgments"]
                    for row in rationales["candidates"]
                }
            for task_id, state in prepared:
                for decision in state["decisions"].values():
                    decision["rationales_revealed"] = True
                self._write_state(task_id, state)
            return revealed

    def final(self, task_id: str, candidate_id: str, label: str) -> dict:
        with self._lock:
            if label not in LABELS:
                raise ValueError("invalid final audit label")
            if self._stopped():
                raise PermissionError("audit stopped after exclusion")
            task, _ = self._sources(task_id)
            state = self._load_state(task_id, task)
            decision = state["decisions"].get(candidate_id)
            if (
                state["complete"]
                or decision is None
                or decision["initial"] != "not_person"
                or not decision["rationales_revealed"]
                or decision["final"] is not None
            ):
                raise PermissionError("final audit judgment is unavailable")
            decision["final"] = label
            self._write_state(task_id, state)
            return decision

    def complete(self, task_id: str) -> dict:
        with self._lock:
            task, _ = self._sources(task_id)
            state = self._load_state(task_id, task)
            if state["complete"]:
                return state
            candidates = self._candidate_index(task)
            if (
                set(state["decisions"]) != set(candidates)
                or any(
                    row["final"] not in LABELS
                    for row in state["decisions"].values()
                )
            ):
                raise ValueError("audit task has unresolved candidates")
            state["complete"] = True
            state["completion_receipt"] = _canonical_sha256({
                key: value for key, value in state.items()
                if key != "completion_receipt"
            })
            self._write_state(task_id, state)
            return state

    def confirm_task(self, task_id: str) -> dict:
        with self._lock:
            if self._stopped():
                raise PermissionError("audit stopped after exclusion")
            task, _ = self._sources(task_id)
            state = self._load_state(task_id, task)
            candidates = self._candidate_index(task)
            if (
                state["complete"]
                or set(state["decisions"]) != set(candidates)
                or any(
                    row["initial"] != "not_person"
                    or not row["rationales_revealed"]
                    or row["final"] not in {None, "not_person"}
                    for row in state["decisions"].values()
                )
            ):
                raise PermissionError(
                    "all rationales must be reviewed before batch confirmation"
                )
            for decision in state["decisions"].values():
                decision["final"] = "not_person"
            state["complete"] = True
            state["completion_receipt"] = _canonical_sha256({
                key: value for key, value in state.items()
                if key != "completion_receipt"
            })
            self._write_state(task_id, state)
            return state

    def confirm_all(self) -> dict:
        with self._lock:
            if self._stopped():
                raise PermissionError("audit stopped after exclusion")
            prepared = []
            for task_id in self.order:
                task, _ = self._sources(task_id)
                state = self._load_state(task_id, task)
                candidates = self._candidate_index(task)
                if state["complete"]:
                    if any(
                        row["final"] != "not_person"
                        for row in state["decisions"].values()
                    ):
                        raise PermissionError(
                            "completed task contains an exclusion"
                        )
                    continue
                if (
                    set(state["decisions"]) != set(candidates)
                    or any(
                        row["initial"] != "not_person"
                        or not row["rationales_revealed"]
                        or row["final"] not in {None, "not_person"}
                        for row in state["decisions"].values()
                    )
                ):
                    raise PermissionError(
                        "all rationales must be reviewed before confirmation"
                    )
                prepared.append((task_id, state))
            for task_id, state in prepared:
                for decision in state["decisions"].values():
                    decision["final"] = "not_person"
                state["complete"] = True
                state["completion_receipt"] = _canonical_sha256({
                    key: value for key, value in state.items()
                    if key != "completion_receipt"
                })
                self._write_state(task_id, state)
            return {
                "complete_tasks": len(self.order),
                "confirmed_candidates": sum(
                    int(self.selected[task_id]["candidates"])
                    for task_id in self.order
                ),
            }


class Handler(SimpleHTTPRequestHandler):
    store: SafeNegativeAuditStore

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def _json(self, status: HTTPStatus, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/safe-negative-audit/index":
                self._json(HTTPStatus.OK, self.store.index())
                return
            if parsed.path == "/api/safe-negative-audit/all":
                self._json(HTTPStatus.OK, self.store.all_payloads())
                return
            if parsed.path == "/api/safe-negative-audit/task":
                task_id = parse_qs(parsed.query).get("task_id", [""])[0]
                self._json(HTTPStatus.OK, self.store.payload(task_id))
                return
            if parsed.path == "/":
                self.path = "/safe_negative_audit.html"
            super().do_GET()
        except Exception as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            body = self._body()
            task_id = str(body.get("task_id", ""))
            if parsed.path == "/api/safe-negative-audit/initial":
                result = self.store.initial(
                    task_id, str(body.get("candidate_id", "")),
                    str(body.get("label", "")),
                )
            elif parsed.path == "/api/safe-negative-audit/reveal":
                result = {"judgments": self.store.reveal(
                    task_id, str(body.get("candidate_id", ""))
                )}
            elif parsed.path == "/api/safe-negative-audit/reveal-task":
                result = {"judgments": self.store.reveal_task(task_id)}
            elif parsed.path == "/api/safe-negative-audit/reveal-all":
                result = {"judgments": self.store.reveal_all()}
            elif parsed.path == "/api/safe-negative-audit/final":
                result = self.store.final(
                    task_id, str(body.get("candidate_id", "")),
                    str(body.get("label", "")),
                )
            elif parsed.path == "/api/safe-negative-audit/confirm-task":
                result = self.store.confirm_task(task_id)
            elif parsed.path == "/api/safe-negative-audit/confirm-all":
                result = self.store.confirm_all()
            elif parsed.path == "/api/safe-negative-audit/complete":
                result = self.store.complete(task_id)
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._json(HTTPStatus.OK, result)
        except PermissionError as error:
            self._json(HTTPStatus.CONFLICT, {"error": str(error)})
        except Exception as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve the Revision-9 blind safe-negative audit."
    )
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    Handler.store = SafeNegativeAuditStore(args.review_dir, args.state_dir)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Safe-negative audit: http://{args.host}:{args.port}/")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
