from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
STATUS = "ml_production_hard_label_human_tasks"
LABELS = {"exact_person", "wrong_boundary", "not_person", "uncertain"}
EXPECTED_CANDIDATES = 298


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HardLabelReviewStore:
    def __init__(self, review_dir: Path, state_dir: Path):
        self.review_dir = review_dir.resolve()
        self.state_dir = state_dir.resolve()
        if self.review_dir == self.state_dir:
            raise ValueError("review and state directories must differ")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.manifest_path = self.review_dir / "manifest.json"
        self.manifest_sha256 = _sha256(self.manifest_path)
        manifest = _read(self.manifest_path)
        routing_manifest_path = (
            self.review_dir.parent / "sealed-routing" / "manifest.json"
        )
        routing_path = self.review_dir.parent / "sealed-routing" / "routing.jsonl"
        routing_manifest = (
            _read(routing_manifest_path)
            if routing_manifest_path.is_file()
            else {}
        )
        if (
            manifest.get("schema_version") != 1
            or manifest.get("status") != STATUS
            or manifest.get("revision") != 12
            or manifest.get("candidate_model_blind") is not True
            or manifest.get("candidate_scores_hidden") is not True
            or manifest.get("original_labels_hidden") is not True
            or manifest.get("ai_judgments_hidden") is not True
            or manifest.get("confirmation_read") is not False
            or not routing_manifest_path.is_file()
            or manifest.get("source_routing_manifest_sha256")
            != _sha256(routing_manifest_path)
            or routing_manifest.get("counts", {}).get(
                "human_review"
            ) != EXPECTED_CANDIDATES
            or not routing_path.is_file()
            or routing_manifest.get("routing_sha256") != _sha256(routing_path)
        ):
            raise ValueError("unsupported hard-label human manifest")
        self.selected = {}
        self.order = []
        human_inventory = set()
        total = 0
        for row in manifest.get("selected", []):
            task_id = str(row.get("task_id", ""))
            task_path = (self.review_dir / str(row.get("task", ""))).resolve()
            try:
                task_path.relative_to(self.review_dir)
            except ValueError as error:
                raise ValueError("hard-label task escapes review root") from error
            task = _read(task_path)
            if (
                task_id in self.selected
                or not task_path.is_file()
                or _sha256(task_path) != row.get("task_sha256")
                or task.get("task_id") != task_id
                or len(task.get("candidates", []))
                != int(row.get("candidates", -1))
            ):
                raise ValueError(f"invalid hard-label human task: {task_id}")
            self._validate_task(task)
            self.selected[task_id] = {**row, "path": task_path}
            self.order.append(task_id)
            for candidate in task["candidates"]:
                key = (task_id, str(candidate["candidate_id"]))
                if key in human_inventory:
                    raise ValueError("duplicate hard-label human candidate")
                human_inventory.add(key)
            total += len(task["candidates"])
        if (
            total != int(manifest.get("counts", {}).get("candidates", -1))
            or total != EXPECTED_CANDIDATES
            or len(self.order)
            != int(manifest.get("counts", {}).get("tasks", -1))
        ):
            raise ValueError("hard-label human inventory differs")
        routing_inventory = set()
        for line in routing_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row.get("route") != "human_review":
                continue
            key = (str(row.get("task_id", "")), str(row.get("candidate_id", "")))
            if key in routing_inventory:
                raise ValueError("duplicate hard-label routed candidate")
            routing_inventory.add(key)
        if routing_inventory != human_inventory:
            raise ValueError("hard-label routed candidate inventory differs")

    @staticmethod
    def _validate_task(task: dict) -> None:
        text = str(task["jie"]["text"])
        paragraphs = {}
        for segment in task["jie"]["segments"]:
            start = int(segment["assembled_start"])
            end = int(segment["assembled_end"])
            para_id = int(segment["para_id"])
            if para_id in paragraphs or not 0 <= start < end <= len(text):
                raise ValueError("invalid hard-label paragraph")
            paragraphs[para_id] = text[start:end]
        seen = set()
        for candidate in task["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            paragraph = paragraphs.get(int(candidate["para_id"]))
            start = int(candidate["start"])
            end = int(candidate["end"])
            if (
                candidate_id in seen
                or paragraph is None
                or not 0 <= start < end <= len(paragraph)
                or paragraph[start:end] != candidate["surface"]
            ):
                raise ValueError("invalid hard-label candidate")
            seen.add(candidate_id)

    def _state_path(self, task_id: str) -> Path:
        return self.state_dir / f"task_{task_id}.json"

    def _task(self, task_id: str) -> dict:
        selected = self.selected[task_id]
        path = selected["path"]
        if _sha256(path) != selected["task_sha256"]:
            raise PermissionError(f"hard-label task changed: {task_id}")
        source = _read(path)
        self._validate_task(source)
        return {
            "schema_version": 1,
            "status": STATUS,
            "phase": "revision-12-blind-human-hard-label-audit",
            "task_id": task_id,
            "juan": int(source["juan"]),
            "jie_index": int(source["jie_index"]),
            "review_scope": "current-numbered-jie-only",
            "jie": {
                "text": str(source["jie"]["text"]),
                "segments": [
                    {
                        "para_id": int(segment["para_id"]),
                        "assembled_start": int(segment["assembled_start"]),
                        "assembled_end": int(segment["assembled_end"]),
                    }
                    for segment in source["jie"]["segments"]
                ],
            },
            "candidates": [
                {
                    key: candidate[key]
                    for key in (
                        "candidate_id", "para_id", "start", "end", "surface"
                    )
                }
                for candidate in source["candidates"]
            ],
            "allowed_labels": sorted(LABELS),
        }

    def _state(self, task_id: str) -> dict:
        path = self._state_path(task_id)
        if path.exists():
            state = _read(path)
        else:
            state = {
                "schema_version": 1,
                "status": STATUS,
                "task_id": task_id,
                "task_sha256": self.selected[task_id]["task_sha256"],
                "decisions": {},
                "complete": False,
            }
        candidate_ids = {
            str(row["candidate_id"])
            for row in self._task(task_id)["candidates"]
        }
        decisions = state.get("decisions")
        if (
            state.get("task_id") != task_id
            or state.get("task_sha256")
            != self.selected[task_id]["task_sha256"]
            or not isinstance(decisions, dict)
            or not set(decisions).issubset(candidate_ids)
            or any(value not in LABELS for value in decisions.values())
            or bool(state.get("complete")) != (set(decisions) == candidate_ids)
        ):
            raise ValueError(f"invalid hard-label state: {task_id}")
        return state

    def _write_state(self, state: dict) -> None:
        path = self._state_path(str(state["task_id"]))
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=self.state_dir,
            prefix=f".{path.name}-",
            suffix=".tmp",
        ) as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(path)

    def all(self) -> dict:
        with self.lock:
            if _sha256(self.manifest_path) != self.manifest_sha256:
                raise PermissionError("hard-label human manifest changed")
            payloads = []
            for task_id in self.order:
                payloads.append({
                    "task": self._task(task_id),
                    "state": self._state(task_id),
                })
            return {"payloads": payloads}

    def decide(self, task_id: str, candidate_id: str, label: str) -> dict:
        with self.lock:
            if task_id not in self.selected or label not in LABELS:
                raise ValueError("invalid hard-label decision")
            task = self._task(task_id)
            candidate_ids = {
                str(row["candidate_id"]) for row in task["candidates"]
            }
            if candidate_id not in candidate_ids:
                raise ValueError("unknown hard-label candidate")
            lock_path = self.state_dir / f".task_{task_id}.lock"
            try:
                descriptor = os.open(
                    lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
            except FileExistsError as error:
                raise PermissionError(
                    "hard-label task is being updated; retry"
                ) from error
            try:
                os.close(descriptor)
                state = self._state(task_id)
                if candidate_id in state["decisions"]:
                    raise PermissionError(
                        "hard-label first judgment is immutable"
                    )
                state["decisions"][candidate_id] = label
                state["complete"] = set(state["decisions"]) == candidate_ids
                self._write_state(state)
                return state
            finally:
                lock_path.unlink(missing_ok=True)


class Handler(SimpleHTTPRequestHandler):
    store: HardLabelReviewStore

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def _json(self, status: HTTPStatus, value: dict) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        try:
            path = urlparse(self.path).path
            if path == "/":
                self.path = "/hard_label_audit.html"
                return super().do_GET()
            if path == "/api/hard-label/all":
                return self._json(HTTPStatus.OK, self.store.all())
            return super().do_GET()
        except Exception as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def do_POST(self) -> None:
        try:
            if urlparse(self.path).path != "/api/hard-label/decision":
                return self._json(
                    HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"}
                )
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            state = self.store.decide(
                str(payload.get("task_id", "")),
                str(payload.get("candidate_id", "")),
                str(payload.get("label", "")),
            )
            self._json(HTTPStatus.OK, state)
        except PermissionError as error:
            self._json(HTTPStatus.CONFLICT, {"error": str(error)})
        except Exception as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve the Revision-12 blind hard-label review."
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    Handler.store = HardLabelReviewStore(args.review, args.state)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
