from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
UI_GEOMETRY_VERSION = 4


class AnnotationStore:
    def __init__(
        self,
        blind_dir: Path,
        recall_dir: Path,
        role_audit_dir: Path,
        state_dir: Path,
        assisted_dir: Path | None = None,
    ):
        self.blind_dir = blind_dir
        self.recall_dir = recall_dir
        self.role_audit_dir = role_audit_dir
        self.state_dir = state_dir
        self.assisted_dir = assisted_dir
        self._lock = threading.RLock()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._read(blind_dir / "manifest.json")
        self.selected = {
            int(row["juan"]): row for row in manifest["selected"]
        }
        self.expansion = any(
            "mode" in row for row in manifest["selected"]
        )

    @staticmethod
    def _read(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def _blind_task(self, juan: int) -> dict:
        self._require_juan(juan)
        path = self.blind_dir / f"blind_juan_{juan:03d}.json"
        selection = self.selected[juan]
        expected_sha256 = selection.get("task_sha256")
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str)
            or hashlib.sha256(path.read_bytes()).hexdigest()
            != expected_sha256
        ):
            raise PermissionError(
                f"task hash differs for juan {juan}"
            )
        task = self._read(path)
        task.pop("selection_role", None)
        return task

    def _recall_pack(self, juan: int) -> dict:
        path = self.recall_dir / f"recall_juan_{juan:03d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"recall pack is missing: {path}")
        expected_sha256 = self.selected[juan].get("pack_sha256")
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str)
            or hashlib.sha256(path.read_bytes()).hexdigest()
            != expected_sha256
        ):
            raise PermissionError(
                f"recall pack hash differs for juan {juan}"
            )
        return self._read(path)

    def _role_audit_pack(self, juan: int) -> dict:
        path = self.role_audit_dir / f"role_audit_juan_{juan:03d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"role-audit pack is missing: {path}")
        return self._read(path)

    def _assisted_pack(self, juan: int) -> dict:
        if self.assisted_dir is None:
            raise FileNotFoundError("assisted pack directory is not configured")
        path = self.assisted_dir / f"assisted_juan_{juan:03d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"assisted pack is missing: {path}")
        return self._read(path)

    def _assisted_pack_sha256(self, juan: int) -> str:
        if self.assisted_dir is None:
            raise FileNotFoundError("assisted pack directory is not configured")
        path = self.assisted_dir / f"assisted_juan_{juan:03d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"assisted pack is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = self.selected[juan].get("pack_sha256")
        if expected is not None and expected != digest:
            raise PermissionError(
                f"assisted pack hash differs for juan {juan}"
            )
        return digest

    def _bind_assisted_pack(self, juan: int, state: dict) -> None:
        digest = self._assisted_pack_sha256(juan)
        bound = state["assisted"].get("pack_sha256")
        if bound is not None and bound != digest:
            raise PermissionError(
                "assisted pack changed after annotation began"
            )
        if bound is None:
            state["assisted"]["pack_sha256"] = digest
            self._write_state(juan, state)

    def _state_path(self, juan: int) -> Path:
        return self.state_dir / f"juan_{juan:03d}.json"

    def _require_juan(self, juan: int) -> None:
        if juan not in self.selected:
            raise ValueError(f"juan {juan} is not in this pilot")

    def state(self, juan: int) -> dict:
        self._require_juan(juan)
        path = self._state_path(juan)
        if path.is_file():
            state = self._read(path)
        else:
            state = {
                "schema_version": 1,
                "juan": juan,
                "blind": {"complete": False, "annotations": []},
                "recall": {
                    "complete": False,
                    "annotations": [],
                    "decisions": {},
                    "note_decisions": {},
                },
            }
        state.setdefault("role_audit", {
            "complete": False,
            "initialized": False,
            "annotations": [],
            "decisions": {},
        })
        state.setdefault("assisted", {
            "complete": False,
            "initialized": False,
            "annotations": [],
            "decisions": {},
        })
        state["assisted"].setdefault("initialized", False)
        return state

    def _mode(self, juan: int) -> str:
        return str(self.selected[juan].get("mode", "legacy"))

    def _is_assisted_mode(self, juan: int) -> bool:
        return self._mode(juan) in {
            "assisted", "diagnostic_assisted", "active_assisted",
        }

    def _is_adjudication_mode(self, juan: int) -> bool:
        return self._mode(juan) == "adjudication"

    def _blind_anchors_complete(self) -> bool:
        anchors = [
            juan for juan in self.selected
            if self._mode(juan) == "blind_anchor"
        ]
        return bool(anchors) and all(
            self.state(juan)["blind"]["complete"] for juan in anchors
        )

    def index(self) -> dict:
        rows = []
        for juan, selection in self.selected.items():
            state = self.state(juan)
            row = {
                "juan": juan,
                "blind_complete": state["blind"]["complete"],
                "recall_complete": state["recall"]["complete"],
                "role_audit_complete": state["role_audit"]["complete"],
            }
            if self.expansion:
                mode = self._mode(juan)
                row["mode"] = mode
                if mode in {"blind_anchor", "sealed_blind"}:
                    row["initial_phase"] = "blind"
                elif mode == "adjudication":
                    row["initial_phase"] = "recall"
                else:
                    row["initial_phase"] = "assisted"
                row["assisted_complete"] = state["assisted"]["complete"]
            if state["blind"]["complete"]:
                row["role"] = selection["role"]
            rows.append(row)
        return {"juans": rows, "boundary_guide": "BOUNDARY_GUIDE.md"}

    def payload(self, juan: int, phase: str) -> dict:
        with self._lock:
            return self._payload(juan, phase)

    def _payload(self, juan: int, phase: str) -> dict:
        state = self.state(juan)
        if self._is_assisted_mode(juan) and phase != "assisted":
            raise PermissionError(
                "assisted juans do not expose this phase; only assisted annotation"
            )
        if self._is_adjudication_mode(juan) and phase != "recall":
            raise PermissionError(
                "adjudication tasks expose only source-hidden recall"
            )
        if self._mode(juan) == "sealed_blind" and phase != "blind":
            raise PermissionError(
                "sealed tasks expose only candidate-blind annotation"
            )
        if phase == "blind":
            if self._is_assisted_mode(juan):
                raise PermissionError(
                    "assisted juans do not expose a blind phase"
                )
            return {
                "task": self._blind_task(juan),
                "state": state["blind"],
                "locked": state["blind"]["complete"],
                "sealed": self._mode(juan) == "sealed_blind",
            }
        if phase == "assisted":
            if not self._is_assisted_mode(juan):
                raise PermissionError("this juan is not an assisted task")
            if (
                self._mode(juan) == "assisted"
                and not self._blind_anchors_complete()
            ):
                raise PermissionError(
                    "assisted annotation is locked until blind anchors complete"
                )
            self._bind_assisted_pack(juan, state)
            pack = self._assisted_pack(juan)
            if (
                self._mode(juan) in {
                    "diagnostic_assisted", "active_assisted",
                }
                and not state["assisted"]["initialized"]
            ):
                annotations = pack.get("initial_annotations")
                if annotations is None:
                    annotations = [
                        {
                            "para_id": row["para_id"],
                            "start": row["start"],
                            "end": row["end"],
                            "surface": row["surface"],
                        }
                        for row in pack["candidates"]
                    ]
                state["assisted"]["annotations"] = (
                    self._validate_annotations(juan, annotations)
                )
                default_decisions = {
                    row["id"]: "accept"
                    for row in pack["candidates"]
                    if row.get("confidence") != "low"
                }
                state["assisted"]["decisions"] = pack.get(
                    "initial_decisions", default_decisions
                )
                state["assisted"]["initialized"] = True
                self._write_state(juan, state)
            return {
                "task": self._blind_task(juan),
                "review": pack,
                "state": state["assisted"],
                "locked": state["assisted"]["complete"],
            }
        if phase == "role_audit":
            if not state["recall"]["complete"]:
                raise PermissionError(
                    "role audit is locked until recall annotation completes"
                )
            if not state["role_audit"]["initialized"]:
                state["role_audit"]["annotations"] = list(
                    state["recall"]["annotations"]
                )
                state["role_audit"]["initialized"] = True
                self._write_state(juan, state)
            return {
                "task": self._blind_task(juan),
                "review": self._role_audit_pack(juan),
                "state": state["role_audit"],
                "locked": state["role_audit"]["complete"],
            }
        if phase != "recall":
            raise ValueError(f"unknown phase: {phase}")
        if not state["blind"]["complete"]:
            raise PermissionError("recall is locked until blind annotation completes")
        blind_geometry = {
            (row["para_id"], row["start"], row["end"])
            for row in state["blind"]["annotations"]
        }
        changed = False
        for candidate in self._recall_pack(juan)["candidates"]:
            geometry = (
                candidate["para_id"], candidate["start"], candidate["end"]
            )
            if (
                not self._is_adjudication_mode(juan)
                and geometry in blind_geometry
                and candidate["id"] not in state["recall"]["decisions"]
            ):
                state["recall"]["decisions"][candidate["id"]] = "accept"
                changed = True
        if changed:
            self._write_state(juan, state)
        return {
            "task": self._blind_task(juan),
            "review": self._recall_pack(juan),
            "state": state["recall"],
            "locked": state["recall"]["complete"],
            "adjudication": self._is_adjudication_mode(juan),
        }

    def _validate_annotations(self, juan: int, rows: object) -> list[dict]:
        if not isinstance(rows, list):
            raise ValueError("annotations must be a list")
        task = self._blind_task(juan)
        text_by_pid = {}
        for jie in task["jies"]:
            for segment in jie["segments"]:
                start, end = segment["assembled_start"], segment["assembled_end"]
                text_by_pid[int(segment["para_id"])] = jie["text"][start:end]
        result = []
        geometries = set()
        for raw in rows:
            para_id = int(raw["para_id"])
            start, end = int(raw["start"]), int(raw["end"])
            text = text_by_pid.get(para_id)
            if text is None or not 0 <= start < end <= len(text):
                raise ValueError("annotation geometry is outside main text")
            surface = text[start:end]
            if raw.get("surface") != surface:
                raise ValueError("annotation surface does not match main text")
            geometry = (para_id, start, end)
            if geometry in geometries:
                raise ValueError("duplicate annotation geometry")
            geometries.add(geometry)
            result.append({
                "para_id": para_id,
                "start": start,
                "end": end,
                "surface": surface,
                "status": str(raw.get("status", "person")),
                "note": str(raw.get("note", "")),
            })
        by_para: dict[int, list[tuple[int, int]]] = {}
        for row in result:
            by_para.setdefault(row["para_id"], []).append(
                (row["start"], row["end"])
            )
        for spans in by_para.values():
            spans.sort()
            if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
                raise ValueError("annotations may not overlap")
        return sorted(result, key=lambda row: (
            row["para_id"], row["start"], row["end"]
        ))

    def save(self, juan: int, phase: str, payload: dict) -> dict:
        with self._lock:
            return self._save(juan, phase, payload)

    def _save(self, juan: int, phase: str, payload: dict) -> dict:
        state = self.state(juan)
        if self._is_assisted_mode(juan) and phase != "assisted":
            raise PermissionError(
                "assisted juans do not expose this phase; only assisted annotation"
            )
        if self._is_adjudication_mode(juan) and phase != "recall":
            raise PermissionError(
                "adjudication tasks expose only source-hidden recall"
            )
        if self._mode(juan) == "sealed_blind" and phase != "blind":
            raise PermissionError(
                "sealed tasks expose only candidate-blind annotation"
            )
        if phase not in {"blind", "recall", "role_audit", "assisted"}:
            raise ValueError(f"unknown phase: {phase}")
        if state[phase]["complete"]:
            raise PermissionError(f"{phase} phase is locked")
        if phase == "recall" and not state["blind"]["complete"]:
            raise PermissionError("recall is locked until blind annotation completes")
        if phase == "role_audit" and not state["recall"]["complete"]:
            raise PermissionError(
                "role audit is locked until recall annotation completes"
            )
        if phase == "assisted":
            if not self._is_assisted_mode(juan):
                raise PermissionError("this juan is not an assisted task")
            if (
                self._mode(juan) == "assisted"
                and not self._blind_anchors_complete()
            ):
                raise PermissionError(
                    "assisted annotation is locked until blind anchors complete"
                )
            self._bind_assisted_pack(juan, state)
        state[phase]["annotations"] = self._validate_annotations(
            juan, payload.get("annotations")
        )
        if phase in {"recall", "role_audit", "assisted"}:
            pack = (
                self._recall_pack(juan)
                if phase == "recall"
                else (
                    self._role_audit_pack(juan)
                    if phase == "role_audit"
                    else self._assisted_pack(juan)
                )
            )
            valid_ids = {
                row["id"] for row in pack["candidates"]
            }
            decisions = payload.get("decisions", {})
            if not isinstance(decisions, dict):
                raise ValueError("decisions must be an object")
            if not set(decisions).issubset(valid_ids):
                raise ValueError("decision references unknown candidate")
            if any(value not in {"accept", "reject", "unsure"}
                   for value in decisions.values()):
                raise ValueError("invalid recall decision")
            if self._is_adjudication_mode(juan):
                if any(
                    value not in {"accept", "reject"}
                    for value in decisions.values()
                ):
                    raise ValueError(
                        "adjudication decisions require accept or reject"
                    )
                annotation_geometry = {
                    (row["para_id"], row["start"], row["end"])
                    for row in state[phase]["annotations"]
                }
                for candidate in pack["candidates"]:
                    decision = decisions.get(candidate["id"])
                    geometry = (
                        int(candidate["para_id"]),
                        int(candidate["start"]),
                        int(candidate["end"]),
                    )
                    if (
                        decision == "accept"
                        and geometry not in annotation_geometry
                    ):
                        raise ValueError(
                            "accepted adjudication candidate is not annotated"
                        )
                    if (
                        decision == "reject"
                        and geometry in annotation_geometry
                    ):
                        raise ValueError(
                            "rejected adjudication candidate remains annotated"
                        )
            state[phase]["decisions"] = decisions
            if phase == "recall":
                note_decisions = payload.get("note_decisions", {})
                if not isinstance(note_decisions, dict):
                    raise ValueError("note_decisions must be an object")
                state["recall"]["note_decisions"] = note_decisions
        self._write_state(juan, state)
        return state[phase]

    def complete(self, juan: int, phase: str) -> dict:
        with self._lock:
            return self._complete(juan, phase)

    def _complete(self, juan: int, phase: str) -> dict:
        state = self.state(juan)
        if self._is_assisted_mode(juan) and phase != "assisted":
            raise PermissionError(
                "assisted juans do not expose this phase; only assisted annotation"
            )
        if self._is_adjudication_mode(juan) and phase != "recall":
            raise PermissionError(
                "adjudication tasks expose only source-hidden recall"
            )
        if self._mode(juan) == "sealed_blind" and phase != "blind":
            raise PermissionError(
                "sealed tasks expose only candidate-blind annotation"
            )
        if phase == "blind":
            if state["blind"]["complete"]:
                return state["blind"]
            state["blind"]["complete"] = True
            if self._mode(juan) != "sealed_blind":
                state["recall"]["annotations"] = list(
                    state["blind"]["annotations"]
                )
        elif phase == "recall":
            if not state["blind"]["complete"]:
                raise PermissionError("blind phase must complete first")
            candidate_ids = {
                row["id"] for row in self._recall_pack(juan)["candidates"]
            }
            unresolved = candidate_ids - set(state["recall"]["decisions"])
            if unresolved:
                raise ValueError(
                    f"recall has {len(unresolved)} unresolved candidates"
                )
            if (
                self._is_adjudication_mode(juan)
                and any(
                    value not in {"accept", "reject"}
                    for value in state["recall"]["decisions"].values()
                )
            ):
                raise ValueError(
                    "adjudication candidates require accept or reject"
                )
            if self._is_adjudication_mode(juan):
                annotation_geometry = {
                    (row["para_id"], row["start"], row["end"])
                    for row in state["recall"]["annotations"]
                }
                for candidate in self._recall_pack(juan)["candidates"]:
                    geometry = (
                        int(candidate["para_id"]),
                        int(candidate["start"]),
                        int(candidate["end"]),
                    )
                    accepted = (
                        state["recall"]["decisions"][candidate["id"]]
                        == "accept"
                    )
                    if accepted != (geometry in annotation_geometry):
                        raise ValueError(
                            "adjudication decisions differ from annotations"
                        )
            state["recall"]["complete"] = True
        elif phase == "role_audit":
            if not state["recall"]["complete"]:
                raise PermissionError("recall phase must complete first")
            candidate_ids = {
                row["id"] for row in self._role_audit_pack(juan)["candidates"]
            }
            unresolved = candidate_ids - set(state["role_audit"]["decisions"])
            if unresolved:
                raise ValueError(
                    f"role audit has {len(unresolved)} unresolved candidates"
                )
            state["role_audit"]["complete"] = True
        elif phase == "assisted":
            if (
                self._mode(juan) == "assisted"
                and not self._blind_anchors_complete()
            ):
                raise PermissionError("blind anchors must complete first")
            self._bind_assisted_pack(juan, state)
            candidate_ids = {
                row["id"] for row in self._assisted_pack(juan)["candidates"]
            }
            unresolved = candidate_ids - set(state["assisted"]["decisions"])
            if unresolved:
                raise ValueError(
                    f"assisted review has {len(unresolved)} unresolved candidates"
                )
            state["assisted"]["complete"] = True
        else:
            raise ValueError(f"unknown phase: {phase}")
        self._write_state(juan, state)
        return state[phase]

    def _write_state(self, juan: int, state: dict) -> None:
        path = self._state_path(juan)
        temp = path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, path)


class Handler(SimpleHTTPRequestHandler):
    store: AnnotationStore

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
            if parsed.path == "/api/index":
                self._json(HTTPStatus.OK, self.store.index())
                return
            if parsed.path == "/api/task":
                query = parse_qs(parsed.query)
                juan = int(query["juan"][0])
                phase = query.get("phase", ["blind"])[0]
                self._json(HTTPStatus.OK, self.store.payload(juan, phase))
                return
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
                    "annotation UI is outdated; refresh the browser page"
                )
            juan = int(payload["juan"])
            phase = str(payload["phase"])
            if self.path == "/api/save":
                result = self.store.save(juan, phase, payload)
            elif self.path == "/api/complete":
                result = self.store.complete(juan, phase)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._json(HTTPStatus.OK, result)
        except (
            KeyError, ValueError, PermissionError, FileNotFoundError, json.JSONDecodeError
        ) as error:
            self._error(error)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local P0 annotation UI.")
    parser.add_argument("--blind-dir", type=Path, required=True)
    parser.add_argument("--recall-dir", type=Path, required=True)
    parser.add_argument("--role-audit-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--assisted-dir", type=Path)
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    Handler.store = AnnotationStore(
        args.blind_dir,
        args.recall_dir,
        args.role_audit_dir,
        args.state_dir,
        args.assisted_dir,
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"annotation UI: http://127.0.0.1:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
