from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_revision17_plan import _read, _sha256
from production_precision_revision18_overlay import (
    BLOCKED_STATUS as OVERLAY_BLOCKED_STATUS,
    _geometry,
    _overlaps,
)
from production_train import _make_read_only


REVISION = 19
TASK_STATUS = "ml_production_precision_revision19_conflict_task"
TASKS_STATUS = "ml_production_precision_revision19_conflict_tasks"
EXPECTED_CONFLICTS = 39
EXPECTED_COMPONENTS = 36
EXPECTED_PARAGRAPHS = 35


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


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple, tuple] = {}

    def add(self, value: tuple) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: tuple) -> tuple:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: tuple, right: tuple) -> None:
        self.add(left)
        self.add(right)
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def conflict_components(
    decisions: list[dict],
    exact_owners: list[dict],
    conflicts: list[dict],
) -> list[dict]:
    decision_by_id = {
        str(row["candidate_id"]): row for row in decisions
    }
    owners_by_geometry = {
        tuple(row["geometry"]): [str(value) for value in row["candidate_ids"]]
        for row in exact_owners
    }
    if len(decision_by_id) != len(decisions):
        raise ValueError("Revision-19 decision inventory differs")

    conflict_geometries: set[tuple] = set()
    conflict_edges: list[tuple[tuple, tuple]] = []
    for row in conflicts:
        if row["type"] == "overlapping_exact_additions":
            left, right = tuple(row["left"]), tuple(row["right"])
        elif row["type"] == "semantic_overlaps_exact_addition":
            left, right = tuple(row["semantic"]), tuple(row["exact"])
        else:
            raise ValueError("Revision-19 conflict type differs")
        conflict_geometries.update((left, right))
        conflict_edges.append((left, right))

    candidate_geometries = {
        candidate_id: _geometry(row["candidate"])
        for candidate_id, row in decision_by_id.items()
    }
    relevant_paragraphs = {
        (geometry[0], geometry[1]) for geometry in conflict_geometries
    }
    all_geometries = set(conflict_geometries)
    all_geometries.update(
        geometry
        for geometry in owners_by_geometry
        if (geometry[0], geometry[1]) in relevant_paragraphs
    )
    all_geometries.update(
        geometry
        for geometry in candidate_geometries.values()
        if (geometry[0], geometry[1]) in relevant_paragraphs
    )

    union = _UnionFind()
    for geometry in all_geometries:
        union.add(("geometry",) + geometry)
    for left, right in conflict_edges:
        union.union(("geometry",) + left, ("geometry",) + right)
    for geometry, owners in owners_by_geometry.items():
        if geometry not in all_geometries:
            continue
        for candidate_id in owners:
            if candidate_id not in decision_by_id:
                raise ValueError("Revision-19 exact owner differs")
            candidate_node = ("candidate", candidate_id)
            union.union(("geometry",) + geometry, candidate_node)
            union.union(
                candidate_node,
                ("geometry",) + candidate_geometries[candidate_id],
            )
    for candidate_id, geometry in candidate_geometries.items():
        if geometry in all_geometries:
            union.union(
                ("candidate", candidate_id),
                ("geometry",) + geometry,
            )

    by_paragraph: dict[tuple, list[tuple]] = defaultdict(list)
    for geometry in all_geometries:
        by_paragraph[(geometry[0], geometry[1])].append(geometry)
    for geometries in by_paragraph.values():
        for index, left in enumerate(geometries):
            for right in geometries[index + 1:]:
                if _overlaps(left, right):
                    union.union(("geometry",) + left, ("geometry",) + right)

    conflict_roots = {
        union.find(("geometry",) + geometry)
        for geometry in conflict_geometries
    }
    grouped: dict[tuple, dict] = {}
    for node in union.parent:
        root = union.find(node)
        if root not in conflict_roots:
            continue
        component = grouped.setdefault(root, {
            "geometries": set(),
            "candidate_ids": set(),
        })
        if node[0] == "geometry":
            component["geometries"].add(node[1:])
        else:
            component["candidate_ids"].add(node[1])
    result = []
    for component in grouped.values():
        component["geometries"] = sorted(component["geometries"])
        component["candidate_ids"] = sorted(component["candidate_ids"])
        result.append(component)
    return sorted(
        result,
        key=lambda row: (row["geometries"][0], row["candidate_ids"]),
    )


def _surface(example: dict, geometry: tuple) -> str:
    _, para_id, start, end = geometry
    segment = next(
        (
            row for row in example["segments"]
            if int(row["para_id"]) == int(para_id)
        ),
        None,
    )
    if segment is None:
        raise ValueError("Revision-19 geometry paragraph differs")
    assembled_start = int(segment["assembled_start"])
    assembled_end = int(segment["assembled_end"])
    paragraph = example["text"][assembled_start:assembled_end]
    if not (0 <= start < end <= len(paragraph)):
        raise ValueError("Revision-19 geometry bounds differ")
    return paragraph[start:end]


def _task_id(overlay_sha256: str, component: dict) -> str:
    payload = json.dumps(
        {
            "geometries": component["geometries"],
            "candidate_ids": component["candidate_ids"],
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(
        f"revision-19:{overlay_sha256}:{payload}".encode("ascii")
    ).hexdigest()[:24]


def freeze_tasks(overlay_root: Path, output_dir: Path) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Revision-19 tasks exist: {output_dir}")
    manifest_path = overlay_root / "manifest.json"
    decisions_path = overlay_root / "decisions.jsonl"
    examples_path = overlay_root / "examples.jsonl"
    owners_path = overlay_root / "exact-owners.jsonl"
    conflicts_path = overlay_root / "conflicts.jsonl"
    manifest = _read(manifest_path)
    if (
        manifest.get("status") != OVERLAY_BLOCKED_STATUS
        or manifest.get("counts", {}).get("conflicts") != EXPECTED_CONFLICTS
        or manifest.get("outputs", {}).get("decisions_sha256")
        != _sha256(decisions_path)
        or manifest.get("outputs", {}).get("examples_sha256")
        != _sha256(examples_path)
        or manifest.get("outputs", {}).get("exact_owners_sha256")
        != _sha256(owners_path)
        or manifest.get("outputs", {}).get("conflicts_sha256")
        != _sha256(conflicts_path)
    ):
        raise ValueError("Revision-19 overlay binding differs")
    decisions = _read_jsonl(decisions_path)
    examples = _read_jsonl(examples_path)
    examples_by_id = {str(row["id"]): row for row in examples}
    components = conflict_components(
        decisions,
        _read_jsonl(owners_path),
        _read_jsonl(conflicts_path),
    )
    if (
        len(components) != EXPECTED_COMPONENTS
        or len({
            (row["geometries"][0][0], row["geometries"][0][1])
            for row in components
        }) != EXPECTED_PARAGRAPHS
    ):
        raise ValueError("Revision-19 closed component inventory differs")

    overlay_sha256 = _sha256(manifest_path)
    prepared = []
    for component in components:
        task_id = _task_id(overlay_sha256, component)
        source_id = str(component["geometries"][0][0])
        source_para = int(component["geometries"][0][1])
        if any(
            str(geometry[0]) != source_id
            or int(geometry[1]) != source_para
            for geometry in component["geometries"]
        ):
            raise ValueError("Revision-19 component crossed jie paragraph scope")
        example = examples_by_id.get(source_id)
        if example is None:
            raise ValueError("Revision-19 source jie differs")
        shown = []
        for geometry in component["geometries"]:
            opaque_id = hashlib.sha256(
                f"{task_id}:{geometry}".encode("utf-8")
            ).hexdigest()[:12]
            shown.append({
                "item_id": opaque_id,
                "para_id": int(geometry[1]),
                "start": int(geometry[2]),
                "end": int(geometry[3]),
                "surface": _surface(example, geometry),
            })
        shown.sort(key=lambda row: row["item_id"])
        prepared.append((task_id, component, example, shown))
    if len({row[0] for row in prepared}) != len(prepared):
        raise ValueError("Revision-19 opaque task ID collision")
    prepared.sort(key=lambda row: row[0])

    git_commit = _git_commit_clean()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        tasks_dir = staging / "reviewer-tasks"
        sealed_dir = staging / "sealed-source"
        tasks_dir.mkdir()
        sealed_dir.mkdir()
        hashes = []
        sealed = []
        for task_id, component, example, shown in prepared:
            task = {
                "schema_version": 1,
                "status": TASK_STATUS,
                "phase": "revision-19-blind-overlap-conflict-adjudication",
                "conflict_task_id": task_id,
                "review_scope": "current-numbered-jie-only",
                "protocol": {
                    "decision": (
                        "Return the complete set of exact individual-person "
                        "geometries that overlap any shown geometry."
                    ),
                    "evidence": "Use only the complete numbered jie in this task.",
                    "independence": (
                        "Return one first judgment without seeking prior or "
                        "sibling tasks."
                    ),
                },
                "jie": {
                    "text": example["text"],
                    "segments": example["segments"],
                },
                "shown_geometries": shown,
                "response_schema": {
                    "uncertain": "boolean",
                    "exact_people": [{
                        "para_id": "integer",
                        "start": "integer",
                        "end": "integer",
                        "surface": "string",
                    }],
                    "rationale": "nonempty string",
                },
            }
            path = tasks_dir / f"task_{task_id}.json"
            path.write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            hashes.append({
                "conflict_task_id": task_id,
                "task_sha256": _sha256(path),
            })
            sealed.append({
                "conflict_task_id": task_id,
                "candidate_ids": component["candidate_ids"],
                "geometries": [list(row) for row in component["geometries"]],
            })
        hashes_path = staging / "task-hashes.jsonl"
        sealed_path = sealed_dir / "components.jsonl"
        _write_jsonl(hashes_path, hashes)
        _write_jsonl(sealed_path, sealed)
        task_manifest = {
            "schema_version": 1,
            "status": TASKS_STATUS,
            "revision": REVISION,
            "fit_only": True,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "formal_reserve_text_read": False,
            "prior_judgments_visible": False,
            "reviewer_progress_disclosure": False,
            "git_commit": git_commit,
            "bindings": {
                "overlay_manifest_sha256": overlay_sha256,
                "decisions_sha256": _sha256(decisions_path),
                "examples_sha256": _sha256(examples_path),
                "exact_owners_sha256": _sha256(owners_path),
                "conflicts_sha256": _sha256(conflicts_path),
            },
            "counts": {
                "source_conflicts": EXPECTED_CONFLICTS,
                "closed_components": len(prepared),
                "tasks": len(prepared),
                "candidate_ids": len({
                    candidate_id
                    for component in components
                    for candidate_id in component["candidate_ids"]
                }),
                "shown_geometries": sum(len(row[3]) for row in prepared),
            },
            "outputs": {
                "task_hashes_sha256": _sha256(hashes_path),
                "sealed_components_sha256": _sha256(sealed_path),
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(task_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        _make_read_only(staging)
        staging.replace(output_dir)
    return task_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze Revision-19 blind conflict tasks."
    )
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_tasks(args.overlay, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
