from __future__ import annotations

import argparse
import hashlib
import json
import stat
from pathlib import Path

from core import assemble_jies


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TEXT = REPO_ROOT / "web" / "public" / "text"
REQUIRED_STATUSES = {
    "candidate_blind_copilot_double_pass_tasks_before_labeling",
    "round7_controlled_training_dataset",
    "round10_independent_dev_tasks_before_labeling",
    "round11_fresh_promotion_tasks_before_labeling",
    "round13_compact_challenge_tasks_before_labeling",
}
CANONICAL_WHOLE_JUAN_EXCLUSIONS = {
    21: "first_leaked_sealed_selection",
    201: "first_leaked_sealed_selection",
    204: "first_leaked_sealed_selection",
    225: "first_leaked_sealed_selection",
    251: "first_leaked_sealed_selection",
}
REQUIRED_ROOT_HASHES = {
    "historical_exclusion_manifest": (
        "b020541f12207244d110f38133e0c2dd1a43a554c18c952a1644a6cef45ec401"
    ),
    "round7_dataset_manifest": (
        "09c2724e346b8df5b0ace423016155167790bc7324bf8876f1bfa5942e958eb5"
    ),
    "round10_dev_task_manifest": (
        "a23614bc61b19093b0db5865825b57329363c06c1ccea75bed27e35eb17660de"
    ),
    "round11_promotion_task_manifest": (
        "ebe06c87674e49eee014624291553cac620b14b42a5ee15ae519c69c16a2d39a"
    ),
    "round13_challenge_task_manifest": (
        "5b5c6b46375ef09ae7447b2af800d3f37eff2905ca28942c532bc14359e7e937"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_files(paths: list[Path]) -> list[Path]:
    files = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing POC artifact: {path}")
        if path.is_dir():
            files.extend(
                child for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() in {".json", ".jsonl"}
            )
        elif path.suffix.lower() in {".json", ".jsonl"}:
            files.append(path)
        else:
            raise ValueError(f"unsupported POC artifact: {path}")
    resolved = sorted({path.resolve() for path in files}, key=str)
    if not resolved:
        raise ValueError("no JSON POC artifacts found")
    return resolved


def _records(path: Path) -> list[object]:
    if path.suffix.lower() == ".jsonl":
        records = []
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL {path}:{number}") from error
        return records
    try:
        return [json.loads(path.read_text(encoding="utf-8"))]
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error


def _is_sealed(path: Path, statuses: set[str]) -> bool:
    text = str(path).lower()
    return (
        any(token in text for token in ("round11", "round13", "promotion", "challenge"))
        or any(
            token in status.lower()
            for status in statuses
            for token in ("formal", "promotion", "challenge")
        )
    )


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value.lower())


def _bound_output_hashes(value: object) -> set[str]:
    hashes = set()
    if isinstance(value, dict):
        outputs = value.get("outputs")
        if isinstance(outputs, dict):
            hashes.update(
                item for item in outputs.values() if _is_sha256(item)
            )
        for key in ("tasks", "selected"):
            rows = value.get(key)
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and _is_sha256(
                        row.get("task_sha256")
                    ):
                        hashes.add(row["task_sha256"])
        for child in value.values():
            hashes.update(_bound_output_hashes(child))
    elif isinstance(value, list):
        for child in value:
            hashes.update(_bound_output_hashes(child))
    return hashes


def _walk(
    value: object,
    *,
    inherited_juan: int | None,
    exact: set[tuple[int, int]],
    claimed_juans: set[int],
    statuses: set[str],
) -> None:
    if isinstance(value, dict):
        juan = inherited_juan
        if "juan" in value:
            juan = int(value["juan"])
        status = value.get("status")
        if isinstance(status, str):
            statuses.add(status)
        if juan is not None and "jie_index" in value:
            exact.add((juan, int(value["jie_index"])))
        jies = value.get("jies")
        if isinstance(jies, list):
            for item in jies:
                if (
                    isinstance(item, dict)
                    and juan is not None
                    and "jie_index" in item
                ):
                    exact.add((juan, int(item["jie_index"])))
        excluded_juans = value.get("excluded_juans")
        if isinstance(excluded_juans, list):
            for raw in excluded_juans:
                claimed_juans.add(int(raw))
        splits = value.get("splits")
        if isinstance(splits, dict):
            for split in splits.values():
                if isinstance(split, dict):
                    for raw in split.get("juans", []):
                        claimed_juans.add(int(raw))
        for child in value.values():
            _walk(
                child,
                inherited_juan=juan,
                exact=exact,
                claimed_juans=claimed_juans,
                statuses=statuses,
            )
    elif isinstance(value, list):
        for child in value:
            _walk(
                child,
                inherited_juan=inherited_juan,
                exact=exact,
                claimed_juans=claimed_juans,
                statuses=statuses,
            )


def build_exclusion_inventory(
    artifact_paths: list[Path],
    output_path: Path,
    *,
    source_dir: Path = TEXT,
) -> dict:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"exclusion inventory exists: {output_path}")
    files = _json_files(artifact_paths)
    hashes_by_path = {path: _sha256(path) for path in files}
    available_hashes = set(hashes_by_path.values())
    exact: set[tuple[int, int]] = set()
    claimed_juans: set[int] = set()
    statuses_by_file: dict[Path, set[str]] = {}
    inputs = []
    bound_output_hashes = set()
    for path in files:
        statuses: set[str] = set()
        for record in _records(path):
            bound_output_hashes.update(_bound_output_hashes(record))
            _walk(
                record,
                inherited_juan=None,
                exact=exact,
                claimed_juans=claimed_juans,
                statuses=statuses,
            )
        statuses_by_file[path] = statuses
        inputs.append({
            "path": str(path),
            "sha256": hashes_by_path[path],
            "statuses": sorted(statuses),
        })

    discovered_statuses = {
        status for statuses in statuses_by_file.values() for status in statuses
    }
    whole_juan_sources = []
    for juan, reason in CANONICAL_WHOLE_JUAN_EXCLUSIONS.items():
        source_path = source_dir / f"juan_{juan:03d}.json"
        if not source_path.is_file():
            raise FileNotFoundError(
                f"missing whole-juan exclusion source: {source_path}"
            )
        source = json.loads(source_path.read_text(encoding="utf-8"))
        jie_indices = [
            int(jie.index)
            for jie in assemble_jies(source["paragraphs"])
            if jie.number is not None
        ]
        if not jie_indices:
            raise ValueError(
                f"whole-juan exclusion has no numbered jies: {juan}"
            )
        exact.update((juan, jie_index) for jie_index in jie_indices)
        whole_juan_sources.append({
            "juan": juan,
            "reason": reason,
            "source_path": str(source_path),
            "source_sha256": _sha256(source_path),
            "numbered_jies": len(jie_indices),
        })
    missing_statuses = sorted(REQUIRED_STATUSES - discovered_statuses)
    missing_roots = sorted(
        name for name, digest in REQUIRED_ROOT_HASHES.items()
        if digest not in available_hashes
    )
    missing_bound_outputs = sorted(bound_output_hashes - available_hashes)
    exact_juans = {juan for juan, _ in exact}
    unresolved_juans = sorted(claimed_juans - exact_juans)
    complete = not (
        missing_statuses
        or missing_roots
        or missing_bound_outputs
        or unresolved_juans
    )
    sealed = set()
    for path, statuses in statuses_by_file.items():
        if not _is_sealed(path, statuses):
            continue
        file_exact: set[tuple[int, int]] = set()
        for record in _records(path):
            _walk(
                record,
                inherited_juan=None,
                exact=file_exact,
                claimed_juans=set(),
                statuses=set(),
            )
        sealed.update(file_exact)

    manifest = {
        "schema_version": 1,
        "status": "ml_production_exact_jie_exclusions",
        "complete": complete,
        "completeness_checks": {
            "required_statuses": sorted(REQUIRED_STATUSES),
            "missing_statuses": missing_statuses,
            "required_root_hashes": REQUIRED_ROOT_HASHES,
            "missing_root_artifacts": missing_roots,
            "missing_bound_output_hashes": missing_bound_outputs,
            "claimed_juans": len(claimed_juans),
            "unresolved_claimed_juans": unresolved_juans,
        },
        "inputs": inputs,
        "whole_juan_exclusions": whole_juan_sources,
        "consumed": [
            {
                "juan": juan,
                "jie_index": jie_index,
                "reason": "poc_artifact_consumed",
            }
            for juan, jie_index in sorted(exact)
        ],
        "sealed": [
            {
                "juan": juan,
                "jie_index": jie_index,
                "reason": "poc_sealed_evaluation",
            }
            for juan, jie_index in sorted(sealed)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_path.chmod(stat.S_IREAD)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an exact-jie exclusion inventory from POC artifacts."
    )
    parser.add_argument(
        "--artifact", type=Path, action="append", required=True,
        help="POC JSON/JSONL file or artifact directory; repeat as needed",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=TEXT)
    args = parser.parse_args()
    manifest = build_exclusion_inventory(
        args.artifact, args.output, source_dir=args.source_dir
    )
    print(json.dumps({
        "complete": manifest["complete"],
        "consumed_jies": len(manifest["consumed"]),
        "sealed_jies": len(manifest["sealed"]),
        **manifest["completeness_checks"],
    }, ensure_ascii=False, indent=2))
    return 0 if manifest["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
