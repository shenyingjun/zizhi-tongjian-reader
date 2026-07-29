from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import tempfile
from collections import Counter
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_pack(task: dict, pack: dict, juan: int) -> Counter:
    if (
        pack.get("juan") != juan
        or pack.get("phase") != "assisted"
        or pack.get("diagnostic_only") is not True
    ):
        raise ValueError(f"invalid diagnostic pack identity for juan {juan}")
    text_by_para = {}
    for jie in task["jies"]:
        text = str(jie["text"])
        for segment in jie["segments"]:
            start = int(segment["assembled_start"])
            end = int(segment["assembled_end"])
            text_by_para[int(segment["para_id"])] = text[start:end]
    counts = Counter()
    seen = set()
    by_para: dict[int, list[tuple[int, int]]] = {}
    for candidate in pack.get("candidates", []):
        para_id = int(candidate["para_id"])
        start = int(candidate["start"])
        end = int(candidate["end"])
        geometry = (para_id, start, end)
        paragraph = text_by_para.get(para_id)
        if (
            geometry in seen
            or paragraph is None
            or not 0 <= start < end <= len(paragraph)
            or paragraph[start:end] != candidate.get("surface")
        ):
            raise ValueError(
                f"invalid diagnostic geometry in juan {juan}: {geometry}"
            )
        if candidate.get("id") != f"copilot:{para_id}:{start}:{end}":
            raise ValueError(f"invalid diagnostic candidate id in juan {juan}")
        confidence = candidate.get("confidence")
        reason = candidate.get("review_reason")
        if confidence not in {"high", "medium", "low"}:
            raise ValueError(f"invalid confidence in juan {juan}")
        if (confidence == "high" and reason) or (
            confidence != "high" and not reason
        ):
            raise ValueError(f"invalid review reason in juan {juan}")
        seen.add(geometry)
        by_para.setdefault(para_id, []).append((start, end))
        counts[confidence] += 1
    for spans in by_para.values():
        spans.sort()
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            raise ValueError(f"overlapping diagnostic candidates in juan {juan}")
    return counts


def prepare_diagnostic(
    sealed_tasks: Path,
    copilot_packs: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"diagnostic output already exists: {output_dir}")
    sealed_manifest = _read(sealed_tasks / "manifest.json")
    selections = sealed_manifest.get("selected", [])
    juans = [int(row["juan"]) for row in selections]
    if len(juans) != 5 or len(set(juans)) != 5:
        raise ValueError("diagnostic workflow requires five distinct juans")
    totals = Counter()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        tasks_dir = staging / "tasks"
        assisted_dir = staging / "assisted"
        tasks_dir.mkdir()
        assisted_dir.mkdir()
        selected = []
        for row in selections:
            juan = int(row["juan"])
            task_source = sealed_tasks / f"blind_juan_{juan:03d}.json"
            if _sha256(task_source) != row.get("task_sha256"):
                raise ValueError(
                    f"sealed task hash differs for juan {juan}"
                )
            pack_source = (
                copilot_packs / f"assisted_juan_{juan:03d}.json"
            )
            task = _read(task_source)
            pack = _read(pack_source)
            totals.update(_validate_pack(task, pack, juan))
            task_target = tasks_dir / task_source.name
            pack_target = assisted_dir / pack_source.name
            shutil.copyfile(task_source, task_target)
            shutil.copyfile(pack_source, pack_target)
            selected.append({
                "juan": juan,
                "mode": "diagnostic_assisted",
                "task": task_target.name,
                "task_sha256": _sha256(task_target),
                "pack_sha256": _sha256(pack_target),
            })
        manifest = {
            "schema_version": 1,
            "status": "copilot_assisted_diagnostic",
            "formal_p3": False,
            "candidate_blind": False,
            "human_review_scope": "low_confidence_candidates",
            "selected": selected,
            "counts": {
                "total": sum(totals.values()),
                "high": totals["high"],
                "medium": totals["medium"],
                "low": totals["low"],
            },
        }
        manifest_path = tasks_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for directory in (tasks_dir, assisted_dir):
            for path in directory.iterdir():
                if path.is_file():
                    path.chmod(stat.S_IREAD)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare low-confidence review for Copilot auto-tags."
    )
    parser.add_argument("--sealed-tasks", type=Path, required=True)
    parser.add_argument("--copilot-packs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_diagnostic(
        args.sealed_tasks, args.copilot_packs, args.output
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
