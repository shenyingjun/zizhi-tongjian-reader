from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
from pathlib import Path

import numpy as np

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from p1_train import MODEL_REVISION
from p3_compact import _git_commit_clean
from p3_compact_evaluate import _model_artifact
from production_precision_grouped_data import GROUPED_DATA_STATUS
from production_precision_grouped_verifier import _score
from production_precision_review import _label_geometries
from production_precision_verifier import _build_head, _extract_features
from production_train import _make_read_only
from production_verifier_lattice import _intrinsic_vetoes


REVISION6_STATUS = "ml_production_precision_grouped_verifier_blocked_ai_assisted"
MINING_STATUS = "ml_production_precision_lexical_mining_ai_assisted"
MAX_LENGTH = 8
MIN_PER_JIE = 8
PER_REFERENCE = 2
MIN_RETAINED = 4000
EXPECTED_FIT_JIES = 189
EXPECTED_FIT_JUANS = 28
EXPECTED_REFERENCES = 2483


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _canonical_key(row: dict) -> tuple[int, int, int, int, int]:
    return (
        int(row["juan"]),
        int(row["jie_index"]),
        int(row["para_id"]),
        int(row["start"]),
        int(row["end"]),
    )


def _candidate_id(key: tuple[int, int, int, int, int]) -> str:
    raw = ":".join(str(value) for value in key).encode("ascii")
    return hashlib.sha256(raw).hexdigest()[:20]


def _overlaps(
    geometry: tuple[int, int, int],
    references: set[tuple[int, int, int]],
) -> bool:
    para_id, start, end = geometry
    return any(
        para_id == ref_para and start < ref_end and ref_start < end
        for ref_para, ref_start, ref_end in references
    )


def _enumerate_candidates(
    example: dict,
    references: set[tuple[int, int, int]],
    existing: set[tuple[int, int, int]],
) -> list[dict]:
    rows = []
    for segment in example["segments"]:
        para_id = int(segment["para_id"])
        assembled_start = int(segment["assembled_start"])
        assembled_end = int(segment["assembled_end"])
        paragraph = str(example["text"])[assembled_start:assembled_end]
        for start in range(len(paragraph)):
            for end in range(start + 1, min(len(paragraph), start + MAX_LENGTH) + 1):
                geometry = (para_id, start, end)
                if (
                    geometry in existing
                    or _overlaps(geometry, references)
                ):
                    continue
                surface = paragraph[start:end]
                if _intrinsic_vetoes(surface):
                    continue
                rows.append({
                    "id": str(example["id"]),
                    "juan": int(example["juan"]),
                    "jie_index": int(example["jie_index"]),
                    "para_id": para_id,
                    "start": start,
                    "end": end,
                    "surface": surface,
                })
    return rows


def _load_revision6(revision6_root: Path, device):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    manifest_path = revision6_root / "manifest.json"
    manifest = _read(manifest_path)
    head_path = revision6_root / "existence" / "head.safetensors"
    scaler_path = revision6_root / "existence" / "scaler.npz"
    encoder_dir = revision6_root / "encoder"
    if (
        manifest.get("status") != REVISION6_STATUS
        or manifest.get("confirmation_read") is not False
        or manifest.get("model_revision") != MODEL_REVISION
        or manifest.get("fit_inventory", {}).get("existence_head_sha256")
        != _sha256(head_path)
        or manifest.get("fit_inventory", {}).get("existence_scaler_sha256")
        != _sha256(scaler_path)
        or manifest.get("encoder_artifact") != _model_artifact(encoder_dir)
    ):
        raise ValueError("revision-7 mining-model binding differs")
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir, use_fast=True)
    encoder = AutoModel.from_pretrained(encoder_dir).to(device)
    head = _build_head(int(manifest["control"]["feature_dim"])).to(device)
    head.load_state_dict(load_file(head_path, device=str(device)))
    head.eval()
    scaler = np.load(scaler_path)
    mean = scaler["mean"].astype(np.float32)
    scale = scaler["scale"].astype(np.float32)
    if len(mean) != int(manifest["control"]["feature_dim"]):
        raise ValueError("revision-7 mining scaler differs")
    return manifest, manifest_path, encoder, tokenizer, head, mean, scale


def mine_lexical_negatives(
    grouped_root: Path,
    revision6_root: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"lexical mining output exists: {output_dir}")
    grouped_manifest_path = grouped_root / "manifest.json"
    grouped_manifest = _read(grouped_manifest_path)
    examples_path = grouped_root / "examples.jsonl"
    existence_path = grouped_root / "existence.jsonl"
    examples = _read_jsonl(examples_path)
    existence = _read_jsonl(existence_path)
    outputs = grouped_manifest.get("outputs", {})
    if (
        grouped_manifest.get("status") != GROUPED_DATA_STATUS
        or grouped_manifest.get("confirmation_read") is not False
        or outputs.get("examples_sha256") != _sha256(examples_path)
        or outputs.get("existence_sha256") != _sha256(existence_path)
        or len(examples) != EXPECTED_FIT_JIES
    ):
        raise ValueError("revision-7 grouped-data binding differs")

    existing_by_id: dict[str, set[tuple[int, int, int]]] = {}
    for row in existence:
        existing_by_id.setdefault(str(row["id"]), set()).add((
            int(row["para_id"]), int(row["start"]), int(row["end"])
        ))
    references_by_id = {
        str(example["id"]): _label_geometries(example) for example in examples
    }
    if sum(map(len, references_by_id.values())) != EXPECTED_REFERENCES:
        raise ValueError("revision-7 fit reference inventory differs")

    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    (
        revision6_manifest,
        revision6_manifest_path,
        encoder,
        tokenizer,
        head,
        mean,
        scale,
    ) = _load_revision6(revision6_root, device)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    retained = []
    jie_counts = []
    pool_hashes = {}
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        pool_dir = staging / "pools"
        pool_dir.mkdir()
        for example in sorted(
            examples, key=lambda row: (int(row["juan"]), int(row["jie_index"]))
        ):
            identity = str(example["id"])
            references = references_by_id[identity]
            candidates = _enumerate_candidates(
                example, references, existing_by_id.get(identity, set())
            )
            if not candidates:
                raise RuntimeError(f"revision-7 empty mining pool: {identity}")
            features = _extract_features(
                {identity: example}, candidates, encoder, tokenizer, device
            )
            scores = _score(
                head, mean, scale, features, device, sigmoid=True
            )
            ranked = sorted(
                zip(candidates, scores),
                key=lambda item: (
                    -float(item[1]),
                    int(item[0]["para_id"]),
                    int(item[0]["start"]),
                    int(item[0]["end"]),
                ),
            )
            quota = min(len(ranked), max(MIN_PER_JIE, PER_REFERENCE * len(references)))
            pool_path = pool_dir / f"{identity}.npz"
            np.savez_compressed(
                pool_path,
                para_id=np.asarray(
                    [int(row["para_id"]) for row, _ in ranked], dtype=np.int32
                ),
                start=np.asarray(
                    [int(row["start"]) for row, _ in ranked], dtype=np.int32
                ),
                end=np.asarray(
                    [int(row["end"]) for row, _ in ranked], dtype=np.int32
                ),
                score=np.asarray(
                    [float(score) for _, score in ranked], dtype=np.float32
                ),
            )
            pool_hashes[identity] = _sha256(pool_path)
            for row, score in ranked[:quota]:
                key = _canonical_key(row)
                retained.append({
                    **row,
                    "candidate_id": _candidate_id(key),
                    "score": float(np.float32(score)),
                })
            jie_counts.append({
                "id": identity,
                "juan": int(example["juan"]),
                "jie_index": int(example["jie_index"]),
                "references": len(references),
                "pool": len(ranked),
                "retained": quota,
                "pool_sha256": pool_hashes[identity],
            })

        retained_juans = {int(row["juan"]) for row in retained}
        if (
            len(retained) < MIN_RETAINED
            or len(retained_juans) != EXPECTED_FIT_JUANS
        ):
            raise RuntimeError(
                "revision-7 lexical mining floors not met: "
                f"{len(retained)} rows / {len(retained_juans)} juans"
            )
        retained_path = staging / "retained.jsonl"
        _write_jsonl(retained_path, retained)
        counts_path = staging / "jie-counts.jsonl"
        _write_jsonl(counts_path, jie_counts)
        git_commit = _git_commit_clean()
        manifest = {
            "schema_version": 1,
            "status": MINING_STATUS,
            "revision": 7,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "git_commit": git_commit,
            "candidate_universe": {
                "length_codepoints": [1, MAX_LENGTH],
                "overlap_reference": False,
                "intrinsic_hard_veto": True,
                "excludes_grouped_existence_geometry": True,
            },
            "quota": {
                "minimum_per_jie": MIN_PER_JIE,
                "per_reference": PER_REFERENCE,
            },
            "counts": {
                "fit_jies": len(examples),
                "fit_juans": len({int(row["juan"]) for row in examples}),
                "fit_references": sum(map(len, references_by_id.values())),
                "pool_candidates": sum(row["pool"] for row in jie_counts),
                "retained_candidates": len(retained),
                "retained_juans": len(retained_juans),
            },
            "floors": {
                "retained_candidates": MIN_RETAINED,
                "retained_juans": EXPECTED_FIT_JUANS,
                "passed": True,
            },
            "bindings": {
                "grouped_manifest_sha256": _sha256(grouped_manifest_path),
                "revision6_manifest_sha256": _sha256(revision6_manifest_path),
                "revision6_existence_head_sha256": revision6_manifest[
                    "fit_inventory"
                ]["existence_head_sha256"],
                "revision6_existence_scaler_sha256": revision6_manifest[
                    "fit_inventory"
                ]["existence_scaler_sha256"],
                "revision6_encoder_artifact": revision6_manifest["encoder_artifact"],
            },
            "outputs": {
                "retained_sha256": _sha256(retained_path),
                "jie_counts_sha256": _sha256(counts_path),
                "pool_files": pool_hashes,
            },
            "claim_limit": (
                "Fit-only model-guided candidate selection. Retained rows are not "
                "negative labels until source-hidden verification completes."
            ),
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
        description="Mine revision-7 fit-only lexical hard-negative candidates."
    )
    parser.add_argument("--grouped-data", type=Path, required=True)
    parser.add_argument("--revision6", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = mine_lexical_negatives(
        args.grouped_data, args.revision6, args.output
    )
    print(json.dumps({
        "status": manifest["status"],
        "counts": manifest["counts"],
        "floors": manifest["floors"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
