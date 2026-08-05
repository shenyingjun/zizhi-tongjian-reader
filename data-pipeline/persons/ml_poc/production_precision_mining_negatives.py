from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_mining_plan import (
    FOLD_NUMBERS,
    MINING_ORDER_SEED,
    PARTITION_STATUS,
)
from production_train import SEEDS, _make_read_only
from production_verifier_lattice import (
    MINIMUM_CONFIDENCE,
    _geometry,
    _intrinsic_vetoes,
    _source_surface,
)


PLAN_STATUS = "ml_production_precision_mining_plan"
OOF_STATUS = "ml_production_precision_mining_oof_predictions"
OOF_RECALL_GATE = 0.90

PLAN_ARTIFACT = "ml-production-precision-mining-plan-v1"
PARTITION_ARTIFACT = "ml-production-precision-partition-v1"


def _holdout_artifact(fold: int, seed: int) -> str:
    return f"ml-production-precision-mining-fold-{fold}-seed-{seed}-holdout-v1"


# Closed negative policies in the exact numbered order of section 5.3.2. Their rank
# defines earliest-primary provenance and the per-jie round-robin schedule.
GENERATOR_MISTAKE = "generator_mistake"
STRICT_PARTIAL = "strict_partial"
OVERREACH = "one_character_overreach"
ADJACENT_MERGE = "adjacent_merge"
POLICY_ORDER = (GENERATOR_MISTAKE, STRICT_PARTIAL, OVERREACH, ADJACENT_MERGE)

# Post-cap training-inventory floors (section 5.3.2). Adjacent merge has no floor.
FLOOR_TOTAL_NEGATIVES = 2000
FLOOR_GENERATOR_MISTAKES = 150
FLOOR_STRICT_PARTIAL = 100
FLOOR_OVERREACH = 100
FLOOR_JUANS_WITH_NEGATIVES = 20
FIT_JUANS = 28

# The negatives-per-positive cap within each jie.
NEGATIVES_PER_POSITIVE = 4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def _paragraph_length(example: dict, para_id: int) -> int:
    matches = [
        segment for segment in example["segments"]
        if int(segment["para_id"]) == para_id
    ]
    if len(matches) != 1:
        raise ValueError(f"candidate paragraph differs: {example['id']}")
    segment = matches[0]
    return int(segment["assembled_end"]) - int(segment["assembled_start"])


def _surface(example: dict, para_id: int, start: int, end: int) -> str | None:
    """Return the source surface for an in-paragraph geometry, or None if the
    geometry is not source-valid (empty or out of the paragraph)."""
    if start < 0 or end <= start:
        return None
    if end > _paragraph_length(example, para_id):
        return None
    return _source_surface(example, (para_id, start, end, ""))


def _overlaps_reference(
    references: set[tuple[int, int, int]],
    para_id: int,
    start: int,
    end: int,
) -> bool:
    return any(
        ref_para == para_id and start < ref_end and ref_start < end
        for ref_para, ref_start, ref_end in references
    )


def _generate_jie_negatives(
    example: dict,
    references: set[tuple[int, int, int]],
    generator_geometries: set[tuple[int, int, int]],
) -> dict[tuple[int, int, int], dict]:
    """Build the four closed negative policies for one fit jie.

    Returns a geometry-keyed mapping with earliest primary provenance and the
    complete ordered policy membership. Geometries equal to a reference are never
    emitted as negatives.
    """
    negatives: dict[tuple[int, int, int], dict] = {}

    def add(geometry: tuple[int, int, int], policy: str) -> None:
        para_id, start, end = geometry
        if geometry in references:
            return
        surface = _surface(example, para_id, start, end)
        if surface is None:
            return
        entry = negatives.get(geometry)
        if entry is None:
            negatives[geometry] = {
                "para_id": para_id,
                "start": start,
                "end": end,
                "surface": surface,
                "policies": [policy],
                "label_noise_overlap": _overlaps_reference(
                    references, para_id, start, end
                ),
            }
        elif policy not in entry["policies"]:
            entry["policies"].append(policy)

    # Policy 1: generator mistakes -- OOF candidates that are not exact references.
    for geometry in sorted(generator_geometries):
        if geometry not in references:
            add(geometry, GENERATOR_MISTAKE)

    for para_id, start, end in sorted(references):
        # Policy 2: strict partials (remove one edge character) for length >= 2.
        if end - start >= 2:
            add((para_id, start + 1, end), STRICT_PARTIAL)
            add((para_id, start, end - 1), STRICT_PARTIAL)
        # Policy 3: one-character overreach left, right, and both directions.
        for candidate in (
            (para_id, start - 1, end),
            (para_id, start, end + 1),
            (para_id, start - 1, end + 1),
        ):
            surface = _surface(example, candidate[0], candidate[1], candidate[2])
            if surface is None or _intrinsic_vetoes(surface):
                continue
            add(candidate, OVERREACH)

    # Policy 4: adjacent merge of two references sharing an interior boundary.
    ordered_refs = sorted(references)
    for left in ordered_refs:
        for right in ordered_refs:
            if left[0] == right[0] and left[2] == right[1] and left != right:
                add((left[0], left[1], right[2]), ADJACENT_MERGE)

    return negatives


def _cap_jie_negatives(
    negatives: dict[tuple[int, int, int], dict],
    positives_count: int,
) -> tuple[list[dict], list[dict]]:
    """Round-robin cap at four negatives per positive within one jie.

    Buckets by primary (earliest) policy, takes ascending geometry within a
    policy, and rotates policies in the numbered order so every available policy
    yields one example before any policy yields its second. Returns
    ``(kept, discarded)`` entries.
    """
    cap = NEGATIVES_PER_POSITIVE * positives_count
    buckets: dict[str, list[dict]] = {policy: [] for policy in POLICY_ORDER}
    for geometry in sorted(negatives):
        entry = negatives[geometry]
        entry["primary_policy"] = entry["policies"][0]
        buckets[entry["primary_policy"]].append(entry)
    for policy in POLICY_ORDER:
        buckets[policy].sort(key=lambda row: (row["para_id"], row["start"], row["end"]))
    kept: list[dict] = []
    cursors = {policy: 0 for policy in POLICY_ORDER}
    while len(kept) < cap:
        progressed = False
        for policy in POLICY_ORDER:
            if len(kept) >= cap:
                break
            cursor = cursors[policy]
            if cursor < len(buckets[policy]):
                kept.append(buckets[policy][cursor])
                cursors[policy] = cursor + 1
                progressed = True
        if not progressed:
            break
    kept_ids = {id(entry) for entry in kept}
    discarded = [
        buckets[policy][index]
        for policy in POLICY_ORDER
        for index in range(len(buckets[policy]))
        if id(buckets[policy][index]) not in kept_ids
    ]
    return kept, discarded


def build_dataset(base: Path) -> dict:
    """Validate mining artifacts and construct the fit-only verifier examples.

    Returns the training inventory plus complete pre-cap, discarded, audit, and
    binding metadata. No filesystem writes and no git access happen here so the
    counts can be measured before the immutable freeze.
    """
    plan_root = base / PLAN_ARTIFACT
    partition_root = base / PARTITION_ARTIFACT
    plan_manifest_path = plan_root / "manifest.json"
    plan_manifest = _read(plan_manifest_path)
    partition_manifest_path = partition_root / "manifest.json"
    partition_manifest = _read(partition_manifest_path)
    fit_path = partition_root / "fit.jsonl"
    fit_raw = fit_path.read_bytes()
    if (
        plan_manifest.get("status") != PLAN_STATUS
        or plan_manifest.get("mining_only") is not True
        or plan_manifest.get("order_seed") != MINING_ORDER_SEED
        or plan_manifest.get("folds") != len(FOLD_NUMBERS)
        or partition_manifest.get("status") != PARTITION_STATUS
        or plan_manifest.get("inputs", {}).get("fit_sha256") != _digest(fit_raw)
        or partition_manifest.get("outputs", {}).get("fit_sha256") != _digest(fit_raw)
    ):
        raise ValueError("hard-negative partition/plan binding differs")

    fit_rows = _read_jsonl(fit_path)
    fit_by_id = {str(row["id"]): row for row in fit_rows}
    if len(fit_by_id) != len(fit_rows) or len(fit_rows) != 189:
        raise ValueError("fit inventory differs from the frozen partition")

    fold_by_juan = {int(j): int(f) for j, f in plan_manifest["fold_by_juan"].items()}
    plan_holdout_sha = {
        int(fold): plan_manifest["outputs"][str(fold)]["holdout_sha256"]
        for fold in FOLD_NUMBERS
    }

    prediction_bindings: dict[str, dict] = {}
    # fold -> seed -> {id -> row}
    fold_seed_predictions: dict[int, dict[int, dict[str, dict]]] = {}
    for fold in FOLD_NUMBERS:
        fold_seed_predictions[fold] = {}
        for seed in SEEDS:
            root = base / _holdout_artifact(fold, seed)
            manifest_path = root / "manifest.json"
            predictions_path = root / "predictions.json"
            manifest = _read(manifest_path)
            rows = _read_json(predictions_path)
            if (
                manifest.get("status") != OOF_STATUS
                or manifest.get("mining_only") is not True
                or manifest.get("fold") != fold
                or manifest.get("seed") != seed
                or manifest.get("plan_manifest_sha256") != _sha256(plan_manifest_path)
                or manifest.get("holdout_sha256") != plan_holdout_sha[fold]
                or manifest.get("predictions_sha256") != _sha256(predictions_path)
                or not isinstance(rows, list)
            ):
                raise ValueError(
                    f"hard-negative OOF binding differs: fold {fold} seed {seed}"
                )
            indexed = {str(row["id"]): row for row in rows}
            if len(indexed) != len(rows):
                raise ValueError(f"duplicate OOF jie: fold {fold} seed {seed}")
            fold_seed_predictions[fold][seed] = indexed
            prediction_bindings[_holdout_artifact(fold, seed)] = {
                "manifest_sha256": _sha256(manifest_path),
                "predictions_sha256": _sha256(predictions_path),
            }

    # Every fit jie must be present in exactly its own fold's holdout, under all
    # three seeds, and nowhere else.
    for fold in FOLD_NUMBERS:
        expected_ids = {
            str(row["id"])
            for row in fit_rows
            if fold_by_juan[int(row["juan"])] == fold
        }
        for seed in SEEDS:
            if set(fold_seed_predictions[fold][seed]) != expected_ids:
                raise ValueError(f"OOF jie inventory differs: fold {fold} seed {seed}")

    positives: list[dict] = []
    kept_negatives: list[dict] = []
    precap_negatives: list[dict] = []
    discarded_negatives: list[dict] = []
    covered_references = 0
    total_references = 0
    precap_policy_membership = dict.fromkeys(POLICY_ORDER, 0)
    postcap_policy_membership = dict.fromkeys(POLICY_ORDER, 0)
    postcap_primary_counts = dict.fromkeys(POLICY_ORDER, 0)
    label_noise_count = 0
    juans_with_negatives: set[int] = set()

    for identity in sorted(fit_by_id):
        example = fit_by_id[identity]
        juan = int(example["juan"])
        jie_index = int(example["jie_index"])
        fold = fold_by_juan[juan]
        seed_predictions = fold_seed_predictions[fold]

        reference_geoms = {
            _geometry(span)
            for span in seed_predictions[SEEDS[0]][identity]["reference_spans"]
        }
        for seed in SEEDS[1:]:
            other = {
                _geometry(span)
                for span in seed_predictions[seed][identity]["reference_spans"]
            }
            if other != reference_geoms:
                raise ValueError(f"OOF seed references differ: {identity}")
        if len(reference_geoms) != int(example["span_count"]):
            raise ValueError(f"reference count differs from span_count: {identity}")

        references = {(g[0], g[1], g[2]) for g in reference_geoms}
        for para_id, start, end in references:
            if _surface(example, para_id, start, end) is None:
                raise ValueError(f"reference geometry not source-valid: {identity}")

        # Concatenated OOF one-seed / >=0.30 lattice for this jie.
        generator_geoms: set[tuple[int, int, int]] = set()
        for seed in SEEDS:
            for span in seed_predictions[seed][identity]["prediction_spans"]:
                if float(span["confidence"]) < MINIMUM_CONFIDENCE:
                    continue
                geometry = _geometry(span)
                if _source_surface(example, geometry) != geometry[3]:
                    raise ValueError(f"OOF candidate surface differs: {identity}")
                generator_geoms.add((geometry[0], geometry[1], geometry[2]))

        total_references += len(references)
        covered_references += len(references & generator_geoms)

        for para_id, start, end in sorted(references):
            surface = _surface(example, para_id, start, end)
            positives.append({
                "id": identity,
                "juan": juan,
                "jie_index": jie_index,
                "para_id": para_id,
                "start": start,
                "end": end,
                "surface": surface,
                "label": 1,
                "policy": "reference",
                "policy_membership": [],
                "label_noise_overlap": False,
                "in_oof_lattice": (para_id, start, end) in generator_geoms,
            })

        jie_negatives = _generate_jie_negatives(
            example, references, generator_geoms
        )
        for entry in jie_negatives.values():
            for policy in entry["policies"]:
                precap_policy_membership[policy] += 1
            precap_negatives.append({
                "id": identity,
                "juan": juan,
                "jie_index": jie_index,
                "para_id": entry["para_id"],
                "start": entry["start"],
                "end": entry["end"],
                "surface": entry["surface"],
                "policy_membership": list(entry["policies"]),
                "primary_policy": entry["policies"][0],
                "label_noise_overlap": entry["label_noise_overlap"],
            })

        kept, discarded = _cap_jie_negatives(jie_negatives, len(references))
        for entry in kept:
            row = {
                "id": identity,
                "juan": juan,
                "jie_index": jie_index,
                "para_id": entry["para_id"],
                "start": entry["start"],
                "end": entry["end"],
                "surface": entry["surface"],
                "label": 0,
                "policy": entry["primary_policy"],
                "policy_membership": list(entry["policies"]),
                "label_noise_overlap": entry["label_noise_overlap"],
                "in_oof_lattice": (
                    (entry["para_id"], entry["start"], entry["end"])
                    in generator_geoms
                ),
            }
            kept_negatives.append(row)
            postcap_primary_counts[entry["primary_policy"]] += 1
            for policy in entry["policies"]:
                postcap_policy_membership[policy] += 1
            if entry["label_noise_overlap"]:
                label_noise_count += 1
            juans_with_negatives.add(juan)
        for entry in discarded:
            discarded_negatives.append({
                "id": identity,
                "juan": juan,
                "jie_index": jie_index,
                "para_id": entry["para_id"],
                "start": entry["start"],
                "end": entry["end"],
                "surface": entry["surface"],
                "policy_membership": list(entry["policies"]),
                "primary_policy": entry["primary_policy"],
                "reason": "per_jie_round_robin_cap",
            })

    oof_recall = covered_references / total_references if total_references else 0.0

    floors = {
        "total_negatives": {
            "value": len(kept_negatives),
            "floor": FLOOR_TOTAL_NEGATIVES,
            "passed": len(kept_negatives) >= FLOOR_TOTAL_NEGATIVES,
        },
        "generator_mistake_membership": {
            "value": postcap_policy_membership[GENERATOR_MISTAKE],
            "floor": FLOOR_GENERATOR_MISTAKES,
            "passed": postcap_policy_membership[GENERATOR_MISTAKE]
            >= FLOOR_GENERATOR_MISTAKES,
        },
        "strict_partial_membership": {
            "value": postcap_policy_membership[STRICT_PARTIAL],
            "floor": FLOOR_STRICT_PARTIAL,
            "passed": postcap_policy_membership[STRICT_PARTIAL] >= FLOOR_STRICT_PARTIAL,
        },
        "overreach_membership": {
            "value": postcap_policy_membership[OVERREACH],
            "floor": FLOOR_OVERREACH,
            "passed": postcap_policy_membership[OVERREACH] >= FLOOR_OVERREACH,
        },
        "juans_with_negatives": {
            "value": len(juans_with_negatives),
            "floor": FLOOR_JUANS_WITH_NEGATIVES,
            "passed": len(juans_with_negatives) >= FLOOR_JUANS_WITH_NEGATIVES,
        },
    }
    recall_gate = {
        "value": oof_recall,
        "gate": OOF_RECALL_GATE,
        "passed": oof_recall >= OOF_RECALL_GATE,
    }

    example_rows = [
        {
            "id": str(row["id"]),
            "juan": int(row["juan"]),
            "jie_index": int(row["jie_index"]),
            "text": row["text"],
            "segments": row["segments"],
            "labels": row["labels"],
        }
        for row in sorted(fit_rows, key=lambda r: str(r["id"]))
    ]

    return {
        "examples": example_rows,
        "positives": positives,
        "negatives": kept_negatives,
        "precap_negatives": precap_negatives,
        "discarded_negatives": discarded_negatives,
        "counts": {
            "fit_jies": len(fit_by_id),
            "positives": len(positives),
            "total_references": total_references,
            "covered_references": covered_references,
            "precap_negatives": len(precap_negatives),
            "postcap_negatives": len(kept_negatives),
            "discarded_negatives": len(discarded_negatives),
            "precap_policy_membership": precap_policy_membership,
            "postcap_policy_membership": postcap_policy_membership,
            "postcap_primary_policy": postcap_primary_counts,
            "label_noise_overlap_negatives": label_noise_count,
            "juans_with_negatives": len(juans_with_negatives),
        },
        "oof_recall_gate": recall_gate,
        "floors": floors,
        "bindings": {
            "plan_manifest_sha256": _sha256(plan_manifest_path),
            "partition_manifest_sha256": _sha256(partition_manifest_path),
            "fit_sha256": _digest(fit_raw),
            "oof_predictions": prediction_bindings,
        },
    }


def freeze_negatives(base: Path, output_dir: Path) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"hard-negative dataset exists: {output_dir}")
    dataset = build_dataset(base)

    failing = [name for name, gate in dataset["floors"].items() if not gate["passed"]]
    if not dataset["oof_recall_gate"]["passed"]:
        raise RuntimeError(
            "OOF candidate recall below 0.90 pipeline-sanity gate: "
            + json.dumps(dataset["oof_recall_gate"])
        )
    if failing:
        raise RuntimeError(
            "hard-negative floors not met; stop before verifier training: "
            + json.dumps({name: dataset["floors"][name] for name in failing})
            + " | full counts: "
            + json.dumps(dataset["counts"])
        )

    veto_source = Path(__file__).with_name("production_verifier_lattice.py")
    git_commit = _git_commit_clean()

    training_rows = sorted(
        dataset["positives"] + dataset["negatives"],
        key=lambda row: (
            row["juan"],
            row["jie_index"],
            row["para_id"],
            row["start"],
            row["end"],
            -int(row["label"]),
            str(row["policy"]),
        ),
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        _write_jsonl(staging / "examples.jsonl", dataset["examples"])
        _write_jsonl(staging / "candidates.jsonl", training_rows)
        _write_jsonl(staging / "precap_negatives.jsonl", dataset["precap_negatives"])
        _write_jsonl(staging / "discarded.jsonl", dataset["discarded_negatives"])
        manifest = {
            "schema_version": 1,
            "status": "ml_production_precision_mining_hard_negatives_ai_assisted",
            "mining_only": True,
            "formal_grade": False,
            "formal_evaluation": False,
            "eligible_for_production": False,
            "eligible_for_production_precision_claim": False,
            "confirmation_read": False,
            "git_commit": git_commit,
            "order_seed": MINING_ORDER_SEED,
            "policy_order": list(POLICY_ORDER),
            "negatives_per_positive_cap": NEGATIVES_PER_POSITIVE,
            "oof_recall_gate": dataset["oof_recall_gate"],
            "floors": dataset["floors"],
            "counts": dataset["counts"],
            "intrinsic_veto": {
                "function": "production_verifier_lattice._intrinsic_vetoes",
                "source_sha256": _sha256(veto_source),
                "minimum_seed_confidence": MINIMUM_CONFIDENCE,
            },
            "bindings": dataset["bindings"],
            "outputs": {
                "examples_sha256": _sha256(staging / "examples.jsonl"),
                "candidates_sha256": _sha256(staging / "candidates.jsonl"),
                "precap_negatives_sha256": _sha256(staging / "precap_negatives.jsonl"),
                "discarded_sha256": _sha256(staging / "discarded.jsonl"),
            },
            "claim_limit": (
                "Fit-only mining hard-negative verifier dataset. Diagnostic "
                "training material only; confirmation is unread and this dataset "
                "cannot authorize deployment or production promotion."
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
        description=(
            "Freeze the revision-4 fit-only mining hard-negative verifier dataset."
        )
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_negatives(args.base, args.output)
    print(json.dumps({
        "status": manifest["status"],
        "counts": manifest["counts"],
        "floors": manifest["floors"],
        "oof_recall_gate": manifest["oof_recall_gate"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
