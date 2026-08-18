from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from p3_compact import _git_commit_clean
from production_precision_mining_negatives import (
    ADJACENT_MERGE,
    GENERATOR_MISTAKE,
    OVERREACH,
    PLAN_ARTIFACT,
    PLAN_STATUS,
    STRICT_PARTIAL,
)
from production_precision_mining_plan import MINING_ORDER_SEED
from production_train import _make_read_only


HARD_NEGATIVE_STATUS = "ml_production_precision_mining_hard_negatives_ai_assisted"
GROUPED_DATA_STATUS = "ml_production_precision_grouped_data_ai_assisted"
RANK_POLICY_ORDER = (
    GENERATOR_MISTAKE,
    STRICT_PARTIAL,
    OVERREACH,
    ADJACENT_MERGE,
)
MAX_NEGATIVES_PER_POSITIVE = 8
MIN_EXISTENCE_POSITIVES = 2000
MIN_EXISTENCE_NEGATIVES = 150
MIN_NEGATIVE_JUANS = 20
MIN_RANK_PAIRS = 2000
MIN_RANK_JUANS = 20
MIN_POLICY_PAIRS = 100


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


def _geometry(row: dict) -> tuple[int, int, int]:
    return int(row["para_id"]), int(row["start"]), int(row["end"])


def _key(row: dict) -> tuple[str, int, int, int]:
    para_id, start, end = _geometry(row)
    return str(row["id"]), para_id, start, end


def _overlaps(left: tuple[int, int, int], right: tuple[int, int, int]) -> bool:
    return (
        left[0] == right[0]
        and left[1] < right[2]
        and right[1] < left[2]
    )


def _candidate(row: dict, fold: int) -> dict:
    return {
        "id": str(row["id"]),
        "juan": int(row["juan"]),
        "jie_index": int(row["jie_index"]),
        "fold": fold,
        "para_id": int(row["para_id"]),
        "start": int(row["start"]),
        "end": int(row["end"]),
        "surface": str(row["surface"]),
    }


def build_grouped_data(base: Path, negatives_root: Path) -> dict:
    manifest_path = negatives_root / "manifest.json"
    manifest = _read(manifest_path)
    examples_path = negatives_root / "examples.jsonl"
    candidates_path = negatives_root / "candidates.jsonl"
    precap_path = negatives_root / "precap_negatives.jsonl"
    examples = _read_jsonl(examples_path)
    candidates = _read_jsonl(candidates_path)
    precap = _read_jsonl(precap_path)
    outputs = manifest.get("outputs", {})
    if (
        manifest.get("status") != HARD_NEGATIVE_STATUS
        or manifest.get("mining_only") is not True
        or manifest.get("confirmation_read") is not False
        or outputs.get("examples_sha256") != _sha256(examples_path)
        or outputs.get("candidates_sha256") != _sha256(candidates_path)
        or outputs.get("precap_negatives_sha256") != _sha256(precap_path)
    ):
        raise ValueError("grouped data hard-negative binding differs")

    plan_root = base / PLAN_ARTIFACT
    plan_manifest_path = plan_root / "manifest.json"
    plan = _read(plan_manifest_path)
    if (
        plan.get("status") != PLAN_STATUS
        or plan.get("mining_only") is not True
        or plan.get("order_seed") != MINING_ORDER_SEED
        or manifest.get("bindings", {}).get("plan_manifest_sha256")
        != _sha256(plan_manifest_path)
    ):
        raise ValueError("grouped data mining-plan binding differs")
    fold_by_juan = {
        int(juan): int(fold) for juan, fold in plan["fold_by_juan"].items()
    }

    examples_by_id = {str(row["id"]): row for row in examples}
    if len(examples_by_id) != len(examples) or len(examples) != 189:
        raise ValueError("grouped data fit inventory differs")
    positives = [row for row in candidates if int(row["label"]) == 1]
    references_by_id: dict[str, list[dict]] = {}
    for row in positives:
        references_by_id.setdefault(str(row["id"]), []).append(row)
    if len(positives) != int(manifest["counts"]["positives"]):
        raise ValueError("grouped data positive inventory differs")

    generator_by_key: dict[tuple, dict] = {}
    for row in positives:
        if row.get("in_oof_lattice") is True:
            generator_by_key[_key(row)] = row
    for row in precap:
        if GENERATOR_MISTAKE in row["policy_membership"]:
            key = _key(row)
            if key in generator_by_key:
                raise ValueError(f"duplicate OOF candidate geometry: {key}")
            generator_by_key[key] = row

    existence_rows = []
    negative_juans = set()
    for key in sorted(generator_by_key):
        row = generator_by_key[key]
        references = references_by_id.get(str(row["id"]), [])
        overlapping = [
            reference
            for reference in references
            if _overlaps(_geometry(row), _geometry(reference))
        ]
        fold = fold_by_juan[int(row["juan"])]
        existence = {
            **_candidate(row, fold),
            "label": int(bool(overlapping)),
            "overlapping_references": [
                {
                    "para_id": int(reference["para_id"]),
                    "start": int(reference["start"]),
                    "end": int(reference["end"]),
                    "surface": str(reference["surface"]),
                }
                for reference in sorted(overlapping, key=_geometry)
            ],
            "policy_membership": list(row.get("policy_membership", [])),
        }
        existence_rows.append(existence)
        if not overlapping:
            negative_juans.add(int(row["juan"]))

    policy_index = {
        policy: index for index, policy in enumerate(RANK_POLICY_ORDER)
    }
    rank_pairs = []
    discarded_pairs = []
    pair_membership = dict.fromkeys(RANK_POLICY_ORDER, 0)
    rank_juans = set()
    precap_by_id: dict[str, list[dict]] = {}
    for row in precap:
        if any(policy in row["policy_membership"] for policy in RANK_POLICY_ORDER):
            precap_by_id.setdefault(str(row["id"]), []).append(row)

    for positive in sorted(positives, key=_key):
        alternatives = [
            row
            for row in precap_by_id.get(str(positive["id"]), [])
            if _overlaps(_geometry(positive), _geometry(row))
        ]
        alternatives.sort(key=lambda row: (
            min(policy_index[p] for p in row["policy_membership"]),
            int(row["para_id"]),
            int(row["start"]),
            int(row["end"]),
        ))
        kept = alternatives[:MAX_NEGATIVES_PER_POSITIVE]
        discarded = alternatives[MAX_NEGATIVES_PER_POSITIVE:]
        fold = fold_by_juan[int(positive["juan"])]
        for negative in kept:
            memberships = [
                policy
                for policy in RANK_POLICY_ORDER
                if policy in negative["policy_membership"]
            ]
            rank_pairs.append({
                "positive": _candidate(positive, fold),
                "negative": {
                    **_candidate(negative, fold),
                    "policy_membership": memberships,
                },
            })
            for policy in memberships:
                pair_membership[policy] += 1
            rank_juans.add(int(positive["juan"]))
        for negative in discarded:
            discarded_pairs.append({
                "positive": _candidate(positive, fold),
                "negative": {
                    **_candidate(negative, fold),
                    "policy_membership": [
                        policy
                        for policy in RANK_POLICY_ORDER
                        if policy in negative["policy_membership"]
                    ],
                },
                "reason": "eight_negatives_per_positive_cap",
            })

    existence_positives = sum(int(row["label"]) for row in existence_rows)
    existence_negatives = len(existence_rows) - existence_positives
    floors = {
        "existence_positives": {
            "value": existence_positives,
            "floor": MIN_EXISTENCE_POSITIVES,
            "passed": existence_positives >= MIN_EXISTENCE_POSITIVES,
        },
        "existence_negatives": {
            "value": existence_negatives,
            "floor": MIN_EXISTENCE_NEGATIVES,
            "passed": existence_negatives >= MIN_EXISTENCE_NEGATIVES,
        },
        "existence_negative_juans": {
            "value": len(negative_juans),
            "floor": MIN_NEGATIVE_JUANS,
            "passed": len(negative_juans) >= MIN_NEGATIVE_JUANS,
        },
        "rank_pairs": {
            "value": len(rank_pairs),
            "floor": MIN_RANK_PAIRS,
            "passed": len(rank_pairs) >= MIN_RANK_PAIRS,
        },
        "rank_juans": {
            "value": len(rank_juans),
            "floor": MIN_RANK_JUANS,
            "passed": len(rank_juans) >= MIN_RANK_JUANS,
        },
        "strict_partial_pairs": {
            "value": pair_membership[STRICT_PARTIAL],
            "floor": MIN_POLICY_PAIRS,
            "passed": pair_membership[STRICT_PARTIAL] >= MIN_POLICY_PAIRS,
        },
        "overreach_pairs": {
            "value": pair_membership[OVERREACH],
            "floor": MIN_POLICY_PAIRS,
            "passed": pair_membership[OVERREACH] >= MIN_POLICY_PAIRS,
        },
    }
    return {
        "examples": examples,
        "existence": existence_rows,
        "rank_pairs": rank_pairs,
        "discarded_rank_pairs": discarded_pairs,
        "counts": {
            "fit_jies": len(examples),
            "fit_references": len(positives),
            "oof_candidates": len(existence_rows),
            "existence_positives": existence_positives,
            "existence_negatives": existence_negatives,
            "existence_negative_juans": len(negative_juans),
            "rank_pairs": len(rank_pairs),
            "discarded_rank_pairs": len(discarded_pairs),
            "rank_juans": len(rank_juans),
            "rank_pair_policy_membership": pair_membership,
        },
        "floors": floors,
        "bindings": {
            "hard_negatives_manifest_sha256": _sha256(manifest_path),
            "mining_plan_manifest_sha256": _sha256(plan_manifest_path),
            "oof_predictions": manifest["bindings"]["oof_predictions"],
        },
    }


def freeze_grouped_data(
    base: Path, negatives_root: Path, output_dir: Path
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"grouped verifier data exists: {output_dir}")
    dataset = build_grouped_data(base, negatives_root)
    failing = [
        name for name, gate in dataset["floors"].items() if not gate["passed"]
    ]
    if failing:
        raise RuntimeError(
            "revision-6 grouped-data floors not met: "
            + json.dumps(
                {name: dataset["floors"][name] for name in failing},
                separators=(",", ":"),
            )
        )

    git_commit = _git_commit_clean()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        _write_jsonl(staging / "examples.jsonl", dataset["examples"])
        _write_jsonl(staging / "existence.jsonl", dataset["existence"])
        _write_jsonl(staging / "rank-pairs.jsonl", dataset["rank_pairs"])
        _write_jsonl(
            staging / "discarded-rank-pairs.jsonl",
            dataset["discarded_rank_pairs"],
        )
        manifest = {
            "schema_version": 1,
            "status": GROUPED_DATA_STATUS,
            "revision": 6,
            "formal_grade": False,
            "eligible_for_production": False,
            "confirmation_read": False,
            "git_commit": git_commit,
            "rank_policy_order": list(RANK_POLICY_ORDER),
            "max_negatives_per_positive": MAX_NEGATIVES_PER_POSITIVE,
            "existence_label": "overlaps_any_fit_reference",
            "counts": dataset["counts"],
            "floors": dataset["floors"],
            "bindings": dataset["bindings"],
            "outputs": {
                "examples_sha256": _sha256(staging / "examples.jsonl"),
                "existence_sha256": _sha256(staging / "existence.jsonl"),
                "rank_pairs_sha256": _sha256(staging / "rank-pairs.jsonl"),
                "discarded_rank_pairs_sha256": _sha256(
                    staging / "discarded-rank-pairs.jsonl"
                ),
            },
            "claim_limit": (
                "Fit-only AI-assisted diagnostic data for revision-6 existence "
                "and boundary heads; confirmation remains unread."
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
        description="Freeze revision-6 existence and boundary training inventories."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--negatives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_grouped_data(args.base, args.negatives, args.output)
    print(json.dumps({
        "status": manifest["status"],
        "counts": manifest["counts"],
        "floors": manifest["floors"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
