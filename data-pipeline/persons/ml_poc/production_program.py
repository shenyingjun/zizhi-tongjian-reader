from __future__ import annotations

import argparse
import hashlib
import json
import random
import stat
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

from core import assemble_jies


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TEXT = REPO_ROOT / "web" / "public" / "text"
BOUNDARY_GUIDE = HERE / "BOUNDARY_GUIDE.md"
PRODUCTION_SPEC = HERE / "PRODUCTION_SPEC.md"

MIN_CHARS = 20
MAX_CHARS = 600
TRAIN_COUNTS = {
    "uniform_random": 95,
    "role_appellation": 20,
    "foreign_title": 5,
    "boundary_anaphora": 20,
}
DEV_COUNTS = {
    "uniform_random": 22,
    "role_appellation": 7,
    "foreign_title": 5,
    "boundary_anaphora": 6,
}
FORMAL_RESERVE = 160
REPLACEMENT_ROUND_RESERVE = sum(TRAIN_COUNTS.values()) + sum(DEV_COUNTS.values())
FORMAL_FOREIGN_RESERVE = 20
CHALLENGE_COHORT = 200

ROLE_TERMS = (
    "太后", "太子", "皇后", "皇帝", "丞相", "大将军", "皇太后",
    "使君", "贵人", "夫人",
)
FOREIGN_TERMS = (
    "单于", "可汗", "谷蠡王", "左贤王", "右贤王", "达干", "叶护",
    "俟斤", "特勒", "大莫弗", "昆弥", "阏氏",
)
BOUNDARY_ANAPHORA_TERMS = (
    "字", "名", "姓", "更名", "弟", "兄", "子", "父", "母", "氏",
    "王", "公主", "皇后", "可汗",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _git_commit_clean() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    ).stdout
    if status:
        raise RuntimeError(
            "refusing to freeze a production program from a dirty worktree"
        )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    ).stdout.strip()


def load_exact_exclusions(path: Path) -> tuple[set[tuple[int, int]], dict]:
    manifest = _load(path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "ml_production_exact_jie_exclusions"
        or manifest.get("complete") is not True
    ):
        raise ValueError("exact-jie exclusion inventory is not complete")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("exclusion inventory must bind its source artifacts")
    consumed = manifest.get("consumed")
    if not isinstance(consumed, list) or not consumed:
        raise ValueError("exclusion inventory has no consumed jies")
    rows: set[tuple[int, int]] = set()
    for item in consumed:
        if not isinstance(item, dict):
            raise ValueError("exclusion row must be an object")
        juan = int(item["juan"])
        jie_index = int(item["jie_index"])
        if not 1 <= juan <= 294 or jie_index < 0:
            raise ValueError(f"invalid exclusion row: {item}")
        key = juan, jie_index
        if key in rows:
            raise ValueError(f"duplicate exclusion row: {key}")
        rows.add(key)
    sealed = {
        (int(item["juan"]), int(item["jie_index"]))
        for item in manifest.get("sealed", [])
    }
    if not sealed.issubset(rows):
        raise ValueError("every sealed jie must also be consumed")
    return rows, manifest


def eligible_jies(
    source_dir: Path,
    excluded: set[tuple[int, int]],
) -> tuple[list[dict], dict[int, Path]]:
    rows = []
    source_paths = {}
    for juan in range(1, 295):
        path = source_dir / f"juan_{juan:03d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing corpus source: {path}")
        source_paths[juan] = path
        source = _load(path)
        for jie in assemble_jies(source["paragraphs"]):
            key = juan, int(jie.index)
            length = len(jie.text)
            if (
                key in excluded
                or jie.number is None
                or not MIN_CHARS <= length <= MAX_CHARS
            ):
                continue
            rows.append({
                "juan": juan,
                "jie_index": int(jie.index),
                "jie_number": int(jie.number),
                "text": jie.text,
                "segments": [asdict(segment) for segment in jie.segments],
                "characters": length,
            })
    return rows, source_paths


def _term_score(row: dict, terms: tuple[str, ...]) -> int:
    return sum(str(row["text"]).count(term) for term in terms)


def select_program_rows(
    frame: list[dict],
    *,
    seed: int,
    train_counts: dict[str, int] = TRAIN_COUNTS,
    dev_counts: dict[str, int] = DEV_COUNTS,
) -> list[dict]:
    available = {
        (int(row["juan"]), int(row["jie_index"])): row for row in frame
    }
    if len(available) != len(frame):
        raise ValueError("sampling frame contains duplicate jies")
    original = dict(available)
    selected = []
    cohorts = {}
    for stratum, terms in (
        ("role_appellation", ROLE_TERMS),
        ("foreign_title", FOREIGN_TERMS),
        ("boundary_anaphora", BOUNDARY_ANAPHORA_TERMS),
    ):
        ranked = sorted(
            (
                (key, row, _term_score(row, terms))
                for key, row in original.items()
            ),
            key=lambda item: (
                item[2],
                int(item[1]["characters"]),
                -item[0][0],
                -item[0][1],
            ),
            reverse=True,
        )
        cohorts[stratum] = [
            item for item in ranked[:CHALLENGE_COHORT] if item[2] > 0
        ]

    def draw_uniform(
        split: str,
        count: int,
        stream: int,
    ) -> None:
        if len(available) < count:
            raise ValueError(f"not enough jies for {split} uniform sample")
        keys = random.Random(seed + stream).sample(sorted(available), count)
        for key in keys:
            selected.append({
                "split": split,
                "stratum": "uniform_random",
                "term_score": None,
                **available.pop(key),
            })

    def draw_challenge(
        split: str,
        stratum: str,
        count: int,
        stream: int,
    ) -> None:
        cohort = [
            item for item in cohorts[stratum] if item[0] in available
        ]
        if len(cohort) < count:
            raise ValueError(f"not enough {stratum} challenge jies")
        chosen = random.Random(seed + stream).sample(cohort, count)
        for key, row, score in chosen:
            selected.append({
                "split": split,
                "stratum": stratum,
                "term_score": score,
                **available.pop(key),
            })

    streams = iter(range(1, 30))
    for split, counts in (("train", train_counts), ("development", dev_counts)):
        draw_uniform(split, counts["uniform_random"], next(streams))
    for split, counts in (("train", train_counts), ("development", dev_counts)):
        draw_challenge(
            split, "role_appellation", counts["role_appellation"], next(streams),
        )
        draw_challenge(
            split, "foreign_title", counts["foreign_title"], next(streams),
        )
        draw_challenge(
            split, "boundary_anaphora",
            counts["boundary_anaphora"], next(streams),
        )

    expected = sum(train_counts.values()) + sum(dev_counts.values())
    if len(selected) != expected:
        raise AssertionError("selected task count differs")
    reserve = FORMAL_RESERVE + REPLACEMENT_ROUND_RESERVE
    if len(available) < reserve:
        raise ValueError(
            f"program leaves {len(available)} eligible jies; "
            f"{reserve} are required for formal and replacement reserves"
        )
    remaining_foreign = sum(
        _term_score(row, FOREIGN_TERMS) > 0 for row in available.values()
    )
    if remaining_foreign < FORMAL_FOREIGN_RESERVE:
        raise ValueError(
            f"program leaves {remaining_foreign} foreign-title jies; "
            f"{FORMAL_FOREIGN_RESERVE} are reserved for formal evaluation"
        )
    return sorted(
        selected,
        key=lambda row: (
            row["split"], row["stratum"], row["juan"], row["jie_index"]
        ),
    )


def _task(row: dict) -> dict:
    juan = int(row["juan"])
    return {
        "schema_version": 1,
        "phase": "copilot_double_pass",
        "candidate_model_blind": True,
        "juan": juan,
        "instructions": (
            "Independently mark every main-text person span in each sampled jie. "
            "Apply BOUNDARY_GUIDE.md using target-jie evidence only. Do not read "
            "another pass, model/rule/v1 output, identities, sealed references, "
            "translations, notes, or text outside this task."
        ),
        "jies": [{
            "jie_index": row["jie_index"],
            "jie_number": row["jie_number"],
            "text": row["text"],
            "segments": row["segments"],
            "annotations": [],
        }],
    }


def prepare_program(
    output_dir: Path,
    exclusion_manifest_path: Path,
    *,
    source_dir: Path = TEXT,
    seed: int,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"production program output exists: {output_dir}")
    git_commit = _git_commit_clean()
    excluded, exclusions = load_exact_exclusions(exclusion_manifest_path)
    frame, source_paths = eligible_jies(source_dir, excluded)
    selected = select_program_rows(frame, seed=seed)
    manifest = {
        "schema_version": 1,
        "status": "ml_production_round_tasks_before_labeling",
        "program_spec_sha256": _sha256(PRODUCTION_SPEC),
        "boundary_guide_sha256": _sha256(BOUNDARY_GUIDE),
        "git_commit": git_commit,
        "selection_commitment_sha256": hashlib.sha256(
            str(seed).encode("ascii")
        ).hexdigest(),
        "context_mode": "target_only",
        "candidate_model_blind": True,
        "model_predictions_generated": False,
        "rules_loaded": False,
        "v1_loaded": False,
        "identity_data_loaded": False,
        "exclusion_manifest_sha256": _sha256(exclusion_manifest_path),
        "excluded_jies": len(excluded),
        "sealed_jies": len(exclusions.get("sealed", [])),
        "sampling_frame": {
            "eligible_jies": len(frame),
            "min_characters": MIN_CHARS,
            "max_characters": MAX_CHARS,
            "formal_reserve": FORMAL_RESERVE,
            "formal_foreign_reserve": FORMAL_FOREIGN_RESERVE,
            "replacement_round_reserve": REPLACEMENT_ROUND_RESERVE,
        },
        "labeling_protocol": {
            "pass_a": "independent_recall_first",
            "pass_b": "independent_exact_boundary_first",
            "pass_visibility": "mutually_hidden_candidate_free_main_text_only",
            "human_review": (
                "all disagreements and explicit-low candidates, deterministic "
                "20pct consensus-positive and consensus-negative audits"
            ),
        },
        "split_counts": {
            "train": sum(row["split"] == "train" for row in selected),
            "development": sum(
                row["split"] == "development" for row in selected
            ),
        },
        "tasks": [],
    }
    private_manifest = {
        "schema_version": 1,
        "status": "ml_production_private_task_roles",
        "selection_seed": seed,
        "selected_jies": [],
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        task_dir = staging / "tasks"
        private_dir = staging / "private"
        task_dir.mkdir()
        private_dir.mkdir()
        for row in selected:
            identity = f"{seed}:{row['juan']}:{row['jie_index']}"
            task_id = hashlib.sha256(identity.encode("ascii")).hexdigest()[:20]
            task_path = task_dir / f"task_{task_id}.json"
            task_path.write_text(
                json.dumps(_task(row), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest["tasks"].append({
                "task_id": task_id,
                "task": str(Path("tasks") / task_path.name),
                "task_sha256": _sha256(task_path),
                "source_sha256": _sha256(source_paths[int(row["juan"])]),
            })
            private_manifest["selected_jies"].append({
                "task_id": task_id,
                **{
                    key: value for key, value in row.items()
                    if key not in {"text", "segments"}
                },
            })
        private_path = private_dir / "selection.json"
        private_path.write_text(
            json.dumps(
                private_manifest, ensure_ascii=False, indent=2
            ) + "\n",
            encoding="utf-8",
        )
        manifest["private_selection_sha256"] = _sha256(private_path)
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in staging.rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IREAD)
        for path in sorted(
            (path for path in staging.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            path.chmod(stat.S_IREAD | stat.S_IEXEC)
        staging.chmod(stat.S_IREAD | stat.S_IEXEC)
        staging.replace(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze fresh candidate-blind ML production train/dev tasks."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-dir", type=Path, default=TEXT)
    args = parser.parse_args()
    manifest = prepare_program(
        args.output,
        args.exclusions,
        source_dir=args.source_dir,
        seed=args.seed,
    )
    print(json.dumps({
        "selected": manifest["split_counts"],
        "eligible_jies": manifest["sampling_frame"]["eligible_jies"],
        "excluded_jies": manifest["excluded_jies"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
