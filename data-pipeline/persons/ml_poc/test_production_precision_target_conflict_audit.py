from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from production_precision_grouped_verifier import _sha256
from production_precision_target_conflict_audit import (
    DECISIONS,
    TASK_STATUS,
    build_tasks,
    derive_conflict_components,
    freeze_decisions,
)


HERE = Path(__file__).parent


def _candidate(
    candidate_id: str,
    start: int,
    end: int,
    label: str,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "juan": 1,
        "jie_index": 1,
        "para_id": 7,
        "start": start,
        "end": end,
        "surface": "甲乙丙丁戊己"[start:end],
        "audited_label": label,
    }


def _target(candidate: dict, start: int, end: int) -> dict:
    return {
        "candidate_id": candidate["candidate_id"],
        "decision": "targets",
        "wrong_candidate": {
            key: candidate[key]
            for key in ("para_id", "start", "end", "surface")
        },
        "targets": [{
            "para_id": 7,
            "start": start,
            "end": end,
            "surface": "甲乙丙丁戊己"[start:end],
        }],
    }


class ConflictDerivationTest(unittest.TestCase):
    def test_candidate_closure_follows_newly_owned_additions(self):
        left = _candidate("left", 0, 2, "exact_person")
        middle = _candidate("middle", 1, 3, "exact_person")
        bridge = _candidate("bridge", 2, 4, "wrong_boundary")
        reached = _candidate("reached", 4, 6, "not_person")
        components = derive_conflict_components(
            [reached, bridge, middle, left],
            [_target(bridge, 3, 5)],
        )
        self.assertEqual(1, len(components))
        self.assertEqual(
            {"left", "middle", "bridge", "reached"},
            set(components[0]["candidate_ids"]),
        )
        self.assertEqual(
            {(0, 2), (1, 3), (3, 5)},
            {
                (row["start"], row["end"])
                for row in components[0]["addition_membership"]
            },
        )

    def test_exact_dedup_retains_all_owners_and_order_invariance(self):
        left = _candidate("left", 0, 2, "exact_person")
        same = _candidate("same", 0, 2, "wrong_boundary")
        overlap = _candidate("overlap", 1, 3, "exact_person")
        targets = [_target(same, 0, 2)]
        forward = derive_conflict_components([left, same, overlap], targets)
        reverse = derive_conflict_components(
            [overlap, same, left], list(reversed(targets))
        )
        self.assertEqual(forward, reverse)
        owners = {
            (row["start"], row["end"]): row["owner_candidate_ids"]
            for row in forward[0]["addition_membership"]
        }
        self.assertEqual(["left", "same"], owners[(0, 2)])

    def test_build_task_exposes_only_neutral_candidate_geometry(self):
        audits = [
            _candidate("left", 0, 2, "exact_person"),
            _candidate("right", 1, 3, "exact_person"),
        ]
        examples = {
            "juan-001-jie-0001": {
                "id": "juan-001-jie-0001",
                "juan": 1,
                "jie_index": 1,
                "text": "甲乙丙",
                "segments": [{
                    "para_id": 7,
                    "assembled_start": 0,
                    "assembled_end": 3,
                }],
            }
        }
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            output = Path(temporary) / "built"
            with (
                patch(
                    "production_precision_target_conflict_audit."
                    "_load_validated_sources",
                    return_value=(examples, audits, [], {"source": "hash"}),
                ),
                patch(
                    "production_precision_target_conflict_audit."
                    "EXPECTED_CONFLICTS",
                    1,
                ),
                patch(
                    "production_precision_target_conflict_audit."
                    "_git_commit_clean",
                    return_value="test-commit",
                ),
            ):
                manifest = build_tasks(
                    Path("grouped"), Path("audit"), Path("targets"), output
                )
            task = json.loads(
                (output / manifest["selected"][0]["task"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                {"neutral_id", "para_id", "start", "end", "surface"},
                set(task["included_candidates"][0]),
            )
            serialized = json.dumps(task, ensure_ascii=False)
            for hidden in (
                "candidate_id",
                "audited_label",
                "original_label",
                "wrong_candidate",
                "rationale",
                "translation",
            ):
                self.assertNotIn(hidden, serialized)


class ConflictFreezeTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, str]:
        tasks = root / "tasks-root"
        raw = root / "raw"
        (tasks / "tasks").mkdir(parents=True)
        raw.mkdir()
        task_id = "component"
        task = {
            "schema_version": 1,
            "status": TASK_STATUS,
            "phase": "revision-13-candidate-closed-conflict-audit",
            "task_id": task_id,
            "juan": 1,
            "jie_index": 1,
            "review_scope": "current-numbered-jie-only",
            "jie": {
                "text": "甲乙丙",
                "segments": [{
                    "para_id": 7,
                    "assembled_start": 0,
                    "assembled_end": 3,
                }],
            },
            "included_candidates": [{
                "neutral_id": "neutral",
                "para_id": 7,
                "start": 0,
                "end": 2,
                "surface": "甲乙",
            }],
            "request": (
                "Return every exact individual-person span in the same "
                "paragraph that overlaps any included candidate."
            ),
            "allowed_decisions": sorted(DECISIONS),
            "component_binding_sha256": "binding",
        }
        task_path = tasks / "tasks" / "task_component.json"
        task_path.write_text(
            json.dumps(task, ensure_ascii=False), encoding="utf-8"
        )
        sealed_path = tasks / "sealed-component-membership.jsonl"
        sealed = {
            "task_id": task_id,
            "candidate_ids": ["upstream"],
            "candidates": [{
                "candidate_id": "upstream",
                "juan": 1,
                "jie_index": 1,
                "para_id": 7,
                "start": 0,
                "end": 2,
                "surface": "甲乙",
            }],
            "component_binding_sha256": "binding",
        }
        sealed_path.write_text(
            json.dumps(sealed, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "status": TASK_STATUS,
            "candidate_closed": True,
            "candidate_model_blind": True,
            "baseline_references_hidden": True,
            "upstream_labels_hidden": True,
            "model_scores_hidden": True,
            "prior_targets_hidden": True,
            "neighboring_jies_hidden": True,
            "translations_hidden": True,
            "knowledge_bases_hidden": True,
            "confirmation_read": False,
            "bindings": {"source": hashlib.sha256(b"source").hexdigest()},
            "counts": {"tasks": 1},
            "selected": [{
                "task_id": task_id,
                "juan": 1,
                "jie_index": 1,
                "task": "tasks/task_component.json",
                "task_sha256": _sha256(task_path),
            }],
            "sealed_component_membership_sha256": _sha256(sealed_path),
        }
        (tasks / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return tasks, raw, task_id

    def test_freeze_allows_target_equal_to_candidate(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            tasks, raw, task_id = self._fixture(root)
            task_hash = _sha256(tasks / "tasks" / "task_component.json")
            (raw / "answer.json").write_text(json.dumps({
                "schema_version": 1,
                "phase": "revision-13-candidate-closed-conflict-audit",
                "adjudicator": "copilot_teacher",
                "task_id": task_id,
                "task_sha256": task_hash,
                "decision": "targets",
                "targets": [{
                    "para_id": 7,
                    "start": 0,
                    "end": 2,
                    "surface": "甲乙",
                }],
                "rationale": "The candidate is the exact person span.",
            }, ensure_ascii=False), encoding="utf-8")
            output = root / "frozen"
            with (
                patch(
                    "production_precision_target_conflict_audit."
                    "EXPECTED_CONFLICTS",
                    1,
                ),
                patch(
                    "production_precision_target_conflict_audit."
                    "_git_commit_clean",
                    return_value="test-commit",
                ),
            ):
                result = freeze_decisions(tasks, raw, output)
            self.assertEqual(0, result["counts"]["uncertain"])
            decision = json.loads(
                (output / "decisions.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(["upstream"], decision[
                "component_candidate_ids"
            ])
            self.assertNotIn("audited_label", decision)


if __name__ == "__main__":
    unittest.main()
