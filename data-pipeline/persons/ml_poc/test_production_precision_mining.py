from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from production_precision_mining_plan import (
    FOLD_NUMBERS,
    MINING_ORDER_SEED,
    STRATA,
    assign_folds,
    plan_folds,
)
from production_precision_mining_train import run_mining_training
from production_precision_mining_infer import infer_holdout


def _synthetic_groups(count: int = 28) -> dict[int, dict[str, int]]:
    return {
        juan: {
            "examples": 5 + juan % 4,
            "spans": 40 + juan,
            "uniform_random": 3 + juan % 3,
            "role_appellation": 1 + juan % 2,
            "foreign_title": int(juan % 5 == 0),
            "boundary_anaphora": 1 + juan % 2,
        }
        for juan in range(1, count + 1)
    }


def _synthetic_fit(count: int = 28) -> tuple[list[dict], dict]:
    rows = []
    stratum_by_key = {}
    strata_cycle = list(STRATA)
    for juan in range(1, count + 1):
        jie_total = 5 + juan % 4
        for jie_index in range(1, jie_total + 1):
            key = (juan, jie_index)
            rows.append({
                "juan": juan,
                "jie_index": jie_index,
                "span_count": 1 + (juan + jie_index) % 3,
            })
            stratum_by_key[key] = strata_cycle[(juan + jie_index) % len(strata_cycle)]
    return rows, stratum_by_key


class MiningFoldAssignmentTest(unittest.TestCase):
    def test_assignment_is_deterministic_and_juan_grouped(self):
        groups = _synthetic_groups()

        first = assign_folds(groups)
        second = assign_folds(groups)

        self.assertEqual(first, second)
        self.assertEqual(set(groups), set(first))
        self.assertEqual(set(FOLD_NUMBERS), set(first.values()))

    def test_ties_resolve_to_lowest_fold_number(self):
        groups = {juan: dict.fromkeys(
            ("examples", "spans", *STRATA), 1
        ) for juan in range(1, 6)}

        assignments = assign_folds(groups)

        first_placed = min(assignments)  # symmetric empty folds -> lowest number
        # The first juan in the deterministic order lands in fold 1 on ties.
        from production_precision_mining_plan import order_juans

        ordered = order_juans(groups, MINING_ORDER_SEED)
        self.assertEqual(assignments[ordered[0]], 1)
        self.assertIn(first_placed, assignments)


class MiningInventoryTest(unittest.TestCase):
    def test_complete_disjoint_and_juan_grouped_holdouts(self):
        fit_rows, stratum_by_key = _synthetic_fit()

        plan = plan_folds(fit_rows, stratum_by_key)

        all_keys = {(r["juan"], r["jie_index"]) for r in fit_rows}
        seen = set()
        for fold in FOLD_NUMBERS:
            holdout_keys = {
                (r["juan"], r["jie_index"]) for r in plan["holdout_rows"][fold]
            }
            train_keys = {
                (r["juan"], r["jie_index"]) for r in plan["train_rows"][fold]
            }
            # Disjoint train/holdout and complement relationship per fold.
            self.assertEqual(holdout_keys | train_keys, all_keys)
            self.assertEqual(holdout_keys & train_keys, set())
            # Every jie appears in exactly one holdout across folds.
            self.assertEqual(holdout_keys & seen, set())
            seen |= holdout_keys
            # No juan is split across train/holdout of the same fold.
            holdout_juans = {r["juan"] for r in plan["holdout_rows"][fold]}
            train_juans = {r["juan"] for r in plan["train_rows"][fold]}
            self.assertEqual(holdout_juans & train_juans, set())
        self.assertEqual(seen, all_keys)

        # Every juan lands in exactly one fold.
        self.assertEqual(set(plan["fold_by_juan"]), {r["juan"] for r in fit_rows})
        self.assertEqual(set(plan["fold_by_juan"].values()), set(FOLD_NUMBERS))

    def test_mismatched_strata_inventory_rejected(self):
        fit_rows, stratum_by_key = _synthetic_fit(count=6)
        stratum_by_key.pop(next(iter(stratum_by_key)))

        with self.assertRaises(ValueError):
            plan_folds(fit_rows, stratum_by_key)


class MiningTrainerControlTest(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(tempfile.mkdtemp(dir=Path(__file__).parent))

    def tearDown(self):
        shutil.rmtree(self.scratch, ignore_errors=True)

    def _fake_plan(self) -> Path:
        plan = self.scratch / "plan"
        fold_dir = plan / "folds" / "fold-1"
        fold_dir.mkdir(parents=True)
        train = fold_dir / "train.jsonl"
        holdout = fold_dir / "holdout.jsonl"
        train.write_text('{"juan":1,"jie_index":1,"span_count":1}\n', encoding="utf-8")
        holdout.write_text('{"juan":2,"jie_index":1,"span_count":1}\n', encoding="utf-8")
        train_sha = hashlib.sha256(train.read_bytes()).hexdigest()
        holdout_sha = hashlib.sha256(holdout.read_bytes()).hexdigest()
        manifest = {
            "status": "ml_production_precision_mining_plan",
            "mining_only": True,
            "eligible_for_deployment": False,
            "order_seed": MINING_ORDER_SEED,
            "outputs": {
                "1": {"train_sha256": train_sha, "holdout_sha256": holdout_sha},
            },
            "fold_by_juan": {"1": 2, "2": 1},
            "fold_summaries": {
                "1": {"holdout_juans": [2], "holdout_examples": 1},
            },
        }
        (plan / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return plan

    def test_rejects_invalid_fold(self):
        with self.assertRaises(ValueError):
            run_mining_training(
                self.scratch / "plan", self.scratch / "out", fold=6, seed=20260727
            )

    def test_rejects_invalid_seed(self):
        with self.assertRaises(ValueError):
            run_mining_training(
                self.scratch / "plan", self.scratch / "out", fold=1, seed=123
            )

    def test_rejects_existing_output(self):
        plan = self._fake_plan()
        out = self.scratch / "out"
        out.mkdir()
        with self.assertRaises(FileExistsError):
            run_mining_training(plan, out, fold=1, seed=20260727)

    def test_rejects_tampered_binding(self):
        plan = self._fake_plan()
        manifest = json.loads((plan / "manifest.json").read_text(encoding="utf-8"))
        manifest["outputs"]["1"]["train_sha256"] = "deadbeef"
        (plan / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            run_mining_training(plan, self.scratch / "out", fold=1, seed=20260727)

    def test_rejects_wrong_status(self):
        plan = self._fake_plan()
        manifest = json.loads((plan / "manifest.json").read_text(encoding="utf-8"))
        manifest["status"] = "something_else"
        (plan / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            run_mining_training(plan, self.scratch / "out", fold=1, seed=20260727)


class MiningInferControlTest(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(tempfile.mkdtemp(dir=Path(__file__).parent))

    def tearDown(self):
        shutil.rmtree(self.scratch, ignore_errors=True)

    def _fake(self, *, fold_in_control: int = 1) -> tuple[Path, Path]:
        plan = self.scratch / "plan"
        fold_dir = plan / "folds" / "fold-1"
        fold_dir.mkdir(parents=True)
        holdout = fold_dir / "holdout.jsonl"
        holdout.write_text('{"juan":2,"jie_index":1,"span_count":1}\n', encoding="utf-8")
        holdout_sha = hashlib.sha256(holdout.read_bytes()).hexdigest()
        (plan / "manifest.json").write_text(json.dumps({
            "status": "ml_production_precision_mining_plan",
            "mining_only": True,
            "order_seed": MINING_ORDER_SEED,
            "outputs": {"1": {"holdout_sha256": holdout_sha}},
            "fold_by_juan": {"1": 2, "2": 1},
            "fold_summaries": {"1": {"holdout_juans": [2], "holdout_examples": 1}},
        }), encoding="utf-8")
        model = self.scratch / "model"
        model.mkdir()
        (model / "report.json").write_text(json.dumps({
            "mining_control": {
                "fold": fold_in_control,
                "seed": 20260727,
                "base_model_revision": "x",
                "checkpoint_selection": "fixed_epoch_5",
                "mining_only": True,
                "eligible_for_deployment": False,
                "eligible_for_production": False,
                "plan_manifest_sha256": "x",
                "holdout_sha256": holdout_sha,
                "model_artifact": "x",
            }
        }), encoding="utf-8")
        return model, plan

    def test_rejects_invalid_fold(self):
        with self.assertRaises(ValueError):
            infer_holdout(
                self.scratch / "m", self.scratch / "p", self.scratch / "o",
                fold=0, seed=20260727,
            )

    def test_rejects_invalid_seed(self):
        with self.assertRaises(ValueError):
            infer_holdout(
                self.scratch / "m", self.scratch / "p", self.scratch / "o",
                fold=1, seed=999,
            )

    def test_rejects_fold_mismatch_between_model_and_request(self):
        model, plan = self._fake(fold_in_control=3)
        with self.assertRaises(ValueError):
            infer_holdout(
                model, plan, self.scratch / "out", fold=1, seed=20260727
            )

    def test_rejects_existing_output(self):
        model, plan = self._fake()
        out = self.scratch / "out"
        out.mkdir()
        with self.assertRaises(FileExistsError):
            infer_holdout(model, plan, out, fold=1, seed=20260727)


if __name__ == "__main__":
    unittest.main()
