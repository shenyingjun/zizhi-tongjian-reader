from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import production_precision_mining_negatives as neg
from production_precision_mining_negatives import (
    ADJACENT_MERGE,
    GENERATOR_MISTAKE,
    OVERREACH,
    STRICT_PARTIAL,
    _cap_jie_negatives,
    _generate_jie_negatives,
    build_dataset,
    freeze_negatives,
)
import production_precision_verifier as verifier
from production_span_verifier import _resolve, _resolve_group


def _example(text: str) -> dict:
    return {
        "id": "juan-002-jie-0012",
        "juan": 2,
        "jie_index": 12,
        "text": text,
        "segments": [
            {"para_id": 1, "assembled_start": 0, "assembled_end": len(text)},
        ],
    }


class NegativeGenerationTest(unittest.TestCase):
    def test_policies_geometry_and_validity(self):
        example = _example("甲乙丙丁戊")
        references = {(1, 0, 2)}
        generator = {(1, 3, 5)}  # 丁戊 -- a generator mistake

        negatives = _generate_jie_negatives(example, references, generator)

        # Strict partials of 甲乙 -> 乙 and 甲.
        self.assertIn((1, 1, 2), negatives)
        self.assertIn((1, 0, 1), negatives)
        # Right overreach 甲乙丙; left/both overreach are out of paragraph.
        self.assertIn((1, 0, 3), negatives)
        self.assertNotIn((1, -1, 2), negatives)
        # Generator mistake retained.
        self.assertIn((1, 3, 5), negatives)
        self.assertEqual(negatives[(1, 3, 5)]["policies"], [GENERATOR_MISTAKE])
        # Surfaces come from source text, never invented.
        self.assertEqual(negatives[(1, 0, 3)]["surface"], "甲乙丙")

    def test_reference_geometry_never_becomes_negative(self):
        example = _example("甲乙丙丁戊")
        references = {(1, 0, 2), (1, 2, 4)}
        # A generator candidate equal to a reference must be dropped.
        negatives = _generate_jie_negatives(example, references, {(1, 0, 2)})
        self.assertNotIn((1, 0, 2), negatives)
        # Adjacent merge of the two touching references -> 甲乙丙丁.
        self.assertIn((1, 0, 4), negatives)
        self.assertEqual(negatives[(1, 0, 4)]["policies"], [ADJACENT_MERGE])

    def test_overreach_veto_blocks_punctuation(self):
        example = _example("甲乙，丁")
        references = {(1, 0, 2)}
        negatives = _generate_jie_negatives(example, references, set())
        # Right overreach 甲乙， contains punctuation and is vetoed.
        self.assertNotIn((1, 0, 3), negatives)

    def test_dedup_membership_keeps_earliest_primary(self):
        example = _example("甲乙丙丁戊")
        references = {(1, 0, 2)}
        # (1,0,1) is both a generator mistake and a left strict partial.
        negatives = _generate_jie_negatives(example, references, {(1, 0, 1)})
        entry = negatives[(1, 0, 1)]
        self.assertEqual(entry["policies"][0], GENERATOR_MISTAKE)
        self.assertIn(STRICT_PARTIAL, entry["policies"])
        self.assertEqual(len(entry["policies"]), len(set(entry["policies"])))


class CapOrderTest(unittest.TestCase):
    def _entry(self, policy, start, end):
        return {
            "para_id": 1,
            "start": start,
            "end": end,
            "surface": "x",
            "policies": [policy],
            "label_noise_overlap": True,
        }

    def test_round_robin_cap_and_discards(self):
        negatives = {}
        for i in range(3):
            negatives[(1, 10 + i, 11 + i)] = self._entry(GENERATOR_MISTAKE, 10 + i, 11 + i)
            negatives[(1, 20 + i, 21 + i)] = self._entry(STRICT_PARTIAL, 20 + i, 21 + i)
            negatives[(1, 30 + i, 31 + i)] = self._entry(OVERREACH, 30 + i, 31 + i)

        kept, discarded = _cap_jie_negatives(negatives, positives_count=1)

        # cap = 4 * 1 positive.
        self.assertEqual(len(kept), 4)
        self.assertEqual(len(discarded), 5)
        # One from each available policy before any policy gets a second.
        self.assertEqual(
            [row["primary_policy"] for row in kept],
            [GENERATOR_MISTAKE, STRICT_PARTIAL, OVERREACH, GENERATOR_MISTAKE],
        )
        # Ascending geometry within the generator policy.
        gen = [row for row in kept if row["primary_policy"] == GENERATOR_MISTAKE]
        self.assertEqual([r["start"] for r in gen], [10, 11])

    def test_cap_scales_with_positive_count(self):
        negatives = {
            (1, i, i + 1): self._entry(STRICT_PARTIAL, i, i + 1) for i in range(10)
        }
        kept, discarded = _cap_jie_negatives(negatives, positives_count=2)
        self.assertEqual(len(kept), 8)  # 4 * 2
        self.assertEqual(len(discarded), 2)


class FreezeStopRuleTest(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(tempfile.mkdtemp(dir=Path(__file__).parent))
        self._original = neg.build_dataset

    def tearDown(self):
        neg.build_dataset = self._original
        shutil.rmtree(self.scratch, ignore_errors=True)

    def _dataset(self, *, recall_ok=True, floor_ok=True):
        return {
            "examples": [], "positives": [], "negatives": [],
            "precap_negatives": [], "discarded_negatives": [],
            "counts": {"postcap_negatives": 10},
            "oof_recall_gate": {
                "value": 0.99 if recall_ok else 0.96,
                "gate": 0.90,
                "passed": recall_ok,
            },
            "floors": {
                "total_negatives": {"value": 10, "floor": 2000, "passed": floor_ok},
            },
            "bindings": {},
        }

    def test_stops_when_recall_gate_fails(self):
        neg.build_dataset = lambda base: self._dataset(recall_ok=False)
        with self.assertRaises(RuntimeError) as ctx:
            freeze_negatives(self.scratch / "base", self.scratch / "out")
        self.assertIn("0.90", str(ctx.exception))

    def test_stops_when_floor_fails(self):
        neg.build_dataset = lambda base: self._dataset(floor_ok=False)
        with self.assertRaises(RuntimeError) as ctx:
            freeze_negatives(self.scratch / "base", self.scratch / "out")
        self.assertIn("floors not met", str(ctx.exception))


class BuildDatasetBindingTest(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(tempfile.mkdtemp(dir=Path(__file__).parent))

    def tearDown(self):
        shutil.rmtree(self.scratch, ignore_errors=True)

    def test_rejects_wrong_plan_status(self):
        base = self.scratch
        plan = base / "ml-production-precision-mining-plan-v1"
        partition = base / "ml-production-precision-partition-v1"
        plan.mkdir()
        partition.mkdir()
        (partition / "fit.jsonl").write_text("", encoding="utf-8")
        (partition / "manifest.json").write_text(json.dumps({
            "status": "ml_production_precision_partition",
            "outputs": {"fit_sha256": "x"},
        }), encoding="utf-8")
        (plan / "manifest.json").write_text(json.dumps({
            "status": "not_a_plan",
            "mining_only": True,
            "order_seed": 20260813,
            "folds": 5,
            "inputs": {"fit_sha256": "x"},
            "fold_by_juan": {},
            "outputs": {},
        }), encoding="utf-8")
        with self.assertRaises(ValueError):
            build_dataset(base)


class Rev4FeatureTest(unittest.TestCase):
    def test_numeric_feature_size_excludes_generator_metadata(self):
        self.assertEqual(verifier.NUMERIC_SIZE, 1 + 6 + 6 + 2)

    def test_feature_matrix_dimension_and_no_metadata_required(self):
        original = verifier._pool_candidate_encodings

        def fake_pool(examples, candidates, encoder, tokenizer, device):
            hidden = 4
            pooled = [
                {
                    "candidate_mean": np.zeros(hidden, dtype=np.float32),
                    "left_hidden": np.zeros(hidden, dtype=np.float32),
                    "right_hidden": np.zeros(hidden, dtype=np.float32),
                    "context_mean": np.zeros(hidden, dtype=np.float32),
                    "left_character": None,
                    "right_character": "甲",
                    "length": 2,
                    "starts_paragraph": True,
                    "ends_paragraph": False,
                }
                for _ in candidates
            ]
            return hidden, pooled

        verifier._pool_candidate_encodings = fake_pool
        try:
            # Candidates deliberately lack support_count / seed_confidences.
            candidates = [{"id": "x", "para_id": 1, "start": 0, "end": 2}]
            features = verifier._extract_features(
                {}, candidates, None, None, None
            )
        finally:
            verifier._pool_candidate_encodings = original
        self.assertEqual(features.shape, (1, 4 * 4 + 15))
        # Left edge bit set (starts paragraph), right edge bit clear.
        self.assertEqual(features[0, -2], 1.0)
        self.assertEqual(features[0, -1], 0.0)


class Rev4BindingTest(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(tempfile.mkdtemp(dir=Path(__file__).parent))

    def tearDown(self):
        shutil.rmtree(self.scratch, ignore_errors=True)

    def _dirs(self):
        neg_dir = self.scratch / "neg"
        lat = self.scratch / "lat"
        ref = self.scratch / "ref"
        for d in (neg_dir, lat, ref):
            d.mkdir()
        (neg_dir / "candidates.jsonl").write_text("", encoding="utf-8")
        (neg_dir / "examples.jsonl").write_text("", encoding="utf-8")
        (lat / "lattice.jsonl").write_text("", encoding="utf-8")
        (ref / "calibration.jsonl").write_text("", encoding="utf-8")
        (neg_dir / "manifest.json").write_text(json.dumps({
            "status": "wrong",
        }), encoding="utf-8")
        (lat / "manifest.json").write_text(json.dumps({
            "status": "x", "counts": {}}), encoding="utf-8")
        (ref / "manifest.json").write_text(json.dumps({
            "status": "x", "outputs": {}}), encoding="utf-8")
        return neg_dir, lat, ref

    def test_rejects_wrong_negatives_status(self):
        neg_dir, lat, ref = self._dirs()
        with self.assertRaises(ValueError):
            verifier.train_verifier(neg_dir, lat, ref, self.scratch / "out")

    def test_rejects_existing_output(self):
        neg_dir, lat, ref = self._dirs()
        out = self.scratch / "out"
        out.mkdir()
        with self.assertRaises(FileExistsError):
            verifier.train_verifier(neg_dir, lat, ref, out)


class ResolverUnchangedTest(unittest.TestCase):
    def _candidate(self, start, end, score, surface):
        return {
            "id": "juan-001-jie-0001",
            "para_id": 4,
            "start": start,
            "end": end,
            "surface": surface,
            "score": score,
            "support_count": 3,
            "seed_confidences": {"a": 0.9, "b": 0.9, "c": 0.9},
        }

    def test_resolve_group_penalizes_weak_fragments(self):
        selected = _resolve_group(
            [
                self._candidate(0, 3, 0.99, "人物名"),
                self._candidate(0, 1, 0.51, "人"),
                self._candidate(1, 3, 0.51, "物名"),
            ],
            0.50,
        )
        self.assertEqual(
            [(r["start"], r["end"]) for r in selected], [(0, 3)]
        )

    def test_resolve_applies_threshold_and_veto(self):
        vetoed = self._candidate(0, 2, 0.99, "甲乙")
        vetoed["intrinsic_hard_vetoes"] = ["numeric_punctuation_or_symbol"]
        weak = self._candidate(2, 4, 0.10, "丙丁")
        kept = self._candidate(4, 6, 0.99, "戊己")
        selected = _resolve([vetoed, weak, kept], 0.50)
        self.assertEqual(selected, {("juan-001-jie-0001", 4, 4, 6, "戊己")})


if __name__ == "__main__":
    unittest.main()
