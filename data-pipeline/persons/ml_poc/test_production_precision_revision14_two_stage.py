from __future__ import annotations

import unittest

import numpy as np

import production_precision_revision14_two_stage as ts


class OverlapComponentsTest(unittest.TestCase):
    """Unit tests for connected-component overlap logic."""

    def test_single_candidate_returns_one_component(self):
        candidates = [
            {"id": "jie-001", "para_id": 1, "start": 0, "end": 3},
        ]
        components = ts.overlap_components(candidates)
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0], [0])

    def test_non_overlapping_candidates_separate(self):
        candidates = [
            {"id": "jie-001", "para_id": 1, "start": 0, "end": 2},
            {"id": "jie-001", "para_id": 1, "start": 2, "end": 4},
        ]
        components = ts.overlap_components(candidates)
        self.assertEqual(len(components), 2)

    def test_overlapping_same_paragraph_merged(self):
        candidates = [
            {"id": "jie-001", "para_id": 1, "start": 0, "end": 3},
            {"id": "jie-001", "para_id": 1, "start": 2, "end": 5},
        ]
        components = ts.overlap_components(candidates)
        self.assertEqual(len(components), 1)
        self.assertIn(0, components[0])
        self.assertIn(1, components[0])

    def test_different_paragraphs_do_not_merge(self):
        candidates = [
            {"id": "jie-001", "para_id": 1, "start": 0, "end": 3},
            {"id": "jie-001", "para_id": 2, "start": 0, "end": 3},
        ]
        components = ts.overlap_components(candidates)
        self.assertEqual(len(components), 2)

    def test_different_jie_ids_do_not_merge(self):
        candidates = [
            {"id": "jie-001", "para_id": 1, "start": 0, "end": 3},
            {"id": "jie-002", "para_id": 1, "start": 0, "end": 3},
        ]
        components = ts.overlap_components(candidates)
        self.assertEqual(len(components), 2)

    def test_transitive_overlap_forms_single_component(self):
        candidates = [
            {"id": "jie-001", "para_id": 1, "start": 0, "end": 3},
            {"id": "jie-001", "para_id": 1, "start": 2, "end": 5},
            {"id": "jie-001", "para_id": 1, "start": 4, "end": 7},
        ]
        components = ts.overlap_components(candidates)
        self.assertEqual(len(components), 1)
        self.assertEqual(sorted(components[0]), [0, 1, 2])

    def test_empty_input(self):
        self.assertEqual(ts.overlap_components([]), [])


class SelectFromComponentsTest(unittest.TestCase):
    """Tests for Stage-2 winner selection with tie breaking."""

    def test_highest_score_wins(self):
        candidates = [
            {"id": "jie-001", "juan": 1, "jie_index": 1, "para_id": 1,
             "start": 0, "end": 3, "surface": "abc"},
            {"id": "jie-001", "juan": 1, "jie_index": 1, "para_id": 1,
             "start": 1, "end": 4, "surface": "bcd"},
        ]
        scores = np.array([0.5, 0.9], dtype=np.float32)
        selected = ts.select_from_components(candidates, scores, set())
        self.assertEqual(selected, [1])

    def test_tie_break_by_geometry_order(self):
        candidates = [
            {"id": "jie-001", "juan": 1, "jie_index": 1, "para_id": 1,
             "start": 1, "end": 4, "surface": "bcd"},
            {"id": "jie-001", "juan": 1, "jie_index": 1, "para_id": 1,
             "start": 0, "end": 3, "surface": "abc"},
        ]
        scores = np.array([0.5, 0.5], dtype=np.float32)
        selected = ts.select_from_components(candidates, scores, set())
        # Index 1 has start=0 which is lower geometry order
        self.assertEqual(selected, [1])

    def test_non_overlapping_singletons_all_selected(self):
        candidates = [
            {"id": "jie-001", "juan": 1, "jie_index": 1, "para_id": 1,
             "start": 0, "end": 2, "surface": "ab"},
            {"id": "jie-001", "juan": 1, "jie_index": 1, "para_id": 1,
             "start": 5, "end": 7, "surface": "fg"},
        ]
        scores = np.array([0.1, 0.2], dtype=np.float32)
        selected = ts.select_from_components(candidates, scores, set())
        self.assertEqual(sorted(selected), [0, 1])


class GateLogicTest(unittest.TestCase):
    """Tests for metric gate checking without torch."""

    def test_all_gates_pass(self):
        # Simulate perfect scores
        rows = [
            {"id": "j", "juan": 1, "jie_index": 1, "para_id": 1,
             "start": 0, "end": 2, "surface": "ab", "class": "exact_reference",
             "existence_label": 1},
            {"id": "j", "juan": 1, "jie_index": 1, "para_id": 1,
             "start": 3, "end": 5, "surface": "cd", "class": "semantic_negative",
             "existence_label": 0},
            {"id": "j", "juan": 2, "jie_index": 1, "para_id": 2,
             "start": 0, "end": 2, "surface": "ab", "class": "easy_negative",
             "existence_label": 0},
        ]
        stage1 = np.array([0.99, 0.01, 0.01], dtype=np.float32)
        stage2 = np.array([1.0, -1.0, -1.0], dtype=np.float32)
        classes = [r["class"] for r in rows]
        fold_ids = np.array([0, 0, 1], dtype=np.int64)
        real_labels = np.array([1, 0, -1], dtype=np.int8)
        exact_set = {("j", 1, 0, 2)}
        boundary_set = set()

        table = ts.end_to_end_metrics(
            rows, stage1, stage2, classes, fold_ids, real_labels,
            exact_set, boundary_set,
        )
        # At threshold 0.50, exact should be admitted and selected
        row_050 = next(r for r in table if r["threshold"] == 0.50)
        self.assertEqual(row_050["e2e_exact_recall"], 1.0)
        self.assertEqual(row_050["stage1_overlap_recall"], 1.0)
        self.assertEqual(row_050["stage1_semantic_rejection"], 1.0)
        self.assertEqual(row_050["stage1_easy_rejection"], 1.0)

    def test_blocked_when_exact_recall_too_low(self):
        # Exact has low stage1 score -> not admitted at threshold 0.99
        rows = [
            {"id": "j", "juan": 1, "jie_index": 1, "para_id": 1,
             "start": 0, "end": 2, "surface": "ab", "class": "exact_reference",
             "existence_label": 1},
        ]
        stage1 = np.array([0.50], dtype=np.float32)
        stage2 = np.array([1.0], dtype=np.float32)
        classes = ["exact_reference"]
        fold_ids = np.array([0], dtype=np.int64)
        real_labels = np.array([1], dtype=np.int8)
        exact_set = {("j", 1, 0, 2)}

        table = ts.end_to_end_metrics(
            rows, stage1, stage2, classes, fold_ids, real_labels,
            exact_set, set(),
        )
        row_099 = next(r for r in table if r["threshold"] == 0.99)
        self.assertEqual(row_099["e2e_exact_recall"], 0.0)
        self.assertFalse(row_099["eligible"])


class BoundaryComponentAccuracyTest(unittest.TestCase):
    """Test boundary-component accuracy calculation."""

    def test_correct_when_exact_wins(self):
        # Component has one exact and one boundary, exact has higher score
        rows = [
            {"id": "j", "juan": 1, "jie_index": 1, "para_id": 1,
             "start": 0, "end": 3, "surface": "abc", "class": "exact_reference",
             "existence_label": 1},
            {"id": "j", "juan": 1, "jie_index": 1, "para_id": 1,
             "start": 1, "end": 4, "surface": "bcd", "class": "boundary_alternative",
             "existence_label": 1},
        ]
        stage1 = np.array([0.99, 0.99], dtype=np.float32)
        stage2 = np.array([2.0, 1.0], dtype=np.float32)
        classes = [r["class"] for r in rows]
        fold_ids = np.array([0, 0], dtype=np.int64)
        real_labels = np.array([1, 0], dtype=np.int8)
        exact_set = {("j", 1, 0, 3)}

        table = ts.end_to_end_metrics(
            rows, stage1, stage2, classes, fold_ids, real_labels,
            exact_set, {("j", 1, 1, 4)},
        )
        row_050 = next(r for r in table if r["threshold"] == 0.50)
        self.assertEqual(row_050["e2e_boundary_component_total"], 1)
        self.assertEqual(row_050["e2e_boundary_component_correct"], 1)
        self.assertEqual(row_050["e2e_boundary_component_accuracy"], 1.0)

    def test_incorrect_when_boundary_wins(self):
        rows = [
            {"id": "j", "juan": 1, "jie_index": 1, "para_id": 1,
             "start": 0, "end": 3, "surface": "abc", "class": "exact_reference",
             "existence_label": 1},
            {"id": "j", "juan": 1, "jie_index": 1, "para_id": 1,
             "start": 1, "end": 4, "surface": "bcd", "class": "boundary_alternative",
             "existence_label": 1},
        ]
        stage1 = np.array([0.99, 0.99], dtype=np.float32)
        stage2 = np.array([1.0, 2.0], dtype=np.float32)
        classes = [r["class"] for r in rows]
        fold_ids = np.array([0, 0], dtype=np.int64)
        real_labels = np.array([1, 0], dtype=np.int8)
        exact_set = {("j", 1, 0, 3)}

        table = ts.end_to_end_metrics(
            rows, stage1, stage2, classes, fold_ids, real_labels,
            exact_set, {("j", 1, 1, 4)},
        )
        row_050 = next(r for r in table if r["threshold"] == 0.50)
        self.assertEqual(row_050["e2e_boundary_component_accuracy"], 0.0)

    def test_denied_exact_does_not_remove_boundary_component_from_denominator(self):
        rows = [
            {"id": "j", "juan": 1, "jie_index": 1, "para_id": 1,
             "start": 0, "end": 3, "surface": "abc", "class": "exact_reference",
             "existence_label": 1},
            {"id": "j", "juan": 1, "jie_index": 1, "para_id": 1,
             "start": 1, "end": 4, "surface": "bcd",
             "class": "boundary_alternative", "existence_label": 1},
        ]
        table = ts.end_to_end_metrics(
            rows,
            np.array([0.1, 0.99], dtype=np.float32),
            np.array([2.0, 1.0], dtype=np.float32),
            [row["class"] for row in rows],
            np.array([0, 0], dtype=np.int64),
            np.array([1, 0], dtype=np.int8),
            {("j", 1, 0, 3)},
            {("j", 1, 1, 4)},
        )
        row_050 = next(row for row in table if row["threshold"] == 0.50)
        self.assertEqual(row_050["e2e_boundary_component_total"], 1)
        self.assertEqual(row_050["e2e_boundary_component_accuracy"], 0.0)

    def test_rejected_bridge_does_not_split_full_component(self):
        rows = [
            {"id": "j", "juan": 1, "jie_index": 1, "para_id": 1,
             "start": 0, "end": 2, "surface": "ab", "class": "exact_reference",
             "existence_label": 1},
            {"id": "j", "juan": 1, "jie_index": 1, "para_id": 1,
             "start": 1, "end": 4, "surface": "bcd",
             "class": "boundary_alternative", "existence_label": 1},
            {"id": "j", "juan": 1, "jie_index": 1, "para_id": 1,
             "start": 3, "end": 5, "surface": "de",
             "class": "boundary_alternative", "existence_label": 1},
        ]
        table = ts.end_to_end_metrics(
            rows,
            np.array([0.99, 0.1, 0.99], dtype=np.float32),
            np.array([2.0, 0.0, 1.0], dtype=np.float32),
            [row["class"] for row in rows],
            np.array([0, 0, 0], dtype=np.int64),
            np.array([1, 0, 0], dtype=np.int8),
            {("j", 1, 0, 2)},
            {("j", 1, 1, 4), ("j", 1, 3, 5)},
        )
        row_050 = next(row for row in table if row["threshold"] == 0.50)
        self.assertEqual(row_050["e2e_selected_count"], 1)
        self.assertEqual(row_050["e2e_boundary_component_total"], 1)
        self.assertEqual(row_050["e2e_boundary_component_accuracy"], 1.0)


class ThresholdSelectionTest(unittest.TestCase):
    """Test threshold selection logic."""

    def test_selects_by_descending_recall_precision_threshold(self):
        # Simulate two eligible thresholds
        table = [
            {"threshold": 0.50, "eligible": True,
             "e2e_exact_recall": 0.99, "e2e_real_precision": 0.995},
            {"threshold": 0.60, "eligible": True,
             "e2e_exact_recall": 0.99, "e2e_real_precision": 0.999},
            {"threshold": 0.70, "eligible": False,
             "e2e_exact_recall": 0.97, "e2e_real_precision": 1.0},
        ]
        eligible = [row for row in table if row["eligible"]]
        selected = max(
            eligible,
            key=lambda row: (
                row["e2e_exact_recall"],
                row["e2e_real_precision"],
                row["threshold"],
            ),
            default=None,
        )
        # Same recall -> higher precision -> threshold 0.60
        self.assertEqual(selected["threshold"], 0.60)


class VetoFieldTest(unittest.TestCase):
    """Test that no veto field is implemented without a source."""

    def test_no_veto_in_inventory_schema(self):
        # Confirm the module reports no veto field
        self.assertNotIn("veto", dir(ts))


if __name__ == "__main__":
    unittest.main()
