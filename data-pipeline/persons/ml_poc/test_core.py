import unittest

from core import Span, assemble_jies, decode_bio, sanitize_note_mentions, score_spans
from pilot import build_blind_task, score_juan, select_pilot


class JieAssemblyTest(unittest.TestCase):
    def test_numbered_paragraph_starts_jie_and_following_paragraph_carries(self):
        rows = [
            {"id": 0, "main": "年号"},
            {"id": 1, "main": "①甲"},
            {"id": 2, "main": "乙"},
            {"id": 3, "main": "②丙"},
        ]

        jies = assemble_jies(rows)

        self.assertEqual([None, 1, 2], [jie.number for jie in jies])
        self.assertEqual("①甲\n乙", jies[1].text)
        self.assertEqual([1, 2], [row.para_id for row in jies[1].segments])
        self.assertEqual((0, 2), (
            jies[1].segments[0].assembled_start,
            jies[1].segments[0].assembled_end,
        ))
        self.assertEqual((3, 4), (
            jies[1].segments[1].assembled_start,
            jies[1].segments[1].assembled_end,
        ))


class NoteSanitizationTest(unittest.TestCase):
    def test_identity_fields_are_not_serialized(self):
        paragraphs = [{
            "id": 6,
            "main": "①生立。",
            "notes": [{"after": 2, "text": "苻生即位。"}],
        }]
        mentions = [{
            "pid": 6,
            "source": "hu",
            "note_index": 0,
            "start": 0,
            "end": 2,
            "surface": "苻生",
            "person_id": "p:fu-sheng",
            "kind": "alias",
        }]

        rows = sanitize_note_mentions(paragraphs, mentions)

        self.assertEqual([{
            "para_id": 6,
            "note_index": 0,
            "after": 2,
            "start": 0,
            "end": 2,
            "surface": "苻生",
        }], rows)
        self.assertNotIn("person_id", rows[0])

    def test_note_surface_must_match(self):
        paragraphs = [{
            "id": 1,
            "main": "①生立。",
            "notes": [{"after": 2, "text": "苻生即位。"}],
        }]
        mention = {
            "pid": 1,
            "source": "hu",
            "note_index": 0,
            "start": 0,
            "end": 2,
            "surface": "姚苌",
        }
        with self.assertRaisesRegex(ValueError, "does not match"):
            sanitize_note_mentions(paragraphs, [mention])


class BioDecodeTest(unittest.TestCase):
    def test_stray_i_is_promoted_and_separator_closes_span(self):
        labels = ["I-PER", "I-PER", "I-PER", "I-PER", "O"]
        separators = [False, False, True, False, False]

        self.assertEqual(
            [(0, 2), (3, 4)],
            decode_bio(labels, separators=separators),
        )

    def test_unowned_character_cannot_join_or_start_span(self):
        labels = ["B-PER", "I-PER", "I-PER", "I-PER"]
        owned = [True, True, False, True]

        self.assertEqual([(0, 2), (3, 4)], decode_bio(labels, owned=owned))


class SpanMetricsTest(unittest.TestCase):
    def test_exact_geometry(self):
        reference = [Span(1, 0, 2), Span(1, 3, 4)]
        predictions = [Span(1, 0, 2), Span(1, 3, 5)]

        metrics = score_spans(reference, predictions)

        self.assertEqual(1, metrics.true_positive)
        self.assertEqual(0.5, metrics.precision)
        self.assertEqual(0.5, metrics.recall)
        self.assertEqual(0.5, metrics.f1)

    def test_overlap_is_one_to_one(self):
        reference = [Span(1, 0, 2), Span(1, 2, 4)]
        predictions = [Span(1, 0, 4)]

        metrics = score_spans(reference, predictions, overlap=True)

        self.assertEqual(1, metrics.true_positive)
        self.assertEqual(1.0, metrics.precision)
        self.assertEqual(0.5, metrics.recall)


class PilotPreparationTest(unittest.TestCase):
    def test_scoring_uses_exact_geometry_and_challenge_features(self):
        source = {
            "paragraphs": [{"id": 1, "main": "①可汗甲，甲至。"}]
        }
        v1 = {
            "mentions": [
                {
                    "pid": 1,
                    "source": "main",
                    "start": 1,
                    "end": 4,
                    "kind": "feng",
                },
                {
                    "pid": 1,
                    "source": "main",
                    "start": 5,
                    "end": 6,
                    "kind": "anaphora",
                },
            ]
        }
        rules = {
            "occurrences": [
                {"para_id": 1, "start": 1, "end": 4, "field": "main"},
                {"para_id": 1, "start": 5, "end": 7, "field": "main"},
            ]
        }

        score = score_juan(source, v1, rules)

        self.assertEqual(1, score["exact_agreements"])
        self.assertEqual(2, score["exact_disagreements"])
        self.assertEqual(1, score["feng"])
        self.assertEqual(1, score["foreign_titles"])
        self.assertEqual(1, score["single_char_anaphora"])
        self.assertEqual(26, score["challenge_score"])

    def test_selection_roles_are_distinct_and_reproducible(self):
        scores = {
            1: {"disagreement_rate": 0.1, "exact_disagreements": 2,
                "challenge_score": 1, "feng": 0, "foreign_titles": 0},
            2: {"disagreement_rate": 0.9, "exact_disagreements": 3,
                "challenge_score": 0, "feng": 0, "foreign_titles": 0},
            3: {"disagreement_rate": 0.2, "exact_disagreements": 1,
                "challenge_score": 9, "feng": 1, "foreign_titles": 2},
            4: {"disagreement_rate": 0.3, "exact_disagreements": 1,
                "challenge_score": 2, "feng": 0, "foreign_titles": 1},
        }

        first = select_pilot(scores, seed=7)
        second = select_pilot(scores, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(3, len({juan for _, juan in first}))

    def test_blind_task_contains_no_candidate_or_note_payload(self):
        source = {
            "paragraphs": [{
                "id": 1,
                "main": "①甲至。",
                "notes": [{"after": 2, "text": "甲，某也。"}],
            }]
        }

        task = build_blind_task(1, source, "random")
        encoded = str(task)

        self.assertNotIn("某也", encoded)
        self.assertNotIn("candidate", encoded)
        self.assertNotIn("selection_role", task)
        self.assertEqual([], task["jies"][0]["annotations"])


if __name__ == "__main__":
    unittest.main()
