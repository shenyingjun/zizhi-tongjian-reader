import json
import tempfile
import unittest
from pathlib import Path

from export_translation_scope import export_scope
from teacher_evidence import RTM, bounded_evidence


class TeacherEvidenceTest(unittest.TestCase):
    def test_exported_scope_contains_no_identity_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = root / "mapping.json"
            output = root / "scope.json"
            mapping.write_text(json.dumps({
                "sources": [{"juan": 1, "source_sha256": "abc"}],
                "all_candidates": [{
                    "juan": 1,
                    "source_page": "https://example.invalid/#pair-2",
                    "repo_jie_index": 3,
                    "identity_surface": "曹操",
                    "translation_ner_name": "曹操",
                }],
            }), encoding="utf-8")

            result = export_scope(mapping, output)

            serialized = output.read_text(encoding="utf-8")
            self.assertFalse(result["identity_fields_present"])
            self.assertEqual({"1": {"1": [3]}}, result["pair_jies"])
            self.assertNotIn("曹操", serialized)
            self.assertNotIn("identity_surface", serialized)

    def test_limits_notes_and_translation_to_one_unique_jie(self):
        source = {
            "paragraphs": [
                {
                    "id": 1,
                    "main": "①曹操至。",
                    "notes": [{"after": 3, "text": "操，小字阿瞒。"}],
                },
                {
                    "id": 2,
                    "main": "②刘备至。",
                    "notes": [{"after": 3, "text": "备，字玄德。"}],
                },
            ],
        }
        pairs = (
            RTM.SourcePair(0, "曹操至。", "曹操到达。"),
            RTM.SourcePair(1, "刘备至。", "刘备到达。"),
            RTM.SourcePair(2, "曹操至。刘备至。", "二人到达。"),
        )

        result = bounded_evidence(
            1,
            1,
            source,
            pairs,
            source_url="https://example.invalid/1",
            source_sha256="abc",
            approved_pair_jies={0: {1}, 1: {2}, 2: {1, 2}},
        )

        self.assertEqual([1], result["paragraph_ids"])
        self.assertEqual(
            ["操，小字阿瞒。"],
            [row["text"] for row in result["hu_sansheng_notes"]],
        )
        self.assertEqual(
            ["曹操到达。"],
            [row["translation"] for row in result["translations"]],
        )
        self.assertEqual(
            ["target", "context_only"],
            [row["authorization"] for row in result["juan_context"]],
        )
        self.assertNotIn("person_id", str(result))


if __name__ == "__main__":
    unittest.main()
