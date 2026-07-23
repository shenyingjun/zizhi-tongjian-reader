from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import translation_evidence as TE


class TranslationEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.text_dir = self.root / "text"
        self.text_dir.mkdir()
        self.paragraphs = [
            {"id": 0, "main": "①魏斯至。"},
            {"id": 1, "main": "魏斯曰。"},
            {"id": 2, "main": "②赵籍至。"},
        ]
        (self.text_dir / "juan_001.json").write_text(
            json.dumps({"paragraphs": self.paragraphs}, ensure_ascii=False),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _mapping(self, **overrides):
        row = {
            "juan": 1,
            "repo_para_id": 1,
            "repo_jie_index": 1,
            "repo_jie_number": 1,
            "identity_surface": "魏斯",
            "translation_ner_name": "魏斯",
            "original_start": 0,
            "original_end": 2,
            "original_surface": "魏斯",
            "normalized_original_surface": "魏斯",
            "transfer_mode": "exact",
            "mapping_status": "mapped_exact",
            "source_kind": "modern_chinese_translation",
            "source_page": "source-1",
            "translation_ner_score": 0.99,
        }
        row.update(overrides)
        return {
            "method": {"alignment_scope": "canonical_numbered_jie"},
            "sources": [{"juan": 1}],
            "all_candidates": [row],
        }

    def _build(self, mapping):
        mapping_path = self.root / "mapping.json"
        evidence_dir = self.root / "evidence"
        mapping_path.write_text(
            json.dumps(mapping, ensure_ascii=False),
            encoding="utf-8",
        )
        with patch.object(TE, "TEXT", self.text_dir):
            TE.build(mapping_path, evidence_dir)
        return evidence_dir

    def test_build_and_load_validate_numbered_jie(self):
        evidence_dir = self._build(self._mapping())

        with patch.object(TE, "TEXT", self.text_dir):
            evidence = TE.load_juan(evidence_dir, 1, self.paragraphs)

        self.assertEqual("魏斯", evidence[1][0]["identity_surface"])
        payload = json.loads(
            (evidence_dir / "juan_001.json").read_text(encoding="utf-8")
        )
        self.assertEqual(1, payload["paragraphs"]["1"]["jie_index"])
        self.assertEqual(1, payload["paragraphs"]["1"]["jie_number"])

    def test_build_rejects_cross_jie_mapping(self):
        mapping = self._mapping(repo_jie_index=2, repo_jie_number=2)

        with self.assertRaisesRegex(ValueError, "belongs to jie 1"):
            self._build(mapping)

    def test_load_rejects_stale_jie_geometry_even_with_valid_hash(self):
        evidence_dir = self._build(self._mapping())
        path = evidence_dir / "juan_001.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["paragraphs"]["1"]["jie_index"] = 2
        encoded = (
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        path.write_bytes(encoded)
        manifest_path = evidence_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["evidence_sha256_by_juan"]["1"] = TE._sha256_bytes(encoded)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "jie mismatch"):
            TE.load_juan(evidence_dir, 1, self.paragraphs)


if __name__ == "__main__":
    unittest.main()
