from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest
from unittest import mock
import uuid

import pos_giv


class PosGivCacheTest(unittest.TestCase):
    def setUp(self):
        self.cache_dir = (
            Path(__file__).resolve().parent
            / f".test_pos_giv_cache_{uuid.uuid4().hex}"
        )
        self.cache_dir.mkdir()
        self.paras = [{"id": 7, "main": "張雲云"}]
        self.path = self.cache_dir / "juan_001.json"

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def _write(self, blob):
        self.path.write_text(
            json.dumps(blob, ensure_ascii=False), encoding="utf-8"
        )

    def test_reads_v1_and_derives_best_effort_spans(self):
        self._write({
            "sha": pos_giv._sha_of(self.paras),
            "giv": {"7": [0, 1, 3]},
        })

        evidence = pos_giv.giv_for_juan(
            1, self.paras, self.cache_dir
        )

        self.assertEqual({0, 1, 3}, evidence[7])
        self.assertEqual(((0, 2), (3, 4)), evidence[7].spans)

    def test_reads_v2_spans_unchanged(self):
        self._write({
            "version": 2,
            "sha": pos_giv._sha_of(self.paras),
            "giv": {"7": [1]},
            "giv_spans": {"7": [[1, 2]]},
        })

        evidence = pos_giv.giv_for_juan(
            1, self.paras, self.cache_dir
        )

        self.assertEqual({1}, evidence[7])
        self.assertEqual(((1, 2),), evidence[7].spans)

    def test_v3_evidence_is_derived_from_tokens(self):
        self._write({
            "version": 3,
            "sha": pos_giv._sha_of(self.paras),
            "model": {"id": pos_giv.MODEL},
            "paragraphs": {
                "7": {
                    "sentences": [{
                        "start": 0,
                        "end": 3,
                        "tokens": [
                            {
                                "text": "張",
                                "start": 0,
                                "end": 1,
                                "pos": "PROPN",
                                "tag": "PROPN|NameType=Sur",
                                "score": 0.9,
                            },
                            {
                                "text": "雲",
                                "start": 1,
                                "end": 2,
                                "pos": "PROPN",
                                "tag": "PROPN|NameType=Giv",
                                "score": 0.8,
                            },
                            {
                                "text": "云",
                                "start": 2,
                                "end": 3,
                                "pos": "VERB",
                                "tag": "VERB",
                                "score": 0.99,
                            },
                        ],
                    }],
                },
            },
            "giv": {"7": [2]},
            "giv_spans": {"7": [[2, 3]]},
        })

        evidence = pos_giv.giv_for_juan(
            1, self.paras, self.cache_dir
        )

        self.assertEqual({1}, evidence[7])
        self.assertEqual(((1, 2),), evidence[7].spans)
        self.assertEqual(3, len(evidence[7].tokens))
        self.assertEqual("PROPN", evidence[7].token_at(1).pos)
        self.assertTrue(evidence[7].token_at(1).is_giv)
        self.assertEqual("VERB", evidence[7].token_at(2).pos)
        self.assertFalse(evidence[7].token_at(2).is_giv)

    def test_normal_read_never_loads_model(self):
        with mock.patch.object(
            pos_giv, "_get_pipe", side_effect=AssertionError("loaded model")
        ):
            with self.assertRaises(pos_giv.CacheMissError):
                pos_giv.giv_for_juan(
                    1, self.paras, self.cache_dir
                )

    def test_failed_atomic_write_preserves_previous_cache(self):
        self.path.write_text("previous", encoding="utf-8")

        with self.assertRaises(TypeError):
            pos_giv._write_atomic(self.path, {"invalid": object()})

        self.assertEqual(
            "previous", self.path.read_text(encoding="utf-8")
        )
        self.assertEqual(
            [self.path],
            list(self.cache_dir.iterdir()),
        )


if __name__ == "__main__":
    unittest.main()
