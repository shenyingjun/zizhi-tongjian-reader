from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import build_v2 as B


class BuildV2IndependenceTest(unittest.TestCase):
    def test_build_does_not_preserve_v1_only_mentions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v1 = root / "persons"
            v2 = root / "persons-v2"
            (v1 / "mentions").mkdir(parents=True)
            (v1 / "people.json").write_text(
                json.dumps({"version": 1, "people": []}),
                encoding="utf-8",
            )
            (v1 / "appearances.json").write_text(
                json.dumps({"version": 1, "appearances": {}}),
                encoding="utf-8",
            )
            (v1 / "mentions" / "juan_001.json").write_text(
                json.dumps({
                    "juan_no": 1,
                    "version": 1,
                    "mentions": [{
                        "pid": 1,
                        "ce_year": 1,
                        "source": "main",
                        "start": 0,
                        "end": 2,
                        "surface": "旧误",
                        "person_id": "legacy-only",
                        "kind": "alias",
                        "confidence": "high",
                    }],
                }),
                encoding="utf-8",
            )
            (root / "juan_001.json").write_text(
                json.dumps({
                    "juan_no": 1,
                    "paragraphs": [{"id": 1, "main": "旧误新名", "ce_year": 1}],
                }),
                encoding="utf-8",
            )
            card = {
                "juan": 1,
                "para_id": 1,
                "start": 2,
                "end": 4,
                "surface": "新名",
                "ce_year": 1,
                "evidence": "giv2",
                "person_id": "new:新名",
            }

            with (
                mock.patch.object(B, "ROOT", str(root)),
                mock.patch.object(B, "V1", str(v1)),
                mock.patch.object(B, "V2", str(v2)),
                mock.patch.object(
                    B.R,
                    "load_lexicons",
                    return_value=([], set(), {}, set()),
                ),
                mock.patch.object(
                    B.R,
                    "load_current_mentions",
                    return_value={1: [{"person_id": "reference-only"}]},
                ),
                mock.patch.object(
                    B.stage1,
                    "detect_juan",
                    return_value=([dict(card)], {}),
                ),
                mock.patch.object(
                    B.stage2,
                    "build_reference",
                    return_value=({}, {}, {}),
                ),
                mock.patch.object(
                    B.stage2,
                    "consolidate",
                    return_value=([dict(card)], {"new:新名": "新名"}),
                ),
            ):
                B.main(["1"])

            output = json.loads(
                (v2 / "mentions" / "juan_001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(["新名"], [row["surface"] for row in output["mentions"]])

    def test_checked_in_juan266_does_not_tag_transit_phrase(self):
        repo = Path(__file__).resolve().parents[3]
        text = json.loads(
            (repo / "web" / "public" / "text" / "juan_266.json").read_text(
                encoding="utf-8"
            )
        )
        paragraph = next(
            row for row in text["paragraphs"] if "间道袭温州" in row["main"]
        )
        start = paragraph["main"].index("道袭")
        target = (paragraph["id"], start, start + 2)

        bound = json.loads(
            (
                repo
                / "web"
                / "public"
                / "text"
                / "persons-v2"
                / "mentions"
                / "juan_266.json"
            ).read_text(encoding="utf-8")
        )
        agent1 = json.loads(
            (
                repo
                / "web"
                / "public"
                / "text"
                / "persons-v2"
                / "agent1"
                / "juan_266.json"
            ).read_text(encoding="utf-8")
        )

        self.assertFalse(any(
            row["pid"] == target[0]
            and row["start"] < target[2]
            and target[1] < row["end"]
            for row in bound["mentions"]
        ))
        self.assertFalse(any(
            row["para_id"] == target[0]
            and row["start"] < target[2]
            and target[1] < row["end"]
            for row in agent1["occurrences"]
        ))


if __name__ == "__main__":
    unittest.main()
