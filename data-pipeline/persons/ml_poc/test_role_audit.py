import json
import tempfile
import unittest
from pathlib import Path

from role_audit import build_role_audit_pack


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class RoleAuditPackTest(unittest.TestCase):
    def test_proposes_uncovered_roles_without_identity_or_existing_spans(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            blind = root / "blind.json"
            recall = root / "recall.json"
            state = root / "state.json"
            write(blind, {
                "jies": [{
                    "text": "①曹操见丞相。丞相府。",
                    "segments": [{
                        "para_id": 2,
                        "assembled_start": 0,
                        "assembled_end": 12,
                    }],
                }],
            })
            write(recall, {
                "candidates": [{
                    "id": "2:4:6",
                    "para_id": 2,
                    "start": 4,
                    "end": 6,
                    "surface": "丞相",
                    "channels": ["rules"],
                }],
            })
            write(state, {
                "recall": {
                    "complete": True,
                    "annotations": [{
                        "para_id": 2,
                        "start": 1,
                        "end": 3,
                        "surface": "曹操",
                    }],
                    "decisions": {"2:4:6": "reject"},
                },
            })

            pack = build_role_audit_pack(1, blind, recall, state)

            self.assertEqual("role_audit", pack["phase"])
            self.assertEqual(
                [(row["start"], row["end"], row["surface"])
                 for row in pack["candidates"]],
                [(4, 6, "丞相")],
            )
            self.assertNotIn("person_id", json.dumps(pack, ensure_ascii=False))
            self.assertEqual(
                ["rejected_recall", "role_lexicon"],
                pack["candidates"][0]["channels"],
            )

    def test_does_not_propose_role_adjacent_to_tagged_name_core(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            blind = root / "blind.json"
            recall = root / "recall.json"
            state = root / "state.json"
            write(blind, {
                "jies": [{
                    "text": "校尉马贤至，马贤校尉还。",
                    "segments": [{
                        "para_id": 2,
                        "assembled_start": 0,
                        "assembled_end": 13,
                    }],
                }],
            })
            write(recall, {"candidates": []})
            write(state, {
                "recall": {
                    "complete": True,
                    "annotations": [
                        {"para_id": 2, "start": 2, "end": 4, "surface": "马贤"},
                        {"para_id": 2, "start": 6, "end": 8, "surface": "马贤"},
                    ],
                    "decisions": {},
                },
            })

            pack = build_role_audit_pack(1, blind, recall, state)

            self.assertEqual([], pack["candidates"])

    def test_keeps_only_longest_nested_role_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            blind = root / "blind.json"
            recall = root / "recall.json"
            state = root / "state.json"
            write(blind, {
                "jies": [{
                    "text": "皇帝曰。",
                    "segments": [{
                        "para_id": 2,
                        "assembled_start": 0,
                        "assembled_end": 4,
                    }],
                }],
            })
            write(recall, {"candidates": []})
            write(state, {
                "recall": {
                    "complete": True,
                    "annotations": [],
                    "decisions": {},
                },
            })

            pack = build_role_audit_pack(1, blind, recall, state)

            self.assertEqual(
                [(0, 2, "皇帝")],
                [(row["start"], row["end"], row["surface"])
                 for row in pack["candidates"]],
            )

    def test_excludes_titles_in_conferral_predicates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            blind = root / "blind.json"
            recall = root / "recall.json"
            state = root / "state.json"
            text = (
                "以黄尚为司徒，王卓为司空。拜祝良为九真太守，"
                "张乔为交趾刺史。自立为呼揭单于。"
                "以不疑为侍中、奉车都尉。司徒奏。"
            )
            write(blind, {
                "jies": [{
                    "text": text,
                    "segments": [{
                        "para_id": 2,
                        "assembled_start": 0,
                        "assembled_end": len(text),
                    }],
                }],
            })
            write(recall, {"candidates": []})
            write(state, {
                "recall": {
                    "complete": True,
                    "annotations": [
                        {"para_id": 2, "start": 1, "end": 3, "surface": "黄尚"},
                        {"para_id": 2, "start": 7, "end": 9, "surface": "王卓"},
                        {"para_id": 2, "start": 14, "end": 16, "surface": "祝良"},
                        {"para_id": 2, "start": 22, "end": 24, "surface": "张乔"},
                    ],
                    "decisions": {},
                },
            })

            pack = build_role_audit_pack(1, blind, recall, state)

            self.assertEqual(
                [(50, 52, "司徒")],
                [(row["start"], row["end"], row["surface"])
                 for row in pack["candidates"]],
            )

    def test_excludes_plural_and_institutional_role_uses(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            blind = root / "blind.json"
            recall = root / "recall.json"
            state = root / "state.json"
            text = "凡五单于，两单于，单于庭，单于号。单于曰。"
            write(blind, {
                "jies": [{
                    "text": text,
                    "segments": [{
                        "para_id": 2,
                        "assembled_start": 0,
                        "assembled_end": len(text),
                    }],
                }],
            })
            write(recall, {"candidates": []})
            write(state, {
                "recall": {
                    "complete": True,
                    "annotations": [],
                    "decisions": {},
                },
            })

            pack = build_role_audit_pack(1, blind, recall, state)

            self.assertEqual(
                [(17, 19, "单于")],
                [(row["start"], row["end"], row["surface"])
                 for row in pack["candidates"]],
            )


if __name__ == "__main__":
    unittest.main()
