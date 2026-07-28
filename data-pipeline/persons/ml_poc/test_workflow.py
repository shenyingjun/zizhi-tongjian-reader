import json
import tempfile
import unittest
from pathlib import Path

from server import AnnotationStore


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


class AnnotationStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.blind = root / "blind"
        self.recall = root / "recall"
        self.role_audit = root / "role-audit"
        self.state = root / "state"
        write(self.blind / "manifest.json", {
            "selected": [{
                "juan": 1,
                "role": "random",
                "scores": {"must_not_leak": 99},
            }]
        })
        write(self.blind / "blind_juan_001.json", {
            "schema_version": 1,
            "phase": "blind",
            "juan": 1,
            "jies": [{
                "jie_index": 0,
                "jie_number": 1,
                "text": "①曹操至。丞相奏。",
                "segments": [{
                    "para_id": 2,
                    "assembled_start": 0,
                    "assembled_end": 9,
                }],
                "annotations": [],
            }],
        })
        write(self.recall / "recall_juan_001.json", {
            "schema_version": 1,
            "phase": "recall",
            "juan": 1,
            "candidates": [{
                "id": "2:1:3",
                "para_id": 2,
                "start": 1,
                "end": 3,
                "surface": "曹操",
                "channels": ["rules", "v1"],
            }],
            "note_evidence": [],
        })
        write(self.role_audit / "role_audit_juan_001.json", {
            "schema_version": 1,
            "phase": "role_audit",
            "juan": 1,
            "candidates": [{
                "id": "role:2:5:7",
                "para_id": 2,
                "start": 5,
                "end": 7,
                "surface": "丞相",
                "channels": ["role_lexicon"],
            }],
            "note_evidence": [],
        })
        self.store = AnnotationStore(
            self.blind, self.recall, self.role_audit, self.state
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_public_index_does_not_expose_selection_scores(self):
        index = self.store.index()

        self.assertEqual([{
            "juan": 1,
            "blind_complete": False,
            "recall_complete": False,
            "role_audit_complete": False,
        }], index["juans"])
        self.assertNotIn("scores", str(index))
        self.assertNotIn("role", index["juans"][0])

    def test_recall_is_inaccessible_until_blind_completion(self):
        with self.assertRaisesRegex(PermissionError, "locked"):
            self.store.payload(1, "recall")

    def test_save_validates_surface_and_locks_completed_blind_phase(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.store.save(1, "blind", {
                "annotations": [{
                    "para_id": 2, "start": 1, "end": 3,
                    "surface": "刘备",
                }]
            })

        saved = self.store.save(1, "blind", {
            "annotations": [{
                "para_id": 2, "start": 1, "end": 3,
                "surface": "曹操",
            }]
        })
        self.assertEqual("曹操", saved["annotations"][0]["surface"])

        self.store.complete(1, "blind")
        with self.assertRaisesRegex(PermissionError, "locked"):
            self.store.save(1, "blind", {"annotations": []})

    def test_recall_starts_from_blind_and_accepts_known_decision_only(self):
        self.store.save(1, "blind", {
            "annotations": [{
                "para_id": 2, "start": 1, "end": 3,
                "surface": "曹操",
            }]
        })
        self.store.complete(1, "blind")

        payload = self.store.payload(1, "recall")
        self.assertEqual(
            payload["state"]["annotations"],
            self.store.state(1)["blind"]["annotations"],
        )
        self.assertEqual(
            "accept", payload["state"]["decisions"]["2:1:3"]
        )
        with self.assertRaisesRegex(ValueError, "unknown candidate"):
            self.store.save(1, "recall", {
                "annotations": payload["state"]["annotations"],
                "decisions": {"unknown": "accept"},
            })
        saved = self.store.save(1, "recall", {
            "annotations": payload["state"]["annotations"],
            "decisions": {"2:1:3": "accept"},
        })
        self.assertEqual("accept", saved["decisions"]["2:1:3"])
        self.assertTrue(self.store.complete(1, "recall")["complete"])

    def test_role_audit_preserves_recall_and_locks_independently(self):
        self.store.save(1, "blind", {
            "annotations": [{
                "para_id": 2, "start": 1, "end": 3,
                "surface": "曹操",
            }]
        })
        self.store.complete(1, "blind")
        recall = self.store.payload(1, "recall")
        self.store.save(1, "recall", {
            "annotations": recall["state"]["annotations"],
            "decisions": {"2:1:3": "accept"},
        })
        self.store.complete(1, "recall")
        recall_snapshot = self.store.state(1)["recall"]

        audit = self.store.payload(1, "role_audit")
        self.assertEqual(
            recall_snapshot["annotations"], audit["state"]["annotations"]
        )
        annotations = audit["state"]["annotations"] + [{
            "para_id": 2, "start": 5, "end": 7, "surface": "丞相",
        }]
        saved = self.store.save(1, "role_audit", {
            "annotations": annotations,
            "decisions": {"role:2:5:7": "accept"},
        })
        self.assertEqual("accept", saved["decisions"]["role:2:5:7"])
        self.assertTrue(self.store.complete(1, "role_audit")["complete"])
        self.assertEqual(recall_snapshot, self.store.state(1)["recall"])
        with self.assertRaisesRegex(PermissionError, "locked"):
            self.store.save(1, "role_audit", {
                "annotations": annotations,
                "decisions": {"role:2:5:7": "accept"},
            })


if __name__ == "__main__":
    unittest.main()
