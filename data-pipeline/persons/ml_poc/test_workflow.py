import json
import hashlib
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

    def test_assisted_pack_is_globally_locked_by_blind_anchor(self):
        root = Path(self.temp.name) / "expansion"
        tasks = root / "tasks"
        assisted = root / "assisted"
        state = root / "state"
        write(tasks / "manifest.json", {
            "selected": [
                {"juan": 1, "role": "anchor", "mode": "blind_anchor"},
                {"juan": 2, "role": "assisted", "mode": "assisted"},
            ],
        })
        for juan in (1, 2):
            write(tasks / f"blind_juan_{juan:03d}.json", {
                "juan": juan,
                "jies": [{
                    "jie_index": 0,
                    "text": "①曹操至。",
                    "segments": [{
                        "para_id": 2,
                        "assembled_start": 0,
                        "assembled_end": 5,
                    }],
                }],
            })
        write(assisted / "assisted_juan_002.json", {
            "phase": "assisted",
            "juan": 2,
            "candidates": [{
                "id": "2:1:3",
                "para_id": 2,
                "start": 1,
                "end": 3,
                "surface": "曹操",
                "channels": ["ml_constrained"],
            }],
        })
        store = AnnotationStore(
            tasks, root / "recall", root / "roles", state, assisted
        )

        with self.assertRaisesRegex(PermissionError, "blind anchors"):
            store.payload(2, "assisted")
        with self.assertRaisesRegex(PermissionError, "do not expose"):
            store.payload(2, "blind")
        store.complete(1, "blind")
        payload = store.payload(2, "assisted")
        self.assertEqual("曹操", payload["review"]["candidates"][0]["surface"])
        pack_path = assisted / "assisted_juan_002.json"
        expected_hash = hashlib.sha256(pack_path.read_bytes()).hexdigest()
        self.assertEqual(
            expected_hash, store.state(2)["assisted"]["pack_sha256"]
        )
        store.save(2, "assisted", {
            "annotations": [{
                "para_id": 2,
                "start": 1,
                "end": 3,
                "surface": "曹操",
            }],
            "decisions": {"2:1:3": "accept"},
        })
        self.assertTrue(store.complete(2, "assisted")["complete"])

        pack = json.loads(pack_path.read_text(encoding="utf-8"))
        pack["candidates"][0]["surface"] = "刘备"
        write(pack_path, pack)
        with self.assertRaisesRegex(PermissionError, "changed"):
            store.payload(2, "assisted")

    def test_sealed_blind_task_never_initializes_recall(self):
        root = Path(self.temp.name) / "sealed"
        tasks = root / "tasks"
        state = root / "state"
        write(tasks / "manifest.json", {
            "selected": [{
                "juan": 1,
                "role": "probability_random",
                "mode": "sealed_blind",
                "task_sha256": "",
            }],
        })
        task_path = tasks / "blind_juan_001.json"
        write(task_path, {
            "juan": 1,
            "jies": [{
                "jie_index": 0,
                "text": "①曹操至。",
                "segments": [{
                    "para_id": 2,
                    "assembled_start": 0,
                    "assembled_end": 5,
                }],
            }],
        })
        manifest = json.loads(
            (tasks / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["selected"][0]["task_sha256"] = hashlib.sha256(
            task_path.read_bytes()
        ).hexdigest()
        write(tasks / "manifest.json", manifest)
        store = AnnotationStore(
            tasks, root / "recall", root / "roles", state
        )

        index = store.index()["juans"][0]
        self.assertEqual("blind", index["initial_phase"])
        self.assertNotIn("role", index)
        self.assertTrue(store.payload(1, "blind")["sealed"])
        with self.assertRaisesRegex(PermissionError, "candidate-blind"):
            store.payload(1, "recall")
        store.save(1, "blind", {
            "annotations": [{
                "para_id": 2, "start": 1, "end": 3, "surface": "曹操",
            }],
        })
        store.complete(1, "blind")

        final = store.state(1)
        self.assertTrue(final["blind"]["complete"])
        self.assertEqual([], final["recall"]["annotations"])
        with self.assertRaisesRegex(PermissionError, "candidate-blind"):
            store.save(1, "recall", {
                "annotations": [], "decisions": {},
            })

        task = json.loads(task_path.read_text(encoding="utf-8"))
        task["jies"][0]["text"] = "①刘备至。"
        write(task_path, task)
        with self.assertRaisesRegex(PermissionError, "hash differs"):
            store.payload(1, "blind")

    def test_adjudication_exposes_only_hash_bound_binary_recall(self):
        root = Path(self.temp.name) / "adjudication"
        tasks = root / "tasks"
        recall = root / "recall"
        state = root / "state"
        task_path = tasks / "blind_juan_001.json"
        write(task_path, {
            "juan": 1,
            "jies": [{
                "jie_index": 1,
                "text": "①曹操至。",
                "segments": [{
                    "para_id": 2,
                    "assembled_start": 0,
                    "assembled_end": 5,
                }],
            }],
        })
        pack_path = recall / "recall_juan_001.json"
        write(pack_path, {
            "juan": 1,
            "candidates": [{
                "id": "hidden",
                "para_id": 2,
                "start": 1,
                "end": 3,
                "surface": "曹操",
                "channels": ["source_hidden"],
            }],
            "note_evidence": [],
        })
        write(tasks / "manifest.json", {
            "selected": [{
                "juan": 1,
                "role": "post_sealed_adjudication",
                "mode": "adjudication",
                "task_sha256": hashlib.sha256(
                    task_path.read_bytes()
                ).hexdigest(),
                "pack_sha256": hashlib.sha256(
                    pack_path.read_bytes()
                ).hexdigest(),
            }],
        })
        write(state / "juan_001.json", {
            "juan": 1,
            "blind": {
                "complete": True,
                "annotations": [{
                    "para_id": 2, "start": 1, "end": 3,
                    "surface": "曹操",
                }],
            },
            "recall": {
                "complete": False,
                "annotations": [{
                    "para_id": 2, "start": 1, "end": 3,
                    "surface": "曹操",
                }],
                "decisions": {},
                "note_decisions": {},
            },
        })
        store = AnnotationStore(
            tasks, recall, root / "roles", state
        )

        index = store.index()["juans"][0]
        self.assertEqual("recall", index["initial_phase"])
        payload = store.payload(1, "recall")
        self.assertTrue(payload["adjudication"])
        self.assertEqual({}, payload["state"]["decisions"])
        with self.assertRaisesRegex(PermissionError, "source-hidden"):
            store.payload(1, "blind")
        with self.assertRaisesRegex(ValueError, "accept or reject"):
            store.save(1, "recall", {
                "annotations": [],
                "decisions": {"hidden": "unsure"},
                "note_decisions": {},
            })
        with self.assertRaisesRegex(ValueError, "not annotated"):
            store.save(1, "recall", {
                "annotations": [],
                "decisions": {"hidden": "accept"},
                "note_decisions": {},
            })
        store.save(1, "recall", {
            "annotations": [],
            "decisions": {"hidden": "reject"},
            "note_decisions": {},
        })
        self.assertTrue(store.complete(1, "recall")["complete"])
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
        pack["candidates"][0]["surface"] = "刘备"
        write(pack_path, pack)
        with self.assertRaisesRegex(PermissionError, "hash differs"):
            store.payload(1, "recall")

    def test_diagnostic_assisted_initializes_only_low_confidence_unresolved(self):
        root = Path(self.temp.name) / "diagnostic"
        tasks = root / "tasks"
        assisted = root / "assisted"
        state = root / "state"
        task_path = tasks / "blind_juan_001.json"
        write(task_path, {
            "juan": 1,
            "jies": [{
                "jie_index": 0,
                "text": "①曹操与帝至。",
                "segments": [{
                    "para_id": 2,
                    "assembled_start": 0,
                    "assembled_end": 8,
                }],
            }],
        })
        write(tasks / "manifest.json", {
            "selected": [{
                "juan": 1,
                "mode": "diagnostic_assisted",
                "task_sha256": hashlib.sha256(
                    task_path.read_bytes()
                ).hexdigest(),
            }],
        })
        write(assisted / "assisted_juan_001.json", {
            "phase": "assisted",
            "juan": 1,
            "candidates": [
                {
                    "id": "copilot:2:1:3",
                    "para_id": 2, "start": 1, "end": 3,
                    "surface": "曹操", "confidence": "high",
                    "channels": ["copilot_diagnostic"],
                },
                {
                    "id": "copilot:2:4:5",
                    "para_id": 2, "start": 4, "end": 5,
                    "surface": "帝", "confidence": "low",
                    "review_reason": "role ambiguity",
                    "channels": ["copilot_diagnostic"],
                },
            ],
        })
        store = AnnotationStore(
            tasks, root / "recall", root / "roles", state, assisted
        )

        payload = store.payload(1, "assisted")

        self.assertEqual(2, len(payload["state"]["annotations"]))
        self.assertEqual(
            {"copilot:2:1:3": "accept"},
            payload["state"]["decisions"],
        )
        self.assertTrue(payload["state"]["initialized"])
        with self.assertRaisesRegex(PermissionError, "do not expose"):
            store.payload(1, "blind")

    def test_active_assisted_initializes_teacher_not_ml_only(self):
        root = Path(self.temp.name) / "active-assisted"
        tasks = root / "tasks"
        assisted = root / "assisted"
        state = root / "state"
        task_path = tasks / "blind_juan_001.json"
        write(task_path, {
            "juan": 1,
            "jies": [{
                "jie_index": 1,
                "text": "①曹操与刘备。",
                "segments": [{
                    "para_id": 2,
                    "assembled_start": 0,
                    "assembled_end": 7,
                }],
            }],
        })
        pack_path = assisted / "assisted_juan_001.json"
        write(pack_path, {
            "juan": 1,
            "candidates": [{
                "id": "copilot:2:1:3",
                "para_id": 2, "start": 1, "end": 3,
                "surface": "曹操", "confidence": "high",
            }, {
                "id": "copilot:2:4:6",
                "para_id": 2, "start": 4, "end": 6,
                "surface": "刘备", "confidence": "low",
            }],
            "initial_annotations": [{
                "para_id": 2, "start": 1, "end": 3, "surface": "曹操",
            }],
            "initial_decisions": {"copilot:2:1:3": "accept"},
        })
        write(tasks / "manifest.json", {
            "selected": [{
                "juan": 1,
                "mode": "active_assisted",
                "task_sha256": hashlib.sha256(
                    task_path.read_bytes()
                ).hexdigest(),
                "pack_sha256": hashlib.sha256(
                    pack_path.read_bytes()
                ).hexdigest(),
            }],
        })
        store = AnnotationStore(
            tasks, root / "recall", root / "roles", state, assisted
        )

        payload = store.payload(1, "assisted")

        with self.assertRaisesRegex(PermissionError, "only assisted"):
            store.save(1, "blind", {"annotations": []})
        with self.assertRaisesRegex(PermissionError, "only assisted"):
            store.complete(1, "blind")
        with self.assertRaisesRegex(PermissionError, "only assisted"):
            store.payload(1, "recall")
        self.assertEqual(
            ["曹操"],
            [row["surface"] for row in payload["state"]["annotations"]],
        )
        self.assertEqual(
            {"copilot:2:1:3": "accept"},
            payload["state"]["decisions"],
        )


if __name__ == "__main__":
    unittest.main()
