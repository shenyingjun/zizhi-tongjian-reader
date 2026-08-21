import unittest

from production_precision_revision17_freeze import normalize_raw


class Revision17FreezeTest(unittest.TestCase):
    def test_normalizes_exact_raw_schema(self):
        result = normalize_raw({
            "task_id": "task",
            "candidate_id": "candidate",
            "label": "not_person",
            "rationale": "  Used as a polity here.  ",
            "reviewer": "copilot-teacher",
            "model": "gpt-5.6-sol",
        })

        self.assertEqual("Used as a polity here.", result["rationale"])
        self.assertEqual("not_person", result["label"])

    def test_rejects_extra_fields_and_empty_rationale(self):
        base = {
            "task_id": "task",
            "candidate_id": "candidate",
            "label": "not_person",
            "rationale": "context",
            "reviewer": "copilot-teacher",
            "model": "gpt-5.6-sol",
        }
        with self.assertRaisesRegex(ValueError, "fields differ"):
            normalize_raw({**base, "score": 0.9})
        with self.assertRaisesRegex(ValueError, "decision differs"):
            normalize_raw({**base, "rationale": " "})


if __name__ == "__main__":
    unittest.main()
