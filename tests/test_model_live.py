"""Live tests: does the configured model give grounded, repeatable answers?

These call the real model through /api/infer, so they need Ollama running and
are skipped when it is not. They test LLM.MODEL; set GROUNDING_MODEL to test
another one, e.g. GROUNDING_MODEL=llama3.2:1b.
"""
import json
import os
import unittest

import LLM
from tests.grounding import check_answer
from tests.model_eval import CASES, ask, ollama_available

MODEL = os.environ.get("GROUNDING_MODEL", LLM.MODEL)
RUNS = 2


@unittest.skipUnless(ollama_available(), "Ollama is not running")
class ModelGrounding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.answers = {}
        for name, form in CASES.items():
            cls.answers[name] = [ask(MODEL, form) for _ in range(RUNS)]

    def test_every_request_gets_an_answer(self):
        for name, results in self.answers.items():
            for answer, error in results:
                with self.subTest(case=name):
                    self.assertIsNone(error)

    def test_same_input_gives_same_answer(self):
        for name, results in self.answers.items():
            with self.subTest(case=name):
                distinct = {json.dumps(answer, sort_keys=True) for answer, _ in results}
                self.assertEqual(len(distinct), 1)

    def test_evidence_matches_the_form(self):
        for name, results in self.answers.items():
            answer, error = results[0]
            if error:
                continue
            for _, kind, detail, evidence in check_answer(answer, CASES[name]):
                with self.subTest(case=name, evidence=evidence):
                    self.fail(f"{kind}: {detail}")


if __name__ == "__main__":
    unittest.main()
