import unittest
from unittest.mock import patch

import rag


class DomainRoutingTests(unittest.TestCase):
    def call(self, question, previous_status=None, context=None):
        with patch.object(rag, "load_resources") as load_resources, patch.object(
            rag, "retrieve_schemes", return_value=[{"rank": 1, "document": "scheme data"}]
        ) as retrieve, patch.object(
            rag, "_groq_completion", return_value="grounded answer"
        ) as groq:
            load_resources.return_value = (object(), object(), [])
            result = rag.ask_schemesathi(
                question,
                previous_status=previous_status,
                conversation_context=context,
                return_status=True,
            )
        return result, load_resources, retrieve, groq

    def test_out_of_scope_questions_call_neither_backend(self):
        for question in ("Why is paracetamol used?", "Who is the prime minister of India?", "Write Python code."):
            (answer, status), resources, retrieve, groq = self.call(question)
            self.assertEqual(answer, rag.DOMAIN_REFUSAL)
            self.assertEqual(status, "out_of_scope")
            resources.assert_not_called()
            retrieve.assert_not_called()
            groq.assert_not_called()

    def test_out_of_scope_summary_does_not_call_backend(self):
        (answer, status), resources, retrieve, groq = self.call(
            "Give summary in two lines.", "out_of_scope"
        )
        self.assertEqual(answer, rag.NO_SUMMARY_CONTEXT)
        self.assertEqual(status, "out_of_scope")
        resources.assert_not_called()
        retrieve.assert_not_called()
        groq.assert_not_called()

    def test_context_free_followup_does_not_call_backend(self):
        (answer, status), resources, retrieve, groq = self.call("What documents are needed?")
        self.assertEqual(answer, rag.MISSING_SCHEME_CONTEXT)
        self.assertEqual(status, "out_of_scope")
        resources.assert_not_called()
        retrieve.assert_not_called()
        groq.assert_not_called()

    def test_valid_new_question_calls_retrieval_and_groq(self):
        (answer, status), resources, retrieve, groq = self.call(
            "Which scholarships are available for college students?"
        )
        self.assertEqual((answer, status), ("grounded answer", "valid_scheme_answer"))
        resources.assert_called_once()
        retrieve.assert_called_once()
        groq.assert_called_once()

    def test_valid_summary_followup_skips_retrieval(self):
        (answer, status), resources, retrieve, groq = self.call(
            "Summarize it in two lines.",
            "valid_scheme_answer",
            {"previous_user_question": "Which scholarship?", "previous_assistant_answer": "A scheme."},
        )
        self.assertEqual((answer, status), ("grounded answer", "valid_scheme_answer"))
        resources.assert_not_called()
        retrieve.assert_not_called()
        groq.assert_called_once()
        prompt = groq.call_args.args[0][1]["content"]
        self.assertIn("A scheme.", prompt)
        self.assertIn("Summarize it in two lines.", prompt)

    def test_missing_followup_detail_is_not_generated(self):
        (answer, status), resources, retrieve, groq = self.call(
            "What documents are needed?",
            "valid_scheme_answer",
            {"previous_user_question": "Which scheme?", "previous_assistant_answer": "The scheme helps farmers."},
        )
        self.assertEqual(answer, rag.MISSING_FOLLOWUP_DETAIL)
        self.assertEqual(status, "out_of_scope")
        resources.assert_not_called()
        retrieve.assert_not_called()
        groq.assert_not_called()

    def test_backward_compatible_call_returns_string(self):
        with patch.object(rag, "load_resources", return_value=(object(), object(), [])), patch.object(
            rag, "retrieve_schemes", return_value=[]
        ):
            answer = rag.ask_schemesathi("What schemes are available for farmers?")
        self.assertIsInstance(answer, str)

    def test_complete_named_scheme_question_is_valid(self):
        (answer, status), resources, retrieve, groq = self.call(
            "What documents are required for PM-KISAN?"
        )
        self.assertEqual(status, "valid_scheme_answer")
        resources.assert_called_once()
        retrieve.assert_called_once()
        groq.assert_called_once()

    def test_legacy_status_defaults_to_error(self):
        self.assertEqual(rag.normalize_status(None), "error")
        self.assertEqual(rag.normalize_status("response_status"), "error")

    def test_backend_failure_returns_safe_error_status(self):
        with patch.object(rag, "load_resources", side_effect=RuntimeError("backend failed")):
            answer, status = rag.ask_schemesathi(
                "What schemes are available for farmers?", return_status=True
            )
        self.assertEqual(answer, rag.ERROR_MESSAGE)
        self.assertEqual(status, "error")


if __name__ == "__main__":
    unittest.main()