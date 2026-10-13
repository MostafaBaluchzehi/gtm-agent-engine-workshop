import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service
from gtm_agent.gtm_records import OFFERINGS, PROSPECTS


class UpdateProspectInfoTests(unittest.TestCase):
    prospect_id = "TEST-TECH-STACK"

    def setUp(self):
        PROSPECTS[self.prospect_id] = {
            "name": "Test Prospect",
            "email": "test@example.com",
            "tech_stack": ["AWS"],
            "engagement_history": [],
            "account_details": {},
        }
        data_service._PROFILES.pop(self.prospect_id, None)

    def tearDown(self):
        PROSPECTS.pop(self.prospect_id, None)
        data_service._PROFILES.pop(self.prospect_id, None)

    def test_addition_is_persisted_and_returned(self):
        result = data_service.update_prospect_info(self.prospect_id, "Terraform")

        self.assertTrue(result["updated"])
        self.assertEqual(result["tech_stack"], ["AWS", "Terraform"])
        self.assertEqual(PROSPECTS[self.prospect_id]["tech_stack"], ["AWS", "Terraform"])

    def test_update_invalidates_cached_profile(self):
        with patch("deepagents.create_deep_agent"), patch("langchain_openai.ChatOpenAI") as chat_openai:
            chat_openai.return_value.with_structured_output.return_value = SimpleNamespace(invoke=Mock())
            from gtm_agent.gtm_agent import build_prospect_profile

        cached = build_prospect_profile.func(self.prospect_id)["prospect_profile"]
        self.assertEqual(cached["tech_stack"], ["AWS"])

        data_service.update_prospect_info(self.prospect_id, "Terraform")

        self.assertIsNone(data_service.get_profile_from_db(self.prospect_id)["prospect_profile"])
        refreshed = build_prospect_profile.func(self.prospect_id)["prospect_profile"]
        self.assertEqual(refreshed["tech_stack"], ["AWS", "Terraform"])

    def test_missing_prospect_is_not_updated(self):
        result = data_service.update_prospect_info("MISSING-LEAD", "Terraform")

        self.assertEqual(result, {"updated": False, "found": False})
        self.assertNotIn("MISSING-LEAD", PROSPECTS)

    def test_rescore_uses_added_required_technology(self):
        with patch("deepagents.create_deep_agent"), patch("langchain_openai.ChatOpenAI") as chat_openai:
            chat_openai.return_value.with_structured_output.return_value = SimpleNamespace(invoke=Mock())
            from gtm_agent.gtm_agent import score_prospect

        data_service.update_prospect_info(self.prospect_id, "Terraform")
        offering = OFFERINGS["OFFER-10004"]
        observed = {}

        def score(messages):
            profile_text = messages[1]["content"].split("\n\nProspect profile:\n", 1)[1]
            profile = json.loads(profile_text)
            observed["tech_stack"] = profile["tech_stack"]
            missing = [
                technology for technology in offering["required_tech_stack"]
                if technology not in profile["tech_stack"]
            ]
            justification = "Terraform is present."
            if missing:
                justification += " Missing required technologies: " + ", ".join(missing) + "."
            return SimpleNamespace(model_dump=lambda: {
                "score": 70,
                "justification": justification,
            })

        with patch("gtm_agent.gtm_agent._scoring_llm.invoke", side_effect=score):
            result = score_prospect.func(
                {"prospect_id": self.prospect_id, "tech_stack": ["AWS"]},
                offering,
            )

        self.assertEqual(observed["tech_stack"], ["AWS", "Terraform"])
        self.assertIn("Terraform is present", result["justification"])
        self.assertNotIn("Terraform", result["justification"].split("Missing required technologies:")[-1])
        self.assertIn("Kubernetes", result["justification"])


if __name__ == "__main__":
    unittest.main()
