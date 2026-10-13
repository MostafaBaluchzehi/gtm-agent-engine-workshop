import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import gtm_agent
from gtm_agent import data_service


class ProspectPrivacyTests(unittest.TestCase):
    def test_get_prospect_returns_only_safe_fields(self):
        result = gtm_agent.get_prospect.func("LEAD-12853")

        self.assertTrue(result["found"])
        self.assertEqual(
            set(result["prospect"]),
            {
                "prospect_id",
                "name",
                "email",
                "disqualified",
                "annual_revenue",
                "enrichment_source",
            },
        )
        self.assertNotIn("billing_qualification", result["prospect"])

    def test_profile_cache_lookup_sanitizes_cached_record(self):
        cached_profile = {
            "prospect_id": "LEAD-CACHE",
            "billing_qualification": {"tax_id": "sensitive"},
            "annual_revenue": 90000000,
        }

        with patch.dict(data_service._PROFILES, {"LEAD-CACHE": cached_profile}):
            result = data_service.get_profile_from_db("LEAD-CACHE")
            self.assertNotIn("billing_qualification", data_service._PROFILES["LEAD-CACHE"])

        self.assertNotIn("billing_qualification", result["prospect_profile"])

    def test_built_and_cached_profiles_exclude_billing_qualification(self):
        with patch.object(data_service, "get_profile_from_db", return_value={"prospect_profile": None}), \
                patch.object(data_service, "save_profile_to_db") as save_profile:
            built = gtm_agent.build_prospect_profile.func("LEAD-12853")

        self.assertTrue(built["found"])
        self.assertNotIn("billing_qualification", built["prospect_profile"])
        self.assertIn("annual_revenue", built["prospect_profile"])
        self.assertIn("tech_stack", built["prospect_profile"])
        self.assertNotIn("billing_qualification", save_profile.call_args.args[1])

        cached_profile = {
            "prospect_id": "LEAD-12853",
            "name": "Omar Okafor",
            "email": "omar.okafor@lakesideanalytics.com",
            "billing_qualification": {"tax_id": "sensitive"},
        }
        with patch.object(
            data_service,
            "get_profile_from_db",
            return_value={"prospect_profile": cached_profile},
        ), patch.object(data_service, "save_profile_to_db") as save_cached_profile:
            cached = gtm_agent.build_prospect_profile.func("LEAD-12853")

        self.assertTrue(cached["found"])
        self.assertNotIn("billing_qualification", cached["prospect_profile"])
        self.assertNotIn("billing_qualification", save_cached_profile.call_args.args[1])

    def test_scoring_prompt_contains_only_rubric_prospect_fields(self):
        prospect_profile = {
            "prospect_id": "LEAD-12853",
            "name": "Omar Okafor",
            "email": "omar.okafor@lakesideanalytics.com",
            "billing_qualification": {"tax_id": "sensitive"},
            "annual_revenue": 90000000,
            "account_details": [{"industry": "Software", "sub_industry": "Analytics Platforms"}],
            "tech_stack": ["Snowflake"],
        }
        offering = {
            "required_tech_stack": ["Snowflake"],
            "min_annual_revenue": 50000000,
            "description": "A test offering.",
        }
        score = Mock()
        score.model_dump.return_value = {"score": 90}

        with patch.object(data_service, "fetch_tech_stack", return_value=["Snowflake", "Kafka"]), \
                patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.return_value = score
            result = gtm_agent.score_prospect.func(prospect_profile, offering)

        self.assertEqual(result, {"score": 90})
        messages = scoring_llm.invoke.call_args.args[0]
        prompt_profile = json.loads(messages[1]["content"].split("\n\nProspect profile:\n", 1)[1])
        self.assertEqual(set(prompt_profile), {"annual_revenue", "tech_stack", "account_details"})
        self.assertEqual(prompt_profile["annual_revenue"], 90000000)
        self.assertEqual(prompt_profile["tech_stack"], ["Snowflake", "Kafka"])
        self.assertNotIn("billing_qualification", messages[1]["content"])
        self.assertNotIn("omar.okafor@lakesideanalytics.com", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
