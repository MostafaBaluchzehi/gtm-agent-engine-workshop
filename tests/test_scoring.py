import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent import gtm_agent


class ScoreProspectTests(unittest.TestCase):
    def setUp(self):
        self.profile = {
            "annual_revenue": 80,
            "tech_stack": ["Snowflake", "Tableau", "dbt"],
        }
        self.offering = {
            "required_tech_stack": ["Snowflake", "Tableau", "dbt", "Kafka"],
            "min_annual_revenue": 100,
            "description": "An offering for the enterprise segment.",
        }
        self.assessment = SimpleNamespace(
            segment_fit=80,
            justification="The prospect fits the segment and has three required technologies.",
        )

    def score(self, profile=None, offering=None):
        with patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.return_value = self.assessment
            return gtm_agent.score_prospect.func(
                profile if profile is not None else self.profile,
                offering if offering is not None else self.offering,
            )

    def test_identical_inputs_return_identical_scores(self):
        first = self.score()
        second = self.score()
        self.assertEqual(first, second)

    def test_three_of_four_technologies_match_is_75_percent(self):
        result = self.score()
        self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 75)

    def test_revenue_fit_is_capped_at_100(self):
        profile = {**self.profile, "annual_revenue": 200}
        result = self.score(profile=profile)
        self.assertEqual(result["rubric_breakdown"]["revenue_fit"], 100)

    def test_score_is_documented_weighted_average(self):
        result = self.score()
        rubric = result["rubric_breakdown"]
        expected = sum(
            rubric[component] * weight
            for component, weight in gtm_agent.SCORING_WEIGHTS.items()
        )
        self.assertEqual(result["score"], expected)

    def test_empty_technology_requirements_have_full_fit(self):
        offering = {**self.offering, "required_tech_stack": []}
        result = self.score(offering=offering)
        self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 100)

    def test_missing_or_nonpositive_revenue_threshold_has_full_fit(self):
        for threshold in (None, 0, -1):
            with self.subTest(threshold=threshold):
                offering = {**self.offering, "min_annual_revenue": threshold}
                result = self.score(offering=offering)
                self.assertEqual(result["rubric_breakdown"]["revenue_fit"], 100)

    def test_llm_supplies_justification_and_segment_fit(self):
        result = self.score()
        self.assertEqual(result["justification"], self.assessment.justification)
        self.assertEqual(result["rubric_breakdown"]["segment_fit"], 80)


if __name__ == "__main__":
    unittest.main()
