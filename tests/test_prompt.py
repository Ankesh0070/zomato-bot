"""
Unit tests for the recommendation engine: prompt builder, parser, and engine.

Tests cover:
- Prompt construction and token estimation
- JSON extraction from various LLM output formats
- Schema adaptation (alternative key names)
- Hallucination detection
- Single recommendation parsing
- Rule-based fallback
- Response caching
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.schema import CleanColumns, RawColumns
from src.data.preprocessing import preprocess
from src.input.models import UserPreferences
from src.recommendation.prompt_builder import (
    build_prompt,
    build_retry_prompt,
    estimate_prompt_tokens,
    SYSTEM_PROMPT,
)
from src.recommendation.parser import (
    ParseError,
    Recommendation,
    RecommendationResult,
    parse_response,
    _extract_json,
    _find_recommendations_array,
    _parse_single_recommendation,
)
from src.recommendation.engine import (
    _cache_key,
    _rule_based_fallback,
    clear_cache,
)


# ===================================================================
# Test Fixtures
# ===================================================================

def _make_sample_candidates() -> pd.DataFrame:
    """Create a small preprocessed DataFrame for testing."""
    raw_df = pd.DataFrame({
        RawColumns.RESTAURANT_NAME: [
            "La Piazza", "Olive Garden", "Sushi House",
            "Curry Palace", "Dragon Wok",
        ],
        RawColumns.ONLINE_ORDER: ["Yes", "Yes", "No", "Yes", "Yes"],
        RawColumns.BOOK_TABLE: ["Yes", "No", "Yes", "No", "No"],
        RawColumns.RATING: ["4.5/5", "4.3/5", "4.7/5", "4.0/5", "3.8/5"],
        RawColumns.VOTES: [1245, 890, 320, 560, 200],
        RawColumns.LOCATION: ["Delhi", "Delhi", "Delhi", "Delhi", "Delhi"],
        RawColumns.REST_TYPE: ["Casual Dining"] * 5,
        RawColumns.CUISINES: [
            "Italian, Continental",
            "Italian",
            "Japanese, Sushi",
            "North Indian, Mughlai",
            "Chinese, Thai",
        ],
        RawColumns.COST_FOR_TWO: ["800", "650", "1200", "500", "400"],
        RawColumns.LISTED_IN_TYPE: ["Dine-out"] * 5,
        RawColumns.LISTED_IN_CITY: ["Delhi"] * 5,
    })
    return preprocess(raw_df)


def _make_valid_llm_response() -> str:
    """Create a valid JSON LLM response string."""
    return json.dumps({
        "recommendations": [
            {
                "rank": 1,
                "restaurant_name": "Sushi House",
                "cuisine": "Japanese, Sushi",
                "rating": 4.7,
                "estimated_cost": "Rs.1200 for two",
                "explanation": "Highest rated restaurant with excellent sushi options."
            },
            {
                "rank": 2,
                "restaurant_name": "La Piazza",
                "cuisine": "Italian, Continental",
                "rating": 4.5,
                "estimated_cost": "Rs.800 for two",
                "explanation": "Great Italian dining with high ratings."
            },
            {
                "rank": 3,
                "restaurant_name": "Olive Garden",
                "cuisine": "Italian",
                "rating": 4.3,
                "estimated_cost": "Rs.650 for two",
                "explanation": "Solid Italian option at a reasonable price."
            },
        ],
        "summary": "Top 3 restaurants in Delhi based on your preferences."
    })


@pytest.fixture
def candidates_df():
    return _make_sample_candidates()


@pytest.fixture
def user_prefs():
    return UserPreferences(
        location="Delhi",
        budget="medium",
        cuisine="Italian",
        min_rating=4.0,
        preferences=["family-friendly"],
    )


# ===================================================================
# Prompt Builder Tests
# ===================================================================

class TestPromptBuilder:
    """Tests for prompt construction."""

    def test_build_prompt_returns_both_parts(self):
        prompt = build_prompt(
            user_context="Location: Delhi\nBudget: medium",
            restaurant_data="1. La Piazza | Italian | 4.5/5",
        )
        assert "system_prompt" in prompt
        assert "user_prompt" in prompt

    def test_system_prompt_contains_json_instruction(self):
        assert "JSON" in SYSTEM_PROMPT
        assert "recommendations" in SYSTEM_PROMPT

    def test_user_prompt_contains_context(self):
        prompt = build_prompt(
            user_context="Location: Delhi",
            restaurant_data="1. La Piazza | Italian",
        )
        assert "Delhi" in prompt["user_prompt"]
        assert "La Piazza" in prompt["user_prompt"]

    def test_user_prompt_contains_instructions(self):
        prompt = build_prompt(
            user_context="Location: Delhi",
            restaurant_data="1. La Piazza",
            max_recommendations=3,
        )
        assert "top 3" in prompt["user_prompt"]

    def test_token_estimation(self):
        prompt = build_prompt(
            user_context="Location: Delhi",
            restaurant_data="1. La Piazza | Italian | 4.5",
        )
        tokens = estimate_prompt_tokens(prompt)
        assert tokens > 0
        assert tokens < 5000  # Should be well under context limit

    def test_retry_prompt_appends_error(self):
        original = build_prompt("context", "data")
        retry = build_retry_prompt(original, "Invalid JSON")
        assert "Invalid JSON" in retry["user_prompt"]
        assert "IMPORTANT" in retry["user_prompt"]
        # System prompt should be unchanged
        assert retry["system_prompt"] == original["system_prompt"]


# ===================================================================
# JSON Extraction Tests
# ===================================================================

class TestJSONExtraction:
    """Tests for extracting JSON from various LLM output formats."""

    def test_clean_json(self):
        raw = '{"recommendations": [], "summary": "test"}'
        result = _extract_json(raw)
        assert result["summary"] == "test"

    def test_json_in_markdown_fences(self):
        raw = '```json\n{"recommendations": [], "summary": "test"}\n```'
        result = _extract_json(raw)
        assert result["summary"] == "test"

    def test_json_in_plain_fences(self):
        raw = '```\n{"recommendations": []}\n```'
        result = _extract_json(raw)
        assert "recommendations" in result

    def test_json_with_preamble(self):
        raw = 'Here are the results:\n{"recommendations": [], "summary": "ok"}'
        result = _extract_json(raw)
        assert result["summary"] == "ok"

    def test_json_with_trailing_text(self):
        raw = '{"recommendations": []}\nHope this helps!'
        result = _extract_json(raw)
        assert "recommendations" in result

    def test_json_array_wrapped(self):
        raw = '[{"rank": 1, "name": "Test"}]'
        result = _extract_json(raw)
        # Should either be wrapped in a dict or returned as-is
        # The downstream _find_recommendations_array handles both
        if isinstance(result, dict):
            assert "recommendations" in result
        else:
            # If returned as list, verify it's a valid list of dicts
            assert isinstance(result, list)
            assert len(result) == 1

    def test_invalid_json_raises(self):
        with pytest.raises(ParseError):
            _extract_json("This is not JSON at all.")

    def test_empty_string_raises(self):
        with pytest.raises(ParseError):
            _extract_json("")


# ===================================================================
# Schema Adaptation Tests
# ===================================================================

class TestSchemaAdaptation:
    """Tests for handling alternative key names in LLM responses."""

    def test_standard_key(self):
        data = {"recommendations": [{"name": "Test"}]}
        result = _find_recommendations_array(data)
        assert len(result) == 1

    def test_alternative_key_results(self):
        data = {"results": [{"name": "Test"}]}
        result = _find_recommendations_array(data)
        assert len(result) == 1

    def test_alternative_key_restaurants(self):
        data = {"restaurants": [{"name": "Test"}]}
        result = _find_recommendations_array(data)
        assert len(result) == 1

    def test_list_input(self):
        data = [{"name": "Test"}]
        result = _find_recommendations_array(data)
        assert len(result) == 1

    def test_empty_dict(self):
        result = _find_recommendations_array({})
        assert result == []

    def test_non_standard_key_with_list(self):
        data = {"picks": [{"name": "Test"}]}
        result = _find_recommendations_array(data)
        assert len(result) == 1


# ===================================================================
# Single Recommendation Parsing Tests
# ===================================================================

class TestSingleRecommendation:
    """Tests for parsing individual recommendation dicts."""

    def test_standard_fields(self):
        item = {
            "rank": 1,
            "restaurant_name": "La Piazza",
            "cuisine": "Italian",
            "rating": 4.5,
            "estimated_cost": "Rs.800 for two",
            "explanation": "Great Italian dining.",
        }
        rec = _parse_single_recommendation(item, 1)
        assert rec.restaurant_name == "La Piazza"
        assert rec.rating == 4.5
        assert rec.rank == 1

    def test_alternative_field_names(self):
        item = {
            "name": "La Piazza",
            "cuisines": "Italian",
            "rate": 4.5,
            "cost": "Rs.800",
            "reason": "Good food.",
        }
        rec = _parse_single_recommendation(item, 1)
        assert rec.restaurant_name == "La Piazza"
        assert rec.cuisine == "Italian"
        assert rec.explanation == "Good food."

    def test_missing_fields_use_defaults(self):
        item = {"name": "Test"}
        rec = _parse_single_recommendation(item, 3)
        assert rec.restaurant_name == "Test"
        assert rec.rank == 3
        assert rec.cuisine == "N/A"
        assert rec.explanation == "No explanation provided."

    def test_non_dict_raises(self):
        with pytest.raises(ValueError):
            _parse_single_recommendation("not a dict", 1)


# ===================================================================
# Full Response Parsing Tests
# ===================================================================

class TestResponseParsing:
    """Tests for end-to-end response parsing."""

    def test_valid_response(self, candidates_df):
        raw = _make_valid_llm_response()
        result = parse_response(raw, candidates_df)
        assert len(result.recommendations) == 3
        assert result.summary != ""

    def test_response_with_fences(self, candidates_df):
        raw = f"```json\n{_make_valid_llm_response()}\n```"
        result = parse_response(raw, candidates_df)
        assert len(result.recommendations) > 0

    def test_truncation_to_max(self, candidates_df):
        # Create response with 10 recommendations
        recs = [
            {
                "rank": i,
                "restaurant_name": f"Restaurant {i}",
                "cuisine": "Test",
                "rating": 4.0,
                "estimated_cost": "Rs.500",
                "explanation": "Test",
            }
            for i in range(1, 11)
        ]
        raw = json.dumps({"recommendations": recs, "summary": "test"})
        result = parse_response(raw, max_recommendations=5)
        assert len(result.recommendations) == 5

    def test_ranks_reassigned(self, candidates_df):
        raw = _make_valid_llm_response()
        result = parse_response(raw, candidates_df)
        ranks = [r.rank for r in result.recommendations]
        assert ranks == [1, 2, 3]

    def test_invalid_json_raises(self):
        with pytest.raises(ParseError):
            parse_response("This is not valid at all.")

    def test_empty_recommendations_raises(self):
        raw = json.dumps({"recommendations": [], "summary": "Nothing"})
        with pytest.raises(ParseError):
            parse_response(raw)


# ===================================================================
# Hallucination Detection Tests
# ===================================================================

class TestHallucinationCheck:
    """Tests for detecting hallucinated restaurants."""

    def test_valid_restaurants_kept(self, candidates_df):
        raw = _make_valid_llm_response()
        result = parse_response(raw, candidates_df)
        # All three are in the candidates
        assert len(result.recommendations) == 3

    def test_hallucinated_restaurant_removed(self, candidates_df):
        response = {
            "recommendations": [
                {
                    "rank": 1,
                    "restaurant_name": "Invented Restaurant XYZ",
                    "cuisine": "Fusion",
                    "rating": 5.0,
                    "estimated_cost": "Rs.2000",
                    "explanation": "This restaurant doesn't exist.",
                },
                {
                    "rank": 2,
                    "restaurant_name": "La Piazza",
                    "cuisine": "Italian",
                    "rating": 4.5,
                    "estimated_cost": "Rs.800",
                    "explanation": "Real restaurant.",
                },
            ],
            "summary": "Test",
        }
        raw = json.dumps(response)
        result = parse_response(raw, candidates_df)
        names = [r.restaurant_name for r in result.recommendations]
        assert "Invented Restaurant XYZ" not in names
        assert any("Piazza" in n for n in names)

    def test_minor_name_variation_corrected(self, candidates_df):
        """LLM might slightly modify names — should still match."""
        response = {
            "recommendations": [
                {
                    "rank": 1,
                    "restaurant_name": "La piazza",  # lowercase
                    "cuisine": "Italian",
                    "rating": 4.5,
                    "estimated_cost": "Rs.800",
                    "explanation": "Test.",
                },
            ],
            "summary": "Test",
        }
        raw = json.dumps(response)
        result = parse_response(raw, candidates_df)
        assert len(result.recommendations) == 1


# ===================================================================
# Rule-Based Fallback Tests
# ===================================================================

class TestRuleBasedFallback:
    """Tests for the rule-based fallback when LLM is unavailable."""

    def test_fallback_returns_results(self, candidates_df):
        result = _rule_based_fallback(candidates_df)
        assert len(result.recommendations) > 0
        assert result.summary != ""

    def test_fallback_respects_max(self, candidates_df):
        result = _rule_based_fallback(candidates_df)
        assert len(result.recommendations) <= 5

    def test_fallback_has_explanations(self, candidates_df):
        result = _rule_based_fallback(candidates_df)
        for rec in result.recommendations:
            assert rec.explanation != ""
            assert rec.restaurant_name != ""

    def test_fallback_mentions_rule_based(self, candidates_df):
        result = _rule_based_fallback(candidates_df)
        assert "rule-based" in result.summary.lower()


# ===================================================================
# Cache Tests
# ===================================================================

class TestCache:
    """Tests for response caching."""

    def test_cache_key_deterministic(self):
        prefs1 = UserPreferences(location="Delhi", budget="medium")
        prefs2 = UserPreferences(location="Delhi", budget="medium")
        assert _cache_key(prefs1) == _cache_key(prefs2)

    def test_cache_key_differs(self):
        prefs1 = UserPreferences(location="Delhi")
        prefs2 = UserPreferences(location="Mumbai")
        assert _cache_key(prefs1) != _cache_key(prefs2)

    def test_clear_cache(self):
        clear_cache()
        # Should not raise


# ===================================================================
# Run tests
# ===================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
