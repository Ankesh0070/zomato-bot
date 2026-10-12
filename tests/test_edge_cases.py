import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import pandas as pd

from src.data.schema import CleanColumns
from src.input.models import UserPreferences
from src.recommendation.engine import get_recommendations
from src.recommendation.llm_client import BaseLLMClient


class BadClient(BaseLLMClient):
    def __init__(self):
        super().__init__(model="bad-model", temperature=0.0, max_tokens=1000)

    def generate(self, user_prompt: str, system_prompt: str = "") -> str:
        # Return invalid (non-JSON) response to force parse failure
        return "This is not JSON and will cause the parser to fail"

    def is_available(self) -> bool:
        return True


def test_no_results_returns_graceful_message():
    empty_df = pd.DataFrame(columns=[
        CleanColumns.RESTAURANT_NAME,
        CleanColumns.LOCATION,
        CleanColumns.RATING,
        CleanColumns.COST_FOR_TWO,
    ])

    prefs = UserPreferences(location="Nowhere")
    result = get_recommendations(empty_df, prefs, use_cache=False)

    assert result["candidates_found"] == 0
    assert result["model_used"] == "none"
    assert "No restaurants found" in result["result"].summary


def test_llm_parse_failure_triggers_rule_based(monkeypatch):
    # Create a small candidates dataframe
    df = pd.DataFrame([
        {
            CleanColumns.RESTAURANT_NAME: "RB1",
            CleanColumns.LOCATION: "TestCity",
            CleanColumns.RATING: 4.0,
            CleanColumns.COST_FOR_TWO: 600,
            CleanColumns.CUISINES: "Indian",
            CleanColumns.CUISINES_LIST: ["Indian"],
            CleanColumns.BUDGET_TIER: "medium",
        }
    ])

    prefs = UserPreferences(location="NonExistentCity", budget=1000, party_size=2)

    # Patch LLM to return invalid content
    import src.recommendation.engine as engine_module
    monkeypatch.setattr(engine_module, "create_llm_client", lambda: BadClient())

    result = get_recommendations(df, prefs, use_cache=False)

    assert result["model_used"] == "rule-based"
    assert result["result"].recommendations


def test_preference_collector_with_party_size_and_budget():
    from src.input.collector import PreferenceCollector

    df = pd.DataFrame([
        {
            CleanColumns.RESTAURANT_NAME: "Resto",
            CleanColumns.LOCATION: "Delhi",
            CleanColumns.RATING: 4.2,
            CleanColumns.COST_FOR_TWO: 800,
            CleanColumns.CUISINES_LIST: ["North Indian"],
        }
    ])
    collector = PreferenceCollector(df)
    prefs, warnings = collector.collect(
        location="Delhi",
        budget="medium",
        party_size=4,
        cuisine="North Indian",
    )
    assert prefs.location == "Delhi"
    assert prefs.budget == 1000
    assert prefs.party_size == 4
    assert prefs.cuisine == "North Indian"


def test_parse_votes_comma_and_malformed_values():
    from src.data.preprocessing import _parse_vote_value, _parse_votes

    assert _parse_vote_value("1,245") == 1245
    assert _parse_vote_value(" 500 ") == 500
    assert _parse_vote_value("NEW") == 0
    assert _parse_vote_value(None) == 0
    assert _parse_vote_value(-10) == 0

    df = pd.DataFrame({
        CleanColumns.VOTES: ["1,000", "95", "NEW", 200]
    })
    cleaned = _parse_votes(df)
    assert cleaned[CleanColumns.VOTES].tolist() == [1000, 95, 0, 200]


def test_deduplication_sorts_by_votes_numerically():
    from src.data.preprocessing import preprocess
    from src.data.schema import RawColumns

    raw_df = pd.DataFrame({
        RawColumns.RESTAURANT_NAME: ["Place A", "Place A"],
        RawColumns.ONLINE_ORDER: ["Yes", "Yes"],
        RawColumns.BOOK_TABLE: ["No", "No"],
        RawColumns.RATING: ["4.0/5", "4.0/5"],
        RawColumns.VOTES: ["95", "1,000"],  # '95' is lexicographically > '1,000'
        RawColumns.LOCATION: ["Delhi", "Delhi"],
        RawColumns.REST_TYPE: ["Cafe", "Cafe"],
        RawColumns.CUISINES: ["Italian", "Italian"],
        RawColumns.COST_FOR_TWO: ["500", "500"],
        RawColumns.LISTED_IN_TYPE: ["Dine-out", "Dine-out"],
        RawColumns.LISTED_IN_CITY: ["Delhi", "Delhi"],
    })
    clean = preprocess(raw_df)
    assert len(clean) == 1
    # Should keep the one with 1000 votes, not 95
    assert clean.iloc[0][CleanColumns.VOTES] == 1000


def test_cache_key_order_independence_for_preferences():
    from src.recommendation.engine import _cache_key

    prefs1 = UserPreferences(location="Delhi", preferences=["romantic", "outdoor seating"])
    prefs2 = UserPreferences(location="Delhi", preferences=["outdoor seating", "romantic"])
    assert _cache_key(prefs1) == _cache_key(prefs2)


def test_cache_preserves_candidate_count_on_hit():
    import src.recommendation.engine as engine_module
    from src.recommendation.engine import clear_cache, get_recommendations

    class FastDummy(BaseLLMClient):
        def __init__(self):
            super().__init__(model="fast", temperature=0.0, max_tokens=100)
        def generate(self, user_prompt, system_prompt=""):
            return json.dumps({
                "recommendations": [{
                    "rank": 1,
                    "restaurant_name": "R1",
                    "cuisine": "Indian",
                    "rating": 4.5,
                    "estimated_cost": "Rs.500",
                    "explanation": "Good"
                }],
                "summary": "Summary"
            })
        def is_available(self):
            return True

    df = pd.DataFrame([{
        CleanColumns.RESTAURANT_NAME: "R1",
        CleanColumns.LOCATION: "TestCity",
        CleanColumns.RATING: 4.5,
        CleanColumns.COST_FOR_TWO: 500,
        CleanColumns.CUISINES: "Indian",
        CleanColumns.CUISINES_LIST: ["Indian"],
        CleanColumns.BUDGET_TIER: "medium",
    }])
    prefs = UserPreferences(location="TestCity")

    clear_cache()
    # Mock LLM
    original_creator = engine_module.create_llm_client
    engine_module.create_llm_client = lambda: FastDummy()
    try:
        res1 = get_recommendations(df, prefs, use_cache=True)
        assert res1["candidates_found"] == 1
        assert res1["cached"] is False

        res2 = get_recommendations(df, prefs, use_cache=True)
        assert res2["cached"] is True
        assert res2["candidates_found"] == 1  # Should not be 0!
    finally:
        engine_module.create_llm_client = original_creator


def test_budget_filter_with_tier_string():
    from src.integration.filter import apply_filters

    df = pd.DataFrame([
        {
            CleanColumns.RESTAURANT_NAME: "CheapEats",
            CleanColumns.LOCATION: "Delhi",
            CleanColumns.RATING: 4.0,
            CleanColumns.COST_FOR_TWO: 300,
            CleanColumns.CUISINES_LIST: ["Indian"],
            CleanColumns.BUDGET_TIER: "low",
        },
        {
            CleanColumns.RESTAURANT_NAME: "FancyDining",
            CleanColumns.LOCATION: "Delhi",
            CleanColumns.RATING: 4.5,
            CleanColumns.COST_FOR_TWO: 1800,
            CleanColumns.CUISINES_LIST: ["Indian"],
            CleanColumns.BUDGET_TIER: "high",
        }
    ])
    # Pass budget="low" directly in UserPreferences
    prefs = UserPreferences(location="Delhi", budget="low", party_size=2)
    filtered, metadata = apply_filters(df, prefs, min_results=1)
    assert len(filtered) == 1
    assert filtered.iloc[0][CleanColumns.RESTAURANT_NAME] == "CheapEats"


def test_validate_budget_tier_names():
    from src.input.validator import _validate_budget

    budget, warnings = _validate_budget("low")
    assert budget == 400
    assert len(warnings) == 0

    budget, warnings = _validate_budget("medium")
    assert budget == 1000
    assert len(warnings) == 0

    budget, warnings = _validate_budget("high")
    assert budget == 2500
    assert len(warnings) == 0

